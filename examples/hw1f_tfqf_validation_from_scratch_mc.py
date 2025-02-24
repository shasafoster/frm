# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for validating Hull-White One Factor model swaption pricing 
against Google's TF Quant Finance implementation.

https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/hull_white/zero_coupon_bond_option_test.py

"""

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pricing of zero coupon bond options using Hull-White model."""

from typing import Callable, Union
import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf

import tf_quant_finance as tff
from tf_quant_finance import types
from tf_quant_finance import utils as tff_utils
from tf_quant_finance.math import random
from tf_quant_finance.models import utils
from tf_quant_finance.models.hjm import zero_coupon_bond_option_util
from tf_quant_finance.models.hull_white import one_factor


def _ncdf(x):
  """Implements the cumulative normal distribution function."""
  return (tf.math.erf(x / _SQRT_2) + 1) / 2


_SQRT_2 = np.sqrt(2.0, dtype=np.float64)


expiries = 1.0
maturities = 5.0
strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
is_call_options = True
mean_reversion = 0.03
volatility = 0.02
num_samples = 500000
time_step = 0.1
random_type = tff.math.random.RandomType.PSEUDO_ANTITHETIC
seed = 1
skip = 0


def discount_rate_fn(t):
    return 0.01 * tf.ones_like(t)

dtype=tf.float64
if dtype is None:
    dtype = tf.convert_to_tensor([0.0]).dtype


strikes = tf.convert_to_tensor(strikes, dtype=dtype, name='strikes')
expiries = tf.convert_to_tensor(expiries, dtype=dtype, name='expiries')
maturities = tf.convert_to_tensor(maturities, dtype=dtype,
                                  name='maturities')
is_call_options = tf.convert_to_tensor(is_call_options, dtype=tf.bool,
                                       name='is_call_options')

model = one_factor.HullWhiteModel1F(
    mean_reversion=mean_reversion,
    volatility=volatility,
    initial_discount_rate_fn=discount_rate_fn,
    dtype=dtype)


if time_step is None:
  raise ValueError('`time_step` must be provided for simulation '
                   'based bond option valuation.')

# def sample_discount_curve_paths_fn(times, curve_times, num_samples):
#   return model.sample_discount_curve_paths(
#       times=times,
#       curve_times=curve_times,
#       num_samples=num_samples,
#       random_type=random_type,
#       seed=seed,
#       skip=skip)

# Shape batch_shape + [1]
# prices = zero_coupon_bond_option_util.options_price_from_samples(
#     strikes, expiries, maturities, is_call_options,
#     sample_discount_curve_paths_fn, num_samples,
#     time_step, dtype=dtype)
# Shape batch_shape

def _prepare_indices(idx0, idx1, idx2, idx3):
  """Prepare indices to get relevant slice from discount curve simulations."""
  len0 = idx0.shape.as_list()[0]
  len1 = idx1.shape.as_list()[0]
  len3 = idx3.shape.as_list()[0]
  idx0 = tf.repeat(idx0, len1 * len3)
  idx1 = tf.tile(tf.repeat(idx1, len3), [len0])
  idx2 = tf.tile(tf.repeat(idx2, len3), [len0])
  idx3 = tf.tile(idx3, [len0 * len1])
  return tf.stack([idx0, idx1, idx2, idx3], axis=-1)


sim_times, _ = tf.unique(tf.reshape(expiries, shape=[-1]))
longest_expiry = tf.reduce_max(sim_times)
sim_times, _ = tf.unique(
    tf.concat(
        [sim_times, tf.range(time_step, longest_expiry, time_step)],
        axis=0))
sim_times = tf.sort(sim_times, name='sort_sim_times')
tau = maturities - expiries
curve_times_builder, _ = tf.unique(tf.reshape(tau, shape=[-1]))
curve_times = tf.sort(curve_times_builder, name='sort_curve_times')

#
# p_t_tau, r_t = model.sample_discount_curve_paths(
#       times=sim_times,
#       curve_times=curve_times,
#       num_samples=num_samples,
#       random_type=random_type,
#       seed=seed,
#       skip=skip)

if not model._is_piecewise_constant:
    raise ValueError('All paramaters `mean_reversion`, `volatility`, and '
                     '`corr_matrix`must be piecewise constant functions.')

times = tf.convert_to_tensor(sim_times, model._dtype, name='times') # [0.1, 0.2, ..., 0.9, 1.0]
times_grid = None
curve_times = tf.convert_to_tensor(curve_times, model._dtype, name='curve_times')
if times_grid is not None:
    times_grid = tf.convert_to_tensor(times_grid, model._dtype, name='times_grid')

mean_reversion = model._mean_reversion(times)
volatility = model._volatility(times)
# Shape [dim, num_sim_times]



def y_integral(t0, t, vol, k):
    """Computes int_t0^t sigma(u)^2 exp(2*k*u) du."""
    return (vol * vol) / (2 * k) * (
            tf.math.exp(2 * k * t) - tf.math.exp(2 * k * t0))

t = times
mr_t = mean_reversion
sigma_t = volatility

t = tf.broadcast_to(t, tf.concat([[model._dim], tf.shape(t)], axis=-1))
time_index = tf.searchsorted(model._jump_locations, t)
y_between_vol_knots = model._y_integral(
    model._padded_knots, model._jump_locations, model._jump_values_vol,
    model._jump_values_mr)
y_at_vol_knots = tf.concat(
    [model._zero_padding,
     utils.cumsum_using_matvec(y_between_vol_knots)], axis=1)

vn = tf.concat(
    [model._zero_padding, model._jump_locations], axis=1)
# y_t = model._y_integral(
#     tf.gather(vn, time_index, batch_dims=1), t, sigma_t, mr_t)
y_t = y_integral(tf.gather(vn, time_index, batch_dims=1), t, sigma_t, mr_t)

y_t = y_t + tf.gather(y_at_vol_knots, time_index, batch_dims=1)
y_t = tf.math.exp(-2 * mr_t * t) * y_t





rate_paths_ = model.sample_paths(
    times=times,
    num_samples=num_samples,
    random_type=random_type,
    skip=skip,
    time_step=time_step,
    times_grid=times_grid,
    normal_draws=None,
    validate_args=False,
    seed=seed)



#%%




#%%
# def conditional_mean_x(model, t, mr_t, sigma_t):
#     """Computes the drift term in [1], Eq. 10.39."""
#     # Shape [dim, num_times]
#     t = tf.broadcast_to(t, tf.concat([[model._dim], tf.shape(t)], axis=-1))
#     time_index = tf.searchsorted(model._jump_locations, t)
#     vn = tf.concat([model._zero_padding, model._jump_locations], axis=1)
#     y_between_vol_knots = model._y_integral(model._padded_knots,
#                                            model._jump_locations,
#                                            model._jump_values_vol,
#                                            model._jump_values_mr)
#
#     y_at_vol_knots = tf.concat(
#         [model._zero_padding,
#          utils.cumsum_using_matvec(y_between_vol_knots)], axis=1)
#
#     ex_between_vol_knots = model._ex_integral(model._padded_knots,
#                                              model._jump_locations,
#                                              model._jump_values_vol,
#                                              model._jump_values_mr,
#                                              y_at_vol_knots[:, :-1])
#
#     ex_at_vol_knots = tf.concat(
#         [model._zero_padding,
#          utils.cumsum_using_matvec(ex_between_vol_knots)], axis=1)
#
#     c = tf.gather(y_at_vol_knots, time_index, batch_dims=1)
#     exp_x_t = model._ex_integral(
#         tf.gather(vn, time_index, batch_dims=1), t, sigma_t, mr_t, c)
#     exp_x_t = exp_x_t + tf.gather(ex_at_vol_knots, time_index, batch_dims=1)
#     exp_x_t = (exp_x_t[:, 1:] - exp_x_t[:, :-1]) * tf.math.exp(
#         -tf.broadcast_to(mr_t, tf.shape(t))[:, 1:] * t[:, 1:])
#     return exp_x_t
#
# def ex_integral(t0, t, vol, k, y_t0):
#     """Function computes the integral for the drift calculation."""
#     # Computes int_t0^t (exp(k*s)*y(s)) ds,
#     # where y(s)=y(t0) + int_t0^s exp(-2*(s-u)) vol(u)^2 du."""
#     value = (np.exp(k * t) - np.exp(k * t0) + np.exp(2 * k * t0) * (np.exp(-k * t) - np.exp(-k * t0)))
#     value = value * vol**2 / (2 * k * k) + y_t0 * (np.exp(-k * t0) - np.exp(-k * t)) / k
#     return value
#
# def conditional_mean_x_scalar(t, mr_t, sigma_t):
#     exp_x_t = ex_integral(t0=0, t=t, vol=sigma_t, k=mr_t, y_t0=0)
#     exp_x_t = (exp_x_t[:, 1:] - exp_x_t[:, :-1]) * np.exp(-np.broadcast_to(mr_t, t.shape)[:, 1:] * t[:, 1:])
#     return exp_x_t
#
#
# t = times
# mr_t = mean_reversion
# sigma_t = volatility
#
# t = tf.broadcast_to(t, tf.concat([[model._dim], tf.shape(t)], axis=-1))
# var_x_between_vol_knots = model._variance_int(model._padded_knots,
#                                              model._jump_locations,
#                                              model._jump_values_vol,
#                                              model._jump_values_mr)
# varx_at_vol_knots = tf.concat(
#     [model._zero_padding,
#      utils.cumsum_using_matvec(var_x_between_vol_knots)],
#     axis=1)
#
# time_index = tf.searchsorted(model._jump_locations, t)
# vn = tf.concat(
#     [model._zero_padding,
#      model._jump_locations], axis=1)
#
#
# def variance_int(t0, t, vol, k):
#     """Computes int_t0^t exp(2*k*s) vol(s)^2 ds."""
#     return vol * vol / (2 * k) * (
#             tf.math.exp(2 * k * t) - tf.math.exp(2 * k * t0))
#
# var_x_t = variance_int(t0=0, t=t, vol=sigma_t, k=mr_t)
# var_x_t = (var_x_t[:, 1:] - var_x_t[:, :-1]) * tf.math.exp(
#     -2 * tf.broadcast_to(mr_t, tf.shape(t))[:, 1:] * t[:, 1:])



#%%

initial_x = tf.zeros((num_samples, model._dim), dtype=model._dtype)
f0_t = model._instant_forward_rate_fn(times[0])

normal_draws = None
corr_matrix_root = None
num_requested_times = sim_times.shape[0] # shape of 'sim_times'
record_samples = True

new_sim_times = tf.concat([tf.constant([0.0], dtype=tf.float64), sim_times], axis=0) # [0.0, 0.1, 0.2, ..., 1.0]

dt = new_sim_times[1:] - new_sim_times[:-1]
keep_mask = tf.constant([False] + [True] * len(sim_times)) # [False, True, True, ..., True]. 1 false, 10 trues
written_count = 0

mean_reversion = model._mean_reversion(new_sim_times)
volatility = model._volatility(new_sim_times)

exp_x_t = model._conditional_mean_x(new_sim_times, mean_reversion, volatility)
var_x_t = model._conditional_variance_x(new_sim_times, mean_reversion, volatility)

element_shape = initial_x.shape
rate_paths = tf.TensorArray(dtype=new_sim_times.dtype,
                            size=num_requested_times,
                            element_shape=element_shape,
                            clear_after_read=False)
# Include initial state, if necessary
rate_paths = rate_paths.write(written_count, initial_x + f0_t)


#%%

if dt.shape.is_fully_defined():
    steps_num = dt.shape.as_list()[-1]
else:
    steps_num = tf.shape(dt)[-1]


def cond_fn(i, written_count, *args):
    # It can happen that `times_grid[-1] > times[-1]` in which case we have
    # to terminate when `written_count` reaches `num_requested_times`
    del args
    return tf.math.logical_and(i < tf.size(dt), written_count < num_requested_times)


def body_fn(i, written_count, current_x, rate_paths):
    """Simulate hull-white process to the next time point."""
    if normal_draws is None:
        normals = random.mv_normal_sample(
            (num_samples,),
            mean=tf.zeros((model._dim,), dtype=mean_reversion.dtype),
            random_type=random_type, seed=seed)
    else:
        normals = normal_draws[i]

    if corr_matrix_root is not None:
        normals = tf.linalg.matvec(corr_matrix_root[i], normals)
    vol_x_t = tf.math.sqrt(tf.nn.relu(tf.transpose(var_x_t)[i]))
    # If numerically `vol_x_t == 0`, the gradient of `vol_x_t` becomes `NaN`.
    # To prevent this, we explicitly set `vol_x_t` to zero tensor at zero
    # values so that the gradient is set to zero at this values.
    vol_x_t = tf.where(vol_x_t > 0.0, vol_x_t, 0.0)
    next_x = (tf.math.exp(-tf.transpose(mean_reversion)[i + 1] * dt[i])
              * current_x
              + tf.transpose(exp_x_t)[i]
              + vol_x_t * normals)
    f_0_t = model._instant_forward_rate_fn(times[i + 1])

    # Update `rate_paths`
    if record_samples:
        rate_paths = rate_paths.write(written_count, next_x + f_0_t)
    else:
        rate_paths = next_x + f_0_t
    written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
    return (i + 1, written_count, next_x, rate_paths)



# TODO(b/157232803): Use tf.cumsum instead?
# Sample paths

_, _, _, rate_paths = tf.while_loop(
    cond_fn, body_fn, (0, written_count, initial_x, rate_paths))

if not record_samples:
    # shape [num_samples, 1, dim]
    rate_paths = tf.expand_dims(rate_paths, axis=-2)
# Shape [num_time_points] + [num_samples, dim]
rate_paths = rate_paths.stack()
# transpose to shape [num_samples, num_time_points, dim]
n = rate_paths.shape.rank
perm = list(range(1, n - 1)) + [0, n - 1]
rate_paths = tf.transpose(rate_paths, perm)


#%%

normal_draws = None

def _prepare_grid(times, times_grid, *params):
  """Prepares grid of times for path generation.

  Args:
    times:  Rank 1 `Tensor` of increasing positive real values. The times at
      which the path points are to be evaluated.
    times_grid: An optional rank 1 `Tensor` representing time discretization
      grid. If `times` are not on the grid, then the nearest points from the
      grid are used.
    *params: Parameters of the Heston model. Either scalar `Tensor`s of the
      same `dtype` or instances of `PiecewiseConstantFunc`.

  Returns:
    Tuple `(all_times, mask)`.
    `all_times` is a 1-D real `Tensor` containing all points from 'times`, the
    uniform grid of points between `[0, times[-1]]` with grid size equal to
    `time_step`, and jump locations of piecewise constant parameters The
    `Tensor` is sorted in ascending order and may contain duplicates.
    `mask` is a boolean 1-D `Tensor` of the same shape as 'all_times', showing
    which elements of 'all_times' correspond to THE values from `times`.
    Guarantees that times[0]=0 and mask[0]=False.
  """
  if times_grid is None:
    additional_times = []
    for param in params:
      if hasattr(param, 'is_piecewise_constant'):
        if param.is_piecewise_constant:
          # Flatten all jump locations
          additional_times.append(tf.reshape(param.jump_locations(), [-1]))
    zeros = tf.constant([0], dtype=times.dtype)
    all_times = tf.concat([zeros] + [times] + additional_times, axis=0)
    all_times = tf.sort(all_times)
    time_indices = tf.searchsorted(all_times, times, out_type=tf.int32)
  else:
    all_times = times_grid
    time_indices = tf.searchsorted(times_grid, times, out_type=tf.int32)
    # Adjust indices to bring `times` closer to `times_grid`.
    times_diff_1 = tf.gather(times_grid, time_indices) - times
    times_diff_2 = tf.gather(times_grid, tf.nn.relu(time_indices-1)) - times
    time_indices = tf.where(
        tf.math.abs(times_diff_2) > tf.math.abs(times_diff_1),
        time_indices,
        tf.nn.relu(time_indices - 1))
  # Create a boolean mask to identify the iterations that have to be recorded.
  mask = tf.scatter_nd(
      indices=tf.expand_dims(tf.cast(time_indices, dtype=tf.int64), axis=1),
      updates=tf.fill(tf.shape(times), True),
      shape=tf.shape(all_times, out_type=tf.int64))
  return all_times, mask

validate_args = False

# Note: all the notations below are the same as in [1].
num_requested_times = tff_utils.get_shape(times)[0]
params = [model._mean_reversion, model._volatility]
if model._corr_matrix is not None:
    params = params + [model._corr_matrix]
times, keep_mask = _prepare_grid(
    times, times_grid, *params)
# Add zeros as a starting location
dt = times[1:] - times[:-1]
if dt.shape.is_fully_defined():
    steps_num = dt.shape.as_list()[-1]
else:
    steps_num = tf.shape(dt)[-1]
    # TODO(b/148133811): Re-enable Sobol test when TF 2.2 is released.
    if random_type == random.RandomType.SOBOL:
        raise ValueError('Sobol sequence for Euler sampling is temporarily '
                         'unsupported when `time_step` or `times` have a '
                         'non-constant value')
if normal_draws is None:
    # In order to use low-discrepancy random_type we need to generate the
    # sequence of independent random normals upfront. We also precompute
    # random numbers for stateless random type in order to ensure independent
    # samples for multiple function calls whith different seeds.
    if random_type in (random.RandomType.SOBOL,
                       random.RandomType.HALTON,
                       random.RandomType.HALTON_RANDOMIZED,
                       random.RandomType.STATELESS,
                       random.RandomType.STATELESS_ANTITHETIC):
        normal_draws = utils.generate_mc_normal_draws(
            num_normal_draws=model._dim, num_time_steps=steps_num,
            num_sample_paths=num_samples, random_type=random_type,
            seed=seed,
            dtype=model._dtype, skip=skip)
    else:
        normal_draws = None
else:
    if validate_args:
        draws_times = tf.shape(normal_draws)[0]
        asserts = tf.assert_equal(
            draws_times, tf.shape(times)[0] - 1,  # We have added `0` to `times`
            message='`tf.shape(normal_draws)[1]` should be equal to the '
                    'number of all `times` plus the number of all jumps of '
                    'the piecewise constant parameters.')
        with tf.compat.v1.control_dependencies([asserts]):
            normal_draws = tf.identity(normal_draws)
# The below is OK because we support exact discretization with piecewise
# constant mr and vol.
mean_reversion = model._mean_reversion(times)
volatility = model._volatility(times)
if model._corr_matrix is not None:
    pass
else:
    corr_matrix_root = None

exp_x_t = model._conditional_mean_x(times, mean_reversion, volatility)
var_x_t = model._conditional_variance_x(times, mean_reversion, volatility)
if model._dim == 1:
    mean_reversion = tf.expand_dims(mean_reversion, axis=0)

# Initial state
initial_x = tf.zeros((num_samples, model._dim), dtype=model._dtype)
f0_t = model._instant_forward_rate_fn(times[0])
# Prepare results format
written_count = 0
if isinstance(num_requested_times, int) and num_requested_times == 1:
    record_samples = False
    rate_paths = initial_x + f0_t
else:
    # If more than one sample has to be recorded, create a TensorArray
    record_samples = True
    element_shape = initial_x.shape
    rate_paths = tf.TensorArray(dtype=times.dtype,
                                size=num_requested_times,
                                element_shape=element_shape,
                                clear_after_read=False)
    # Include initial state, if necessary
    rate_paths = rate_paths.write(written_count, initial_x + f0_t)
written_count += tf.cast(keep_mask[0], dtype=tf.int32)


# Define sampling while_loop body function
def cond_fn(i, written_count, *args):
    # It can happen that `times_grid[-1] > times[-1]` in which case we have
    # to terminate when `written_count` reaches `num_requested_times`
    del args
    return tf.math.logical_and(i < tf.size(dt),
                               written_count < num_requested_times)


def body_fn(i, written_count, current_x, rate_paths):
    """Simulate hull-white process to the next time point."""
    if normal_draws is None:
        normals = random.mv_normal_sample(
            (num_samples,),
            mean=tf.zeros((model._dim,), dtype=mean_reversion.dtype),
            random_type=random_type, seed=seed)
    else:
        normals = normal_draws[i]

    if corr_matrix_root is not None:
        normals = tf.linalg.matvec(corr_matrix_root[i], normals)
    vol_x_t = tf.math.sqrt(tf.nn.relu(tf.transpose(var_x_t)[i]))
    # If numerically `vol_x_t == 0`, the gradient of `vol_x_t` becomes `NaN`.
    # To prevent this, we explicitly set `vol_x_t` to zero tensor at zero
    # values so that the gradient is set to zero at this values.
    vol_x_t = tf.where(vol_x_t > 0.0, vol_x_t, 0.0)
    next_x = (tf.math.exp(-tf.transpose(mean_reversion)[i + 1] * dt[i])
              * current_x
              + tf.transpose(exp_x_t)[i]
              + vol_x_t * normals)
    f_0_t = model._instant_forward_rate_fn(times[i + 1])

    # Update `rate_paths`
    if record_samples:
        rate_paths = rate_paths.write(written_count, next_x + f_0_t)
    else:
        rate_paths = next_x + f_0_t
    written_count += tf.cast(keep_mask[i + 1], dtype=tf.int32)
    return (i + 1, written_count, next_x, rate_paths)


# TODO(b/157232803): Use tf.cumsum instead?
# Sample paths
_, _, _, rate_paths = tf.while_loop(
    cond_fn, body_fn, (0, written_count, initial_x, rate_paths))
if not record_samples:
    # shape [num_samples, 1, dim]
    rate_paths_2 =  tf.expand_dims(rate_paths, axis=-2)
# Shape [num_time_points] + [num_samples, dim]
rate_paths = rate_paths.stack()
# transpose to shape [num_samples, num_time_points, dim]
n = rate_paths.shape.rank
perm = list(range(1, n - 1)) + [0, n - 1]
rate_paths_2 = tf.transpose(rate_paths, perm)





#%%

short_rate = tf.expand_dims(rate_paths, axis=1)

# Reshape all `Tensor`s so that they have the dimensions same as (or
# broadcastable to) the output shape
# [num_samples, num_curve_times, num_sim_times, dim].
num_curve_nodes = tf.shape(curve_times)[0]  # m
num_sim_steps = tf.shape(times)[0]  # k
times = tf.reshape(times, (1, 1, num_sim_steps))
curve_times = tf.reshape(curve_times, (1, num_curve_nodes, 1))
# curve_times = tf.repeat(curve_times, self._dim, axis=-1)

mean_reversion = tf.reshape(
    mean_reversion, (1, 1, model._dim, num_sim_steps))
# Transpose so the `dim` is the trailing dimension.
mean_reversion = tf.transpose(mean_reversion, [0, 1, 3, 2])

# Calculate the variable `y(t)` (described in [1], section 10.1.6.1)
# so that we have the full Markovian state to compute the P(t,T).
y_t = tf.reshape(tf.transpose(y_t), (1, 1, num_sim_steps, model._dim))
# Shape [num_samples, num_curve_times, num_sim_steps, dim]
p_t_tau, r_t = model._bond_reconstitution(times, times + curve_times,
                                 mean_reversion, short_rate,
                                 y_t), rate_paths



dim = p_t_tau.shape[-1]

dt_builder = tf.concat(
    axis=0,
    values=[
        tf.convert_to_tensor([0.0], dtype=dtype),
        sim_times[1:] - sim_times[:-1] # This results in the 1st simulated zero rate (at time 0.1) having a dt of 0.
    ])
dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
discount_factors_builder = tf.math.exp(-r_t * dt)
tmp1 = pd.Series(discount_factors_builder[-1].numpy().reshape(-1))

# Transpose before (and after) because we want the cumprod along axis=1
# and `matvec` operates on the last axis. The shape before and after would
# be `(num_samples, len(times), dim)`
discount_factors_builder = tf.transpose(
    utils.cumprod_using_matvec(
        tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])
tmp2 = pd.Series(discount_factors_builder[-1].numpy().reshape(-1))  # SF: This is just the cumprod of the discount factors

# make discount factors the same shape as `p_t_tau`. This involves adding
# an extra dimenstion (corresponding to `curve_times`).
discount_factors_builder_ =  tf.identity(discount_factors_builder)
discount_factors_builder = tf.expand_dims(discount_factors_builder, axis=1)
discount_factors_simulated = tf.repeat(
    discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)

# `sim_times` and `curve_times` are sorted for simulation. We need to
# select the indices corresponding to our input.
sim_time_index = tf.searchsorted(sim_times, tf.reshape(expiries, [-1]))
curve_time_index = tf.searchsorted(curve_times, tf.reshape(tau, [-1]))
# Broadcast shapes of strikes, expiries and maturities
curve_time_index, sim_time_index = tff_utils.broadcast_tensors(
    curve_time_index, sim_time_index)
gather_index = _prepare_indices(
    tf.range(0, num_samples), curve_time_index, sim_time_index,
    tf.range(0, dim))

# The shape after `gather_nd` would be (num_samples*num_strikes*dim,)
payoff_discount_factors_builder = tf.gather_nd(discount_factors_simulated,
                                               gather_index) # This is the final discount factor (at the 1Y point)
# Reshape to `[num_samples] + strikes.shape + [dim]`
payoff_discount_factors = tf.reshape(payoff_discount_factors_builder,
                                     [num_samples] + strikes.shape + [dim])
payoff_bond_price_builder = tf.gather_nd(p_t_tau, gather_index)
payoff_bond_price = tf.reshape(payoff_bond_price_builder,
                               [num_samples] + strikes.shape + [dim])

is_call_options = tf.reshape(
    tf.broadcast_to(is_call_options, strikes.shape),
    [1] + strikes.shape + [1])

strikes = tf.reshape(strikes, [1] + strikes.shape + [1])
payoff = tf.where(is_call_options,
                  tf.math.maximum(payoff_bond_price - strikes, 0.0),
                  tf.math.maximum(strikes - payoff_bond_price, 0.0))
option_value = tf.math.reduce_mean(payoff_discount_factors * payoff, axis=0) # discount the payments back to the present value


px_monte_carlo = tf.squeeze(option_value, axis=-1)
print(f"Google Simulation Price: {float(px_monte_carlo):.8f}")
assert np.isclose(float(px_monte_carlo), 0.02823332, atol=1e-4), "Monte Carlo price does not match Google's implementation"

