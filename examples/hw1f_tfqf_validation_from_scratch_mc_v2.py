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

#%%

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


rate_paths = model.sample_paths(
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

average_rate = np.mean(rate_paths[:,-1,0])
print("Average of last rate: ", average_rate)
assert np.isclose(average_rate, 0.010194103663829423, atol=1e-4), "Average rate does not match Google's implementation"

short_rate = tf.expand_dims(rate_paths, axis=1)

# Reshape all `Tensor`s so that they have the dimensions same as (or
# broadcastable to) the output shape
# [num_samples, num_curve_times, num_sim_times, dim].
num_curve_nodes = tf.shape(curve_times)[0]  # m
num_sim_steps = tf.shape(times)[0]  # k
times = tf.reshape(times, (1, 1, num_sim_steps))
curve_times = tf.reshape(curve_times, (1, num_curve_nodes, 1))
# curve_times = tf.repeat(curve_times, self._dim, axis=-1)

#%%

mean_reversion = tf.reshape(mean_reversion, (1, 1, model._dim, num_sim_steps))
# Transpose so the `dim` is the trailing dimension.
mean_reversion = tf.transpose(mean_reversion, [0, 1, 3, 2])

# Calculate the variable `y(t)` (described in [1], section 10.1.6.1)
# so that we have the full Markovian state to compute the P(t,T).
y_t = tf.reshape(tf.transpose(y_t), (1, 1, num_sim_steps, model._dim))
# Shape [num_samples, num_curve_times, num_sim_steps, dim]
p_t_tau, r_t = model._bond_reconstitution(times, times + curve_times,
                                 mean_reversion, short_rate,
                                 y_t), rate_paths


#%% maturities = times + curve_times
#
#
# # """Compute discount bond prices using Eq. 10.18 in Ref [2]."""
#
# # Shape `times.shape + [dim]`
# f_0_t = model._instant_forward_rate_fn(times)
# # Shape `short_rate.shape`
# x_t = short_rate - f_0_t
# discount_rates_times = model._initial_discount_rate_fn(times)
# times_expand = tf.expand_dims(times, axis=-1)
# # Shape `times.shape + [dim]`
# p_0_t = tf.math.exp(-discount_rates_times * times_expand)
# # Shape `times.shape + [dim]`
# discount_rates_maturities = model._initial_discount_rate_fn(maturities)
# maturities_expand = tf.expand_dims(maturities, axis=-1)
# # Shape `times.shape + [dim]`
# p_0_t_tau = tf.math.exp(
#     -discount_rates_maturities * maturities_expand) / p_0_t
# # Shape `times.shape + [dim]`
# g_t_tau = (1. - tf.math.exp(
#     -mean_reversion * (maturities_expand - times_expand))) / mean_reversion
# # Shape `x_t.shape`
# term1 = x_t * g_t_tau
# term2 = y_t * g_t_tau ** 2
# # Shape `short_rate.shape`
# p_t_tau = p_0_t_tau * tf.math.exp(-term1 - 0.5 * term2)
# # Shape `short_rate.shape`

print("Average:", np.mean(p_t_tau[:,:,-1,:]))
print("SD:", np.std(p_t_tau[:,:,-1,:]))



#%%

dim = p_t_tau.shape[-1]

dt_builder = tf.concat(
    axis=0,
    values=[
        tf.convert_to_tensor([0.0], dtype=dtype),
        sim_times[1:] - sim_times[:-1] # This results in the 1st simulated zero rate (at time 0.1) having a dt of 0.
    ])
dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
discount_factors_builder = tf.math.exp(-r_t * dt)
#tmp1 = pd.Series(discount_factors_builder[-1].numpy().reshape(-1))



# Transpose before (and after) because we want the cumprod along axis=1
# and `matvec` operates on the last axis. The shape before and after would
# be `(num_samples, len(times), dim)`
discount_factors_builder = tf.transpose(
    utils.cumprod_using_matvec(
        tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])
#tmp2 = pd.Series(discount_factors_builder[-1].numpy().reshape(-1))  # SF: This is just the cumprod of the discount factors

#%%

# make discount factors the same shape as `p_t_tau`. This involves adding
# an extra dimenstion (corresponding to `curve_times`).
discount_factors_builder_ =  tf.identity(discount_factors_builder)
discount_factors_builder = tf.expand_dims(discount_factors_builder, axis=1)
discount_factors_simulated = tf.repeat(
    discount_factors_builder, p_t_tau.shape.as_list()[1], axis=1)

#%%

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

