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

# # Google's Implementation - Simulation
# price_google_simulation = tff.models.hull_white.bond_option_price(
#     strikes=strike,
#     expiries=expiry,
#     maturities=maturity,
#     mean_reversion=0.03,
#     volatility=0.02,
#     discount_rate_fn=discount_rate_fn,
#     use_analytic_pricing=False,
#     num_samples=500000,
#     time_step=0.1,
#     random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
#     dtype=tf.float64
# )

expiries = 1.0
maturities = 5.0
strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)
is_call_options = True
mean_reversion = 0.03
volatility = 0.02
num_samples = 500000
time_step = 0.1
random_type = tff.math.random.RandomType.PSEUDO_ANTITHETIC
seed = None
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

def analytic_valuation(discount_rate_fn, model, strikes, expiries, maturities,
                        is_call_options):
  """Performs analytic valuation."""
  # Shape `expiry.shape`
  discount_rates_expiries = discount_rate_fn(expiries)
  discount_factor_expiries = tf.math.exp(
      -discount_rates_expiries * expiries)
  input_shape = tff_utils.common_shape(strikes, expiries, maturities)
  variance = bond_option_variance(
      model, tf.reshape(expiries, shape=[-1]), tf.reshape(maturities, [-1]))
  # Reshape to original shape
  variance = tf.reshape(variance, input_shape)
  discount_rates_maturities = discount_rate_fn(maturities)
  # Shape `expiries.shape`
  discount_factor_maturity = tf.math.exp(-discount_rates_maturities
                                         * maturities)
  forward_bond_price = discount_factor_maturity / discount_factor_expiries

  sqrt_variance = tf.math.sqrt(variance)
  # Shape `expiries.shape`
  log_moneyness = tf.math.log(forward_bond_price / strikes)
  d1 = tf.math.divide_no_nan(log_moneyness + 0.5 * variance, sqrt_variance)
  d2 = d1 - tf.math.sqrt(variance)
  option_value_call = (discount_factor_maturity * _ncdf(d1)
                       - strikes * discount_factor_expiries* _ncdf(d2))
  option_value_put = (strikes * discount_factor_expiries * _ncdf(-d2)
                      - discount_factor_maturity * _ncdf(-d1))

  intrinsic_value = tf.where(
      is_call_options,
      tf.math.maximum(forward_bond_price - strikes, 0),
      tf.math.maximum(strikes - forward_bond_price, 0))
  option_value = tf.where(
      maturities < expiries, tf.zeros_like(maturities),
      tf.where(sqrt_variance > 0.0,
               tf.where(is_call_options, option_value_call, option_value_put),
               intrinsic_value))
  return option_value

def bond_option_variance(model, option_expiry, bond_maturity):
  """Computes black equivalent variance for bond options.

  Black equivalent variance is defined as the variance to use in the Black
  formula to obtain the model implied price of European bond options.

  Args:
    model: An instance of `VectorHullWhiteModel`.
    option_expiry: A rank 1 `Tensor` of real dtype specifying the time to
      expiry of each option.
    bond_maturity: A rank 1 `Tensor` of real dtype specifying the time to
      maturity of underlying zero coupon bonds.

  Returns:
    A rank 1 `Tensor` of same dtype and shape as the inputs with computed
    Black-equivalent variance for the underlying options.
  """
  # pylint: disable=protected-access
  if model._sample_with_generic:
      raise ValueError('The paramerization of `mean_reversion` and/or '
                       '`volatility` does not support analytic computation '
                       'of bond option variance.')
  mean_reversion = model.mean_reversion(option_expiry)
  volatility = model.volatility(option_expiry)

  # Shape [num_times]
  var_between_vol_knots = model._variance_int(model._padded_knots,
                                              model._jump_locations,
                                              model._jump_values_vol,
                                              model._jump_values_mr)[0]
  # Shape [num_times]
  varx_at_vol_knots = tf.concat(
      [tf.zeros([1], dtype=var_between_vol_knots.dtype),
       utils.cumsum_using_matvec(var_between_vol_knots)],
      axis=-1)
  # Shape [num_times + 1]
  time_index = tf.searchsorted(model._jump_locations[0], option_expiry)
  # Shape [1, num_times + 1]
  vn = tf.concat(
      [model._zero_padding,
       model._jump_locations], axis=-1)

  # Shape [num_times]
  var_expiry = model._variance_int(
      tf.gather(vn, time_index, axis=-1), option_expiry,
      volatility, mean_reversion)[0]
  var_expiry = var_expiry + tf.gather(
      varx_at_vol_knots, time_index)
  var_expiry = var_expiry * (
          tf.math.exp(-mean_reversion * option_expiry) - tf.math.exp(
      -mean_reversion * bond_maturity)) ** 2 / mean_reversion ** 2
  # gpylint: enable=protected-access
  # shape [num_times]
  return var_expiry


px_analytic = analytic_valuation(discount_rate_fn, model, strikes, expiries, maturities, is_call_options)
print(f"Google Analytical Price: {float(px_analytic):.8f}") # Should be 0.02817777


if time_step is None:
  raise ValueError('`time_step` must be provided for simulation '
                   'based bond option valuation.')

def sample_discount_curve_paths_fn(times, curve_times, num_samples):
  return model.sample_discount_curve_paths(
      times=times,
      curve_times=curve_times,
      num_samples=num_samples,
      random_type=random_type,
      seed=seed,
      skip=skip)

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

p_t_tau, r_t = sample_discount_curve_paths_fn(
    times=sim_times, curve_times=curve_times, num_samples=num_samples)
dim = p_t_tau.shape[-1]

dt_builder = tf.concat(
    axis=0,
    values=[
        tf.convert_to_tensor([0.0], dtype=dtype),
        sim_times[1:] - sim_times[:-1]
    ])
dt = tf.expand_dims(tf.expand_dims(dt_builder, axis=-1), axis=0)
discount_factors_builder = tf.math.exp(-r_t * dt)
# Transpose before (and after) because we want the cumprod along axis=1
# and `matvec` operates on the last axis. The shape before and after would
# be `(num_samples, len(times), dim)`
discount_factors_builder = tf.transpose(
    utils.cumprod_using_matvec(
        tf.transpose(discount_factors_builder, [0, 2, 1])), [0, 2, 1])

# make discount factors the same shape as `p_t_tau`. This involves adding
# an extra dimenstion (corresponding to `curve_times`).
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
                                               gather_index)
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
option_value = tf.math.reduce_mean(payoff_discount_factors * payoff, axis=0)


px_monte_carlo = tf.squeeze(option_value, axis=-1)
print(f"Google Simulation Price: {float(px_monte_carlo):.8f}")
assert np.isclose(float(px_monte_carlo), 0.02823332, atol=1e-8), "Monte Carlo price does not match Google's implementation"

