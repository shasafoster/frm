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

# Validate the HW1F ZCBO pricing against TF Quant Finance for both analytical and monte carlo methods

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_quant_finance as tff
from frm.term_structures.zero_curve import ZeroCurve
from frm.enums import ZeroCurveInterpMethod, CompoundingFreq
from frm.pricing_engine.hw1f import HullWhite1Factor

# Create flat 1% zero curve
years = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
rates = 0.01 * np.ones_like(years)  # Flat 1% curve
zero_curve = ZeroCurve(
    years=years,
    rates=rates,
    compounding_freq=CompoundingFreq.CONTINUOUS,
    interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR
)

# Test parameters
test_cases = [
    {"name": "Base case", "mean_rev": 0.03, "volatility": 0.01},
    {"name": "High volatility", "mean_rev": 0.03, "volatility": 0.05},
    {"name": "Low volatility", "mean_rev": 0.03, "volatility": 0.002}
]

# Swaption parameters matching Google's test cases
expiries = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])
float_leg_start_times = np.array([
    [0.5, 1.0, 1.5, 2.0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],  # 6M x 2Y
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  # 6M x 5Y
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # 1Y x 2Y
    [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],  # 1Y x 5Y
    [2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # 2Y x 2Y
    [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],  # 2Y x 5Y
])
float_leg_end_times = float_leg_start_times + 0.5
fixed_leg_payment_times = float_leg_end_times
float_leg_daycount_fractions = float_leg_end_times - float_leg_start_times
fixed_leg_daycount_fractions = float_leg_daycount_fractions
fixed_leg_coupon = 0.01 * np.ones_like(fixed_leg_payment_times)

# Zero rate function for TF Quant Finance
zero_rate_fn = lambda x: 0.01 * tf.ones_like(x)

print("\nValidating Hull-White One Factor Model Swaption Pricing")
print("=" * 80)

for test_case in test_cases:
    print(f"\nTest Case: {test_case['name']}")
    print(f"Mean Reversion = {test_case['mean_rev']}, Volatility = {test_case['volatility']}")
    print("-" * 80)

    # Initialize your model with test parameters
    hw1f_model = HullWhite1Factor(
        zero_curve=zero_curve,
        mean_rev_lvl=test_case['mean_rev'],
        vol=test_case['volatility'],
        dt=1e-4,
        num=1000
    )

    # Get TF Quant Finance prices
    tf_prices = tff.models.hull_white.swaption_price(
        expiries=expiries,
        floating_leg_start_times=float_leg_start_times,
        floating_leg_end_times=float_leg_end_times,
        fixed_leg_payment_times=fixed_leg_payment_times,
        floating_leg_daycount_fractions=float_leg_daycount_fractions,
        fixed_leg_daycount_fractions=fixed_leg_daycount_fractions,
        fixed_leg_coupon=fixed_leg_coupon,
        reference_rate_fn=zero_rate_fn,
        notional=100.,
        mean_reversion=test_case['mean_rev'],
        volatility=[test_case['volatility']],
        use_analytic_pricing=True,
        dtype=tf.float64
    ).numpy()

    # Get your model's prices
    your_prices = []
    for i in range(len(expiries)):
        price = hw1f_model.price_swaption(
            swaption_expiry=expiries[i],
            coupon_payment_times=fixed_leg_payment_times[i][fixed_leg_payment_times[i] > 0],
            coupon_year_fractions=fixed_leg_daycount_fractions[i][fixed_leg_daycount_fractions[i] > 0],
            notional_payment_time=float(fixed_leg_payment_times[i][-1]),
            fixed_rate=0.01,
            is_payer=True,
            notional=100.
        )
        your_prices.append(price)
    your_prices = np.array(your_prices)

    # Compare results
    rel_diff = np.abs(tf_prices - your_prices) / tf_prices
    max_rel_diff = np.max(rel_diff)

    print(f"{'Expiry':<10} {'TF Price':>15} {'Your Price':>15} {'Rel Diff %':>15}")
    print("-" * 80)
    for i in range(len(expiries)):
        print(f"{expiries[i]:<10.2f} {tf_prices[i]:>15.6f} {your_prices[i]:>15.6f} {rel_diff[i] * 100:>15.6f}")
    print("-" * 80)
    print(f"Maximum Relative Difference: {max_rel_diff * 100:.6f}%")