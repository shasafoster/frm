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

import numpy as np
from frm.term_structures.zero_curve import ZeroCurve
from frm.enums import ZeroCurveInterpMethod, CompoundingFreq
import pandas as pd

# Create flat 1% zero curve for your implementation
years = np.linspace(0, 10, 11)
rates = np.full_like(years, 0.01)
zero_curve = ZeroCurve(
    curve_date=pd.Timestamp('2024-04-01'),
    pillar_df=pd.DataFrame({'years': years, 'zero_rate': rates}),
    compounding_freq=CompoundingFreq.CONTINUOUS,
    interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR
)

# Initialize your Hull-White model
from frm.pricing_engine.hw1f import HullWhite1Factor
hw_model = HullWhite1Factor(
    zero_curve=zero_curve,
    mean_rev_lvl=0.03,
    vol=0.02
)

# Test parameters (same as Google's test)
expiry = 1.0
maturity = 5.0
strike = np.exp(-0.01 * maturity) / np.exp(-0.01 * expiry)

print("\nTest Case 1: Single Option - Your Implementation")
# Your analytical price
price_analytical = hw_model.price_zero_coupon_bond_option(
    option_expiry=expiry,
    bond_maturity=maturity,
    K=strike,
    cp=1
)

# Your simulation price
sim_results = hw_model.simulate(
    tau=maturity,
    nb_steps=int(maturity/0.1),  # dt = 0.1 as per Google's test
    nb_simulations=500000
)

expiry_idx = np.searchsorted(sim_results['years_grid'], expiry)
maturity_idx = np.searchsorted(sim_results['years_grid'], maturity)

df_expiry = sim_results['sim_dsc_factor'][expiry_idx]
df_maturity = sim_results['sim_dsc_factor'][maturity_idx]

forward_bond_price = df_maturity / df_expiry
payoff = np.maximum(forward_bond_price - strike, 0)
price_simulation = np.mean(payoff) * df_expiry.mean()

print(f"Your Analytical Price: {price_analytical.item():.8f}")
print(f"Your Simulation Price: {price_simulation.item():.8f}")


#%%

import tensorflow.compat.v2 as tf
import tf_quant_finance as tff

print("\nTest Case 1: Single Option - Google TF Quant Finance")
# Google's Implementation - Analytical
def discount_rate_fn(t):
    return 0.01 * tf.ones_like(t)

price_google_analytical = tff.models.hull_white.bond_option_price(
    strikes=strike,
    expiries=expiry,
    maturities=maturity,
    mean_reversion=0.03,
    volatility=0.02,
    discount_rate_fn=discount_rate_fn,
    use_analytic_pricing=True,
    dtype=tf.float64
)

# Google's Implementation - Simulation
price_google_simulation = tff.models.hull_white.bond_option_price(
    strikes=strike,
    expiries=expiry,
    maturities=maturity,
    mean_reversion=0.03,
    volatility=0.02,
    discount_rate_fn=discount_rate_fn,
    use_analytic_pricing=False,
    num_samples=500000,
    time_step=0.1,
    random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
    dtype=tf.float64
)

print(f"Google Analytical Price: {float(price_google_analytical):.8f}")
print(f"Google Simulation Price: {float(price_google_simulation):.8f}")


#%%


