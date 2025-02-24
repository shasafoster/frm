# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import pandas as pd
import numpy as np
import time
from frm.pricing_engine import HullWhite1Factor
from frm.term_structures import ZeroCurve
from frm.enums import CompoundingFreq, ZeroCurveInterpMethod
import matplotlib.pyplot as plt

# Tests for test_zero_coupon_bond_option_pricing()
# Test details copied from tf_quant_finance
# https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/hull_white/zero_coupon_bond_option_test.py

#% Setup ZeroCurve and HW1F object

mean_reversion = 0.03
volatility = 0.02

# Create flat zero curve at 1%
curve_date = pd.Timestamp('2024-04-01')
years = np.array(list(range(21)))
rates = np.array([0.01 for _ in range(21)])
pillar_df = pd.DataFrame(np.array([years, rates]).T, columns=['years', 'zero_rate'])

zero_curve = ZeroCurve(curve_date=curve_date,
                       pillar_df=pillar_df,
                       compounding_freq=CompoundingFreq.CONTINUOUS,
                       interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)

# Initialize model
hw1f = HullWhite1Factor(
    zero_curve=zero_curve,
    mean_rev_lvl=mean_reversion,
    vol=volatility
)

hw1f.setup_theta()

#% Scalar test case

# Test parameters
expiry = 10
maturity = 15
K = np.exp(-0.01 * maturity) / np.exp(-0.01 * expiry)  # Strike calculation from test
cp = 1 # call option

# Price call option
analytical_price = hw1f.price_zero_coupon_bond_option(
    option_expiry=expiry,
    bond_maturity=maturity,
    K=K,
    cp=cp
)

expected_price = 8.720313 / 100

print("Price: ", analytical_price[0])
print("Expected Price: ", expected_price)
print("Difference: ", analytical_price[0] - expected_price)
print(f"Analytical ZCBO Price (%): {100 * analytical_price[0]:.6f}")

# Simulate to maturity
sim_result = hw1f.simulate(
    tau=expiry,
    nb_steps=expiry * 365,
    nb_simulations=25_000,
    flag_apply_antithetic_variates=True,
    method='exact'
)

print("Convergence:")
print(sim_result['averages_df'].iloc[-1],'\n')
R_expiry = sim_result['R'][-1,:] # Simulated Δt rate, R, at expiry (discretization of the short rate)

#%%

def calculate_zcbo_payoff(R_expiry, hw1f, expiry, maturity, K, cp):
    # Calculate P(T,S) at option expiry for each simulation path
    P_T_S = np.array([hw1f.calc_discount_factor_by_solving_ode_1(t1=expiry, t2=maturity, r_t1=r) for r in R_expiry])

    # Calculate option payoff
    payoff = np.maximum(cp * (P_T_S - K), 0)

    # Discount payoff to t=0
    discount_factor_T = hw1f.zero_curve.get_discount_factors(years=expiry)
    present_value = payoff * discount_factor_T

    return present_value


# Calculate MC price
present_values = calculate_zcbo_payoff(R_expiry, hw1f, expiry, maturity, K, cp)
mc_price = np.mean(present_values)
mc_std_error = np.std(present_values) / np.sqrt(len(present_values))

print(f"\nMonte Carlo Results:")
print(f"Monte Carlo ZCBO Price (%): {100 * mc_price:.6f}")
print(f"Monte Carlo Std Error (%): {100 * mc_std_error:.6f}")
print(f"95% CI: [{100 * (mc_price - 1.96 * mc_std_error):.6f}, {100 * (mc_price + 1.96 * mc_std_error):.6f}]")
print(f"Difference from Analytical (%): {100 * (mc_price - analytical_price[0]):.6f}")

# Optional: Plot histogram of present values
plt.figure(figsize=(10, 6))
plt.hist(present_values * 100, bins=50, density=True, alpha=0.7, color='blue')
plt.axvline(mc_price * 100, color='red', linestyle='dashed', label='MC Mean')
plt.axvline(analytical_price[0] * 100, color='green', linestyle='dashed', label='Analytical')
plt.xlabel('Option Price (%)')
plt.ylabel('Density')
plt.title('Distribution of Monte Carlo ZCBO Prices')
plt.legend()
plt.grid(True)
plt.show()


#%%

bond_prices_expiry = sim_result['sim_dsc_factor'][-1, :]
bond_prices_expiry_df = pd.DataFrame(bond_prices_expiry, columns=['bond_prices_expiry'])

bond_prices_maturity = hw1f.calc_discount_factor_by_solving_ode_2(t1=expiry, t2=maturity, r_t1=R_expiry)
bond_prices_maturity_df = pd.DataFrame(bond_prices_maturity, columns=['bond_prices_maturity'])


payoffs = np.maximum(cp * (bond_prices_maturity - K), 0)

print(f"Analytical ZCBO Price (%): {100 * analytical_price[0]:.6f}")
print(f"MC Price (%): {100 * np.mean(payoffs):.6f}")

#%%

# Test case with a simulated short rate
test_r = R_expiry[0]
t1, t2 = expiry, maturity

# Components
df_t = hw1f.zero_curve.get_discount_factors(years=t1)
df_T = hw1f.zero_curve.get_discount_factors(years=t2)
f_t = hw1f.zero_curve.get_instantaneous_forward_rate(years=t1)
B_t_T = hw1f.calc_b(t1, t2)
α = hw1f.mean_rev_lvl
σ = hw1f.vol

# Print each component
print(f"df_t: {df_t}")
print(f"df_T: {df_T}")
print(f"f_t: {f_t}")
print(f"B_t_T: {B_t_T}")
print(f"Forward DF from curve: {df_T/df_t}")
print(f"r(t1): {test_r}")

# Calculate final price
a = (df_T/df_t) * np.exp(B_t_T * f_t - (σ**2/(4*α)) * (1-np.exp(-2*α*t1)) * B_t_T**2)
p = a * np.exp(-B_t_T * test_r)
print(f"Final price: {p}")

#%%

df_t1_t2 = hw1f.calc_discount_factor_by_solving_ode_2(t1=10, t2=15, r_t1=0.01)
