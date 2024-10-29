# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hw1f import HullWhite1Factor
import matplotlib.pyplot as plt
import scipy

df = pd.read_excel('C:/Users/shasa/Documents/frm_private/hull_white (in progress)/zero_data.xlsx')

# running this - working through getting really clean functions
zero_curve = ZeroCurve(curve_date=pd.Timestamp(2023,6,30),
               data=df[['years','discount_factor']],
               interpolation_method='cubic_spline_on_zero_rates')

# HW1F model parameters
r0 = zero_curve.data['nacc'].iloc[0]
mean_rev_lvl = 0.19
vol = 0.0196

# Chosen settings
grid_length = 50


hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=mean_rev_lvl, vol=vol)
hw1f.setup_theta(num=grid_length)


years_grid = np.linspace(zero_curve.data['years'].min(), zero_curve.data['years'].max(),grid_length)

hw1f_instantaneous_forward_rate  = [hw1f.get_instantaneous_forward_rate(t) for t in years_grid]
hw1f_zero_rates = [hw1f.get_zero_rate(0,t) for t in years_grid]
hw1f_discount_factors = [hw1f.get_discount_factor(0, t) for t in years_grid]

plt.figure(figsize=(16,4))
plt.subplot(121)
plt.plot(years_grid, hw1f_instantaneous_forward_rate, label ='calculated (instantaneous forward rates)')
plt.plot(years_grid, hw1f_zero_rates, label='calculated zero rates')
plt.plot(zero_curve.data['years'], zero_curve.data['nacc'], marker='.', linestyle='None', label='Source data zero rates')
plt.xlabel(r'Maturity $T$ ')
plt.title(r'continously compounded $r(t,T)$ and $f(t,T)$')
plt.grid(True)
plt.legend()
plt.ylabel('spot and forward rates')

plt.subplot(122)
plt.plot(years_grid, hw1f.get_thetas(years_grid), marker='.',label =r'$\theta(t)$')
plt.xlabel(r'Maturity $T$')
plt.title(r'$function$ $\theta(t)$')
plt.grid(True)
plt.ylabel(r'$\theta(t)$')
plt.show()

plt.figure(figsize=(16,4))

plt.plot(zero_curve.data['years'], zero_curve.data['discount_factor'], marker='*', label = "Source data discount factors")
plt.plot(years_grid, hw1f_discount_factors, label ='HW1F model discount factors')
plt.xlabel(r'Maturity $T$ ')
plt.title(r'Source vs HW1F model discount factors')
plt.grid(True)
plt.legend()
plt.ylabel('Prices')
plt.show()


tau = 5
nb_simulations = 10000
nb_steps = tau*252

# R, years_grid = hw1f.simulate(tau=tau,
#                               nb_steps=nb_steps,
#                               nb_simulations=nb_simulations,
#                               flag_apply_antithetic_variates=True,
#                               random_seed=1500)
#
# terminal_discount_factor = np.full(nb_simulations, np.nan)
#
# # Calculate the terminal discount factor for each simulation
# for i in range(nb_simulations):
#     terminal_discount_factor[i] = np.exp(-(scipy.integrate.simpson(y=R[:, i], x=years_grid)))
#
# terminal_zero_rates = -1 * np.log(terminal_discount_factor) / tau
#
# print('Simulation average: ', round(100 * -1 * np.log(terminal_discount_factor.mean()) / 5.0,8))
# print(' Analytical result: ', round(100 *hw1f.get_zero_rate(t=0,T=5),8))


results = hw1f.simulate(tau=tau,
                        nb_steps=nb_steps,
                        nb_simulations=nb_simulations,
                        flag_apply_antithetic_variates=True,
                        random_seed=1500)


#%%
#
# simulation_discount_factors = np.full(R.shape, np.nan)
#
# for step_nb in range(1,nb_steps):
#     print('step_nb: ', step_nb)
#     for i in range(nb_simulations):
#         simulation_discount_factors[step_nb, i] = np.exp(-(scipy.integrate.simpson(y=R[:step_nb, i], x=years_grid[:step_nb])))

#%% Vectorised version

simulation_discount_factors = np.full(R.shape, np.nan)
simulation_discount_factors[0, :] = 1.0

for step_nb in range(1, nb_steps+1):
    print('step_nb:', step_nb)
    integrated_R = scipy.integrate.simpson(y=R[:(step_nb+1), :], x=years_grid[:(step_nb+1)], axis=0)
    simulation_discount_factors[step_nb, :] = np.exp(-integrated_R)

avg_discount_factors = np.mean(simulation_discount_factors, axis=1)
avg_zero_rates = -1 * np.log(avg_discount_factors) / years_grid

print('Vectorised ZR[-1]:', round(100*avg_zero_rates[-1],8))

df_v = pd.DataFrame({'years': years_grid, 'zero_rates': avg_zero_rates, 'discount_factors': avg_discount_factors})
df_v.to_clipboard()

#%% Cumulative, vectorised version

simulation_discount_factors_CUM = np.full(R.shape, np.nan)
simulation_discount_factors_CUM[0, :] = 1.0
cumulative_integrated_R = np.full(R.shape, np.nan)

# Initial integration value at step 1
step_nb = 1
cumulative_integrated_R[step_nb] = scipy.integrate.simpson(
    y=R[(step_nb-1):(step_nb+1), :], x=years_grid[:(step_nb+1)], axis=0)
simulation_discount_factors_CUM[step_nb, :] = np.exp(-cumulative_integrated_R[step_nb])

for step_nb in range(2, nb_steps+1):
    cumulative_integrated_R[step_nb] = cumulative_integrated_R[step_nb - 1] + scipy.integrate.simpson(
        y=R[(step_nb-1):(step_nb+1), :],x=years_grid[(step_nb-1):(step_nb+1)],axis=0)
    simulation_discount_factors_CUM[step_nb, :] = np.exp(-cumulative_integrated_R[step_nb])

avg_simulation_discount_factors_CUM = np.mean(simulation_discount_factors_CUM, axis=1)
avg_zero_rates_CUM = -1 * np.log(avg_simulation_discount_factors_CUM) / years_grid

print('CUM ZR[-1]:', round(100*avg_zero_rates_CUM[-1],8))

df_cum = pd.DataFrame({'years': years_grid, 'zero_rates': avg_zero_rates_CUM, 'discount_factors': avg_simulation_discount_factors_CUM})
df_cum.to_clipboard()
