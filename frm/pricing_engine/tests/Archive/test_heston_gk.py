# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    import sys
    
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve()) 
    sys.path.append(os.getcwd())
    print('__main__ - current working directory:', os.getcwd())

        
from frm.pricing_engine.heston_gk import heston1993_price_fx_vanilla_european, heston_fit_vanilla_fx_smile, heston_cos_vanilla_european, heston_carr_madan_fx_vanilla_european

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from time import time
import pandas as pd

#% STF2hes07 - fit smile

# Initialize
standalone = 0  # set to 0 to make plots as seen in STF2
delta_plt = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # forward deltas
Δ = np.array([-0.1, -0.25, 0.5, 0.25, 0.1])
r_d = np.array([0.31100, 0.32875, 0.49781, 0.70075, 1.08, 1.08]) * 0.01
r_f = np.array([0.58, 0.631, 0.884, 1.131, 1.399, 1.399]) * 0.01
tau = np.array([7 / 365, 1 / 12, 3 / 12, 6 / 12, 1, 2])
S = 1.2779
cp = np.array([-1, -1, 1, 1, 1])
Δ_convention = 'regular_forward_Δ'

tenors = ['1W','1M','3M','6M','1Y','2Y']

σ_market_set = [
    np.array([13.948, 13.192, 13.35, 14.042, 15.4355]) * 0.01,      # 1W 
    np.array([12.45, 12.249, 12.7, 13.749, 15.225]) * 0.01,         # 1M
    np.array([12.6885, 12.3945, 13.0495, 14.5445, 16.666]) * 0.01,  # 3M
    np.array([13.688, 12.899, 13.55, 15.299, 18.1275]) * 0.01,      # 6M
    np.array([13.6705, 12.668, 13.187, 14.718, 17.2705]) * 0.01,    # 1Y
    np.array([12.139, 11.784, 12.174, 13.284, 14.764]) * 0.01       # 2Y
]

start_time = time()
results = []

#pricing_method = 'heston_analytical_1993'
pricing_method = 'heston_carr_madan_gauss_kronrod_quadrature'
#pricing_method = 'heston_carr_madan_fft_w_simpsons'
#pricing_method = 'heston_cos'


# Main loop for various smiles
for i, σ_market in enumerate(σ_market_set):
    if i > -5:
        #delta_spot = np.exp(-r_f[i] * tau[i]) * delta
        v0, vv, kappa, theta, lambda_, rho, IV, SSE = heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ_market, S, r_f[i], r_d[i], tau[i], cp, pricing_method=pricing_method)
        results.append([v0, vv, kappa, theta, lambda_, rho, IV, SSE])
        
        # Displaying output
        print(f'=== {tenors[i]} calibration results ===')
        print(f'v0, vv, kappa, theta, rho: {v0, vv, kappa, theta, rho}')
        print(f'[IV (10, 25, ATM, 75, 90), SSE] * 100%: {IV*100, SSE*100}')
        
        # Plotting
        plt.figure(i+1)
        plt.plot(delta_plt * 100, σ_market * 100, 'ko-', linewidth=1)
        plt.plot(delta_plt * 100, IV * 100, 'rs--', linewidth=1)
        plt.legend([f'{tenors[i]} smile', 'Heston fit'], loc='upper left')
        plt.xlabel('Delta [%]')
        plt.ylabel('Implied volatility [%]')
        plt.xticks(delta_plt * 100)
        plt.title(pricing_method)
        plt.show()

end_time = time()
print(f"Elapsed time: {end_time - start_time:.2f} seconds")

#%%








#%%

# #%%%

# strikes = np.array([1.03413767, 1.15064198, 1.28873193, 1.466967  , 1.6959051 ])

# # Inputs
# v0 = 0.014820628
# vv = 0.293475456
# kappa = 1.5
# theta = 0.017132031
# rho = 0.232453505225961

# S = 1.2779
# tau = 2
# cp = 1

# i = 3
# rd = r_d[3] # 0.0070075
# rf = r_f[3] # 0.01131

# #%%

# print('Heston1993',heston1993_price_fx_vanilla_european(cp, S, strikes[i], tau, rd, rf, v0, vv, kappa, theta, rho, 0))
# print('CM_GQ:',heston_carr_madan_fx_vanilla_european(cp, S, strikes[i], tau, rd, rf, kappa, theta, vv, rho, v0, integration_method=0))
# print('CM_FFT:',heston_carr_madan_fx_vanilla_european(cp, S, strikes[i], tau, rd, rf, kappa, theta, vv, rho, v0, integration_method=1))
# print('COS:',heston_cos_vanilla_european(cp, S, strikes[i], tau, rd, rf, v0, vv, kappa, theta, rho))

# #%%

# import frm.pricing_engine.AllFunctions as func

# # In[6]: COS with Fang & Oosterlee (2008) Version of Heston's Characteristic Function

# r      = rd                  # assumption Risk-free rate
# mu     = r #annualisedMean  # Mean rate of drift
# sigma  = v0 # Initial Vola of underyling at time 0; also called u0 or a
# S0     = S       # Today's stock price
# tau    = tau           # Time to expiry in years
# q      = 0                  # Divindend Yield
# lm     = kappa             # The speed of mean reversion
# v_bar  = theta  # Mean level of variance of the underlying
# volvol =  vv            # Volatility of the volatiltiy process
# rho    = rho            # Covariance between the log stock and the variance process

# # Truncation Range
# L       = 120
# a, b    = func.truncationRange(L, mu, tau, sigma, v_bar, lm, rho, volvol)
# bma     = b-a

# # Number of Points
# N       = 15
# k       = np.arange(np.power(2,N))

# # Input for the Characterstic Function Phi
# u       = k * np.pi/bma

# K = [strikes[i]]

# #  In[4]: COS-FFT Value Function for Put
# UkPut  = 2 / bma * ( func.cosSer1(a,b,a,0,k) - func.cosSerExp(a,b,a,0,k) )
# UkCall = 2 / bma * ( func.cosSerExp(a,b,0,b,k) - func.cosSer1(a,b,0,b,k) )

# charactersticFunctionHFO = func.charFuncHestonFO(mu, r, u, tau, sigma, v_bar, lm, rho, volvol)

# C_COS_HFO = np.zeros((np.size(K)))
# P_COS_HFO = np.zeros((np.size(K)))
# C_COS_PCP = np.zeros((np.size(K)))

# for m in range(0, np.size(K)):
#     x  = np.log(S0/K[m])
#     addIntegratedTerm = np.exp(1j * k * np.pi * (x-a)/bma)
#     Fk = np.real(charactersticFunctionHFO * addIntegratedTerm)
#     Fk[0] = 0.5 * Fk[0]						
#     C_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkCall)) * np.exp(-r * tau)
#     P_COS_HFO[m] = K[m] * np.sum(np.multiply(Fk, UkPut)) * np.exp(-r * tau)
#     C_COS_PCP[m] = P_COS_HFO[m] + S0 * np.exp(-q * tau) - K[m] * np.exp(-r * tau)

# print(C_COS_HFO)
# print(P_COS_HFO)
# print(C_COS_PCP)


#%% STF2Hes01

# standalone = 0
# spot = 4
# mu = 0.02
# kappa = 2
# theta = 0.04
# sigma = 0.3
# rho = -0.05
# days = 100
# time = np.arange(0, days+1) / days

# np.random.seed([3621423255, 1292471671])
# no = np.random.normal(0, 1, (len(time)-1, 2))

# no = [
# [1.01324345095392, -0.142225283569095],
# [0.407344920643207, -0.151153289330984],
# [1.14314593040433, 0.318719872913361],
# [-0.853448402446952, -1.11877736557267],
# [-1.13309912333958, 0.483345180106232],
# [0.341397730575067, 0.508989589576643],
# [0.0498175119356292, -0.854868829818002],
# [-0.951439318747714, 0.0357402644626707],
# [-0.0672896257201675, -0.356285396191052],
# [-1.68511345286972, 0.849046371733583],
# [0.676759291387555, 0.135437683910847],
# [0.256137129570204, 0.759927977382943],
# [0.664414127701385, -1.47841589097176],
# [1.17493644019628, 0.219831847372367],
# [-0.661590129562252, -1.20538979497377],
# [-0.45175502708869, 0.208091637876728],
# [-0.0980938536639713, -0.679038919867732],
# [0.686432233941189, 0.636458630901011],
# [1.06463516603529, -0.808282650526677],
# [0.392008134801371, 0.0811519196324566],
# [-0.0547560571351863, 0.146932270725739],
# [1.15029196280438, -0.229011029944057],
# [-0.402510498759189, 1.67881792773982],
# [1.42852448407727, 1.23808893154354],
# [-1.18197317843088, -0.662686103970816],
# [-0.235361822006365, -1.99322251427358],
# [-0.401425133041613, -2.23845877545728],
# [-2.30914261567867, -0.860818001414857],
# [-0.754180911273729, -2.12717840498656],
# [-0.139636287156667, -0.245694813800602],
# [-0.462451250610224, 0.650783000210527],
# [0.766265809397281, 1.48273100479755],
# [-0.527690825871072, 0.5689576702246],
# [0.639955180431011, -0.672287960736216],
# [-0.417233981097452, -1.37665909765066],
# [-1.09947730581632, 1.22047960886136],
# [1.12202028403938, -0.171468619698071],
# [0.395820934298721, 2.81046621555568],
# [-1.01405034998498, -0.0646086198599094],
# [0.258564108227897, -0.571518288520795],
# [-0.0346640460292493, -0.537699007513607],
# [-0.3992497079824, 1.43882655065703],
# [-0.60924932586135, -0.690012226637994],
# [0.509389576682374, 0.318715070502666],
# [1.02381723196008, 2.33602455645169],
# [-0.350474064102046, 0.598985751890907],
# [1.41404403584175, 0.241508552260953],
# [0.321473516734551, 1.19821872203099],
# [1.02052831200169, 0.910920022350408],
# [-0.527588016113858, -0.16371200200994],
# [0.54449721756301, 0.445351927077147],
# [-1.80482988371221, -0.225746797552568],
# [2.19512788032745, -0.0787849382078636],
# [-0.207959821278037, -0.818672828429031],
# [0.714554753711663, -0.142687092383126],
# [1.25351682614961, -0.210809008352346],
# [0.773500861269606, 0.808903722025248],
# [-1.11925058689156, -0.189081009190524],
# [-0.27183872353785, -0.0463393453088817],
# [-0.288876395923868, 1.38239118044999],
# [0.384386222202134, 0.969852614732045],
# [0.155190310066371, -1.06218180916389],
# [0.0804386563612439, 2.40162029078653],
# [1.95740584233107, 1.64707486395158],
# [1.95979278879262, -1.03901759257765],
# [2.24328958181418, 0.198349066597986],
# [1.13843129392206, 0.448170486265491],
# [-0.888835852132315, 0.0876338232010184],
# [-1.74920621532156, 0.398644808205792],
# [-0.760661674377894, -0.675765712870144],
# [0.707542949204636, 0.492586749794515],
# [-0.0649033962224585, -0.0397569610959917],
# [-0.0790601757661689, 0.721367071231378],
# [-0.163978074165912, 0.948906041915486],
# [1.27171694764777, 0.377075931877217],
# [1.02575225809093, 0.167857676562958],
# [-0.117641875671809, 1.5415572025322],
# [0.522076514442483, 0.299562523387065],
# [1.55451111362411, 0.150994359739978],
# [-0.220156364751855, -0.0845102140559424],
# [-2.86391118170739, 1.49122135530833],
# [1.06650084722725, -0.51971530636786],
# [-3.04614375545796, 0.315528559789131],
# [0.185652270058855, 1.28583337713792],
# [-0.0832543036536954, -1.47577502381802],
# [-0.311791454070877, -0.997167926287072],
# [-0.673659400121962, 0.519440818571086],
# [0.603345759008325, 0.635619259407623],
# [1.27688012834029, 0.101970476724232],
# [0.559122693322207, 0.552007838919491],
# [0.43436725759261, -1.54226471033052],
# [-0.149302239543807, -0.424030281209169],
# [0.991991473021708, -0.99615119018846],
# [0.577261285265419, -0.712566273553651],
# [-2.14033315580991, -1.67488894772361],
# [0.352145450551827, 0.262131493990169],
# [0.485832349620454, -0.298310201117196],
# [-0.48766076758097, 0.643137681407291],
# [-0.0609415845690305, -1.72661202597394],
# [-0.665612917361851, -0.895132048958458],
#       ]
# no = np.array(no)


# y = simGBM(1, spot, mu, np.sqrt(theta), 1/days, no[:,0], 0)
# x = simHeston(1, spot, theta, mu, kappa, theta, sigma, rho, 1/days, no, 0)

# if standalone:
#     plt.figure(1)
# else:
#     plt.subplot(2, 2, 1)
# plt.plot(time, y, 'r--', time, x[:, 0], 'b', linewidth=1)
# if standalone: plt.title('GBM vs. Heston dynamics')
# plt.xlabel('Time [years]')
# plt.ylabel('FX rate')
# plt.ylim([3.5, 6])
# plt.legend(['GBM', 'Heston'], loc=2)

# if standalone:
#     plt.figure(2)
# else:
#     plt.subplot(2, 2, 2)
# plt.plot(time, 100 * np.sqrt(x[:, 1]), 'b', time, np.zeros(len(time)) + np.sqrt(theta) * 100, 'r--', linewidth=1)
# if standalone: plt.title('GBM vs. Heston volatility')
# plt.xlabel('Time [years]')
# plt.ylabel('Volatility [%]')
# plt.ylim([5, 35])
# plt.show()


#%% STF2hes02

# standalone = 0  # set to 0 to make plots as seen in STF2

# # Sample input:
# kappa = 2
# theta = 0.04
# sigma = 0.3
# rho = -0.05
# x = np.arange(-2, 2, 0.02)

# # Compute Heston marginal pdf at values in x (assuming the function pdfHeston is defined as in the previous code snippet)
# y = pdfHeston(x, theta, kappa, sigma, rho, 1, 0)

# # Compute GBM marginal pdf at values in x
# z = norm.pdf(x, 0, 0.2)

# # Compare marginal pdf with Gaussian density of N(0,0.2)
# if standalone:
#     plt.figure(1)
# else:
#     plt.subplot(2, 2, 1)

# plt.plot(x, z, 'r--', x, y, 'b', linewidth=1)
# if standalone: plt.title('Gaussian vs. Heston densities')
# plt.xlabel('x')
# plt.ylabel('PDF(x)')
# plt.legend(['GBM', 'Heston'], loc=2)
# plt.xlim([-1, 1])

# if standalone:
#     plt.figure(2)
# else:
#     plt.subplot(2, 2, 2)

# plt.semilogy(x, z, 'r--', x, y, 'b', linewidth=1)
# if standalone: plt.title('Gaussian vs. Heston log-densities')
# plt.xlabel('x')
# plt.ylabel('PDF(x)')
# plt.ylim([1e-8, 10])
# plt.yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])

# plt.show()



