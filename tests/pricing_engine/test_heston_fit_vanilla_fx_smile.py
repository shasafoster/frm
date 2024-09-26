# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 


# This test script is a check to "STF2hes07.m" which is a support to
# "FX smile in the Heston model" by A Janek, 2010.
# Source: https://ideas.repec.org/c/wuu/hscode/zip10001.html
# WaybackMachine save: 
# https://web.archive.org/web/20240926111306/https://ideas.repec.org/c/wuu/hscode/zip10001.html

from frm.pricing_engine.heston_garman_kohlhagen import heston_fit_vanilla_fx_smile, VALID_HESTON_PRICING_METHODS
import numpy as np
import matplotlib.pyplot as plt


def test_heston_vanilla_fx_smile():   
    # Initialize
    Δ_for_plt = np.array([0.1, 0.25, 0.5, 0.75, 0.9]) # forward deltas
    
    Δ = np.array([0.1, 0.25, 0.5, -0.25, -0.1])
    r_d = np.array([0.31100, 0.32875, 0.49781, 0.70075, 1.08, 1.08]) * 0.01
    r_f = np.array([0.58, 0.631, 0.884, 1.131, 1.399, 1.399]) * 0.01
    tau = np.array([7 / 365, 1 / 12, 3 / 12, 6 / 12, 1, 2])
    S0 = 1.2779
    
    cp = np.array([1, 1, 1, -1, -1])
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
    
    # Results of STF2hes07.m
    # [IV (10, 25, ATM, 75, 90), SSE] * 100%
    MATLAB_IV_SSE = [
        np.array([13.9211, 13.2659, 13.2548, 14.1217, 15.4063, 0.0002]), # 1W 
        np.array([12.4581, 12.2439, 12.6723, 13.8030, 15.1972, 0.0000]), # 1M
        np.array([12.6938, 12.4002, 13.0049, 14.6137, 16.6331, 0.0001]), # 3M
        np.array([13.6671, 12.9641, 13.4527, 15.3880, 18.0948, 0.0002]), # 6M
        np.array([13.6178, 12.7871, 13.0836, 14.7562, 17.2699, 0.0003]),  # 1Y
        np.array([12.1160, 11.8254, 12.1683, 13.2481, 14.7866, 0.0000])  # 2Y
    ]
    
    # Results of STF2hes07.m 
    # v0, vv, kappa, theta, rho, 2*kappa*theta - vv^2
    MATLAB_heston_params = [
        np.array([0.0178, 1.2971, 1.5000, 0.1650, -0.1577, -1.1874]),  # 1W 
        np.array([0.0161, 0.5836, 1.5000, 0.0487, -0.3143, -0.19460]), # 1M
        np.array([0.0170, 0.4804, 1.5000, 0.0362, -0.3763, -0.1224]),  # 3M
        np.array([0.0184, 0.5251, 1.5000, 0.0354, -0.3572, -0.1695]),  # 6M
        np.array([0.0174, 0.4724, 1.5000, 0.0277, -0.3114, -0.1400]),  # 1Y
        np.array([0.0148, 0.3159, 1.5000, 0.0190, -0.3004, -0.0430])   # 2Y
    ]
    
    pricing_method ='heston_cosine'
    assert pricing_method in VALID_HESTON_PRICING_METHODS
    
    # Main loop for various smiles
    for i, σ_market in enumerate(σ_market_set):
        
        var0, vv, kappa, theta, rho, lambda_, IV, SSE = heston_fit_vanilla_fx_smile(Δ, Δ_convention, σ_market, S0, r_f[i], r_d[i], tau[i], cp, pricing_method=pricing_method)        
        params = np.array([var0, vv, kappa, theta, rho])
        var0 = round(var0.item(), 6)
        vv = round(vv.item(), 6)
        theta = round(theta.item(), 6)
        rho = round(rho.item(), 6)
        
        IV *= 100
        IV_list = [round(v.item(), 6) for v in list(IV)]
        SSE = round(100 * SSE.item(), 6) 
        
        IV_MATLAB = MATLAB_IV_SSE[i][:-1]
        
        # Check frm fit matches the MATLAB fit
        assert (np.abs(IV - IV_MATLAB) < 0.01).all() # 0.01 = 0.01%
        assert (np.abs(params - MATLAB_heston_params[i][:-1]) / params  < 0.01).all() # Params are within 1%
        
        # Want to run if running in script, but not in pytest
        if __name__ == "__main__":
        
            # Displaying output
            print(f'=== {tenors[i]} calibration results ===')
            print(f'v0, vv, kappa, theta, rho: {var0, vv, kappa, theta, rho}')
            print(f'[IV (10, 25, ATM, 75, 90), SSE] * 100%: {IV_list, SSE}')
            
            # Plot 
            plt.figure(i+1)
            plt.plot(Δ_for_plt * 100, σ_market * 100, 'ko-', linewidth=1)
            plt.plot(Δ_for_plt * 100, IV, 'bs-', linewidth=1)
            plt.plot(Δ_for_plt * 100, MATLAB_IV_SSE[i][:-1], 'rs--', linewidth=1)
            plt.legend([f'{tenors[i]} smile', f'frm - Heston fit via {pricing_method}', 'MATLAB - Heston fit'], loc='upper left')
            plt.xlabel('Delta [%]')
            plt.ylabel('Implied volatility [%]')
            plt.xticks(Δ_for_plt * 100)
            plt.title('Comparison b/t market vols and calculated implied vols')
            plt.show()
    

if __name__ == "__main__":
    test_heston_vanilla_fx_smile()