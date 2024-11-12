# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import numpy as np
from frm.pricing_engine.garman_kohlhagen import garman_kohlhagen_price, garman_kohlhagen_solve_strike_from_delta, garman_kohlhagen_solve_implied_vol

def test_garman_kohlhagen_price_and_solve_implied_vol():

    epsilon_px = 0.0006 # 0.06% of notional
    epsilon_σ = 0.0001 # 0.0001%    
    ##### AUDUSD Tests, function results in USD per 1 AUD ######
    
    # 1Y AUDUSD Call, 30 June 2023, London 8am
    px_test = 0.00133006 # 1330.06 USD per A$1,000,000 notional
    S0 = 00.6629
    vol = 0.098408
    r_d = 0.05381 # USD risk free rate
    r_f = 0.0466 # AUD risk free rate
    tau = 1.0 # Time to expiry in years
    cp = 1
    K = 0.7882
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    assert abs(px_test - px) < epsilon_px
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ
    
    # 1Y AUDUSD Put, 30 June 2023, London 8am
    px_test = 0.11533070 # 115,330.7 USD per A$1,000,000 notional
    S0=0.662866
    vol=0.0984338
    r_d=0.05381
    r_f=0.0466
    tau=1.0
    cp=-1
    K=0.7882
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,vol=vol)['price']
    assert abs(px_test - px) < epsilon_px
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ 
    
    # 1Y AUDUSD Call, 30 June 2023, London 8am
    px_test = 0.12284766 
    S0 = 0.662866
    vol = 0.1354523
    r_d = 0.05381 # USD risk free rate
    r_f = 0.0466 # AUD risk free rate
    tau = 1.0 # Time to expiry in years
    cp = 1
    K = 0.5405
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    assert abs(px_test - px) < epsilon_px  
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ
          
    # 1Y AUDUSD Put, 30 June 2023, London 8am
    px_test = 0.00200167
    S0 = 0.662866
    vol = 0.1354523
    r_d = 0.05381 # USD risk free rate
    r_f = 0.0466 # AUD risk free rate
    tau = 1.0 # Time to expiry in years
    cp = -1
    K = 0.5405
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    assert abs(px_test - px) < epsilon_px     
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ    
    
    ##### AUDUSD Tests, function results in AUD per 1 USD ######
    
    # 1Y USDAUD Put, 30 June 2023, London 8am
    px_test = 0.00133006
    S0 = 1.0 / 0.662866
    vol = 0.0984338
    r_d = 0.0466 # AUD risk free rate
    r_f = 0.05381 # USD risk free rate
    tau = 1.0 # Time to expiry in years
    cp = -1
    K = 1.0 / 0.7882
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    px_in_AUD_per_1_AUD = px / S0 / K
    assert abs(px_test - px_in_AUD_per_1_AUD) < epsilon_px
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ    
    
    # 1Y USDAUD Call, 30 June 2023, London 8am
    px_test = 0.11533070 
    S0 = 1.0 / 0.662866
    vol = 0.0984338
    r_d = 0.0466
    r_f = 0.05381
    tau = 1.0
    cp = 1
    K = 1.0 / 0.7882
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    px_in_AUD_per_1_AUD = px / S0 / K
    assert abs(px_test - px_in_AUD_per_1_AUD) < epsilon_px
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ   

    # 1Y USDAUD Put, 30 June 2023, London 8am
    px_test = 0.12284766
    S0 = 1.0 / 0.662866
    vol = 0.1354523
    r_d = 0.0466
    r_f = 0.05381
    tau = 1.0
    cp = -1
    K = 1.0 / 0.5405
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    px_in_AUD_per_1_AUD = px / S0 / K
    assert abs(px_test - px_in_AUD_per_1_AUD) < epsilon_px 
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau,r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ   

    # 1Y USDAUD Call, 30 June 2023, London 8am
    px_test = 0.00200167
    S0 = 1.0 / 0.662866
    vol = 0.1354523
    r_d = 0.0466
    r_f = 0.05381
    tau = 1.0
    cp = 1
    K = 1.0 / 0.5405
    px = garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['price']
    px_in_AUD_per_1_AUD = px / S0 / K
    assert abs(px_test - px_in_AUD_per_1_AUD) < epsilon_px
    IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=px, vol_guess=0.1)
    assert 100* abs(vol - IV) < epsilon_σ  


def test_garman_kohlhagen_solve_strike_from_delta():
    
    epsilon_px = 0.001 # 0.1 % 

    # AUDUSD 1Y 30 Δ Put, 30 June 2023 London 10pm 
    strike_test = 0.64100
    S0 = 0.6662
    vol = 0.1064786
    r_f = 0.04655
    r_d = 0.05376
    tau = 1.0
    signed_delta = -0.3
    delta_convention = 'regular_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['spot_delta'].iloc[0] - signed_delta) < 1e-8

    # AUDUSD 1Y 30 Δ Call, 30 June 2023 London 10pm 
    strike_test = 0.70690
    S0 = 0.6662
    vol = 0.0963895
    r_f = 0.04655
    r_d = 0.05376
    tau = 1.0
    signed_delta = 0.3
    delta_convention = 'regular_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['spot_delta'].iloc[0] - signed_delta) < 1e-8

    # AUDUSD 1Y 5 Δ Put, 30 June 2023 London 10pm 
    strike_test = 0.54280
    S0 = 0.6662
    vol = 0.1359552
    r_f = 0.04655
    r_d = 0.05376
    tau = 1.0
    signed_delta = -0.05
    delta_convention = 'regular_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['spot_delta'].iloc[0] - signed_delta) < 1e-8

    # AUDUSD 1Y 5 Δ Call, 30 June 2023 London 10pm 
    epsilon_px = 0.0015 # 0.15 % higher tolerance for this one 
    strike_test = 0.79300
    S0 = 0.6662
    vol = 0.0990869
    r_f = 0.04655
    r_d = 0.05376
    tau = 1.0
    signed_delta = 0.05
    delta_convention = 'regular_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['spot_delta'].iloc[0] - signed_delta) < 1e-8


    # AUDUSD 5Y 30 Δ Put, 30 June 2023 London 10pm 
    epsilon_px = 0.0034 # 0.34 % higher tolerance for this one 
    strike_test = 0.59180
    S0 = 0.6662
    vol = 0.1161961
    r_f = 0.04287
    r_d = 0.03902
    tau = 5.0
    signed_delta = -0.3
    delta_convention = 'regular_forward'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['forward_delta'].iloc[0] - signed_delta) < 1e-8


    # AUDUSD 5Y 30 Δ Put, 30 June 2023 London 10pm 
    epsilon_px = 0.0036 # 0.36 % higher tolerance for this one 
    strike_test = 0.75820
    S0 = 0.6662
    vol = 0.1016627
    r_f = 0.04287
    r_d = 0.03902
    tau = 5.0
    signed_delta = 0.3
    delta_convention = 'regular_forward'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px
    result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=np.sign(signed_delta), K=strike, vol=vol, analytical_greeks=True)
    assert abs(result['analytical_greeks']['forward_delta'].iloc[0] - signed_delta) < 1e-8

    epsilon_px = 0.001
    
    # 2Y USDJPY call, (USD call, JPY Put), 30 June 2023, London 10pm data
    strike_test = 145.83
    S0 = 144.32 # 20 Δ Call
    vol = 0.0960777
    r_f = 0.04812
    r_d = -0.00509
    tau = 2.0
    F = 129.7958
    signed_delta = 0.2
    delta_convention = 'premium_adjusted_forward'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)
    assert abs(strike_test - strike) / strike_test < epsilon_px


    # 9M USDJPY call, (USD call, JPY Put), 30 June 2023, London 10pm data
    strike_test = 148.11
    S0 = 144.32 # 20 Δ Call
    vol = 0.0985008
    r_f = 0.05413
    r_d = -0.00528
    tau = 0.75
    F = 138.031
    signed_delta = 0.2
    delta_convention = 'premium_adjusted_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention, F=F)
    assert abs(strike_test - strike) / strike_test < epsilon_px
        
        
def test_gk_price_greeks():
    
    pass
    # 1Y AUDUSD Call, data from 30 June 2023, London 8am
    # S0=0.6629
    # vol=9.84251/100
    # r_f=0.0466
    # r_d=0.05381
    # tau=1.0
    # cp=1
    # K=0.7882
    # F=0.667962
    # result = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d,r_f=r_f, cp=cp, K=K, vol=vol, F=F, analytical_greeks_flag=True, numerical_greeks_flag=True)
    #print('X:',X)
    #print('greeks_analytical:',greeks_analytical)   
    #print('greeks_numerical:',greeks_numerical)   



if __name__ == '__main__':
    test_garman_kohlhagen_price_and_solve_implied_vol()
    test_garman_kohlhagen_solve_strike_from_delta()
    


    epsilon_px = 0.001 # 0.1 %

    # AUDUSD 1Y 30 Δ Put, 30 June 2023 London 10pm
    strike_test = 0.64100
    S0 = 0.6662
    vol = 0.1064786
    r_f = 0.04655
    r_d = 0.05376
    tau = 1.0
    signed_delta = -0.3
    delta_convention = 'regular_spot'
    strike = garman_kohlhagen_solve_strike_from_delta(S0=S0,tau=tau,r_d=r_d,r_f=r_f,vol=vol,signed_delta=signed_delta,delta_convention=delta_convention)



    assert abs(strike_test - strike) / strike_test < epsilon_px
        
    # S0=0.6629
    # vol=0.0984688
    # r_f=0.0466
    # r_d=0.05381
    # tau= 1.00957592339261
    # cp=1
    # K=0.7882
    # F=0.667962
    # result  = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=vol, F=F, analytical_greeks_flag=True, numerical_greeks_flag=True)
    
    # X = result['option_value'].item()
    
    # IV = garman_kohlhagen_solve_implied_vol(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, X=X, vol_guess=0.1)

    

    
    