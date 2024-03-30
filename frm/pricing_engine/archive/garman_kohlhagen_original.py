# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def garman_kohlhagen(S: float, 
                    σ: float, 
                    r_f: float,                    
                    r_d: float,  
                    tau: float, 
                    cp: int = None,
                    K: float = None,
                    Δ = None,
                    Δ_convention = None,
                    F: float = None,
                    task: str='px') -> float:
    '''
    GarmanKohlhagen European FX option pricing formula.
    :param S: FX spot, specified in # of units of domestic per 1 unit of foreign currency
    :param σ: volatility
    :param r_d: domestic risk free interest rate (annualised)
    :param r_f: foreign riskless interest rate (annualised)
    :param t: time to np.expiry (in years)
    :param cp: option type (1 for call option (default), -1 for put option)
    :param K: strike, specified in # of units of term currency per 1 unit of base currency
    :param Δ, signed delta, must be in range [0,0.5] for calls, [-0.5,0] for puts
    :param Δ_convention: delta quote convention {'regular_spot_Δ','regular_forward_Δ','premium_adjusted_spot_Δ','premium_adjusted_forward_Δ'}
    :param F: market forward rate, if None is calculated under interest rate parity 
    :param task: task to perform; 'px' (default), 'strike',  
    :return: option price (in the domestic currency, per 1 unit of foreign currency notional), 
             strike, calculated from delta
             greeks,
    '''
  
    
    S = np.atleast_1d(S).astype(float)
    assert (S > 0.0).all(), S
    
    σ = np.atleast_1d(σ).astype(float)
    # For numerical solving, may need to allow negative vol
    # assert (σ > 0.0).all(), σ
    
    tau = np.atleast_1d(tau).astype(float)
    assert (tau > 0.0).all(), tau  
  
    assert task in {'px','strike','greeks'}, task  
  
    if task == 'px':
        assert cp is not None, "'cp' is a required input for task 'px'"
        assert Δ is None, "'Δ' is not an input for task 'px'"
        assert Δ_convention is None, "'Δ_convention' is not an input for task 'px'"
        cp = np.atleast_1d(cp).astype(float)
        assert cp.shape == σ.shape
        assert np.all(np.isin(cp, [1, -1]) | np.isnan(cp))
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * σ**2) * tau) / (σ * np.sqrt(tau))
        d2 = d1 - σ * np.sqrt(tau)
    elif task == 'strike':
        assert Δ is not None, "'Δ' is a required input for task 'strike'"
        assert Δ_convention is not None, "'Δ_convention' is a required input for task 'strike'"
        assert cp is None, "'cp' is calculated from the 'Δ' input"
        Δ = np.atleast_1d(Δ).astype(float)
        assert Δ.shape == σ.shape
        assert (Δ >= -0.5).all() and (Δ <= 0.5).all()
        cp = np.sign(Δ)
        Δ_convention = Δ_convention.replace('delta','Δ')
        assert Δ_convention in {'regular_spot_Δ',
                                'regular_forward_Δ',
                                'premium_adjusted_spot_Δ',
                                'premium_adjusted_forward_Δ'}, Δ_convention

    if F is not None: 
        F = np.atleast_1d(F).astype(float)
        assert (F > 0.0).all() 
        
    if task == 'px':
        return cp * (S * np.exp(-r_f * tau) * norm.cdf(cp * d1) - K * np.exp(-r_d * tau) * norm.cdf(cp * d2))

    elif task == 'strike':
        result = np.zeros(shape=Δ.shape)
        if F == None:
            F = np.atleast_1d(S * np.exp((r_d - r_f) * tau)) # if not supplied, calculate the forward rate per interest rate parity 
        
        if Δ_convention in {'regular_spot_Δ','regular_forward_Δ'}:
            bool_cond = Δ == 0.5
            if bool_cond.any():
                # at-the-money Δ-neutral strike, for regular spot/forward Δ
                tmp = F * np.exp(0.5 * σ**2 * tau)
                result[bool_cond] = tmp[bool_cond]
            bool_cond = Δ != 0.5
            if bool_cond.any():
                if Δ_convention == 'regular_spot_Δ':
                    tmp = F * np.exp(-cp * norm.ppf(cp * Δ * np.exp(r_f * tau)) * σ * np.sqrt(tau) + (0.5 * σ**2) * tau)
                    result[bool_cond] = tmp[bool_cond]
                elif Δ_convention == 'regular_forward_Δ':
                    tmp = F * np.exp(-cp * norm.ppf(cp * Δ) * σ * np.sqrt(tau) + (0.5 * σ**2) * tau)
                    result[bool_cond] = tmp[bool_cond]
        
        elif Δ_convention in {'premium_adjusted_spot_Δ','premium_adjusted_forward_Δ'}:
            bool_cond = Δ == 0.5
            if bool_cond.any():
                # at-the-money Δ-neutral strike, for premium adjusted spot/forward Δ
                tmp = F * np.exp(-1 * 0.5 * σ**2 * tau)
                result[bool_cond] = tmp[bool_cond]
                
            bool_cond = Δ != 0.5                
            if bool_cond.any(): 
                # For premium adjusted quotes the solution must be solved numerically
                # Please refer to 'A Guide to FX Options Quoting Conventions' by Uwe Wystub for full details
    
                # For the premimum adjusted call Δ, due to non-monotonicity, two strikes can be solved numerically. 
                # To avoid this, we solve bounds to get the correct solution
                
                # 1. Solve the K_max, the upper bound for the numerical solver
                # The premium adjusted Δ is always below the regular (non premium adjusted) Δ
                K_max = garman_kohlhagen(S=S, 
                                         σ=σ, 
                                         r_f=r_f, 
                                         r_d=r_d, 
                                         tau=tau, 
                                         cp=cp, 
                                         Δ=Δ, 
                                         Δ_convention=Δ_convention.replace('premium_adjusted','regular'),
                                         F=F, 
                                         task='strike')  
                
                # 2. Solve K_min, the lower bound for the numerical solver
                # The lower bound is the 'maximum' Δ, hence we numerically solve the maximum Δ
                def solve_K_min(K, σ, cp, F, t):
                    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * σ**2) * t) / (σ * np.sqrt(t))
                    d2 = d1 - σ * np.sqrt(t)
                    return σ * np.sqrt(t) * norm.cdf(d2) - norm.pdf(d2)
                solution = root_scalar(solve_K_min, args=(σ, cp, F, tau), x0=F ,bracket=[0.00001, K_max])
                
                if solution.converged:
                    K_min = solution.root
                else:
                    raise ValueError('the numerical solver for K_min did not converge')
                
                if Δ_convention == 'premium_adjusted_spot_Δ':
                    def solve_Δ(K, σ, cp, F, tau, Δ):
                        return np.exp(r_d * tau) * (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ                
                elif Δ_convention == 'premium_adjusted_forward_Δ':
                    def solve_Δ(K, σ, cp, F, tau, Δ):
                        return (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ
                    
                solution = root_scalar(solve_Δ, args=(σ, cp, F, tau, Δ), x0=F, bracket=[K_min,K_max])
                
                if solution.converged:
                    return solution.root
                else:
                    raise ValueError('the numerical solver for K_min did not converge')
                    
                    
        return result
                    
                
    elif task == 'greeks':
        # INCOMPLETE
        greeks = {}
        
        # Δ appears to be as a % 
        # vega appears to be for 1 unit of term ccy compared to SD, check bbg
        # theta appears to be for 1 unit of base ccy compared to SD, check bbg
        
        # S = 1.1 #EURUSD
        # f = 1.101070 #EURUSD
        # K = 1.101
        # r_f = 0.05 # EUR
        # r_d = 0.1 # USD
        # σ = 0.1
        # days = 7
        # t = days/365
 
        # d1 = (log(S / K) + (r_d - r_f + 0.5 * σ**2) * t) / (σ * np.sqrt(t))
        # d2 = d1 - σ * np.sqrt(t)
    
        
        greeks['spot_Δ'] = cp * np.exp(-r_f * tau) * norm.cdf(cp * d1)
        greeks['fwd_Δ'] = cp * norm.cdf(cp * d1) 
        greeks['vega'] = S * np.sqrt(tau) * norm.pdf(d1) * np.exp(-r_f * tau) # identical for calls and puts
        greeks['theta'] = -(S*np.exp(-r_d*tau)*norm.pdf(d1 * cp)*σ)/(2*np.sqrt(tau)) \
            + r_f * np.exp(-r_f * tau) * S * norm.cdf(d1 * cp) \
            - r_d * np.exp(-r_d * tau) * K * norm.cdf(d2 * cp)
        greeks['gamma'] = np.exp(-r_f * tau) * norm.pdf(d1)  / (S * σ * np.sqrt(tau)) # identical for calls and puts


        return greeks
        
        
#%%


if __name__ == '__main__':
    # AUD put, put on base ccy
    greeks = garman_kohlhagen(S=0.662866,σ=0.1055564,r_f=0.0466,r_d=0.05381,tau=1.0,cp=-1,K=0.64050,task='greeks')
    px = garman_kohlhagen(S=0.662866,σ=0.1055564,r_f=0.0466,r_d=0.05381,tau=1.0,cp=-1,K=0.64050,task='px')






#%%

# # 2Y USDJPY call, (USD call, JPY Put), 30 June 2023, London 10pm data
# S = 144.32 # 20 Δ Call
# K = 145.83
# σ = 0.0960777
# cp = 1
# r_f = 0.04812
# r_d = -0.00509
# t = 2.0
# F = 129.7958
# Δ = 0.2
# Δ_convention = 'premium_adjusted_fwd_Δ'

# K_IDD = 145.83

# result = garman_kohlhagen(S=S,σ=σ,r_f=r_f,r_d=r_d,t=t,cp=cp,Δ=Δ,F=F,task='strike',Δ_convention=Δ_convention)

# print('result', result)
# print('variance', result - K_IDD)
# print('% variance', 100 * (result - K_IDD) / result)

# #%%

# # 9M USDJPY call, (USD call, JPY Put), 30 June 2023, London 10pm data
# S = 144.32 # 20 Δ Call
# K = 145.83
# σ = 0.0960777
# cp = 1
# r_f = 0.05413
# r_d = -0.00528
# t = 0.75
# F = 138.031
# Δ = 0.2
# Δ_convention = 'premium_adjusted_spot_Δ'

# K_IDD = 148.11

# result = garman_kohlhagen(S=S,σ=σ,r_f=r_f,r_d=r_d,t=t,cp=cp,Δ=Δ,F=F,task='strike',Δ_convention=Δ_convention)

# print('result', result)
# print('variance', result - K_IDD)
# print('% variance', 100 * (result - K_IDD) / result)




#%%
















