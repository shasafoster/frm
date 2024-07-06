# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar, newton

def gk_price(S0: float, 
             tau: float, 
             r_f: float,  
             r_d: float,               
             cp: int,
             K: float,
             σ: float, 
             F: float = None,
             analytical_greeks_flag: bool=False,
             numerical_greeks_flag: bool=False,
             intrinsic_time_split_flag: bool=False
             ) -> float:
    """
    GarmanKohlhagen European FX option pricing formula.
    - if tau is 0, intrinsic value is returned
    x
    :param S0: FX spot, specified in # of units of domestic per 1 unit of foreign currency
    :param tau: time to np.expiry (in years)
    :param r_d: domestic risk free interest rate (annualised)
    :param r_f: foreign riskless interest rate (annualised)
    :param cp: option type (1 for call option (default), -1 for put option)
    :param K: strike, specified in # of units of term currency per 1 unit of base currency
    :param σ: volatility
    :param F: market forward rate, if None this is calculated under interest rate parity     
    :param analytical_greeks_flag: boolean flag for in/excluding analytically calculated greeks
    :param numerical_greeks_flag: boolean flag for in/excluding numerically calculated greeks
    :return: option price (in the domestic currency, per 1 unit of foreign currency notional), 
             analytical greeks (if analytical_greeks_flag=True),
             numerical greeks (if numerical_greeks_flag=True)
             
    """
  
    S0 = np.atleast_1d(S0).astype(float)
    tau = np.atleast_1d(tau).astype(float)
    r_f = np.atleast_1d(r_f).astype(float)
    r_d = np.atleast_1d(r_d).astype(float)
    cp = np.atleast_1d(cp).astype(float)
    K = np.atleast_1d(K).astype(float)
    σ = np.atleast_1d(σ).astype(float)
    
    # Sensical value checks
    # No >0 check for σ, as when doing numerical solving, need to allow for -ve σ
    assert (S0 > 0.0).all(), S0
    assert (tau >= 0.0).all(), tau # if tau is 0.0, return instrinsic value
    assert np.all(np.isin(cp, [1, -1])), cp
    assert cp.shape == σ.shape
    assert cp.shape == K.shape
    
    results = dict()
    
    if F is not None: 
        F = np.atleast_1d(F).astype(float)
        assert (F > 1e-8).all() 
        
        # Use market forward rate and imply basis-adjusted domestic interest rate        
        r_d_basis_adj = np.log(F / S0) / tau + r_f # from F = S0 * exp((r_d - r_f) * tau)
        d1 = (np.log(F / K) + (0.5 * σ**2) * tau) / (σ * np.sqrt(tau))
        d2 = d1 - σ * np.sqrt(tau)    
        X = cp * np.exp(-r_d_basis_adj * tau) * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))   
        
        # If time-to-maturity is 0.0, set to intrinsic value 
        X[tau==0] = np.maximum(0, cp * (S0 - K))[tau==0]
        results['option_value'] = X
        
        if intrinsic_time_split_flag:
            X_intrinsic = np.full_like(X, np.nan)
            X_time = np.full_like(X, np.nan)            
            X_intrinsic[tau>0] = np.maximum(0, cp * np.exp(-r_d_basis_adj * tau) * (F - K))[tau>0]
            X_intrinsic[tau==0] = X[tau==0]
            X_time = X - X_intrinsic
            results['intrinsic_value'] = X_intrinsic
            results['time_value'] = X_time
    else:
        # Under interest rate parity 
        F = S0 * np.exp((r_d - r_f) * tau)
        d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * σ**2) * tau) / (σ * np.sqrt(tau)) # Not actually used in the pricing
        d2 = d1 - σ * np.sqrt(tau)    
        X = cp * (S0 * np.exp(-r_f * tau) * norm.cdf(cp * d1) - K * np.exp(-r_d * tau) * norm.cdf(cp * d2))
        
        # If time-to-maturity is 0.0, set to intrinsic value 
        X[tau==0] = np.maximum(0, cp * (S0 - K))[tau==0]
        results['option_value'] = X
        
        if intrinsic_time_split_flag:
            X_intrinsic = np.full_like(X, np.nan)
            X_time = np.full_like(X, np.nan)            
            X_intrinsic[tau>0] = np.maximum(0, cp * (S0 * np.exp(-r_f * tau) - K * np.exp(-r_d * tau)))[tau>0]
            X_intrinsic[tau==0] = X[tau==0]
            X_time = X - X_intrinsic 
            results['intrinsic_value'] = X_intrinsic
            results['time_value'] = X_time            
        
    if not (analytical_greeks_flag or numerical_greeks_flag):
        return results
    else:
        Δ_shift = 1 / 100 # 1% shift
        σ_shift = 1 / 100 # 1% shift
        θ_shift = 1 / 365.25 # 1 Day
        ρ_shift = 1 / 100 # 1% shift
    
        if analytical_greeks_flag:
            analytical_greeks = {}
            
            ######################## 1st Order Greeks #########################
            
            # Delta, Δ, is the change in an option's price for a small change in the underlying assets price.
            # Δ := ∂X/∂S ≈ (X(S_plus) − X(S_minus)) / (S_plus - S_minus)
            # where X is the option price (whose units is DOM),
            # and S0 is the fx spot price (whose units is DOM/FOR)
            # We multiply by S0 so to return the Δ a % applicable to  to the foreign currency notional
            analytical_greeks['spot_delta'] = S0 * cp * np.exp(-r_f * tau) * norm.cdf(cp * d1)
            analytical_greeks['forward_delta'] = S0 * cp * norm.cdf(cp * d1) 
    
            # Vega, ν, is the change in an options price for a small change in the volatility input
            # ν = ∂X/∂σ ≈ (X(σ_plus) − X(σ_minus)) / (σ_plus - σ_minus)
            # In practice, vega is normalised to measure the change in price for a 1% change in the volatility input
            # Hence we have scaled the numerical vega to a 1% change
            analytical_formula = S0 * np.sqrt(tau) * norm.pdf(d1) * np.exp(-r_f * tau) # identical for calls and puts
            analytical_greeks['vega'] = analytical_formula * 0.01 # normalised to 1% change
    
            # Theta, θ, is the change in an options price for a small change in the time to expiry
            # In practice, theta is normalised to measure the change in price, for a 1 day shorter expiry 
            # Theta includes
            # 1. Time decay: the change in the option's value as time moves forward, that is, as the expiry date moves closer, 
            #               i.e., the time value price tomorrow minus the time value price today.
            # 2. Cost of carry: the interest rate sensitivity that causes the value of the portfolio to change as time progresses, 
            #                   e.g., the cost of carry on spots and forwards is included in the theta value.
            analytical_formula = -(S0*np.exp(-r_d*tau)*norm.pdf(d1 * cp)*σ)/(2*np.sqrt(tau)) \
                + r_f * np.exp(-r_f * tau) * S0 * norm.cdf(d1 * cp) \
                - r_d * np.exp(-r_d * tau) * K * norm.cdf(d2 * cp)
            analytical_greeks['theta'] = analytical_formula * θ_shift
                
            # Gamma, Γ, is the change in an option's delta for a small change in the underlying assets price.
            # Gamma := ∂Δ/∂S ≈ (Δ(S_plus) − Δ(S_minus)) / (S_plus - S_minus)
            # In practice, gamma is  normalised to measure the change in Δ, for a 1% change in the underlying assets price.
            # Hence we have multiplied the analystical gamma formula by 'S * 0.01'
            analytical_formula = np.exp(-r_f * tau) * norm.pdf(d1)  / (S0 * σ * np.sqrt(tau)) # identical for calls and puts
            analytical_greeks['gamma'] = (0.01 * S0) * analytical_formula # normalised for 1% change
            
            # Rho, ρ, is the rate at which the price of an option changes relative to a change in the interest rate. 
            # In practice, Rho is normalised to measure the change in price for a 1% change in the underlying interest rate.
            analytical_formula = K * tau * np.exp(-r_f * tau) * norm.cdf(cp * d2)
            analytical_greeks['rho'] = analytical_formula * 0.01
            
            analytical_greeks = pd.DataFrame.from_dict(analytical_greeks)
            results['analytical_greeks'] = analytical_greeks

    
        if numerical_greeks_flag:
            numerical_greeks = {}
            
            if F != None:
                F_upshift = F*(1+Δ_shift)
                F_downshift = F*(1-Δ_shift)
            else:
                F_upshift = None
                F_downshift = None
            results_S0_plus = gk_price(S0=S0*(1+Δ_shift), tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ, F=F_upshift,analytical_greeks_flag=True) 
            X_S_plus, analytical_greeks_S0_plus = results_S0_plus['option_value'], results_S0_plus['analytical_greeks']
            results_S0_minus = gk_price(S0=S0*(1-Δ_shift), tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ, F=F_downshift,analytical_greeks_flag=True)
            X_S_minus, analytical_greeks_S0_minus = results_S0_minus['option_value'], results_S0_minus['analytical_greeks']
            
            numerical_greeks['spot_delta'] = (X_S_plus - X_S_minus) / (2 * S0 * Δ_shift)
            numerical_greeks['forward_delta'] = numerical_greeks['spot_delta'] / np.exp(-r_f * tau)
            
            X_σ_plus = gk_price(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ+σ_shift, F=F)['option_value']
            X_σ_minus = gk_price(S0=S0, tau=tau, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ-σ_shift, F=F)['option_value'] 
            numerical_greeks['vega'] = (X_σ_plus - X_σ_minus) / (2 * (σ_shift / 0.01))
            
            X_plus = gk_price(S0=S0, tau=tau+θ_shift, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ, F=F)['option_value']     
            X_minus = gk_price(S0=S0, tau=tau-θ_shift, r_f=r_f, r_d=r_d, cp=cp, K=K, σ=σ, F=F)['option_value']  
            numerical_greeks['theta'] = (X_minus - X_plus) / 2 
            
            numerical_greeks['gamma'] = (analytical_greeks_S0_plus['spot_delta'] - analytical_greeks_S0_minus['spot_delta']) / (2 * (Δ_shift / 0.01))
            
            # This formulae will yield meaningfully different results on whether the forward rate is an input (or if it is calculated from the interest rate differential)
            X_ρ_plus = gk_price(S0=S0, tau=tau, r_f=r_f+ρ_shift, r_d=r_d, cp=cp, K=K, σ=σ+σ_shift, F=F)['option_value'] 
            X_ρ_minus = gk_price(S0=S0, tau=tau, r_f=r_f+ρ_shift, r_d=r_d, cp=cp, K=K, σ=σ-σ_shift, F=F)['option_value'] 
            numerical_greeks['rho'] = (X_ρ_plus - X_ρ_minus) / 2 
            
            numerical_greeks = pd.DataFrame.from_dict(numerical_greeks)
            results['numerical_greeks'] = numerical_greeks
            
        return results
            
def gk_solve_strike(S0: float, 
                   tau: float,                
                   r_f: float,                    
                   r_d: float,  
                   σ: float,                     
                   Δ: float,
                   Δ_convention: str,
                   F: float = None
                   ) -> float:
    """
    GarmanKohlhagen European FX option pricing formula.
    :param S0: FX spot, specified in # of units of domestic per 1 unit of foreign currency
    :param tau: time to np.expiry (in years)
    :param r_d: domestic risk free interest rate (annualised)
    :param r_f: foreign riskless interest rate (annualised)
    :param cp: option type (1 for call option (default), -1 for put option)
    :param σ: volatility
    :param F: market forward rate, if None is calculated under interest rate parity     
    :param Δ, signed delta, must be in range [0,0.5] for calls, [-0.5,0] for puts
    :param Δ_convention: delta quote convention {'regular_spot_Δ','regular_forward_Δ','premium_adjusted_spot_Δ','premium_adjusted_forward_Δ'}
    :return: strike, calculated from Δ
    """
  
    S0 = np.atleast_1d(S0).astype(float)
    tau = np.atleast_1d(tau).astype(float)
    r_f = np.atleast_1d(r_f).astype(float)
    r_d = np.atleast_1d(r_d).astype(float)
    σ = np.atleast_1d(σ).astype(float)
    Δ = np.atleast_1d(Δ).astype(float)
    
    # Sensical value checks
    assert (S0 > 0.0).all(), S0
    assert (tau > 0.0).all(), tau
    assert (σ > 0.0).all(), σ
    assert (Δ >= -0.5).all() and (Δ <= 0.5).all()
    assert Δ.shape == σ.shape
    cp = np.sign(Δ)
    Δ_convention = Δ_convention.replace('delta','Δ')
    assert Δ_convention in {'regular_spot_Δ',
                            'regular_forward_Δ',
                            'premium_adjusted_spot_Δ',
                            'premium_adjusted_forward_Δ'}, Δ_convention    

    if F is not None: 
        F = np.atleast_1d(F).astype(float)
        assert (F > 0.0).all() 

    result = np.zeros(shape=Δ.shape)
    if F == None:
        F = np.atleast_1d(S0 * np.exp((r_d - r_f) * tau)) # if not supplied, calculate the forward rate per interest rate parity 
    
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
            for i in range(len(Δ)):
                if Δ[i] != 0.5:
                    
                    # For premium adjusted quotes the solution must be solved numerically
                    # Please refer to 'A Guide to FX Options Quoting Conventions' by Uwe Wystub for full details
        
                    if Δ_convention == 'premium_adjusted_spot_Δ':
                        def solve_Δ(K, σ, cp, F, tau, Δ):
                            return np.exp(-r_f * tau) * (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ                
                    elif Δ_convention == 'premium_adjusted_forward_Δ':
                        def solve_Δ(K, σ, cp, F, tau, Δ):
                            return (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ
        
                    # Solve the upper bound, 'K_max' for the numerical solver
                    # The strike of a premium adjusted Δ-σ quote is ALWAYS below the regular (non premium adjusted) Δ-σ quote
                    # Hence, we analytically calculate the K, assuming the σ was a regular Δ quote
                    K_max = gk_solve_strike(S0=S0, 
                                            tau=tau,
                                            r_f=r_f, 
                                            r_d=r_d,  
                                            σ=σ[i], 
                                            Δ=Δ[i], 
                                            Δ_convention=Δ_convention.replace('premium_adjusted','regular'),
                                            F=F)
                    K_max = K_max.item()
    
                    # Put Option
                    if Δ[i] < 0: 
                        solution = root_scalar(solve_Δ, args=(σ[i], cp[i], F, tau, Δ[i]), x0=F, bracket=[0.00001, K_max])
                        
                    # Call Option
                    if Δ[i] > 0: 
                        # For the premimum adjusted call Δ, due to non-monotonicity, two strikes can be solved numerically for call options (but not put options).
                        # To avoid this, we solve a lower bound, 'K_min' to guarantee we get the correct solution in the numerical solver.
                        # The lower bound is the 'maximum' Δ, hence we numerically solve the maximum Δ
                        def solve_K_min(K, σ, cp, F, t):
                            d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * σ**2) * t) / (σ * np.sqrt(t))
                            d2 = d1 - σ * np.sqrt(t)
                            return σ * np.sqrt(t) * norm.cdf(d2) - norm.pdf(d2)
                        
                        try: 
                            solution = root_scalar(solve_K_min, args=(σ[i], cp[i], F, tau), x0=F ,bracket=[0.00001, K_max])
                        except ValueError('the numerical solve for K_min, for Δ', Δ[i], ', resulted in an error'):
                            pass
                    
                        if solution.converged:
                            K_min = solution.root
                        else:
                            raise ValueError('the numerical solver for K_min, for Δ', Δ[i], ', did not converge')
                        
                        try:
                            solution = root_scalar(solve_Δ, args=(σ[i], cp[i], F, tau, Δ[i]), x0=K_max, bracket=[K_min,K_max])
                        except ValueError('the numerical solve for premium adjusted strike for Δ', Δ[i], ', resulted in an error'):
                            print('This is likely due to an error in the input, for example typos or specifying the Δ_convention as spot Δ when it is actually forward Δ')
                        
                    if solution.converged:
                        result[i] = solution.root
                    else:
                        raise ValueError('the numerical solver for the premium adjusted strike for Δ', Δ[i], ', did not converge')
                
                
    return result
                    

def gk_solve_implied_σ(S0: float, 
                       tau: float, 
                       r_f: float,                       
                       r_d: float, 
                       cp: int,
                       K: float, 
                       X: float, 
                       σ_guess: float) -> float:
    """
    Solve the implied volatility using the Garman-Kohlhagen model for a European Vanilla FX option.

    Parameters:
    - S0 (float): Spot price
    - tau (float): Time to maturity in years
    - r_f (float): Foreign interest rate    
    - r_d (float): Domestic interest rate
    - cp (int): Option type (1 for call, -1 for put)
    - K (float): Strike price
    - X (float): Option price
    - σ_guess (float): Initial guess for volatility

    Returns:
    - float: Implied volatility

    Attempts to find implied volatility using Newton's method initially and falls back to Brent's method in case of failure.
    """
    try:
        # Try Netwon's method first (it's faster but less robust)
        return newton(lambda σ: (gk_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,σ=σ)['option_value']  - X), x0=σ_guess, tol=1e-4, maxiter=50)
    except RuntimeError:
        # Fallback to Brent's method
        try:
            return root_scalar(lambda σ: (gk_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,σ=σ)['option_value']  - X), bracket=[0.0001, 2], method='brentq').root
        except ValueError:
            return np.inf        
        
        
#%%

if __name__ == '__main__':
    pass

    S0=0.6438
    σ=0 #0.0953686
    r_d=0.05408
    r_f=0.04189
    tau=0.33403698
    cp=-1
    K=0.71000
    F = 0.646478
    p1 = gk_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,σ=σ,F=F)['option_value']
    print('F specified:', p1)
    p2 = gk_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,σ=σ)['option_value']
    print('IR parity', p2)


    # check if instrinisc value is returned. Use IR parity. 
    p3 = gk_price(S0=S0,tau=tau,r_f=r_f,r_d=r_d,cp=cp,K=K,σ=σ)['option_value']
    print('if tau=0 intrinisc?:', p3)    












