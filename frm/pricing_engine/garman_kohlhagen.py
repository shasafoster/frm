# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar, newton


def to_np_array(*args):
    return [np.atleast_1d(arg).astype(float) for arg in args]


def gk_price(S0: float, 
             tau: float,
             r_d: float,
             r_f: float,
             cp: int,
             K: float,
             vol: float,
             F: float = None,
             analytical_greeks: bool=False,
             numerical_greeks: bool=False,
             intrinsic_time_split: bool=False
             ) -> float:
    """
    Garman-Kohlhagen European FX option pricing formula for:
    - option total, intrinsic, and time value (in the domestic currency per 1 unit of foreign currency notional).
    - analytical and numerical greeks (normalized to shifts applied per market convention).

    Parameters
    ----------
    S0 : float
        FX spot rate (specified in # of units of domestic currency per 1 unit of foreign currency).
    tau : float
        Time to expiry (in years).
    r_d : float
        Domestic risk-free interest rate (annualized continuously compounded).
    r_f : float
        Foreign risk-free interest rate (annualized continuously compounded).
    cp : int
        Option type: 1 for call option, -1 for put option.
    K : float
        Strike price (in units of domestic currency per 1 unit of foreign currency).
    vol : float
        Volatility (annualized).
    F : float, optional
        Market forward rate. If None, it will be calculated using interest rate parity (default is None).
    analytical_greeks : bool, optional
        If True, analytical greeks will be calculated and returned (default is False).
    numerical_greeks : bool, optional
        If True, numerical greeks will be calculated and returned using finite differences (default is False).
    intrinsic_time_split : bool, optional
        If True, splits option value into intrinsic and time value components (default is False).

    Returns
    -------
    results : dict
        Dictionary containing the option price and, if requested, analytical or numerical greeks.
        - 'option_value' : Option price.
        - 'intrinsic_value' : Intrinsic value (if `intrinsic_time_split_flag` is True).
        - 'time_value' : Time value (if `intrinsic_time_split_flag` is True).
        - 'analytical_greeks' : DataFrame with analytical greeks (if `analytical_greeks_flag` is True).
        - 'numerical_greeks' : DataFrame with numerical greeks (if `numerical_greeks_flag` is True).
    
    Notes
    -----
    1. The option is priced under the assumption of continuous interest rate compounding.
    2. If the `F` parameter is provided, the domestic risk-free rate is adjusted to match the forward rate.
    3. The analytical greeks calculated are: delta, vega, gamma, theta, and rho.
    4. Numerical greeks are calculated using finite differences with small shifts (e.g., 1% for delta and vega, 1 day for theta).
    5. If tau equals 0, time value is set to zero. 
    """
    results = dict()

    # Set to Greek letter so code review to analytical formulae is easier
    σ = vol

    # Convert to arrays. Function is vectorised.
    S0, tau, r_d, r_f, cp, K, σ = [np.atleast_1d(arg).astype(float) for arg in [S0, tau, r_d, r_f, cp, K, σ]]
    
    # Sensical value checks
    # No >0 check for σ, as when doing numerical solving, need to allow for -ve σ
    assert (S0 > 0.0).all(), S0
    assert (tau >= 0.0).all(), tau 
    assert np.all(np.isin(cp, [1, -1])), cp
    assert cp.shape == σ.shape
    assert cp.shape == K.shape
    
    if F is not None: 
        # Use market forward rate and imply the currency basis-adjusted domestic interest rate
        F = np.atleast_1d(F).astype(float)
        r_d_basis_adj = np.log(F / S0) / tau + r_f # from F = S0 * exp((r_d - r_f) * tau)
        r = r_d_basis_adj
        q = r_f
    else:
        # By interest rate parity
        F = S0 * np.exp((r_d - r_f) * tau)   
        r = r_d
        q = r_f
    assert (F > 0.0).all()   
    
    μ = r - q
    d1 = (np.log(S0 / K) + (μ + 0.5 * σ**2) * tau) / (σ * np.sqrt(tau))
    d2 = d1 - σ * np.sqrt(tau)   
    X = cp * (S0 * np.exp(-q * tau) * norm.cdf(cp * d1) - K * np.exp(-r * tau) * norm.cdf(cp * d2))

    X[tau==0] = np.maximum(0, cp * (S0 - K))[tau==0] # If time-to-maturity is 0.0, set to intrinsic value 
    results['option_value'] = X
    
    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (S0 * np.exp(-q * tau) - K * np.exp(-r * tau)))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    # Checks to alternative formulae
    epsilon = 1e-10
    assert (abs(d1 - (np.log(F / K) + (0.5 * σ**2) * tau) / (σ * np.sqrt(tau))) < epsilon).all()
    assert (abs(X - cp * np.exp(-r * tau) * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))) < epsilon).all()
    if intrinsic_time_split:
        assert (abs(intrinsic[tau>0] - np.maximum(0, cp * np.exp(-r * tau) * (F - K))[tau>0]) < epsilon).all()


    if analytical_greeks:
        results['analytical_greeks'] = pd.DataFrame()

        # Delta, Δ, is the change in an option's price for a small change in the underlying assets price.
        # Δ := ∂X/∂S ≈ (X(S_plus) − X(S_minus)) / (S_plus - S_minus)
        # where X is the option price (whose units is DOM),
        # and S0 is the fx spot price (whose units is DOM/FOR)
        # We multiply by S0 so to return the Δ a percentage applicable to the foreign currency notional
        # Multiplication of S0 removed
        results['analytical_greeks']['spot_delta'] = cp * np.exp(-q * tau) * norm.cdf(cp * d1)
        results['analytical_greeks']['forward_delta'] = cp * norm.cdf(cp * d1)

        # Vega, ν, is the change in an options price for a small change in the volatility input
        # ν = ∂X/∂σ ≈ (X(σ_plus) − X(σ_minus)) / (σ_plus - σ_minus)
        # In practice, vega is normalised to measure the change in price for a 1% change in the volatility input
        # Hence we have scaled the numerical vega to a 1% change
        analytical_formula = S0 * np.sqrt(tau) * norm.pdf(d1) * np.exp(-q * tau) # identical for calls and puts
        results['analytical_greeks']['vega'] = analytical_formula * 0.01 # normalised to 1% change

        # Theta, θ, is the change in an options price for a small change in the time to expiry
        # In practice, theta is normalised to measure the change in price, for a 1 day shorter expiry
        # Theta includes
        # 1. Time decay: the change in the option's value as time moves forward, that is, as the expiry date moves closer,
        #               i.e., the time value price tomorrow minus the time value price today.
        # 2. Cost of carry: the interest rate sensitivity that causes the value of the portfolio to change as time progresses,
        #                   e.g., the cost of carry on spots and forwards is included in the theta value.
        analytical_formula = -(S0*np.exp(-r*tau)*norm.pdf(d1 * cp)*σ)/(2*np.sqrt(tau)) \
            + r_f * np.exp(-q * tau) * S0 * norm.cdf(d1 * cp) \
            - r_d * np.exp(-r * tau) * K * norm.cdf(d2 * cp)
        results['analytical_greeks']['theta'] = analytical_formula * 1/365.25 # normalised to 1 calendar day

        # Gamma, Γ, is the change in an option's delta for a small change in the underlying assets price.
        # Gamma := ∂Δ/∂S ≈ (Δ(S_plus) − Δ(S_minus)) / (S_plus - S_minus)
        # In practice, gamma is  normalised to measure the change in Δ, for a 1% change in the underlying assets price.
        # Hence, we have multiplied the analytical gamma formula by 'S * 0.01'
        analytical_formula = np.exp(-q * tau) * norm.pdf(d1)  / (S0 * σ * np.sqrt(tau)) # identical for calls and puts
        results['analytical_greeks']['gamma'] = (0.01 * S0) * analytical_formula # normalised for 1% change

        # Rho, ρ, is the rate at which the price of an option changes relative to a change in the interest rate.
        # In practice, Rho is normalised to measure the change in price for a 1% change in the underlying interest rate.
        # Hence, we have multiplied the analytical formula result by 0.01 (i.e. 1%)
        analytical_formula = K * tau * np.exp(-q * tau) * norm.cdf(cp * d2)
        results['analytical_greeks']['rho'] = analytical_formula * 0.01


    if numerical_greeks:
        numerical_greeks = pd.DataFrame()

        Δ_shift = 1 / 100  # 1% shift
        σ_shift = 1 / 100  # 1% shift
        θ_shift = 1 / 365.25  # 1 Day
        ρ_shift = 1 / 100  # 1% shift

        if F is not None:
            F_upshift, F_downshift = F * (1 + Δ_shift), F * (1 - Δ_shift)
        else:
            F_upshift, F_downshift = None, None

        results_S0_plus = gk_price(S0=S0*(1+Δ_shift), tau=tau, r_d=r_d, r_f=r_f , cp=cp, K=K, vol=σ, F=F_upshift,analytical_greeks=True)
        X_S_plus, analytical_greeks_S0_plus = results_S0_plus['option_value'], results_S0_plus['analytical_greeks']
        results_S0_minus = gk_price(S0=S0*(1-Δ_shift), tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F_downshift,analytical_greeks=True)
        X_S_minus, analytical_greeks_S0_minus = results_S0_minus['option_value'], results_S0_minus['analytical_greeks']

        results['numerical_greeks']['spot_delta'] = (X_S_plus - X_S_minus) / (2 * S0 * Δ_shift)
        results['numerical_greeks']['forward_delta'] = numerical_greeks['spot_delta'] / np.exp(-r_f * tau)

        X_σ_plus = gk_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ+σ_shift, F=F)['option_value']
        X_σ_minus = gk_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ-σ_shift, F=F)['option_value']
        results['numerical_greeks']['vega'] = (X_σ_plus - X_σ_minus) / (2 * (σ_shift / 0.01))

        X_plus = gk_price(S0=S0, tau=tau+θ_shift, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F)['option_value']
        X_minus = gk_price(S0=S0, tau=tau-θ_shift, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F)['option_value']
        results['numerical_greeks']['theta'] = (X_minus - X_plus) / 2

        results['numerical_greeks']['gamma'] = (analytical_greeks_S0_plus['spot_delta'] - analytical_greeks_S0_minus['spot_delta']) / (2 * (Δ_shift / 0.01))

        # This formulae will yield meaningfully different results on whether the forward rate is an input (or if it is calculated from the interest rate differential)
        X_ρ_plus = gk_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f+ρ_shift, cp=cp, K=K, vol=σ+σ_shift, F=F)['option_value']
        X_ρ_minus = gk_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f+ρ_shift, cp=cp, K=K, vol=σ-σ_shift, F=F)['option_value']
        results['numerical_greeks']['rho'] = (X_ρ_plus - X_ρ_minus) / 2

    return results
            
def gk_solve_strike(S0: float,
                    tau: float,
                    r_d: float,
                    r_f: float,
                    vol: [float, np.ndarray],
                    signed_delta: [float, np.ndarray],
                    delta_convention: str,
                    F: float = None,
                    atm_delta_convention: str='forward') -> float:
    """
    Solves the strike for a given Delta-volatility (Δ-σ) quate.

    Parameters
    ----------
    S0 : float
        FX spot rate (specified in # of units of domestic currency per 1 unit of foreign currency).
    tau : float
        Time to expiry (in years).
    r_d : float
        Domestic risk-free interest rate (annualized continuously compounded).
    r_f : float
        Foreign risk-free interest rate (annualized continuously compounded).
    vol : float
        Volatility (annualized standard deviation of FX returns).
    signed_delta : float
        Signed delta, restricted to [0, 0.5] for calls and [-0.5, 0] for puts.
    delta_convention : str
        Delta convention used for the tenors quoting, one of {'regular_spot', 'regular_forward', 'premium_adjusted_spot', 'premium_adjusted_forward'}.
    F : float, optional
        Market forward rate. If None, it will be calculated using interest rate parity (default is None).
    atm_delta_convention : str, optional
        At-the-money delta convention used for the tenors quoting, one of {'forward','per_delta_convention'} (default is 'forward').
        Market convention, is at-the-money delta is quoted in terms of the forward delta, regardless of the delta convention used for the tenors quoting.
    Returns
    -------
    float
        Calculated strike price based on the provided delta.
        
    References
    -------
    [1] Reiswich, Dimitri & Wystup, Uwe. (2010). A Guide to FX Options Quoting Conventions. The Journal of Derivatives. 18. 58-68. 10.3905jod.2010.18.2.058
    """

    # Set to greek letter so code review to reference [1] is easier
    σ = vol
    Δ = signed_delta

    # Convert to arrays. Function is vectorised.
    S0, tau, r_f, r_d, σ, Δ = to_np_array(S0, tau, r_f, r_d, σ, Δ)
    if F is not None: 
        F = np.atleast_1d(F).astype(float)
        assert (F > 0.0).all() 

    # Value checks
    assert (S0 > 0.0).all(), S0
    assert (tau > 0.0).all(), tau
    assert (σ > 0.0).all(), σ
    assert (Δ >= -0.5).all() and (Δ <= 0.5).all()
    assert Δ.shape == σ.shape
    cp = np.sign(Δ)
    assert delta_convention in {'regular_spot',
                                'regular_forward',
                                'premium_adjusted_spot',
                                'premium_adjusted_forward'}, delta_convention
    assert atm_delta_convention in {'forward', 'per_delta_convention'}, atm_delta_convention

    result = np.zeros(shape=Δ.shape)
    if F is None:
        # If market forward rate not supplied, calculate it per interest rate parity
        F = np.atleast_1d(S0 * np.exp((r_d - r_f) * tau))  
    
    mask_atm = Δ == 0.5
    mask_not_atm = np.logical_not(mask_atm)

    if delta_convention in {'regular_spot','regular_forward'}:

        if delta_convention == 'regular_spot':
            norm_func = norm.ppf(cp * Δ * np.exp(r_f * tau))
        elif delta_convention == 'regular_forward':
            norm_func = norm.ppf(cp * Δ) # Note: norm.ppf(0.5) = 0. Applicable to for atm-delta-neutral quotes.
        result = (F * np.exp(-cp * norm_func * σ * np.sqrt(tau) + 0.5 * σ**2 * tau))

        if atm_delta_convention == 'per_delta_convention':
            pass
        elif atm_delta_convention == 'forward':
            result[mask_atm] = (F * np.exp(0.5 * σ[mask_atm]**2 * tau))

    elif delta_convention in {'premium_adjusted_spot','premium_adjusted_forward'}:
        
        if np.any(mask_atm):
            # at-the-money Δ-neutral strike, for premium adjusted spot/forward Δ
            result[mask_atm] = (F * np.exp(-1 * 0.5 * σ**2 * tau))[mask_atm]
                       
        if np.any(mask_not_atm):
        
            for i in range(len(Δ)):
                if Δ[i] != 0.5:
                    # For premium adjusted quotes the solution must be solved numerically
                    # Please refer to Reference [1] for full details
                    match delta_convention:
                        case 'premium_adjusted_spot':
                            def solve_delta(K, σ, cp, F, tau, Δ):
                                return np.exp(-r_f * tau) * (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ
                        case 'premium_adjusted_forward':
                            def solve_delta(K, σ, cp, F, tau, Δ):
                                return (cp * K / F) * norm.cdf(cp * (np.log(F/K) - 0.5 * σ ** 2 * tau) / (σ * np.sqrt(tau))) - Δ
        
                    # Solve the upper bound, 'K_max' for the numerical solver
                    # The strike of a premium adjusted Δ-σ quote is ALWAYS below the regular (non premium adjusted) Δ-σ quote
                    # Hence, we analytically calculate the K, assuming the σ was a regular Δ quote
                    delta_convention_adj = delta_convention.replace('premium_adjusted','regular')
                    K_max = gk_solve_strike(S0=S0,tau=tau, r_d=r_d, r_f=r_f, vol=σ[i], signed_delta=Δ[i], delta_convention=delta_convention_adj, F=F)
                    K_max = K_max.item()
    
                    # Put Option
                    if Δ[i] < 0: 
                        solution = root_scalar(solve_delta, args=(σ[i], cp[i], F, tau, Δ[i]), x0=F, bracket=[0.00001, K_max])
                        
                    # Call Option
                    if Δ[i] > 0: 
                        # For the premium adjusted call Δ, due to non-monotonicity, two strikes can be solved numerically for call options (but not put options).
                        # To avoid this, we solve a lower bound, 'K_min' to guarantee we get the correct solution in the numerical solver.
                        # The lower bound is the 'maximum' Δ, hence we numerically solve the maximum Δ
                        def solve_K_min(K, σ, cp, F, tau): # TODO test and remove cp, F
                            d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * σ**2) * tau) / (σ * np.sqrt(tau))
                            d2 = d1 - σ * np.sqrt(tau)
                            return σ * np.sqrt(tau) * norm.cdf(d2) - norm.pdf(d2)
                        
                        try: 
                            solution = root_scalar(solve_K_min, args=(σ[i], cp[i], F, tau), x0=F ,bracket=[0.00001, K_max])
                        except ValueError('the numerical solve for K_min, for delta', Δ[i], ', resulted in an error'):
                            pass
                    
                        if solution.converged:
                            K_min = solution.root
                        else:
                            raise ValueError('the numerical solver for K_min, for delta', Δ[i], ', did not converge')
                        
                        try:
                            solution = root_scalar(solve_delta, args=(σ[i], cp[i], F, tau, Δ[i]), x0=K_max, bracket=[K_min,K_max])
                        except ValueError('the numerical solve for premium adjusted strike for delta', Δ[i], ', resulted in an error'):
                            print('This is likely due to an error in the input, for example typos or specifying the delta_convention as spot-delta when it is actually forward-delta')
                        
                    if solution.converged:
                        result[i] = solution.root
                    else:
                        raise ValueError('the numerical solver for the premium adjusted strike for delta', Δ[i], ', did not converge')
    else:
        raise ValueError("'delta_convention' must be one of {regular_spot, regular_forward, premium_adjusted_spot, premium_adjusted_forward}", delta_convention)


    return result
                    

def gk_solve_implied_volatility(S0: float,
                                tau: float,
                                r_d: float,
                                r_f: float,
                                cp: int,
                                K: float,
                                X: float,
                                vol_guess: float) -> float:
    """
    Solve for the implied volatility using the Garman-Kohlhagen model for a European Vanilla FX option.

    This function attempts to compute the implied volatility by matching the observed option price 
    with the Garman-Kohlhagen model price. It first uses Newton's method for faster convergence 
    and falls back to Brent's method if necessary.

    Parameters
    ----------
    S0 : float
        FX spot price (specified in # of units of domestic currency per 1 unit of foreign currency).
    tau : float
        Time to expiry in years.
    r_d : float
        Domestic risk-free interest rate (annualized continuously compounded).
    r_f : float
        Foreign risk-free interest rate (annualized continuously compounded).
    cp : int
        Option type: 1 for call, -1 for put.
    K : float
        Strike price of the option (specified in # of units of domestic currency per 1 unit of foreign currency).
    X : float
        Observed market option price.
    vol_guess : float
        Initial guess for the volatility.

    Returns
    -------
    float
        The implied volatility, or np.inf if no solution is found.

    Raises
    ------
    RuntimeError
        If Newton's method fails to converge within the iteration limit.
    ValueError
        If Brent's method fails to find a solution within the specified bracket.

    Notes
    -----
    - The function first tries Newton's method, which is faster but less robust, with a tolerance of 1e-4.
    - If Newton's method fails, Brent's method is used for fallback, searching between 0.0001 and 2.
    - If neither method succeeds, the function returns np.inf to indicate failure.
    """
    try:
        # Try Netwon's method first (it's faster but less robust)
        return newton(lambda vol: (gk_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['option_value']  - X), x0=vol_guess, tol=1e-4, maxiter=50).item()
    except RuntimeError:
        # Fallback to Brent's method
        try:
            return root_scalar(lambda vol: (gk_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['option_value']  - X), bracket=[0.0001, 2], method='brentq').root
        except ValueError:
            return np.inf        
        
        
if __name__ == '__main__':
    pass

    S0=0.6438
    vol=0 #0.0953686
    r_d=0.05408
    r_f=0.04189
    tau=0.33403698
    cp=-1
    K=0.71000
    F = 0.646478
    p1 = gk_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol,F=F)['option_value']
    print('F specified:', p1)
    p2 = gk_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['option_value']
    print('IR parity', p2)


    # check if instrinisc value is returned. Use IR parity. 
    p3 = gk_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol)['option_value']
    print('if tau=0 intrinisc?:', p3)    












