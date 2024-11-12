# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar, newton

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

def garman_kohlhagen_price(
        S0: [float, np.ndarray],
        tau: [float, np.ndarray],
        r_d: [float, np.ndarray],
        r_f: [float, np.ndarray],
        cp: [int, np.ndarray],
        K: [float, np.ndarray],
        vol: [float, np.ndarray],
        F: [float, np.ndarray] = None,
        analytical_greeks: bool=False,
        numerical_greeks: bool=False,
        intrinsic_time_split: bool=False
        ) -> dict:
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
        Prices are in units of domestic currency per 1 unit of foreign currency.
        - 'price' : Option unit price
        - 'intrinsic' : Intrinsic unit price (if `intrinsic_time_split_flag` is True).
        - 'time' : Time unit price (if `intrinsic_time_split_flag` is True).
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


    # Convert to arrays. Function is vectorised.
    S0, tau, r_d, r_f, cp, K, σ = map(lambda x: np.atleast_1d(x).astype(float), (S0, tau, r_d, r_f, cp, K, vol))

    # Value bounds checks. No >0 check for σ, as when doing numerical solving, need to allow for -ve σ
    assert (S0 > 0.0).all(), S0
    assert (tau >= 0.0).all(), tau 
    assert np.all(np.isin(cp, [1, -1])), cp
    assert cp.shape == σ.shape
    assert cp.shape == K.shape
    
    if F is not None: 
        # Use market forward rate and imply the currency basis-adjusted domestic interest rate
        F = np.atleast_1d(F).astype(float)
        assert (F > 0.0).all()
        r_d_basis_adj = np.log(F / S0) / tau + r_f # from F = S0 * exp((r_d - r_f) * tau)
        r = r_d_basis_adj
        q = r_f
    else:
        r = r_d
        q = r_f

    μ = r - q
    d1 = (np.log(S0 / K) + (μ + 0.5 * σ**2) * tau) / (σ * np.sqrt(tau))
    d2 = d1 - σ * np.sqrt(tau)   
    X = cp * (S0 * np.exp(-q * tau) * norm.cdf(cp * d1) - K * np.exp(-r * tau) * norm.cdf(cp * d2))

    X[tau==0] = np.maximum(0, cp * (S0 - K))[tau==0] # If time to maturity is 0.0, set to intrinsic value
    results = {'price': X}

    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (S0 * np.exp(-q * tau) - K * np.exp(-r * tau)))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    if analytical_greeks:
        results['analytical_greeks'] = pd.DataFrame()

        # These formulae produce the delta applicable to the domestic currency notional.
        # To get the Δ % applicable to the foreign currency notional multiply the below by S0.
        results['analytical_greeks']['spot_delta'] = cp * np.exp(-q * tau) * norm.cdf(cp * d1)
        results['analytical_greeks']['forward_delta'] = cp * norm.cdf(cp * d1)

        # In practice, vega is normalised to measure the change in price for a 1% change in the volatility input
        # Hence we have scaled the analytical vega to a 1% change
        analytical_formula = S0 * np.sqrt(tau) * norm.pdf(d1) * np.exp(-q * tau) # identical for calls and puts
        results['analytical_greeks']['vega'] = analytical_formula * 0.01 # normalised to 1% change

        # Per market convention, theta is normalised to the price decay for 1 calendar day.
        analytical_formula = -(S0*np.exp(-r*tau)*norm.pdf(d1 * cp)*σ)/(2*np.sqrt(tau)) \
            + q * np.exp(-q * tau) * S0 * norm.cdf(d1 * cp) \
            - r * np.exp(-r * tau) * K * norm.cdf(d2 * cp)
        results['analytical_greeks']['theta'] = analytical_formula * 1/365.25 # normalised to 1 calendar day

        # In practice, gamma is  normalised to measure the change in Δ, for a 1% change in the underlying assets price.
        # Hence, we have multiplied the analytical gamma formula by 'S * 0.01'
        analytical_formula = np.exp(-q * tau) * norm.pdf(d1)  / (S0 * σ * np.sqrt(tau)) # identical for calls and puts
        results['analytical_greeks']['gamma'] = (0.01 * S0) * analytical_formula # normalised for 1% change

        # In practice, Rho is normalised to measure the change in price for a 1% change in the underlying interest rate.
        # Hence, we have multiplied the analytical formula result by 0.01 (i.e. 1%)
        analytical_formula = K * tau * np.exp(-q * tau) * norm.cdf(cp * d2)
        results['analytical_greeks']['rho'] = analytical_formula * 0.01


    if numerical_greeks:
        numerical_greeks = pd.DataFrame()

        Δ_shift = 1 / 100  # 1% shift
        σ_shift = 1 / 100  # 1% shift
        θ_shift = 1 / 365.25  # 1 calendar day
        ρ_shift = 1 / 100  # 1% shift

        if F is not None:
            F_upshift, F_downshift = F * (1 + Δ_shift), F * (1 - Δ_shift)
        else:
            F_upshift, F_downshift = None, None

        results_S0_plus = garman_kohlhagen_price(S0=S0*(1+Δ_shift), tau=tau, r_d=r_d, r_f=r_f , cp=cp, K=K, vol=σ, F=F_upshift,analytical_greeks=True)
        X_S_plus, analytical_greeks_S0_plus = results_S0_plus['price'], results_S0_plus['analytical_greeks']
        results_S0_minus = garman_kohlhagen_price(S0=S0*(1-Δ_shift), tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F_downshift,analytical_greeks=True)
        X_S_minus, analytical_greeks_S0_minus = results_S0_minus['price'], results_S0_minus['analytical_greeks']

        # Delta, Δ, is the change in an option's price for a small change in the underlying assets price.
        # Δ := ∂X/∂S ≈ (X(S_plus) − X(S_minus)) / (S_plus - S_minus)
        # where X is the option price (whose units is DOM),
        # and S0 is the fx spot price (whose units is DOM/FOR)
        # To get the Δ % applicable to the foreign currency notional multiply the below by S0.
        results['numerical_greeks']['spot_delta'] = (X_S_plus - X_S_minus) / (2 * S0 * Δ_shift)
        results['numerical_greeks']['forward_delta'] = numerical_greeks['spot_delta'] / np.exp(-r_f * tau)

        # Vega, ν, is the change in an options price for a small change in the volatility input
        # ν = ∂X/∂σ ≈ (X(σ_plus) − X(σ_minus)) / (σ_plus - σ_minus).
        # We divide by 0.01 (1%) to normalise the vega to a 1% change in the volatility input, per market convention.
        X_σ_plus = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ+σ_shift, F=F)['price']
        X_σ_minus = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ-σ_shift, F=F)['price']
        results['numerical_greeks']['vega'] = (X_σ_plus - X_σ_minus) / (2 * (σ_shift / 0.01))

        # Theta, θ, is the change in an options price for a small change in the time to expiry
        # In practice, theta is normalised to measure the change in price, for a 1 day shorter expiry
        # Theta includes
        # 1. Time decay: the change in the option's value as time moves forward, that is, as the expiry date moves closer,
        #               i.e., the time value price tomorrow minus the time value price today.
        # 2. Cost of carry: the interest rate sensitivity that causes the value of the portfolio to change as time progresses,
        #                   e.g., the cost of carry on spots and forwards is included in the theta value.
        X_plus = garman_kohlhagen_price(S0=S0, tau=tau+θ_shift, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F)['price']
        X_minus = garman_kohlhagen_price(S0=S0, tau=tau-θ_shift, r_d=r_d, r_f=r_f, cp=cp, K=K, vol=σ, F=F)['price']
        results['numerical_greeks']['theta'] = (X_minus - X_plus) / 2

        # Gamma, Γ, is the change in an option's delta for a small change in the underlying assets price.
        # Gamma := ∂Δ/∂S ≈ (Δ(S_plus) − Δ(S_minus)) / (S_plus - S_minus).
        results['numerical_greeks']['gamma'] = (analytical_greeks_S0_plus['spot_delta'] - analytical_greeks_S0_minus['spot_delta']) / (2 * (Δ_shift / 0.01))

        # Rho, ρ, is the rate at which the price of an option changes relative to a change in the interest rate.
        # This formulae will yield meaningfully different results on whether the forward rate is an input (or if it is calculated from the interest rate differential)
        X_ρ_plus = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f+ρ_shift, cp=cp, K=K, vol=σ+σ_shift, F=F)['price']
        X_ρ_minus = garman_kohlhagen_price(S0=S0, tau=tau, r_d=r_d, r_f=r_f+ρ_shift, cp=cp, K=K, vol=σ-σ_shift, F=F)['price']
        results['numerical_greeks']['rho'] = (X_ρ_plus - X_ρ_minus) / 2

    return results


def garman_kohlhagen_solve_strike_from_delta(
        S0: float,
        tau: float,
        r_d: float,
        r_f: float,
        vol: [float, np.ndarray],
        signed_delta: [float, np.ndarray],
        delta_convention: str,
        F: float = None,
        atm_delta_convention: str='forward') -> float:
    """
    Solves the strike for a given Delta-volatility (Δ-σ) quote.

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

    # Convert to arrays. Function is vectorised.
    S0, tau, r_f, r_d, σ, Δ = map(lambda x: np.atleast_1d(x).astype(float), (S0, tau, r_f, r_d, vol, signed_delta))

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
    assert atm_delta_convention in {'forward',
                                    'per_delta_convention'}, atm_delta_convention

    if F is None:
        # If market forward rate not supplied, calculate it per interest rate parity
        F = np.atleast_1d(S0 * np.exp((r_d - r_f) * tau))
    else:
        F = np.atleast_1d(F).astype(float)
        assert (F > 0.0).all()

    result = np.zeros(shape=Δ.shape)
    mask_atm = np.abs(Δ) == 0.5
    mask_not_atm = np.logical_not(mask_atm)

    if delta_convention in {'regular_spot','regular_forward'}:

        if delta_convention == 'regular_spot':
            norm_func = norm.ppf(cp * Δ * np.exp(r_f * tau))
        else:
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
                    # For premium adjusted quotes the strike must be solved numerically. Refer to [1] for full details.

                    def solve_delta(K_):
                        if delta_convention == 'premium_adjusted_spot':
                            multiplier = np.exp(-r_f * tau)
                        else:
                            multiplier = 1
                        return (multiplier * (cp[i] * K_ / F) * norm.cdf(cp[i] * (np.log(F/K_) - 0.5 * σ[i]**2 * tau) / (σ[i] * np.sqrt(tau)))) - Δ[i]

                    # Solve the upper bound, 'K_max' for the numerical solver
                    # The strike of a premium adjusted Δ-σ quote is ALWAYS below the regular (non premium-adjusted) Δ-σ quote
                    # Hence, we analytically calculate the K, assuming the σ was a regular Δ quote
                    delta_convention_adj = delta_convention.replace('premium_adjusted','regular')
                    K_max = garman_kohlhagen_solve_strike_from_delta(
                        S0=S0,tau=tau, r_d=r_d, r_f=r_f, vol=σ[i], signed_delta=Δ[i], delta_convention=delta_convention_adj, F=F).item()
    
                    # Put Option
                    if Δ[i] < 0: 
                        solution = root_scalar(solve_delta, x0=F, bracket=[0.000001, K_max])
                        
                    # Call Option
                    if Δ[i] > 0: 
                        # For the premium adjusted call Δ, due to non-monotonicity, two strikes can be solved numerically for call options (but not put options).
                        # To avoid this, we solve a lower bound, 'K_min' to guarantee we get the correct solution in the numerical solver.
                        # The lower bound is the 'maximum' Δ, hence we numerically solve the maximum Δ
                        def solve_k_lower_bound(K): # TODO test and remove cp, F
                            d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * σ[i]**2) * tau) / (σ[i] * np.sqrt(tau))
                            d2 = d1 - σ[i] * np.sqrt(tau)
                            return σ[i] * np.sqrt(tau) * norm.cdf(d2) - norm.pdf(d2)
                        
                        try: 
                            solution = root_scalar(solve_k_lower_bound, x0=F ,bracket=[0.00001, K_max])
                        except ValueError('the numerical solve for K_min, for delta', Δ[i], ', resulted in an error'):
                            pass
                    
                        if solution.converged:
                            K_min = solution.root
                        else:
                            raise ValueError('the numerical solver for K_min, for delta', Δ[i], ', did not converge')
                        
                        try:
                            solution = root_scalar(solve_delta, x0=K_max, bracket=[K_min,K_max])
                        except ValueError('the numerical solve for premium adjusted strike for delta', Δ[i], ', resulted in an error'):
                            print('This is likely due to an error in the input, for example typos or specifying the delta_convention as spot-delta when it is actually forward-delta')
                        
                    if solution.converged:
                        result[i] = solution.root
                    else:
                        raise ValueError('the numerical solver for the premium adjusted strike for delta', Δ[i], ', did not converge')
    else:
        raise ValueError("'delta_convention' must be one of {regular_spot, regular_forward, premium_adjusted_spot, premium_adjusted_forward}", delta_convention)


    return result
                    

def garman_kohlhagen_solve_implied_vol(
        S0: float,
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
    def error_function(vol_):
        return garman_kohlhagen_price(S0=S0,tau=tau,r_d=r_d,r_f=r_f,cp=cp,K=K,vol=vol_)['price']  - X

    try:
        # Try Netwon's method first (it's faster but less robust)
        return newton(lambda vol_: error_function(vol_), x0=vol_guess, tol=1e-4, maxiter=50).item()
    except RuntimeError:
        # Fallback to Brent's method
        try:
            return root_scalar(lambda vol_: error_function(vol_), bracket=[0.0001, 2], method='brentq').root
        except ValueError:
            return np.inf












