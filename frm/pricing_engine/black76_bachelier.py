# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

VOL_SLN_BOUNDS = (0.1 / 100, 1000 / 100)  # 0.1% to 1000% (0.001 to 10)
VOL_N_BOUNDS = (0.01 / 100, 100 / 100)  # 0.01% to 100% (0.0001 to 1)

def black76_price(
        F: [float, np.array],
        tau: [float, np.array],
        cp: [float, np.array],
        K: [float, np.array],
        vol_sln: [float, np.array],
        ln_shift: [float, np.array],
        annuity_factor: [float, np.array]=1,
        intrinsic_time_split: bool=False,
        analytical_greeks: bool=False,
        numerical_greeks: bool=False):
    """
    Black76 pricing + greeks.

    The function has the parameters 'annuity_factor' instead of a risk-free rate.
    This adjustment allows for more generic applications of Black76 to caplets/floorlets and swaptions instead of just European options delivered at expiry.

    Parameters
    ----------
    F : float
        Forward price.
    tau : float
        Time to expiry (in years).
    K : float
        Strike price.
    cp : int
        Option type: 1 for call option, -1 for put option.
    vol_sln : float
        Volatility (annualized).
    ln_shift : float, optional
        Log-normal shift, applied to forward price and strike (default is 0).
    annuity_factor : float, optional
        Multiplier to adjust the Black76 forward price to present value (default is 1).
        This is composed of the discount factor and the accrual period fraction.
    analytical_greeks : bool, optional
        If True, analytical greeks will be calculated and returned (default is False).
    numerical_greeks : bool, optional
        If True, numerical greeks will be calculated and returned using finite differences (default is False).
    intrinsic_time_split : bool, optional
        If True, splits option value into intrinsic and time value components (default is False).

    Returns
    -------
    results : dict
          Dictionary containing the following key-value pairs:
        - 'price' : float
            Option price.
        - 'analytical_greeks' : pd.DataFrame
            Analytical greeks; columns: 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'numerical_greeks' : pd.DataFrame
            Numerical greeks; columns: 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'intrinsic_time_split' : float
            Dictionary of intrinsic and time value components; 'intrinsic', 'time'.
    """

    # Convert to arrays. Function is vectorised.
    F, tau, cp, K, σB, ln_shift, annuity_factor = map(np.atleast_1d, (F, tau, cp, K, vol_sln, ln_shift, annuity_factor))

    # Validation checks
    assert F.shape == tau.shape
    shapes = set([param.shape for param in (F, tau, cp, K, σB, ln_shift, annuity_factor)])
    assert len(shapes) in [1, 2]
    if len(shapes) == 2:
        assert (1,) in shapes

    F = F + ln_shift
    K = K + ln_shift

    # Price per Black76 formula
    d1 = (np.log(F/K) + (0.5 * σB**2 * tau)) / (σB*np.sqrt(tau))
    d2 = d1 - σB*np.sqrt(tau)
    X = annuity_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))

    results = {'price': X}

    # Intrinsic / time value split
    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (F - K))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    if analytical_greeks:
        results['analytical_greeks'] = pd.DataFrame()

        results['analytical_greeks']['delta'] = cp * norm.cdf(cp * d1) * annuity_factor

        analytical_vega = F * np.sqrt(tau) * norm.pdf(d1)
        # In practice, vega is displayed as normalised to a 1% shift, hence the 0.01 multiplier.
        results['analytical_greeks']['vega'] = 0.01 * analytical_vega *  annuity_factor

    if numerical_greeks:
        results['numerical_greeks'] = pd.DataFrame()

        # Delta, Δ, is the change in an option's price for a small change in the underlying assets (forward) price.
        # Δ := ∂X/∂F ≈ (X(F_plus) − X(F_minus)) / (F_plus - F_minus)
        Δ_shift = 0.01
        results_F_plus = black76_price(F=F*(1+Δ_shift), tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift,
                           annuity_factor=annuity_factor, analytical_greeks=True)
        results_F_minus = black76_price(F=F*(1-Δ_shift), tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift,
                            annuity_factor=annuity_factor, analytical_greeks=True)
        results['numerical_greeks']['delta'] = (results_F_plus['price'] - results_F_minus['price']) / (2 * F * Δ_shift)

        # Vega, ν, is the change in an options price for a small change in the volatility input
        # ν = ∂X/∂σ ≈ (X(σ_plus) − X(σ_minus)) / (σ_plus - σ_minus).
        # We divide by 0.01 (1%) to normalise the vega to a 1% change in the volatility input, per market convention.
        σ_shift = 0.01
        results_σ_plus = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln+σ_shift, ln_shift=ln_shift,
                           annuity_factor=annuity_factor, analytical_greeks=True)
        results_σ_minus = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln-σ_shift, ln_shift=ln_shift,
                            annuity_factor=annuity_factor, analytical_greeks=True)
        results['numerical_greeks']['vega'] = (results_σ_plus['price'] - results_σ_minus['price']) / (2 * (σ_shift / 0.01))

    return results


def black76_solve_implied_vol(
        F: [float, np.array],
        tau: [float, np.array],
        cp: [float, np.array],
        K: [float, np.array],
        ln_shift: [float, np.array],
        X: float,
        vol_sln_guess: float=0.1,
        annuity_factor: [float, np.array]=1,
        ) -> float:
    """Solve the implied normal volatility with the Black76 pricing formula."""

    def error_function(vol_):
        # Return the errors (relative to the price, as we want invariance to the price) to be minimised.
        # For gradient-based optimisation methods, return squared error to ensure differentiability.
        relative_error = (black76_price(F=F, tau=tau, cp=cp, K=K, vol_sln=vol_, ln_shift=ln_shift, annuity_factor=annuity_factor)['price'].sum() - X) / X
        error = relative_error**power
        if abs(error) < obj_func_tol**power:  # Check against objective function tolerance
            raise StopIteration(vol_)  # Use StopIteration to terminate optimization early if tolerance is met
        return error

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    xtol = 1e-6
    obj_func_tol = 1e-5
    x0 = vol_sln_guess
    bounds = VOL_SLN_BOUNDS

    # Note Brent's does not work for a SSE obj. function as f(a) and f(b) must have different signs.
    # f(a), f(b) are the function values of the bracket.
    try:
        power = 1 # Use linear error for the root_scalar method
        res = root_scalar(lambda vol_: error_function(vol_), x0=x0, bracket=bounds, xtol=xtol, method='brentq')
        if abs(error_function(res.root)) < obj_func_tol**power:
            return res.root
        else:
            raise RuntimeError
    except StopIteration as e:
        # Return the root if we meet the objective tolerance early
        return e.value
    except RuntimeError:
        try:
            # Fallback to L-BFGS-B if root_scalar fails
            options = {'ftol': obj_func_tol, 'gtol': 0} # gtol set to 0, so optimisation is terminated based on ftol
            power = 2 # Use squared error for the optimisation
            res = minimize(fun=error_function, x0=np.atleast_1d(x0), bounds=[bounds], method='L-BFGS-B', options=options)
            # 2nd condition required as we have overridden options to terminate based on ftol
            if res.success or abs(res.fun) < obj_func_tol**power:
                return res.x[0]
            else:
                raise ValueError('Optimisation to solve normal volatility did not converge.')
        except StopIteration as e:
            # Return the root if we meet the objective tolerance early
            return e.value[0]


def bachelier_price(
        F: [float, np.array],
        tau: [float, np.array],
        cp: [float, np.array],
        K: [float, np.array],
        vol_n: [float, np.array],
        annuity_factor: [float, np.array] = 1,
        intrinsic_time_split: bool=False,
        analytical_greeks: bool=False,
        numerical_greeks: bool=False):
    """
    Bachelier pricing + greeks.

    The function has the parameters 'annuity_factor' instead of a risk-free rate.
    This adjustment allows for more generic applications of Bachelier to caplets/floorlets and swaptions instead of just European options delivered at expiry.

    Parameters
    ----------
    F : float
        Forward price.
    tau : float
        Time to expiry (in years).
    cp : int
        Option type: 1 for call option, -1 for put option.
    K : float
        Strike price.
    vol_n : float
        Volatility (annualized).
    annuity_factor : float, optional
        Multiplier to adjust the Black76 forward price to present value (default is 1).
        This is composed of the discount factor and the accrual period fraction.
    intrinsic_time_split : bool, optional
        If True, splits option value into intrinsic and time value components (default is False).
    analytical_greeks : bool, optional
        If True, analytical greeks will be calculated and returned (default is False).
    numerical_greeks : bool, optional
        If True, numerical greeks will be calculated and returned using finite differences (default is False).

    Returns
    -------
    results : dict
          Dictionary containing the following key-value pairs:
        - 'price' : float
            Option price.
        - 'analytical_greeks' : float
            Dictionary of analytical greeks; 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'numerical_greeks' : float
            Dictionary of numerical greeks; 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'intrinsic_time_split' : float
            Dictionary of intrinsic and time value components; 'intrinsic', 'time'.
    """

    # Convert to arrays. Function is vectorised.
    F, tau, cp, K, σN, annuity_factor = map(np.atleast_1d, (F, tau, cp, K, vol_n, annuity_factor))

    # Validation checks
    assert F.shape == tau.shape
    shapes = set([param.shape for param in (F, tau, cp, K, σN, annuity_factor)])
    assert len(shapes) in [1, 2]
    if len(shapes) == 2:
        assert (1,) in shapes

    # Price per Bachelier formula
    d = (F - K) / (σN * np.sqrt(tau))
    X = annuity_factor * ( cp * (F - K) * norm.cdf(cp * d) + σN * np.sqrt(tau) * norm.pdf(d) )

    results = {'price': X}

    # Intrinsic / time value split
    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (F - K))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    if analytical_greeks:
        results['analytical_greeks'] = pd.DataFrame()

        results['analytical_greeks']['delta'] = cp * norm.cdf(cp * d) * annuity_factor

        # Market convention is to adjust normal vega to a 1 basis point impact hence the 0.0001 multiplier.
        results['analytical_greeks']['vega'] = np.sqrt(tau) * norm.pdf(cp * d) * annuity_factor * 0.0001

        # Normalised to 1 calendar day.
        results['analytical_greeks']['theta'] = (-0.5 * norm.pdf(d) * σN / np.sqrt(tau)) * annuity_factor / (1/365.25)

    return results


def bachelier_solve_implied_vol(
        F: [float, np.array],
        tau: [float, np.array],
        cp: [float, np.array],
        K: [float, np.array],
        X: float,
        vol_n_guess: float=0.01,
        annuity_factor: [float, np.array]=1,
        ) -> float:
    """Solve the implied normal volatility with the Bachelier pricing formula."""

    def error_function(vol_):
        # Return the errors (relative to the price, as we want invariance to the price) to be minimised.
        # For gradient-based optimisation methods, return squared error to ensure differentiability.
        relative_error = (bachelier_price(F=F, tau=tau, cp=cp, K=K, vol_n=vol_, annuity_factor=annuity_factor)['price'].sum() - X) / X
        error = relative_error**power
        if abs(error) < obj_func_tol**power:  # Check against objective function tolerance
            raise StopIteration(vol_)  # Use StopIteration to terminate optimization early if tolerance is met
        return error

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    xtol = 1e-6
    obj_func_tol = 1e-5
    x0 = vol_n_guess
    bounds = VOL_N_BOUNDS

    # Note Brent's does not work for a SSE obj. function as f(a) and f(b) must have different signs.
    # f(a), f(b) are the function values of the bracket.
    try:
        power = 1 # Use linear error for the root_scalar method
        res = root_scalar(lambda vol_: error_function(vol_), x0=x0, bracket=bounds, xtol=xtol, method='brentq')
        if abs(error_function(res.root)) < obj_func_tol**power:
            return res.root
        else:
            raise RuntimeError
    except StopIteration as e:
        # Return the root if we meet the objective tolerance early
        return e.value
    except RuntimeError:
        try:
            # Fallback to L-BFGS-B if root_scalar fails
            options = {'ftol': obj_func_tol, 'gtol': 0} # gtol set to 0, so optimisation is terminated based on ftol
            power = 2 # Use squared error for the optimisation
            res = minimize(fun=error_function, x0=np.atleast_1d(x0), bounds=[bounds], method='L-BFGS-B', options=options)
            # 2nd condition required as we have overridden options to terminate based on ftol
            if res.success or abs(res.fun) < obj_func_tol**power:
                return res.x[0]
            else:
                raise ValueError('Optimisation to solve normal volatility did not converge.')
        except StopIteration as e:
            # Return the root if we meet the objective tolerance early
            return e.value[0]



def black76_sln_to_normal_vol_analytical(
        F: float,
        tau: float,
        K: float,
        vol_sln: float,
        ln_shift: float
    ):
    """
    Calculates the normal volatility from the Black76 log-normal volatility.

    Parameters
    ----------
    F : float
        Forward price.
    tau : float
        Time to expiry (in years).
    K : float
        Strike price.
    vol_sln : float
        Log-normal volatility (annualized).
    ln_shift : float, optional
        Log-normal shift, applied to forward price and strike (default is 0).

    Returns
    -------
    float
        Normal volatility (annualized).

    References:
    [1] Hagan, Patrick & Lesniewski, Andrew & Woodward, Diana. (2002). Managing Smile Risk. Wilmott Magazine. 1. 84-108.
    """
    F = F + ln_shift
    K = K + ln_shift
    σB = vol_sln

    # Per B.63. in reference [1].
    # σN = σB * (F - K) * (1 - σB**2 * tau / 24) / np.log(F / K)

    # Per B.64 in reference [1]. Slightly more accurate than B.63.
    ln_F_K = np.log(F / K)
    σN = σB * np.sqrt(F*K) \
         * (1 + (1/24) * ln_F_K**2 + 1/1920 * ln_F_K**4) \
         / (1 + (1/24) * (1 - (1/120) * ln_F_K**2) * σB**2 * tau + (1/5760) * σB**4 * tau**2)
    return σN


def black76_sln_to_normal_vol(
        F: float,
        tau: float,
        K: float,
        vol_sln: float,
        ln_shift: float
    ) -> float:

    # If the forward and strike are equal, we can equate the Black76 & Bachelier formulae, and solve analytically.
    if abs(F - K) < 1e-10:
        F = F + ln_shift # K is not used as K=F
        res = (2.0 * F * norm.cdf((vol_sln * np.sqrt(tau)) / 2.0) - F) / (np.sqrt(tau) * norm.pdf(0))
    else:
        # If needed, faster method detailed in: Le Floc'h, Fabien, Fast and Accurate Analytic Basis Point Volatility (April 10, 2016).
        # This solve is invariant to the call/put perspective and the risk-free rate / annuity factor.
        cp = 1
        black76_px = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]
        vol_n_guess = np.atleast_1d(black76_sln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift))
        res = bachelier_solve_implied_vol(F=F, tau=tau, cp=cp, K=K, X=black76_px, vol_n_guess=vol_n_guess.item())
    return np.atleast_1d(res).item()


def shift_black76_vol(
        F: [float, np.float64],
        tau: [float, np.float64],
        K: [float, np.float64],
        vol_sln: [float, np.float64],
        from_ln_shift: [float, np.float64],
        to_ln_shift: [float, np.float64]
    ):

    # The solve is invariant to the call/put perspective, risk-free rate/annuity factor.
    cp = 1

    black76_px = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=from_ln_shift)['price'][0]

    def obj_func_relative_px_error(vol_new_ln_shift):
        # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
        # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
        return ((black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_new_ln_shift, ln_shift=to_ln_shift)['price'][0] - black76_px) / black76_px)**2

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    # We set gtol to zero, so that the optimisation is terminated based on ftol.
    obj_func_tol = 1e-5 ** 2  # 0.001%^2
    options = {'ftol': obj_func_tol, 'gtol': 0}
    res = minimize(fun=obj_func_relative_px_error,
                   x0=np.atleast_1d(vol_sln * (F / (F + (to_ln_shift - from_ln_shift)))),
                   bounds=[VOL_SLN_BOUNDS],
                   method='L-BFGS-B',
                   options=options)
    # 2nd condition required as we have overridden options to terminate based on ftol
    if res.success or abs(res.fun) < obj_func_tol:
        return res.x[0]
    else:
        # Further development could include:
        # (i) fallback to other optimisation methods (e.g. 'trust-constr')
        # (ii) addition of 'xtol' parameter into the optimisation (it is a parameter that is not available in 'L-BFGS-B')
        #      'xtol' allows termination of the optimisation when the solved parameter (i.e. vol_sln) only changes by a small amount (e.g. 0.001%)
        vol_new_ln_shift = res.x[0]
        black76_px_shifted = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_new_ln_shift, ln_shift=to_ln_shift)['price'][0]
        print({'F': F, 'tau': tau, 'K': K, 'vol_n': vol_sln, 'ln_shift': from_ln_shift})
        print({'black76_px': black76_px, 'black76_px_shifted': black76_px_shifted, 'vol_new_ln_shift': vol_new_ln_shift,
               'relative_error': obj_func_relative_px_error(vol_sln)})
        print(res)
        raise ValueError('Optimisation to shift Black76 volatility did not converge.')


def normal_vol_atm_to_black76_sln_atm(
        F: [float, np.float64, np.array],
        tau: [float, np.float64, np.array],
        vol_n_atm: [float, np.float64, np.array],
        ln_shift: [float, np.float64, np.array]):
    F = F + ln_shift
    return (2.0 / np.sqrt(tau)) * norm.ppf((vol_n_atm * np.sqrt(tau) * norm.pdf(0) + F) / (2.0 * F))


def normal_vol_to_black76_sln(
        F: [float, np.float64],
        tau: [float, np.float64],
        K: [float, np.float64],
        vol_n: [float, np.float64],
        ln_shift: [float, np.float64]
    ) -> float:
    # If the forward and strike are equal, we use an analytical solution from equating the Black76 & Bachelier formulae
    if abs(F - K) < 1e-10:
        normal_vol_atm_to_black76_sln_atm(F=F, tau=tau, vol_n_atm=vol_n, ln_shift=ln_shift)

    # The solve is invariant to the risk-free rate or the call/put perspective.
    cp = 1
    bachelier_px = bachelier_price(F=F, tau=tau, K=K, cp=cp, vol_n=vol_n)['price'][0]

    def obj_func_relative_px_error(vol_sln):
        # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
        # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
        return ((bachelier_px - black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]) / bachelier_px)**2

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    # We set gtol to zero, so that the optimisation is terminated based on ftol.
    obj_func_tol = 1e-5 ** 2  # 0.001%^2
    options = {'ftol': obj_func_tol, 'gtol': 0}
    res = minimize(fun=obj_func_relative_px_error,
                   x0=np.atleast_1d(vol_n / (F + ln_shift)),
                   bounds=[VOL_SLN_BOUNDS],
                   method='L-BFGS-B',
                   options=options)
    # 2nd condition required as we have overridden options to terminate based on ftol
    if res.success or abs(res.fun) < obj_func_tol:
        return res.x[0]
    else:
        # Further development could include:
        # (i) fallback to other optimisation methods (e.g. 'trust-constr')
        # (ii) addition of 'xtol' parameter into the optimisation (it is a parameter that is not available in 'L-BFGS-B')
        #      'xtol' allows termination of the optimisation when solved parameter (i.e. vol_sln) only changes by a small amount (e.g. 0.001%)
        vol_sln = res.x[0]
        black76_px = black76_price(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]
        print({'F': F, 'tau': tau, 'K': K, 'vol_n': vol_n, 'ln_shift': ln_shift})
        print({'bachelier_px': bachelier_px, 'black76_px': black76_px, 'vol_sln': vol_sln,
               'relative_error': obj_func_relative_px_error(vol_sln)})
        print(res)
        raise ValueError('Optimisation to convert normal volatility to log-normal volatility did not converge.')



