# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from numba import njit
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy


def black76(F: [float, np.array],
            tau: [float, np.array],
            r: [float, np.array],
            cp: [float, np.array],
            K: [float, np.array],
            vol_sln: [float, np.array],
            ln_shift: [float, np.array],
            intrinsic_time_split: bool=False,
            analytical_greeks: bool=False,
            numerical_greeks: bool=False):
    """
    Black76 pricing for European options.

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
    results = dict()

    # Set to Greek letter so code review to analytical formulae is easier
    σB = vol_sln

    # Convert to arrays. Function is vectorised.
    F, tau, r, cp, K, σB, ln_shift = [np.atleast_1d(arg).astype(float) for arg in [F, tau, r, cp, K, σB, ln_shift]]
    F = F + ln_shift
    K = K + ln_shift

    # Price per Black76 formula
    d1 = (np.log(F/K) + (0.5 * σB**2 * tau)) / (σB*np.sqrt(tau))
    d2 = d1 - σB*np.sqrt(tau)
    X = np.exp(-r*tau) * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    results['price'] = X

    # Intrinsic / time value split
    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (F - K))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    if analytical_greeks:
        # To be validated
        results['analytical_greeks'] = pd.DataFrame
        results['analytical_greeks']['delta'] = cp * norm.cdf(cp * d1)
        results['analytical_greeks']['vega'] = F * np.sqrt(tau) * norm.pdf(d1)
        results['analytical_greeks']['theta'] = -F * σB * norm.pdf(d1) / (2 * np.sqrt(tau)) - r * K * np.exp(-r*tau) * norm.cdf(cp * d2)
        results['analytical_greeks']['gamma'] = norm.pdf(d1) / (F * σB * np.sqrt(tau))
        results['analytical_greeks']['rho'] = cp * K * tau * np.exp(-r*tau) * norm.cdf(cp * d2)

    if numerical_greeks:
        results['numerical_greeks'] = pd.DataFrame

    return results


def bachelier(F: [float, np.array],
              tau: [float, np.array],
              r: [float, np.array],
              cp: [float, np.array],
              K: [float, np.array],
              vol_n: [float, np.array],
              intrinsic_time_split: bool=False,
              analytical_greeks: bool=False,
              numerical_greeks: bool=False):
    """
    Bachelier pricing for European options.

    Parameters
    ----------
    F : float
        Forward price.
    tau : float
        Time to expiry (in years).
    r : float
        Risk-free interest rate (annualized continuously compounded).
    cp : int
        Option type: 1 for call option, -1 for put option.
    K : float
        Strike price.
    vol_n : float
        Volatility (annualized).
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
        - 'analytical_greeks' : float
            Dictionary of analytical greeks; 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'numerical_greeks' : float
            Dictionary of numerical greeks; 'delta', 'vega', 'theta', 'gamma', 'rho'.
        - 'intrinsic_time_split' : float
            Dictionary of intrinsic and time value components; 'intrinsic', 'time'.
    """
    results = dict()

    # Set to Greek letter so code review to analytical formulae is easier
    σN = vol_n

    # Convert to arrays. Function is vectorised.
    F, tau, r, cp, K, σN = [np.atleast_1d(arg).astype(float) for arg in [F, tau, r, cp, K, σN]]

    # Price per Bachelier formula
    d = (F - K) / (σN * np.sqrt(tau))
    X = np.exp(-r*tau) * ( cp * (F - K) * norm.cdf(cp * d) + σN * np.sqrt(tau) * norm.pdf(d) )

    results['price'] = X

    # Intrinsic / time value split
    if intrinsic_time_split:
        intrinsic = np.full_like(X, np.nan)
        intrinsic[tau>0] = np.maximum(0, cp * (F - K))[tau>0]
        intrinsic[tau==0] = X[tau==0]
        results['intrinsic'] = intrinsic
        results['time'] = X - intrinsic

    if analytical_greeks:
        results['analytical_greeks'] = pd.DataFrame()

    if numerical_greeks:
        results['numerical_greeks'] = pd.DataFrame()

    return results


def black76_ln_to_normal_vol_analytical(
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
    # If needed, faster method detailed in: Le Floc'h, Fabien, Fast and Accurate Analytic Basis Point Volatility (April 10, 2016).
    # The solve is invariant to the risk-free rate or the call/put perspective.

    r = 0.0
    cp = 1

    black76_px = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price']
    vol_n_guess = np.atleast_1d(black76_ln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift))

    def px_sse(vol_n):
        bachelier_px = bachelier(F=F, tau=tau, r=r, cp=cp, K=K, vol_n=vol_n)['price']
        return 100 * np.sum(black76_px - bachelier_px) ** 2 # Multiply by 100 to help optimizer convergence.

    res = scipy.optimize.minimize(
            fun=px_sse,
            x0=vol_n_guess,
            jac=None,
            options={'gtol': 1e-8,
                     'eps': 1e-9,
                     'maxiter': 10,
                     'disp': False},
            method='CG'
    )
    vol_n = res.x[0]
    return vol_n


def shift_black76_vol(
        F: float,
        tau: float,
        K: float,
        vol_sln: float,
        from_ln_shift: float,
        to_ln_shift: float
    ):

    # The solve is invariant to the risk-free rate or the call/put perspective.
    r = 0
    cp = 1

    black76_px = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=vol_sln, ln_shift=from_ln_shift)['price']

    def px_sse(vol_new_ln_shift):
        black76_px_shift = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=vol_new_ln_shift, ln_shift=to_ln_shift)['price']
        return 100 * np.sum(black76_px - black76_px_shift) ** 2  # Multiply by 100 to help optimizer convergence.

    vol_guess = np.atleast_1d(vol_sln * (F / (F + (to_ln_shift - from_ln_shift))))
    res = scipy.optimize.minimize(fun=px_sse, x0=vol_guess)
    vol_post_shift = res.x[0]
    return vol_post_shift


def normal_vol_to_black76_sln(
        F: float,
        tau: float,
        K: float,
        vol_n: float,
        ln_shift: float
    ):

    # The solve is invariant to the risk-free rate or the call/put perspective.
    r = 0.0
    cp = 1

    bachelier_px = bachelier(F=F, tau=tau, K=K, r=r, cp=cp, vol_n=vol_n)['price']
    vol_sln_guess = np.atleast_1d(vol_n / F)

    def px_sse(vol_sln):
        black76_px = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price']
        return 1000 * np.sum(bachelier_px - black76_px) ** 2 # Multiply by 1000 to help optimizer convergence.

    res = scipy.optimize.minimize(fun=px_sse,x0=vol_sln_guess)
    vol_sln = res.x[0]
    return vol_sln
