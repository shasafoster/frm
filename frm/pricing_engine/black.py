# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy

VOL_SLN_BOUNDS = [(0.1 / 100, 1000 / 100)]  # 0.1% to 1000% (0.001 to 10)
VOL_N_BOUNDS = [(0.01 / 100, 100 / 100)]  # 0.01% to 100% (0.0001 to 1)

def black76(F: [float, np.array],
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
    results = dict()

    # Set to Greek letter so code review to analytical formulae is easier
    σB = vol_sln


    # Convert to arrays. Function is vectorised.
    F, tau, cp, K, σB, ln_shift, annuity_factor = \
        [np.atleast_1d(arg).astype(float) for arg in [F, tau, cp, K, σB, ln_shift, annuity_factor]]
    F = F + ln_shift
    K = K + ln_shift

    # Price per Black76 formula
    d1 = (np.log(F/K) + (0.5 * σB**2 * tau)) / (σB*np.sqrt(tau))
    d2 = d1 - σB*np.sqrt(tau)
    X = annuity_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))

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

        results['analytical_greeks']['delta'] = cp * norm.cdf(cp * d1) * annuity_factor

        analytical_vega = F * np.sqrt(tau) * norm.pdf(d1)
        # In practice, vega is displayed as normalised to a 1% shift, hence the 0.01 multiplier.
        results['analytical_greeks']['vega'] = 0.01 * analytical_vega *  annuity_factor

    if numerical_greeks:
        results['numerical_greeks'] = pd.DataFrame()

        # Delta, Δ, is the change in an option's price for a small change in the underlying assets (forward) price.
        # Δ := ∂X/∂F ≈ (X(F_plus) − X(F_minus)) / (F_plus - F_minus)
        Δ_shift = 0.01
        results_F_plus = black76(F=F*(1+Δ_shift), tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift,
                           annuity_factor=annuity_factor, analytical_greeks=True)
        results_F_minus = black76(F=F*(1-Δ_shift), tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift,
                            annuity_factor=annuity_factor, analytical_greeks=True)
        results['numerical_greeks']['delta'] = (results_F_plus['price'] - results_F_minus['price']) / (2 * F * Δ_shift)

        # Vega, ν, is the change in an options price for a small change in the volatility input
        # ν = ∂X/∂σ ≈ (X(σ_plus) − X(σ_minus)) / (σ_plus - σ_minus).
        # We divide by 0.01 (1%) to normalise the vega to a 1% change in the volatility input, per market convention.
        σ_shift = 0.01
        results_σ_plus = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln+σ_shift, ln_shift=ln_shift,
                           annuity_factor=annuity_factor, analytical_greeks=True)
        results_σ_minus = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln-σ_shift, ln_shift=ln_shift,
                            annuity_factor=annuity_factor, analytical_greeks=True)
        results['numerical_greeks']['vega'] = (results_σ_plus['price'] - results_σ_minus['price']) / (2 * (σ_shift / 0.01))

    return results


def bachelier(F: [float, np.array],
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
    results = dict()

    # Set to Greek letter so code review to analytical formulae is easier
    σN = vol_n

    # Convert to arrays. Function is vectorised.
    F, tau, cp, K, σN, annuity_factor = [np.atleast_1d(arg).astype(float) for arg in [F, tau, cp, K, σN, annuity_factor]]

    # Price per Bachelier formula
    d = (F - K) / (σN * np.sqrt(tau))
    X = annuity_factor * ( cp * (F - K) * norm.cdf(cp * d) + σN * np.sqrt(tau) * norm.pdf(d) )

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

        results['analytical_greeks']['delta'] = cp * norm.cdf(cp * d) * annuity_factor

        # Market convention is to adjust normal vega to a 1 basis point impact hence the 0.0001 multiplier.
        results['analytical_greeks']['vega'] = np.sqrt(tau) * norm.pdf(cp * d) * annuity_factor * 0.0001

        # Normalised to 1 calendar day.
        results['analytical_greeks']['theta'] = (-0.5 * norm.pdf(d) * σN / np.sqrt(tau)) * annuity_factor / (1/365.25)

    return results


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
        return (2.0 * F * norm.cdf((vol_sln * np.sqrt(tau)) / 2.0) - F) / (np.sqrt(tau) * norm.pdf(0))

    # If needed, faster method detailed in: Le Floc'h, Fabien, Fast and Accurate Analytic Basis Point Volatility (April 10, 2016).
    # This solve is invariant to the call/put perspective and the risk-free rate / annuity factor.
    cp = 1
    black76_px = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]
    vol_n_guess = np.atleast_1d(black76_sln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift))

    def obj_func_relative_px_error(vol_n):
        bachelier_px = bachelier(F=F, tau=tau, cp=cp, K=K, vol_n=vol_n)['price'][0]
        # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
        # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
        return ((black76_px - bachelier_px) / black76_px)**2

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    # We set gtol to zero, so that the optimisation is terminated based on ftol.
    obj_func_tol = 1e-5 ** 2  # 0.001%^2
    options = {'ftol': obj_func_tol, 'gtol': 0}
    res = scipy.optimize.minimize(fun=obj_func_relative_px_error,
                                  x0=vol_n_guess,
                                  bounds=VOL_N_BOUNDS,
                                  method='L-BFGS-B',
                                  options=options)
    # 2nd condition required as we have overridden options to terminate based on ftol
    if res.success or abs(res.fun) < obj_func_tol:
        return res.x[0]
    else:
        raise ValueError('Optimisation to solve normal volatility did not converge.')



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

    black76_px = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=from_ln_shift)['price'][0]

    def obj_func_relative_px_error(vol_new_ln_shift):
        # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
        # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
        return ((black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_new_ln_shift, ln_shift=to_ln_shift)['price'][0] - black76_px) / black76_px)**2

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    # We set gtol to zero, so that the optimisation is terminated based on ftol.
    obj_func_tol = 1e-5 ** 2  # 0.001%^2
    options = {'ftol': obj_func_tol, 'gtol': 0}
    res = scipy.optimize.minimize(fun=obj_func_relative_px_error,
                                  x0=np.atleast_1d(vol_sln * (F / (F + (to_ln_shift - from_ln_shift)))),
                                  bounds=VOL_SLN_BOUNDS,
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
        black76_px_shifted = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_new_ln_shift, ln_shift=to_ln_shift)['price'][0]
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
    bachelier_px = bachelier(F=F, tau=tau, K=K, cp=cp, vol_n=vol_n)['price'][0]

    def obj_func_relative_px_error(vol_sln):
        # Return the square of the relative error (to the price, as we want invariance to the price) to be minimised.
        # Squared error ensures differentiability which is critical for gradient-based optimisation methods.
        return ((bachelier_px - black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]) / bachelier_px)**2

    # Want prices to be within 0.001% - i.e. if price is 100,000, acceptable range is (99999, 100001)
    # Hence set obj function tol 'ftol' to (0.001%)**2 = 1e-10 (due to the squaring of the relative error in the obj function)
    # We set gtol to zero, so that the optimisation is terminated based on ftol.
    obj_func_tol = 1e-5 ** 2  # 0.001%^2
    options = {'ftol': obj_func_tol, 'gtol': 0}
    res = scipy.optimize.minimize(fun=obj_func_relative_px_error,
                                  x0=np.atleast_1d(vol_n / (F + ln_shift)),
                                  bounds=VOL_SLN_BOUNDS,
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
        black76_px = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift)['price'][0]
        print({'F': F, 'tau': tau, 'K': K, 'vol_n': vol_n, 'ln_shift': ln_shift})
        print({'bachelier_px': bachelier_px, 'black76_px': black76_px, 'vol_sln': vol_sln,
               'relative_error': obj_func_relative_px_error(vol_sln)})
        print(res)
        raise ValueError('Optimisation to convert normal volatility to log-normal volatility did not converge.')



if __name__ == "__main__":

    # sigma_N = 0.006637982346968922
    # F = 0.03368593485365877 + 0.02
    #
    # # B.63 from Hagan 2002 rearranged as a quartic equation in σ_B, for when f=K.
    # # σ_B^4 + 240σ_B^2 + (-5760*F/σ_N)σ_B + 5760 = 0
    # # Aσ_B^4 + Bσ_B^3 + Cσ_B^2 + Dσ_B + K = 0
    # A = 1
    # B = 0  # There is no σ_B^3 term
    # C = 240
    # D = -1 * 5760 * F / sigma_N
    # K = 5760
    # coeffs = [A, B, C, D, K]
    #
    # # Find all roots of the quartic equation
    # roots = np.roots(coeffs)
    #
    # # Filter out complex roots and negative roots (since volatility must be positive and real)
    # real_positive_roots = [root.real for root in roots if np.isreal(root) and root.real > 0]
    #
    # if not real_positive_roots:
    #     raise ValueError("No positive real roots found for σ_B.")
    #
    # # Choose the smallest positive real root (could also choose based on other criteria)
    # sigma_B = min(real_positive_roots)
    # print(sigma_B)
    #

    #from frm.pricing_engine.black import black76

    F = 3.97565 / 100,
    tau = 1.01944
    vol_sln = 19.96 / 100
    ln_shift = 0.02

    black76(F=F, K=F, tau=tau, cp=1, vol_sln=vol_sln, ln_shift=ln_shift)


    # F = 4.47385 / 100
    # tau = 0.758904109589041
    # K = F
    # vol_sln = 14.07 / 100
    # vol_n = 0.008767
    # ln_shift = 0.02
    # cp = 1
    # discount_factor = 0.95591
    # term_multiplier = 24.9315068493151 / 100
    # annuity_factor = discount_factor * term_multiplier
    #
    # σB = vol_sln
    # σN = vol_n
    #
    # # Convert to arrays. Function is vectorised.
    # F, tau, cp, K, σB, ln_shift, discount_factor, term_multiplier = \
    #     [np.atleast_1d(arg).astype(float) for arg in [F, tau, cp, K, σB, ln_shift, discount_factor, term_multiplier]]
    # F = F + ln_shift
    # K = K + ln_shift
    #
    # # Price per Black76 formula
    # d1 = (np.log(F/K) + (0.5 * σB**2 * tau)) / (σB*np.sqrt(tau))
    # d2 = d1 - σB*np.sqrt(tau)
    # Xb = 100e6 * annuity_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    #
    #
    # # Price per Bachelier formula
    # d = (F - K) / (σN * np.sqrt(tau))
    # Xn = 100e6 * annuity_factor * ( cp * (F - K) * norm.cdf(cp * d) + σN * np.sqrt(tau) * norm.pdf(d) )
    # Xn_ = 100e6 * annuity_factor * σN * np.sqrt(tau) * norm.pdf(0)
    #
    # print(Xb, Xn, Xn_)






    # #delta = cp * norm.cdf(cp * d1)
    #
    # # result = black76(F=F, tau=tau, K=K, cp=cp, vol_sln=vol_sln, ln_shift=ln_shift, discount_factor=discount_factor,
    # #                  term_multiplier=term_multiplier, analytical_greeks=True, numerical_greeks=True)
    # # for k,v in result.items():
    # #     print(k,np.round(v * 100e6,0))
    #
    # result = bachelier(F=F, tau=tau, K=K, cp=cp, vol_n=vol_n, annuity_factor=annuity_factor, analytical_greeks=True)
    #
    # for k,v in result.items():
    #     print(k,np.round(v * 100e6,0))
