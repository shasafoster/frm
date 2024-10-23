# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import scipy
from typing import Optional
import numbers

# TODO additional tests for calc_sln_vol_for_strike_from_sabr_params
# Checks: F&tau must have same shape, K must be scalar or same shape as F&tau
# Want to be able to pass term structure of sabr params / optiolet and get all the vols needed to price cap/floor in one call.
# TODO
# alpha, beta, rho, volvol must have same shape
# TODO


def calc_sln_vol_for_strike_from_sabr_params(
        tau: [float, np.array],
        F: [float, np.array],
        alpha: [float, np.array],
        beta: [float, np.array],
        rho: [float, np.array],
        volvol: [float, np.array],
        K: [float, np.array],
        ln_shift: float) -> float:
    """
    Calculate the log-normal (Black76) volatility for a given strike per the SABR model.
    Calculation is per original Hagan 2002 paper, [1] equations 2.17 and 2.18.

    Parameters:
    -----------
    tau: float
        Time to maturity in years.
    F: float
        Forward rate.
    alpha: float
        Initial volatility level.
    beta: float
        Elasticity of the volatility with respect to the forward rate.
    rho: float
        Correlation between the forward rate and the volatility.
    volvol: float
        Volatility of volatility.
    K: float
        Strike price.
    ln_shift: float
        Log-normal shift.

    Returns:
    --------
    float
        Log-normal volatility for the given strike.

    References:
    [1] Hagan, Patrick & Lesniewski, Andrew & Woodward, Diana. (2002). Managing Smile Risk. Wilmott Magazine. 1. 84-108.
    """

    tau, F, alpha, beta, rho, volvol, K, ln_shift = map(np.atleast_1d, (tau, F, alpha, beta, rho, volvol, K, ln_shift))

    assert tau.shape == F.shape == alpha.shape == beta.shape == rho.shape == volvol.shape, \
        'tau, F, alpha, beta, rho, volvol must have the same shape as they define the SABR smile.'

    if ln_shift.size == 1:
        ln_shift = ln_shift.item()
    assert isinstance(ln_shift, numbers.Real), 'ln_shift must be a valid numeric scalar.'

    F = F + ln_shift
    K = K + ln_shift

    # Set to symbols used in [1] for easier comparison
    α = alpha
    β = beta
    ρ = rho
    v = volvol

    mask_atm = np.abs(np.log(F / K)) < 1e-06
    σB = np.full(mask_atm.shape, np.nan)

    if mask_atm.any():
        # If ATM, use (2.18) in [1].
        # At ATM, F=K which simplifies the formulae in 2.17.
        σB[mask_atm] = (α /
                (F*K)**((1 - β) / 2) *
                (1
                + ( ((1-β)**2 / 24) * α**2 / ((F*K)**(1-β))
                    + (0.25 * ρ * β * v * α / ((F*K)**((1-β)/2)))
                    + ((2 - 3 * ρ**2) * v**2 / 24)
                ) * tau))[mask_atm]

    if ~mask_atm.all():
        # (2.17b) in [1]
        z = (v / α) * (F * K)**((1 - β) / 2) * np.log(F / K)

        def x(z):
            # (2.17c) in [1]
            return np.log((np.sqrt(1 - 2 * ρ * z + z ** 2) + z - ρ) / (1 - ρ))

        # Row 1 in (2.17a) in [1]
        row1 = α * (z / x(z)) / (
                (F*K)**((1 - β) / 2) *
                (1
                 + (1 - β)**2 / 24 * np.log(F / K)**2
                 + (1 - β)**4 / 1920 * np.log(F / K)**4))

        # Row 2 in (2.17a) in [1]
        row2 = (1
                + ( ((1-β)**2 / 24) * (α**2 / ((F*K)**(1-β)))
                    + (0.25 * ρ * β * v * α / ((F*K)**((1-β)/2)))
                    + ((2 - 3 * ρ**2) * v**2 / 24)
                ) * tau)

        σB[~mask_atm] = (row1 * row2)[~mask_atm]

    if σB.size == 1:
        return σB.item()
    else:
        return σB


def solve_alpha_from_sln_vol(
    tau: [float, np.float64],
    F: [float, np.float64],
    beta: [float, np.float64],
    rho: [float, np.float64],
    volvol: [float, np.float64],
    vol_sln_atm: [float, np.float64],
    ln_shift: [float, np.float64] = 0.0) -> float:
    """
    Solve alpha analytically by rearranging (2.18) in [1] for alpha.
    The equation is a 3rd degree polynomial (in alpha).

    Parameters:
    ----------
    tau: float
        Time to maturity in years.
    F: float
        Forward rate.
    beta: float
        Elasticity of the volatility with respect to the forward rate.
    rho: float
        Correlation between the forward rate and the volatility.
    volvol: float
        Volatility of volatility.
    vol_sln_atm: float
        ATM log-normal volatility.
    ln_shift: float
        Log-normal shift.

    Returns:
    --------
    float
        Solved alpha.

    References:
    [1] Hagan, Patrick & Lesniewski, Andrew & Woodward, Diana. (2002). Managing Smile Risk. Wilmott Magazine. 1. 84-108.
    """

    F = F + ln_shift
    β = beta; del beta
    ρ = rho; del rho
    v = volvol; del volvol

    # Rearrange equation 2.18 from Hagan 2002 into  form Aα^3 + Bα^2 + Cα + K = 0
    A = (1/24) * ((1-β)**2) * tau / (F**(2-2*β))
    B = (1/4) * ρ * β * v * tau / (F**(1-β))
    C = (1 + (2-3*ρ**2) * v**2 * tau/ 24)
    K =  -vol_sln_atm * F**(1-β)
    cubic_polynomial = [A, B, C, K]

    roots = np.roots(cubic_polynomial)
    roots_real = np.extract(np.isreal(roots), np.real(roots))

    # Note: the double real roots case is not tested
    alpha_first_guess = vol_sln_atm * F**(1-β)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min].item()


def fit_sabr_params_to_sln_smile(tau: float,
                                 F: float,
                                 K: np.array,
                                 vols_sln: np.array,
                                 ln_shift: float=None,
                                 beta: Optional[float]=None)-> tuple:
    """
    Calibrates SABR parameters to a volatility smile of European option prices using the Hagan 2002 (lognormal) SABR model and the Black76 formula.
    This method is not suitable for caps/floors or swaptions.

    Notes:
    The rho, volvol and (optionally) beta parameters are solved numerically.
    The beta parameter is optional as it is common for this parameter to be fixed by the practitioner.
    The alpha parameter is solved analytically from the other SABR parameters and the ATM volatility.

    Parameters:
    -----------
    tau: float
        Time to maturity in years.
    F: float
        Forward rate.
    ln_shift: float
        Log-normal shift.
    K: np.array
        Strike prices.
    vols_sln: np.array
        (shifted) log-normal volatilities of European options.
    beta: Optional[float]
        Elasticity of the volatility with respect to the forward rate.

    Returns:
    --------
    tuple
        SABR parameters (alpha, beta, rho, volvol), scipy.optimize.OptimizeResult
    """

    def vol_sse(param, tau, F, ln_shift, K, vols_sln, vol_sln_atm, beta=None):
        if beta is None:
            beta, rho, volvol = param
        else:
            rho, volvol = param
        alpha = solve_alpha_from_sln_vol(tau=tau, F=F, beta=beta, rho=rho, volvol=volvol, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)
        sabr_vols = calc_sln_vol_for_strike_from_sabr_params(tau=tau,F=F,alpha=alpha,beta=beta,rho=rho,volvol=volvol,K=K, ln_shift=ln_shift)
        return sum((vols_sln - sabr_vols) ** 2)

    # Index the at-the-money (ATM) volatility
    mask_atm = K == F
    if mask_atm.sum() != 1:
        raise ValueError('ATM strike must be unique and present.')
    vol_sln_atm = vols_sln[mask_atm].item()

    # params = (beta), rho, volvol
    # beta has a valid range of 0≤β≤1
    # rho has a valid range of -1≤ρ≤1
    # volvol has a valid range of 0<v≤∞
    x0 = np.array([0.00, 0.10]) if beta is not None else np.array([0.0, 0.0, 0.1])
    bounds = [(-1.0, 1.0), (0.0001, None)] if beta is not None else [(-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]

    res = scipy.optimize.minimize(
        fun=lambda param: vol_sse(param, tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vols_sln, vol_sln_atm=vol_sln_atm, beta=beta),
        x0=x0,
        bounds=bounds)

    beta, rho, volvol = (beta, *res.x) if beta is not None else res.x
    alpha = solve_alpha_from_sln_vol(tau=tau, F=F, beta=beta, rho=rho, volvol=volvol, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)

    return (alpha, beta, rho, volvol), res





