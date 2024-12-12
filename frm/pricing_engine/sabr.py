# -*- coding: utf-8 -*-
import os
import numpy as np
from numpy.typing import NDArray
import scipy
from typing import Optional
import numbers

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


def calc_sln_vol_for_strike_from_sabr_params(
        tau: [float, NDArray[float]],
        F: [float, NDArray[float]],
        alpha: [float, NDArray[float]],
        beta: [float, NDArray[float]],
        rho: [float, NDArray[float]],
        volvol: [float, NDArray[float]],
        K: [float, NDArray[float]],
        ln_shift: float) -> [float, NDArray[float]]:
    """
    Calculate the log-normal (Black76) volatility for a given strike price using SABR model parameters.

    This implementation follows the Hagan et al. (2002) model for managing smile risk, applying formulas 2.17 and 2.18.

    Parameters:
    -----------
    tau : float or array-like of float
        Time to expiry in years. Must be either scalar or 1D array matching the dimensions of `F`, `alpha`, etc.
    F : float or array-like of float
        Forward rate, scalar or 1D array.
    alpha : float or array-like of float
        Initial volatility level, scalar or 1D array.
    beta : float or array-like of float
        Elasticity of the volatility with respect to the forward rate, scalar or 1D array.
    rho : float or array-like of float
        Correlation between the forward rate and the volatility, scalar or 1D array.
    volvol : float or array-like of float
        Volatility of volatility, scalar or 1D array.
    K : float or array-like of float
        Strike price, scalar or 1D array.
    ln_shift : float
        Log-normal shift to adjust both `F` and `K` to prevent zero or near-zero log values.

    Returns:
    --------
    float or ndarray
        Log-normal volatility for the given strike price(s). Returns a scalar if all inputs are scalars, otherwise returns an ndarray.

    Notes:
    ------
    If the `F` and `K` values are very close (at-the-money condition), an alternate formula (2.18) is applied for higher accuracy.

    References:
    -----------
    [1] Hagan, P., Lesniewski, A., & Woodward, D. (2002). Managing Smile Risk. Wilmott Magazine, 1, 84-108.
    """

    tau, F, alpha, beta, rho, volvol, K, ln_shift = map(np.atleast_1d, (tau, F, alpha, beta, rho, volvol, K, ln_shift))

    # Input must be 1D column vector
    for param in (tau, F, alpha, beta, rho, volvol):
        assert param.ndim == 1, 'Parameters must be 1D arrays or 2D column vectors with shape (n, 1).'

    assert tau.shape == F.shape == alpha.shape == beta.shape == rho.shape == volvol.shape, \
        'tau, F, alpha, beta, rho, volvol must have the same shape as they define the SABR smile for a given expiry.'

    if K.size > 1:
        assert tau.size == K.size or tau.size == 1, \
            'K must be the same shape as the SABR smile parameters (tau, F, alpha, beta, rho, volvol), and/or a scalar, (1,).'

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
    tau: float,
    F: float,
    beta: float,
    rho: float,
    volvol: float,
    vol_sln_atm: float,
    ln_shift: float = 0.0) -> float:
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
        ATMF log-normal volatility.
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
    β, ρ, v = beta, rho, volvol
    del beta, rho, volvol

    # Rearrange equation 2.18 from Hagan 2002 into cubic form: Aα^3 + Bα^2 + Cα + K = 0
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
                                 K: NDArray[float],
                                 vols_sln: NDArray[float],
                                 ln_shift: float=None,
                                 beta_overide: Optional[float]=None)-> tuple:
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

    def vol_sse(param):
        if beta_overide is None:
            beta_, rho_, volvol_ = param
        else:
            rho_, volvol_ = param
            beta_ = beta_overide

        alpha_ = solve_alpha_from_sln_vol(tau=tau, F=F, beta=beta_, rho=rho_, volvol=volvol_, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)
        sabr_vols = calc_sln_vol_for_strike_from_sabr_params(tau=tau, F=F, alpha=alpha_, beta=beta_, rho=rho_, volvol=volvol_ ,K=K, ln_shift=ln_shift)
        return sum((vols_sln - sabr_vols) ** 2)

    # ATM volatility check
    vol_sln_atm = vols_sln[np.isclose(K, F)]
    if vol_sln_atm.size != 1:
        raise ValueError('A unique ATM strike must be present.')
    vol_sln_atm = np.atleast_1d(vol_sln_atm).item()


    # params = (beta), rho, volvol
    # beta has a valid range of 0≤β≤1
    # rho has a valid range of -1≤ρ≤1
    # volvol has a valid range of 0<v≤∞
    x0 = np.array([0.00, 0.10]) if beta_overide is not None else np.array([0.0, 0.0, 0.1])
    bounds = [(-1.0, 1.0), (0.0001, None)] if beta_overide is not None else [(-1.0, 1.0), (-1.0, 1.0), (0.0001, None)]
    res = scipy.optimize.minimize(fun=lambda param: vol_sse(param), x0=x0, bounds=bounds)

    if res.success:
        beta, rho, volvol = (beta_overide, *res.x) if beta_overide is not None else res.x
        alpha = solve_alpha_from_sln_vol(tau=tau, F=F, beta=beta, rho=rho, volvol=volvol, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)
        return (alpha, beta, rho, volvol), res
    else:
        print(res)
        raise ValueError('Optimization of SABR parameters failed.')





