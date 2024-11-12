# -*- coding: utf-8 -*-
import os
import numpy as np
import warnings

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 


def forward_volatility(t1: float | np.ndarray,
                       vol_t1: float | np.ndarray,
                       t2: float| np.ndarray,
                       vol_t2: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate forward volatility from time t1 to time t2 using the consistency condition.
    Consistency condition: vol_t1**2 * t1 + vol_t1_t2**2 * (t2 - t1) = vol_t2**2 * t2

    Parameters:
    - t1 (float or np.array): Time to first maturity in years, must be less than t2.
    - vol_t1 (float or np.array): Annualized volatility to expiry at t1 for a given delta.
    - t2 (float or np.array): Time to second maturity in years, must be greater than t1.
    - vol_t2 (float or np.array): Annualized volatility to expiry at t2 for the same delta.

    Returns:
    - float or np.array: Forward volatility from time t1 to t2.

    Raises:
    - ValueError: If t2 < t1 or if a negative value is encountered under the square root.

    Warnings:
    - If t1 and t2 are equal, NaN values are returned for those instances, with a warning.

    Notes:
    - Forward volatility is computed based on the implied variance between times t1 and t2.
    - Negative values under the square root indicate inconsistent input volatilities and will raise an error.
    """

    tau = t2 - t1
    if np.any(tau == 0):
        mask = tau == 0
        warnings.warn(f"t2 and t1 are equal. NaN will be returned for these values: t1 {t1[mask]}, t2 {t2[mask]}")
    elif np.any(tau < 0):
        raise ValueError("t2 is less than t1.")

    var_t1_t2 = (vol_t2 ** 2 * t2 - vol_t1 ** 2 * t1) / tau
    if np.any(var_t1_t2 < 0):
        raise ValueError("Negative value encountered under square root.")

    return np.sqrt(var_t1_t2)


def flat_forward_interp(t1: float | np.ndarray,
                        vol_t1: float | np.ndarray,
                        t2: float | np.ndarray,
                        vol_t2: float | np.ndarray,
                        t: float | np.ndarray) -> float | np.ndarray:
    """
    Interpolate volatility at a specified time 't' using flat forward interpolation.

    Parameters:
    - t1 (float or np.array): Time to first expiry in years, must be less than t2.
    - vol_t1 (float or np.array): Annualized volatility to expiry at t1 for a given delta.
    - t2 (float or np.array): Time to second expiry in years, must be greater than t1.
    - vol_t2 (float or np.array): Annualized volatility to expiry at t2 for the same delta.
    - t (float or np.array): Time at which to interpolate the volatility. Must satisfy t1 <= t <= t2.

    Returns:
    - float or np.array: Interpolated volatility at time 't'.

    Notes:
    - If t is outside the range [t1, t2], the function will not compute interpolation and will use the input boundaries.
    - Assumes a flat forward interpolation model, where volatilities are averaged in a time-weighted manner.
    """
    vol_t12 = np.zeros_like(vol_t1)
    mask = (t1 != t2).flatten()
    vol_t12[mask] = forward_volatility(t1[mask], vol_t1[mask], t2[mask], vol_t2[mask])
    return np.sqrt((vol_t1 ** 2 * t1 + vol_t12 ** 2 * (t - t1)) / t)

