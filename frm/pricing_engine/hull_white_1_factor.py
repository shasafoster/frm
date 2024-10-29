# -*- coding: utf-8 -*-

import numpy as np
import scipy 

from frm.term_structures.zero_curve import ZeroCurve



class HullWhite1Factor():
    zero_curve: ZeroCurve
    mean_rev_lvl: float
    vol: float

    def __post_init__(self):
        assert self.zero_curve.interpolation_method == 'cubic_spline_on_zero_rates'


def calc_theta(self,
               mean_rev_lvl: float,
               vol: float,
               n: int=100):
    """
    Per a specified mean_reversion_level (α) and volatility (σ), 
    calculate theta (θ), based on the term structure of interest rates

    Parameters
    ----------
    zero_curve : ZeroCurve
    mean_reversion_level : float
        The mean reversion parameter of the Hull-White 1-factor model.
    volatility : float
        The (annualised) volatility parameter of the Hull-White 1-factor model
    n : int, optional
        defines the granularity of the date grid. The default is 100.

    Returns
    -------
    θ_spline_definition : Result of scipy.interpolate.splrep. Input to scipy.interpolate.splev
  
    # To do, to be extended for a term structure of α, σ 
    # Need to specify term structure segments if doing that?

    References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) 
    """
    
    dt = 1e-5
    α = mean_rev_lvl
    σ = vol
    
    years_grid = np.linspace(start=zero_curve.data['years'].min(),stop=zero_curve.data['years'].max(),num=n)
    
    # Calculate the 2nd derivative by numerical differentiation
    f = zero_curve.instantaneous_forward_rate(years=years_grid)
    f_plus_dt = zero_curve.instantaneous_forward_rate(years=years_grid+dt)
    f_minus_dt = zero_curve.instantaneous_forward_rate(years=years_grid-dt)
    df_dt = (f_plus_dt - f_minus_dt) / (2 * dt)

    # Equation (3.34), in Section 3.3.1 'The Short-Rate Dynamics' on page 73 of [1] (page 121/1007 of the pdf)
    theta_grid = df_dt + α * f + (σ**2) * (1-np.exp(-2*α*years_grid)) / (2*α)

    theta_spline = scipy.interpolate.splrep(years_grid, theta_grid)

    return theta_grid, theta_spline
    

def calc_B(t, T, mean_rev_lvl):
    # [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) 
    #     In section 3.3.2 'Bond and Option Pricing', page 75 (page 123/1007 of the pdf)
    # [2] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
    α = mean_rev_lvl
    return (1/α) *(1-np.exp(-α*(T- t)))


def calc_A(t, T, theta_spline, mean_rev_lvl, vol):

    α = mean_rev_lvl
    σ = vol

    # MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41            
    def integrand_1(t):
        return calc_B(t, T, α)**2
    def integrand_2(t):
        θ_values = scipy.interpolate.splev(t, theta_spline)
        return θ_values * calc_B(t, T, α)
        
    integrand_1_res = scipy.integrate.quad(func=integrand_1, a=t, b=T)[0]
    integrand_2_res = scipy.integrate.quad(func=integrand_2, a=t, b=T)[0]
    
    return 0.5 * σ**2 * integrand_1_res - integrand_2_res
        

def get_discount_factor(
        t,
        T,
        theta_spline,
        mean_rev_lvl,
        vol,
        r0):
    # MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41 (Hull Whi)

    B = calc_B(t=t, T=T, mean_rev_lvl=mean_rev_lvl)
    A = calc_A(t=t, T=T, theta_spline=theta_spline, mean_rev_lvl=mean_rev_lvl, vol=vol)
    return np.exp(A - r0*B)


def get_zero_rate(
        t,
        T,
        theta_spline,
        mean_rev_lvl,
        vol,
        r0):
    # MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41 (Hull Whi)
    discount_factor = get_discount_factor(t=t, T=T, theta_spline=theta_spline, mean_rev_lvl=mean_rev_lvl, vol=vol, r0=r0)
    return -np.log(discount_factor) / (T-t)