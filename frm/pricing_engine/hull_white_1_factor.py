# -*- coding: utf-8 -*-

import numpy as np
import scipy 

def calc_theta(zero_curve,
               mean_reversion_level: float,
               volatility: float,
               n: int=100):
    """
    Per a specified mean_reversion_level (α) and volatility (σ), 
    set theta (θ), based on the term structure of interest rates

    Parameters
    ----------
    alpha : float
        α, the mean reversion parameter of the Hull-White 1 factor model 
    sigma : float
        σ, the (annualised) volatility paramater of the Hull-White 1 factor model 
    n : int, optional
        defines the granularity of the date grid. The default is 100.

    Returns
    -------
    θ_spline_definition : Result of scipy.interpolate.splrep. Input to scipy.interpolate.splev
  
    # To do, to be extended for a term structure of α, σ 
    # Need to specify term structure segmenents if doing that? 

    References:
        [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) 
    """
    
    dt = 1e-5
    α = mean_reversion_level
    σ = volatility
    
    years_grid = np.linspace(zero_curve.zero_data['years'].min(),
                             zero_curve.zero_data['years'].max(),n)
    
    # Calculate the 2nd derivative by numerical differentiation
    f = zero_curve.instantaneous_forward_rate(years=years_grid)
    f_plus_dt = zero_curve.instantaneous_forward_rate(years=years_grid+dt)
    f_minus_dt = zero_curve.instantaneous_forward_rate(years=years_grid-dt)
    df_dt = (f_plus_dt - f_minus_dt) / (2 * dt)

    # Equation (3.34), in Section 3.3.1 'The Short-Rate Dynamics' on page 73 of [1] (page 121 of the pdf)
    θ_grid = df_dt + α * f + (σ**2) * (1-np.exp(-2*α*years_grid)) / (2*α)

    θ_spline_definition = scipy.interpolate.splrep(years_grid, θ_grid) 

    return θ_spline_definition
    

def calc_B(t, T, α):
    # [1] Damiano Brigo, Fabio Mercurio - Interest Rate Models Theory and Practice (2001, Springer) 
    #     In section 3.3.2 'Bond and Option Pricing', page 75 (page 123 of the pdf) in [1]
    # [2] MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41
    return (1/α) *(1-np.exp(-α*(T- t)))


def calc_A(t, T, θ_spline_definition, α, σ):         
    # MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41            
    def integrand_1(t, T): 
        return calc_B(t, T)**2    
    def integrand_2(t, T):
        θ_values = scipy.interpolate.splev(t, θ_spline_definition)
        return θ_values * calc_B(t, T)
        
    integrand_1_res = scipy.integrate.quad(integrand_1, t, T)[0]
    integrand_2_res = scipy.integrate.quad(integrand_2, t, T)[0]
    
    return 0.5*(σ**2) * integrand_1_res - integrand_2_res
        

def calc_discount_factor(t, T, θ, α, σ, r0):
    # MAFS525 – Computational Methods for Pricing Structured Products, Slide 3/41 (Hull Whi)
    B = calc_B(t, T, α)
    A = calc_A(t, T, θ, α, σ)
    return np.exp(A - r0*B)
