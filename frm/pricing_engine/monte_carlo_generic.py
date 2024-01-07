# -*- coding: utf-8 -*-
"""
@author: Shasa Foster
https://www.linkedin.com/in/shasafoster
"""

if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())
    
import numpy as np
np.random.seed(0)


def generate_rand_nbs(nb_of_periods: int,
                      nb_of_simulations: int=None,     
                      flag_apply_antithetic_variates: bool=None):
    """
    Generate random numbers for Monte Carlo simulations with an option to apply antithetic variates.

    This function generates normally distributed random numbers for use in Monte Carlo simulations.
    It can optionally generate antithetic variates to help in variance reduction. The function is
    designed to support simulations across multiple periods and can handle a large number of simulations.

    Parameters:
    nb_of_periods (int): The number of periods for which random numbers need to be generated.
    nb_of_simulations (int, optional): The total number of simulations. Default is 100,000.
    flag_apply_antithetic_variates (bool, optional): Flag to indicate whether antithetic variates should
                                                     be applied for variance reduction. Default is True.

    Returns:
    np.array: A 2D array of random numbers. Each row corresponds to a simulation path, and each
              column corresponds to a period.

    Raises:
    ValueError: If the number of simulations exceeds 10 million, as this may lead to memory issues.

    Notes:
    - If `flag_apply_antithetic_variates` is True, the function generates half the number of 
      normal simulations and then creates antithetic variates for these. The results are concatenated
      to form the final set of simulations.
    - The function ensures that the total number of simulations (normal and antithetic) matches the
      specified `nb_of_simulations`.
    - The generated random numbers follow a standard normal distribution (mean 0, standard deviation 1).
    """
    
    if nb_of_simulations is None:
        nb_of_simulations = 1000 # tk need to change this to 100k 
        
    if flag_apply_antithetic_variates is None:
        flag_apply_antithetic_variates = True
    
      
    if (nb_of_periods * nb_of_simulations) > 100 * 1000 * 1000:
        raise ValueError("Too many simulations for one refresh; may lead to memory leak")    
        
    if flag_apply_antithetic_variates:
        nb_of_antithetic_variate_simulations = nb_of_simulations // 2
        nb_of_normal_simulations = nb_of_simulations - nb_of_antithetic_variate_simulations
        rand_nbs_normal = np.random.normal(0, 1, (nb_of_normal_simulations,nb_of_periods)) # standard normal random numbers
        rand_nbs_antithetic_variate = -1 * rand_nbs_normal[:nb_of_antithetic_variate_simulations,:]
        rand_nbs = np.concatenate([rand_nbs_normal, rand_nbs_antithetic_variate], axis=0)
    else:    
        rand_nbs = np.random.normal(0, 1, (nb_of_simulations,nb_of_periods))
    
    return rand_nbs