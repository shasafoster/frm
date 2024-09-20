# -*- coding: utf-8 -*-
import numpy as np
MAX_SIMULATIONS_PER_LOOP = 100e6

def generate_rand_nbs(nb_steps: int,
                      nb_rand_vars: int=1,
                      nb_simulations: int=None,  
                      flag_apply_antithetic_variates: bool=None,
                      random_seed=0):
    
    """
    Generate random numbers for Monte Carlo simulations with an option to apply antithetic variates.

    This function generates normally distributed random numbers for use in Monte Carlo simulations.
    It can optionally generate antithetic variates to help in variance reduction. The function is
    designed to support simulations across multiple periods and can handle a large number of simulations.

    Parameters:
    nb_steps (int): The number of periods for which random numbers need to be generated.
    nb_rand_vars (int) The number of random numbers
    nb_simulations (int, optional): The total number of simulations. Default is 100,000.
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
      specified `nb_simulations`.
    - The generated random numbers follow a standard normal distribution (mean 0, standard deviation 1).
    """
    
    np.random.seed(random_seed)
    
    if nb_simulations is None:
        nb_simulations = 100 * 1000 
        
    if flag_apply_antithetic_variates is None:
        flag_apply_antithetic_variates = False
    
    assert isinstance(nb_steps, int), type(nb_steps) 
    assert isinstance(nb_rand_vars, int), type(nb_rand_vars)
    assert isinstance(nb_simulations, int), type(nb_simulations)
    assert isinstance(flag_apply_antithetic_variates, bool)
    assert nb_steps >= 1, nb_steps
    assert nb_rand_vars >= 1, nb_rand_vars
    assert nb_simulations >= 1, nb_simulations
    
    if (nb_steps * nb_simulations) > MAX_SIMULATIONS_PER_LOOP:
        raise ValueError("Too many steps & simulations for one refresh; may lead to memory leak")    
        
    if flag_apply_antithetic_variates and nb_simulations == 1:
        raise ValueError("Antithetic variates requiries >=2 simulations") 
        
    if flag_apply_antithetic_variates:
        nb_antithetic_variate_simulations = nb_simulations // 2
        nb_normal_simulations = nb_simulations - nb_antithetic_variate_simulations
        rand_nbs_normal = np.random.normal(0, 1, (nb_steps, nb_rand_vars, nb_normal_simulations)) # standard normal random numbers
        rand_nbs_antithetic_variate = -1 * rand_nbs_normal[:,:,:nb_antithetic_variate_simulations]
        rand_nbs = np.concatenate([rand_nbs_normal, rand_nbs_antithetic_variate], axis=2)
    else:    
        rand_nbs = np.random.normal(0, 1, (nb_steps, nb_rand_vars, nb_simulations))
    
    return rand_nbs


if __name__ == "__main__":
    rand_nbs = generate_rand_nbs(nb_steps=20,
                                 nb_rand_vars=1,
                                 nb_simulations=1000,
                                 flag_apply_antithetic_variates=True)


