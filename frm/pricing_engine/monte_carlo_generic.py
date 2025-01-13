# -*- coding: utf-8 -*-
import numpy as np

from frm.utils import MAX_SIMULATIONS_PER_LOOP, DEFAULT_NB_SIMULATIONS

def generate_rand_nbs(nb_steps: int,
                      nb_rand_vars: int=1,
                      nb_simulations: int=DEFAULT_NB_SIMULATIONS,
                      apply_antithetic_variates: bool=False,
                      random_seed: int=0):
    """
    Generate random numbers for Monte Carlo simulations with options for variance reduction.
    # TODO: Consider extenstions for stratified sampling, sobol and halton sequences.

    Parameters:
    -----------
    nb_steps : int
        The number of periods for which random numbers need to be generated.
    nb_rand_vars : int
        The number of random numbers
    nb_simulations : int, optional
        The total number of simulations. Default is 100,000.
    apply_antithetic_variates : bool, optional
        Flag to indicate whether antithetic variates should be applied. Default is False.
    random_seed : int, optional
        Seed for random number generation. Default is 0.

    Returns:
    --------
    np.array: Array of shape (nb_steps, nb_rand_vars, nb_simulations) containing random numbers.

    Raises:
    ValueError: If the number of random numbers to be generated exceeds 10 million, as this can lead to memory issues.
    """

    np.random.seed(random_seed)

    assert isinstance(nb_steps, int), type(nb_steps) 
    assert isinstance(nb_rand_vars, int), type(nb_rand_vars)
    assert isinstance(nb_simulations, int), type(nb_simulations)
    assert isinstance(apply_antithetic_variates, bool)
    assert nb_steps >= 1, nb_steps
    assert nb_rand_vars >= 1, nb_rand_vars
    assert nb_simulations >= 1, nb_simulations
    
    if (nb_steps * nb_simulations) > MAX_SIMULATIONS_PER_LOOP:
        raise ValueError("Too many steps & simulations for one refresh; may lead to memory leak")    
        
    if apply_antithetic_variates and nb_simulations == 1:
        raise ValueError("Antithetic variates requiries >=2 simulations")

    if apply_antithetic_variates:
        nb_pairs = nb_simulations // 2
        rand_nbs_normal = np.random.normal(0, 1, (nb_steps, nb_rand_vars, nb_simulations - nb_pairs)) # standard normal random numbers
        rand_nbs_antithetic_variate = -1 * rand_nbs_normal[:,:,:nb_pairs]
        rand_nbs = np.concatenate([rand_nbs_normal, rand_nbs_antithetic_variate], axis=2)

        # Reindex to pair normal with antithetic variates
        idx = np.column_stack((np.arange(nb_pairs), np.arange(nb_pairs) + nb_pairs)).flatten()
        rand_nbs = rand_nbs[:, :, idx]
    else:    
        rand_nbs = np.random.normal(0, 1, (nb_steps, nb_rand_vars, nb_simulations))
    
    return rand_nbs


def normal_corr(C: np.array, 
               rand_nbs: np.array):
    """
    Generate correlated pseudo random normal variates using the Cholesky factorization.
    
    Parameters:
    C (np.ndarray): Correlation matrix.
    rand_nbs (np.ndarray): Matrix of normally distributed pseudorandom numbers.
    
    Returns:
    np.ndarray: Correlated pseudo random normal variates.
    """    
    
    M = np.linalg.cholesky(C).T
    return (M @ rand_nbs.T).T



if __name__ == "__main__":
    rand_nbs = generate_rand_nbs(nb_steps=20,
                                 nb_rand_vars=1,
                                 nb_simulations=1000,
                                 apply_antithetic_variates=True)


