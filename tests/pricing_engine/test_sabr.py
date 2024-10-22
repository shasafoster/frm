# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

import numpy as np
import pandas as pd
from frm.pricing_engine.sabr import (solve_alpha_from_sln_vol, calc_sln_vol_for_strike_from_sabr_params,
                                     fit_sabr_params_to_sln_smile, calc_sln_vol_for_strike_from_sabr_params)


def test_solve_alpha_from_sln_vol():
    tau = 5.0
    F = 3.229 / 100
    alpha = 3.26 / 100
    beta = 50 /100
    rho = -5.95 / 100
    volvol = 36.35 / 100
    vol_sln_atm = 15.03 / 100
    ln_shift = 0.02

    solved_alpha = solve_alpha_from_sln_vol(tau=tau, F=F, beta=beta, rho=rho, volvol=volvol, vol_sln_atm=vol_sln_atm, ln_shift=ln_shift)
    assert np.isclose(alpha, solved_alpha, atol=1e-3)


def test_calc_sln_vol_for_strike_from_sabr_params():

    # Scalar strike over term structure
    data = {
        'expiry_years': [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25],
        'forward_rate': [0.0447, 0.0412, 0.0413, 0.0409, 0.0423, 0.0457, 0.0479, 0.0496, 0.0487, 0.0448, 0.0382],
        'alpha': [0.1302, 0.0092, 0.0501, 0.0277, 0.0357, 0.0246, 0.0277, 0.0258, 0.0180, 0.0257, 0.0154],
        'beta': [1.0, 0, 0.6, 0.4, 0.5, 0.4, 0.5, 0.5, 0.4, 0.6, 0.4],
        'rho': [-0.2387, 0.0976, -0.1018, -0.0437, -0.0595, 0.0152, -0.0781, -0.0262, -0.0487, -0.0778, -0.0075],
        'volvol': [0.6233, 0.2869, 0.3991, 0.3237, 0.3635, 0.3417, 0.3557, 0.3486, 0.3452, 0.3549, 0.3461],
        'vol_sln_atm': [0.1326, 0.1527, 0.1582, 0.1533, 0.1502, 0.1344, 0.1169, 0.1095, 0.1030, 0.0920, 0.1058]
    }
    df = pd.DataFrame(data)

    for i,row in df.iterrows():
        ln_shift = 0.02
        tau = row['expiry_years'] - 0.25
        F = row['forward_rate']
        alpha = row['alpha']
        beta = row['beta']
        rho = row['rho']
        volvol = row['volvol']
        test_vol_sln_atm = row['vol_sln_atm']

        ln_vol_atm = calc_sln_vol_for_strike_from_sabr_params(tau=tau, F=F, alpha=alpha, beta=beta, rho=rho, volvol=volvol, K=F, ln_shift=ln_shift)
        assert np.isclose(ln_vol_atm, test_vol_sln_atm, atol=5e-4) # 0.05% tolerance

    ln_vol_atm_array = calc_sln_vol_for_strike_from_sabr_params(tau=df.loc[:, 'expiry_years'].values - 0.25,
                                                                F=df.loc[:, 'forward_rate'].values,
                                                                alpha=df.loc[:, 'alpha'].values,
                                                                beta=df.loc[:, 'beta'].values,
                                                                rho=df.loc[:, 'rho'].values,
                                                                volvol=df.loc[:, 'volvol'].values,
                                                                K=df.loc[:, 'forward_rate'].values,
                                                                ln_shift=0.02)

    assert np.allclose(ln_vol_atm_array, df.loc[:, 'vol_sln_atm'].values, atol=5e-4) # 0.05% tolerance

    # Strike array
    tau = 4.75
    F = 4.229 / 100
    alpha = 3.57 / 100
    beta = 50 / 100
    rho = -5.95 / 100
    volvol = 36.35 / 100
    ln_shift = 0.02

    K = F + np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]) / 100
    vol_sln = np.array([18.71, 17.38, 16.33, 15.54, 15.03, 14.75, 14.67, 14.74, 14.91]) / 100

    vol_calc = calc_sln_vol_for_strike_from_sabr_params(tau=tau, F=F, alpha=alpha, beta=beta, rho=rho, volvol=volvol, K=K, ln_shift=ln_shift)
    assert (np.abs(vol_calc - vol_sln) < 5e-4).all() # 0.05% tolerance


def test_fit_sabr_params_to_sln_smile():

    K = np.array([-0.4729, 0.5271, 1.0271, 1.5271, 1.7771, 2.0271, 2.2771,
                  2.4021, 2.5271, 2.6521, 2.7771, 3.0271, 3.2771, 3.5271,
                  4.0271, 4.5271, 5.5271]) / 100
    vols_target = np.array([19.641923, 15.785344, 14.305103, 13.073869,
                            12.550007, 12.088721, 11.691661, 11.517660,
                            11.360133, 11.219058, 11.094293, 10.892464,
                            10.750834, 10.663653, 10.623862, 10.714479,
                            11.103755]) / 100

    F = 2.5271 / 100
    tau = 10
    ln_shift = 0.03
    beta = 0.5

    params, res = fit_sabr_params_to_sln_smile(tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vols_target, beta=0.5)

    check = np.abs(1 - params / np.array([0.0253, 0.5, -0.2463, 0.2908]))
    assert (check < 1e-3).all() # 0.1% tolerance

    tau = 4.75
    F = 4.229 / 100
    alpha = 3.57 / 100
    beta = 50 / 100
    rho = -5.96 / 100
    volvol = 36.35 / 100
    ln_shift = 0.02

    K = F + np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]) / 100
    vol_sln = np.array([18.71, 17.38, 16.33, 15.54, 15.03, 14.75, 14.67, 14.74, 14.91]) / 100
    params, res = fit_sabr_params_to_sln_smile(tau=tau, F=F, ln_shift=ln_shift, K=K, vols_sln=vol_sln, beta=0.5)

    check = np.abs(1 - params / np.array([alpha, beta, rho, volvol]))
    assert (check < 5e-3).all() # 0.1% tolerance


if __name__ == "__main__":
    test_solve_alpha_from_sln_vol()
    test_calc_sln_vol_for_strike_from_sabr_params()
    test_fit_sabr_params_to_sln_smile()



