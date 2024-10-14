# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM')) 

from frm.pricing_engine.black import (black76,
                                      bachelier,
                                      shift_black76_vol,
                                      black76_ln_to_normal_vol_analytical,
                                      black76_ln_to_normal_vol,
                                      normal_vol_to_black76_ln)


def test_black76_bachelier():
    F = 4.47385 / 100
    tau = 0.758904109589041
    K = 4 / 100
    r = 0

    cp = 1
    px = 0.005984
    px_black76 = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=0.2074, ln_shift=0)['price']
    px_bachelier = bachelier(F=F, tau=tau, K=K, r=r, cp=cp, vol_n=0.8767/100)['price']
    assert abs(px_black76 - px) < 1e-6
    assert abs(px_bachelier - px) < 1e-6

    cp = -1
    px = 0.001246
    px_black76 = black76(F=F, tau=tau, K=K, r=r, cp=cp, vol_sln=0.2074, ln_shift=0)['price']
    px_bachelier = bachelier(F=F, tau=tau, K=K, r=r, cp=cp, vol_n=0.8767/100)['price']
    assert abs(px_black76 - px) < 1e-6
    assert abs(px_bachelier - px) < 1e-6


def test_shift_black76_vol():
    F = 4.47385 / 100
    tau = 0.758904109589041
    K = 4 / 100

    vol_from_shift = 0.2074
    from_shift = 0
    to_shift = 0.02
    vol_to_shift = shift_black76_vol(F=F, tau=tau, K=K, vol_sln=vol_from_shift, from_ln_shift=from_shift, to_ln_shift=to_shift)
    assert abs(vol_to_shift - 0.1407) < 1e-4

    vol_from_shift = 0.1407
    from_shift = 0.02
    to_shift = 0
    vol_to_shift = shift_black76_vol(F=F, tau=tau, K=K, vol_sln=vol_from_shift, from_ln_shift=from_shift, to_ln_shift=to_shift)
    assert abs(vol_to_shift - 0.2074) < 1e-4

    vol_from_shift = 0.2074
    from_shift = 0
    to_shift = 0.01
    vol_to_shift = shift_black76_vol(F=F, tau=tau, K=K, vol_sln=vol_from_shift, from_ln_shift=from_shift, to_ln_shift=to_shift)
    assert abs(vol_to_shift - 0.1677) < 1e-4


def test_black76_ln_to_normal_vol():
    F = 4.47385 / 100
    tau = 0.758904109589041
    K = 4 / 100

    vol_sln = 0.2074
    ln_shift = 0
    normal_vol_analytical = black76_ln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    normal_vol_numerical = black76_ln_to_normal_vol(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    assert abs(normal_vol_analytical - 0.008766291621773826) < 1e-10
    assert abs(normal_vol_numerical - 0.008766291621773826) < 1e-10


    vol_sln = 0.1677
    ln_shift = 0.01
    normal_vol_analytical = black76_ln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    normal_vol_numerical = black76_ln_to_normal_vol(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    assert abs(normal_vol_analytical - 0.008768530296484415) < 1e-10
    assert abs(normal_vol_numerical - 0.008768530296484415) < 1e-10

    vol_sln = 0.1407
    ln_shift = 0.02
    normal_vol_analytical = black76_ln_to_normal_vol_analytical(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    normal_vol_numerical = black76_ln_to_normal_vol(F=F, tau=tau, K=K, vol_sln=vol_sln, ln_shift=ln_shift)
    assert abs(normal_vol_analytical - 0.008765643554332224) < 1e-10
    assert abs(normal_vol_numerical - 0.008765643554332224) < 1e-10


def test_normal_vol_to_black76_ln():

    F = 4.47385 / 100
    tau = 0.758904109589041
    K = 4 / 100

    vol_n = 87.67 / 10000
    ln_shift = 0
    vol_sln = normal_vol_to_black76_ln(F=F, tau=tau, K=K, vol_n=vol_n, ln_shift=ln_shift)
    assert abs(vol_sln - 0.2074) < 1e-3 # Needs slightly higher tolerance.

    vol_n = 87.67 / 10000
    ln_shift = 0.01
    vol_sln = normal_vol_to_black76_ln(F=F, tau=tau, K=K, vol_n=vol_n, ln_shift=ln_shift)
    assert abs(vol_sln - 0.1677) < 1e-4

    vol_n = 87.67 / 10000
    ln_shift = 0.02
    vol_sln = normal_vol_to_black76_ln(F=F, tau=tau, K=K, vol_n=vol_n, ln_shift=ln_shift)
    assert abs(vol_sln - 0.1407) < 1e-4



if __name__ == "__main__":
    test_black76_bachelier() # 0.35s for 1000 iterations
    test_shift_black76_vol()   # 4.9s for 1000 iterations
    test_black76_ln_to_normal_vol() # 1.4s for 1000 iterations
    test_normal_vol_to_black76_ln() # 5.4s for 1000 iterations


