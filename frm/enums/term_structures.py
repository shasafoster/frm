# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
    
from enum import Enum


class RFRFixingCalcMethod(Enum):
    DAILY_COMPOUNDED = 'dailycompounded'
    WEIGHTED_AVERAGE = 'weightedaverage'
    SIMPLE_AVERAGE = 'simpleaverage'

class TermRate(Enum):
    SIMPLE = 'simple'
    CONTINUOUS = 'continuous'
    #DAILY = 'daily' # Forward rate formula is not implemented in ZeroCurve
    #WEEKLY = 'weekly' # Forward rate formula is not implemented in ZeroCurve
    #MONTHLY = 'monthly' # Forward rate formula is not implemented in ZeroCurve
    #QUARTERLY = 'quarterly' # Forward rate formula is not implemented in ZeroCurve
    #SEMIANNUAL = 'semiannual' # Forward rate formula is not implemented in ZeroCurve
    ANNUAL = 'annual'


class FXSmileInterpolationMethod(Enum):
    UNIVARIATE_SPLINE = 'univariate_spline'
    CUBIC_SPLINE = 'cubic_spline'
    HESTON_1993 = 'heston_1993'
    HESTON_CARR_MADAN_GAUSS_KRONROD_QUADRATURE = 'heston_carr_madan_gauss_kronrod_quadrature'
    HESTON_CARR_MADAN_FFT_W_SIMPSONS = 'heston_carr_madan_fft_w_simpsons'
    HESTON_LIPTON = 'heston_lipton'
    HESTON_COSINE = 'heston_cosine'


class DeltaConvention(Enum):
    REGULAR_SPOT = 'regular_spot'
    REGULAR_FORWARD = 'regular_forward'
    PREMIUM_ADJUSTED_SPOT = 'premium_adjusted_spot'
    PREMIUM_ADJUSTED_FORWARD = 'premium_adjusted_forward'


