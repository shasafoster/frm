# -*- coding: utf-8 -*-
import os

from frm.enums import TermRate

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


# from frm.instruments.ir.swap import Swap
# from frm.instruments.ir.swap_defaults import USD_LIBOR_SWAP_ABOVE_1Y, USD_LIBOR_SWAP_1Y

import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.enums import CompoundingFreq, TermRate


def test_construction_from_discount_factors():

    # Test yield curve construction from discount factors
    epsilon = 1e-8
    
    df = pd.DataFrame([[pd.Timestamp(2021,9,30),1.0],
                        [pd.Timestamp(2021,10,1),0.99],
                        [pd.Timestamp(2021,10,4),0.999],
                        [pd.Timestamp(2021,10,11),0.998],
                        [pd.Timestamp(2022,1,5),0.982],
                        [pd.Timestamp(2023,1,4),0.953],
                        [pd.Timestamp(2024,1,4),0.926],
                        [pd.Timestamp(2025,1,7),0.905],
                        [pd.Timestamp(2026,1,5),0.883],
                        [pd.Timestamp(2027,1,4),0.866],
                        [pd.Timestamp(2028,1,4),0.844],
                        [pd.Timestamp(2029,1,4),0.827],
                        [pd.Timestamp(2030,1,4),0.810],
                        [pd.Timestamp(2031,1,5),0.789],
                        [pd.Timestamp(2033,1,4),0.749],
                        [pd.Timestamp(2036,1,7),0.699],
                        [pd.Timestamp(2041,1,8),0.622],
                        [pd.Timestamp(2046,1,4),0.559],
                        [pd.Timestamp(2051,1,4),0.513],
                        [pd.Timestamp(2061,1,4),0.451],
                        [pd.Timestamp(2071,1,4),0.410],
                        [pd.Timestamp(2081,1,6),0.354]],columns=['date','discount_factor'])
    
    zc = ZeroCurve(curve_date=pd.Timestamp(2021,9,30),pillar_df=df, interp_method='linear_on_ln_discount')
    for i,row in df.iterrows():
        assert abs(row['discount_factor'] - zc.get_discount_factors(row['date'])[0]) < epsilon
  

def test_construction_from_zero_rates():
    epsilon = 1e-8

    # Construct zero curve term structure
    df = pd.DataFrame([[pd.Timestamp(2022,1,1), 0.05],
                       [pd.Timestamp(2023,1,1), 0.05],
                       [pd.Timestamp(2024,1,1), 0.05],
                       [pd.Timestamp(2025,1,1), 0.05]],columns=['date','zero_rate'])
    zc = ZeroCurve(curve_date=pd.Timestamp(2021, 12, 31),pillar_df=df, compounding_freq=CompoundingFreq.CONTINUOUS, interp_method='linear_on_ln_discount')

    # Test zero rate
    date = pd.Timestamp(2022, 12, 31)
    simple      = zc.get_zero_rates(CompoundingFreq.SIMPLE, dates=date)
    continuous  = zc.get_zero_rates(CompoundingFreq.CONTINUOUS, dates=date)
    daily       = zc.get_zero_rates(CompoundingFreq.DAILY, dates=date)
    monthly     = zc.get_zero_rates(CompoundingFreq.MONTHLY, dates=date)
    quarterly   = zc.get_zero_rates(CompoundingFreq.QUARTERLY, dates=date)
    semi_annual = zc.get_zero_rates(CompoundingFreq.SEMIANNUAL, dates=date)
    annual      = zc.get_zero_rates(CompoundingFreq.ANNUAL, dates=date)

    assert abs(simple[0]      - 0.05127109637602412) < epsilon
    assert abs(continuous[0]  - 0.05) < epsilon
    assert abs(daily[0]       - 0.05000342481392583) < epsilon
    assert abs(monthly[0]     - 0.05010431149342143) < epsilon
    assert abs(quarterly[0]   - 0.05031380616253766) < epsilon
    assert abs(semi_annual[0] - 0.05063024104885771) < epsilon
    assert abs(annual[0]      - 0.05127109637602412) < epsilon

    #% Test forward rates
    epsilon = 1e-8

    d1 = pd.Series([pd.Timestamp(2022,1,15)])
    d2 = pd.Series([pd.Timestamp(2023,1,15)])
    fwd_rate = zc.get_forward_rates(d1,d2, TermRate.SIMPLE)
    assert abs(fwd_rate[0] - simple[0]) < epsilon

    d1 = pd.Series([pd.Timestamp(2022,1,15),pd.Timestamp(2022,1,15)])
    d2 = pd.Series([pd.Timestamp(2022,1,15),pd.Timestamp(2023,1,15)])
    fwd_rate = zc.get_forward_rates(d1,d2, TermRate.SIMPLE)
    assert abs(fwd_rate[1] - simple[0]) < epsilon


    # Test zero curve construction from zero rates
    df = pd.DataFrame([[pd.Timestamp(2022,1,1), 0.0033254],
                       [pd.Timestamp(2022,1,4), 0.0033946],
                       [pd.Timestamp(2022,1,11),0.0042867],
                       [pd.Timestamp(2022,4,5), 0.0096205]],columns=['date','zero_rate'])
    zc = ZeroCurve(curve_date=pd.Timestamp(2021,12,31),pillar_df=df, compounding_freq=CompoundingFreq.ANNUAL, interp_method='linear_on_ln_discount')

    for i,row in df.iterrows():
        assert abs(row['zero_rate'] - zc.get_zero_rates(CompoundingFreq.ANNUAL, row['date'])[0]) < epsilon


if __name__ == "__main__":
    test_construction_from_discount_factors()


#%%

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from frm.term_structures.zero_curve import ZeroCurve
from frm.pricing_engine.hw1f import HullWhite1Factor
from frm.utils import year_frac
from frm.enums import DayCountBasis
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# ESTR swap curve on 1 April 2024 per https://github.com/YANJINI/One-Factor-Hull-White-Model-Calibration-with-CAF
curve_date = pd.Timestamp('2024-04-01')
df = pd.DataFrame({
    'tenor': ['ON', 'SW', '2W', '3W', '1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M', '15M', '18M', '21M', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y', '11Y', '12Y', '15Y', '20Y', '25Y', '30Y'],
    'date': pd.to_datetime(['2-Apr-2024', '10-Apr-2024', '17-Apr-2024', '24-Apr-2024', '3-May-2024', '3-Jun-2024', '3-Jul-2024', '5-Aug-2024', '3-Sep-2024', '3-Oct-2024', '4-Nov-2024', '3-Dec-2024', '3-Jan-2025', '3-Feb-2025', '3-Mar-2025', '3-Apr-2025', '3-Jul-2025', '3-Oct-2025', '5-Jan-2026', '7-Apr-2026', '5-Apr-2027', '3-Apr-2028', '3-Apr-2029', '3-Apr-2030', '3-Apr-2031', '5-Apr-2032', '4-Apr-2033', '3-Apr-2034', '3-Apr-2035', '3-Apr-2036', '4-Apr-2039', '4-Apr-2044', '5-Apr-2049', '3-Apr-2054']),
    'discount_factor': [0.999892, 0.999026, 0.998266, 0.997514, 0.996546, 0.993222, 0.99014, 0.98688, 0.984079, 0.981287, 0.978453, 0.975944, 0.973358, 0.970875, 0.968705, 0.966373, 0.959921, 0.954107, 0.948336, 0.942805, 0.922607, 0.903406, 0.884216, 0.864765, 0.845061, 0.824882, 0.804566, 0.783991, 0.763235, 0.742533, 0.683701, 0.605786, 0.54803, 0.500307]
})
df['years'] = year_frac(curve_date, df['date'], DayCountBasis.ACT_ACT)

zero_curve = ZeroCurve(curve_date=curve_date, pillar_df=df[['years','discount_factor']], interp_method='cubic_spline_on_ln_discount')

# HW1F model parameters
short_rate_mean_rev_lvl = 0.05 # Standard values are 1%-10% annualized
short_rate_vol = 0.0196 # Standard values are 1%-10% annualized
hw1f = HullWhite1Factor(zero_curve=zero_curve, mean_rev_lvl=short_rate_mean_rev_lvl, vol=short_rate_vol)

# Fit the model to the zero curve
grid_length = 50
hw1f.setup_theta(num=grid_length)

# Demonstrate the fit of the model to the zero curve with a table of the errors (in basis points ) for each source data point
avg_error_bps = hw1f.calc_error_for_theta_fit(print_results=True)

#%%




#%%
# #%% Test boostrapping from deposits, futures and par swaps

# curve_date = pd.Timestamp(2022,3,31)
# deposits = [Deposit(effective_date=curve_date,tenor_name='O/N',interest_rate=0.0033254),
#             Deposit(effective_date=curve_date,tenor_name='T/N',interest_rate=0.0033946),
#             Deposit(effective_date=curve_date,tenor_name='1W', interest_rate=0.0042867),
#             Deposit(effective_date=curve_date,tenor_name='3M', interest_rate=0.0096205)]

# futures = [IRFuture(effective_date=pd.Timestamp(2022,6,15), maturity_date=pd.Timestamp(2022,9,15), day_count_basis='ACT/360',price=98.4585),
#            IRFuture(effective_date=pd.Timestamp(2022,9,15), maturity_date=pd.Timestamp(2022,12,21),day_count_basis='ACT/360',price=97.8374),
#            IRFuture(effective_date=pd.Timestamp(2022,12,21),maturity_date=pd.Timestamp(2023,3,21), day_count_basis='ACT/360',price=97.3385),
#            IRFuture(effective_date=pd.Timestamp(2023,3,21), maturity_date=pd.Timestamp(2023,6,15), day_count_basis='ACT/360',price=97.0123),
#            IRFuture(effective_date=pd.Timestamp(2023,6,15), maturity_date=pd.Timestamp(2023,9,21), day_count_basis='ACT/360',price=96.8906),
#            IRFuture(effective_date=pd.Timestamp(2023,9,21), maturity_date=pd.Timestamp(2023,12,20),day_count_basis='ACT/360',price=96.8573),
#            IRFuture(effective_date=pd.Timestamp(2023,12,20),maturity_date=pd.Timestamp(2024,3,20), day_count_basis='ACT/360',price=96.9217),
#            IRFuture(effective_date=pd.Timestamp(2024,3,20), maturity_date=pd.Timestamp(2024,6,20), day_count_basis='ACT/360',price=97.0056)]

# tenor_par_USD_LIBOR_3M_fixed_rates = [('1Y', 0.010237),
#                                       ('2Y', 0.025167),
#                                       ('3Y', 0.026041),
#                                       ('4Y', 0.025436),
#                                       ('5Y', 0.024722),
#                                       ('6Y', 0.024325),
#                                       ('7Y', 0.024108),
#                                       ('8Y', 0.024021),
#                                       ('9Y', 0.023789),
#                                       ('10Y',0.023744),
#                                       ('12Y',0.023818),
#                                       ('15Y',0.023784),
#                                       ('20Y',0.023762),
#                                       ('25Y',0.023144),
#                                       ('30Y',0.022451),
#                                       ('40Y',0.020943),
#                                       ('50Y',0.019437)]

# specified_term_list = [{'transaction_date' : curve_date,
#                         'tenor' : tenor,
#                         'pay_fixed_rate' : fixed_rate} for (tenor,fixed_rate) in tenor_par_USD_LIBOR_3M_fixed_rates]

# swap_1Y = Swap(specified_terms=specified_term_list[0], default_terms=USD_LIBOR_SWAP_1Y)
# swaps = [swap_1Y] + [Swap(specified_terms=item, default_terms=USD_LIBOR_SWAP_ABOVE_1Y) for item in specified_term_list[1:]]

# instruments = {'deposits': deposits, 'futures': futures, 'swaps': swaps}

# zc = ZeroCurve(curve_date=pd.Timestamp(2022,3,31),
#                instruments=instruments)



# #%%

# # Swaps used in bootstrapping must have pricing close to par
# # Exclude 1Y and 2Y swaps from this testing as they were not used in bootstrapping
# for swap in swaps[2:]:
#     swap.set_discount_curve(zc)
#     swap.set_forward_curve(zc)
#     pricing = swap.price(calc_PV01=True)
#     price = int(pricing['price'])
#     PV01 = int(pricing['PV01']['PV01'])
#     msg = '\n' \
#           + 'Maturity date: ' + swap.pay_leg.maturity_date.strftime('%Y-%m-%d') + '\n' \
#           + 'Price: ' + str(price) + '\n' \
#           + 'PV01: ' + str(PV01) + '\n' \
#           + 'Price/PV01: ' + str(round(price/PV01,2)) + '\n'
#     if price < PV01:
#         print('Maturity date:', swap.pay_leg.maturity_date.strftime('%Y-%m-%d'), 'Price / PV01:', str(round(price/PV01,2)))
#     else:
#         print(msg)   
#     #assert price < PV01/5
    

#%%




#%% Test boostrapping from deposits and futures

# curve_date = pd.Timestamp(2022,3,31)
# deposits = [Deposit(effective_date=curve_date,tenor_name='O/N',interest_rate=0.0033254),
#             Deposit(effective_date=curve_date,tenor_name='T/N',interest_rate=0.0033946),
#             Deposit(effective_date=curve_date,tenor_name='1W', interest_rate=0.0042867),
#             Deposit(effective_date=curve_date,tenor_name='3M', interest_rate=0.0096205)]
# futures = [IRFuture(effective_date=pd.Timestamp(2022,6,15), maturity_date=pd.Timestamp(2022,9,15), day_count_basis='ACT/360',price=98.4585),
#            IRFuture(effective_date=pd.Timestamp(2022,9,15), maturity_date=pd.Timestamp(2022,12,21),day_count_basis='ACT/360',price=97.8374),
#            IRFuture(effective_date=pd.Timestamp(2022,12,21),maturity_date=pd.Timestamp(2023,3,21), day_count_basis='ACT/360',price=97.3385),
#            IRFuture(effective_date=pd.Timestamp(2023,3,21), maturity_date=pd.Timestamp(2023,6,15), day_count_basis='ACT/360',price=97.0123),
#            IRFuture(effective_date=pd.Timestamp(2023,6,15), maturity_date=pd.Timestamp(2023,9,21), day_count_basis='ACT/360',price=96.8906),
#            IRFuture(effective_date=pd.Timestamp(2023,9,21), maturity_date=pd.Timestamp(2023,12,20),day_count_basis='ACT/360',price=96.8573),
#            IRFuture(effective_date=pd.Timestamp(2023,12,20),maturity_date=pd.Timestamp(2024,3,20), day_count_basis='ACT/360',price=96.9217),
#            IRFuture(effective_date=pd.Timestamp(2024,3,20), maturity_date=pd.Timestamp(2024,6,20), day_count_basis='ACT/360',price=97.0056)]
# instruments = {'deposits': deposits, 'futures': futures}

# zc = ZeroCurve(curve_date=curve_date,instruments=instruments)
# zc.plot(forward_rate_terms=[90,180])



 
#%%   
 




