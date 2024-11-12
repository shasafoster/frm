# -*- coding: utf-8 -*-

from frm.enums import DayCountBasis

# Global settings
VALUE_DATE = None
INCLUDE_PAYMENTS_ON_VALUE_DATE_IN_NPV  = False
INCLUDE_MONTHS_NON_BUSINESS_DAYS_ACCRUED_INTEREST_IF_VALUE_DATE_IS_LAST_BUSINESS_DAY_OF_MONTH = False
LIMIT_ACCRUED_INTEREST_TO_UNSETTLED_CASHFLOW = False
ROLL_USER_SPECIFIED_DATES  = False

# Day count basis
REPORTING_DAY_COUNT_BASIS = DayCountBasis._30_360 # For current/non-current splits and cashflow bucketing
DATE2YEARFRAC_DAY_COUNT_BASIS = DayCountBasis.ACT_365 #

# Interpolation methods
ZERO_CURVE_INTERPOLATION_METHOD  = 'linear_on_log_of_discount_factors'
# add a few more methods
# - cubic / smoothed spline on ln of discount factors
# - cubic / smoothed  spline on zero rates
# - cubic / smoothed  linear on zero rates
# - flat forward (linear on log of zeros
# https://stackoverflow.com/questions/78054656/what-is-the-difference-between-the-various-spline-interpolators-from-scipy

# MTM CCIRS - notional definition for day 1 or current coupon for single pricer or bulk run

# BBG - IBOR ← → RFR adjustment spreads.

# Method of calculating greeks. (analycal, bump and reprice)


INFLATION_FORWARD_INTERPOLATION_METHOD  = 'linear' # 'exponential'


# Lyashenko, A., & Mercurio, F. (2019). Looking Forward to Backward-Looking Rates: A Modeling Framework for Term Rates Replacing LIBOR. SSRN Electronic Journal. doi:10.2139/ssrn.3330240
RFR_OPTIONLET_EXPIRY_METHOD = ['date_of_last_fixing','3rd_way_through_period']

# Swaption
# Cash settlement method - 'market default annuity' or 'swap curve annuity'.

# Greeks
vega_normalisation = +0.01 # +1%
dv01_adjustment = +0.0001 # +0.01%

# Excel formats
EXCEL_DATE_FORMAT = '%d-%b-%Y'
EXCEL_AMT_FORMAT = '#,##0_-;(#,##0)_-;-_-'


MAX_SIMULATIONS_PER_LOOP = 100_000_000  # Maximum number of total random numbers per loop

# Default number of monte carlo paths

# extroplotation methods
# flat on rate / yield
# do not apply


