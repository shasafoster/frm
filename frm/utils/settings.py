# -*- coding: utf-8 -*-
import numpy as np

# Global settings
value_date = np.nan
include_payments_on_value_date_in_npv = False
include_months_non_business_days_accrued_interest_if_value_date_is_last_business_day_of_month = False
roll_user_specified_dates = False

# Interpolation methods
zero_curve_interpolation_method = 'linear_on_log_of_discount_factors'
# add a few more methods
# - cubic / smoothed spline on ln of discount factors
# - cubic / smoothed  spline on zero rates
# - cubic / smoothed  linear on zero rates
# - flat forward (linear on log of zeros
# https://stackoverflow.com/questions/78054656/what-is-the-difference-between-the-various-spline-interpolators-from-scipy

# MTM CCIRS - notional definition for day 1 or current coupon for single pricer or bulk run

# BBG - IBOR ← → RFR adjustment spreads.

# Method of calculating greeks. (analycal, bump and reprice)


inflation_forward_interpolation_method = 'linear' # 'exponential'


# Lyashenko, A., & Mercurio, F. (2019). Looking Forward to Backward-Looking Rates: A Modeling Framework for Term Rates Replacing LIBOR. SSRN Electronic Journal. doi:10.2139/ssrn.3330240
rfr_optionlet_expiry_method = ['date_of_last_fixing','3rd_way_through_period']

# Swaption
# Cash settlement method - 'market default annuity' or 'swap curve annuity'.

# Greeks
vega_normalisation = +0.01 # +1%
dv01_adjustment = +0.0001 # +0.01%

# Excel formats
excel_date_format = '%d-%b-%Y'
excel_amt_format = '#,##0_-;(#,##0)_-;-_-'


# Default number of monte carlo paths

# extroplotation methods
# flat on rate / yield
# do not apply


