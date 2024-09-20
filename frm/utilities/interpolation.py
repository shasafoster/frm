# -*- coding: utf-8 -*-


# import numpy as np
# import warnings


# def interpolate(X: np.array,
#                 Y: np.array,
#                 x: np.array,
#                 flat_extrapolation: bool=True,
#                 log_Y_values: bool=False):
#     """
#     Interpolate the FX forward curve to match given expiry_dates.
#     Please note expiry date is 2 business days prior to the delivery date
#     """
    
#     unique_dates = x.drop_duplicates()
#     combined_index = self.fx_forward_curve.index.union(unique_dates)
#     result = self.fx_forward_curve.reindex(combined_index).copy()
#     start_date, end_date = self.fx_forward_curve.index.min(), self.fx_forward_curve.index.max()
    
#     if flat_extrapolation:
#         try:                    
#             result['fx_forward_rate'] = result['fx_forward_rate'].interpolate(method='time', limit_area='inside').ffill().bfill()
#         except:
#             pass
#         # Find out of range dates and warn
#         out_of_range_dates = unique_dates[(unique_dates < start_date) | (unique_dates > end_date)]
#         for date in out_of_range_dates:
#             warnings.warn(f"Date {date} is outside the range {start_date} - {end_date}, flat extrapolation applied.")
#     else:
#         result['fx_forward_rate'] = result['fx_forward_rate'].interpolate(method='time', limit_area='inside')
    
#     result = result.reindex(x)
    
#     return result['fx_forward_rate']
# Test