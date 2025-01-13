# # -*- coding: utf-8 -*-
# import os
# if __name__ == "__main__":
#     os.chdir(os.environ.get('PROJECT_DIR_FRM'))
#
# import numpy as np
# from frm.enums import CompoundingFreq
#
#
# def zero_rate_from_discount_factor(
#         years,
#         discount_factor,
#         compounding_freq: CompoundingFreq):
#
#     if compounding_freq == CompoundingFreq.CONTINUOUS:
#             return (- np.log(discount_factor) / years)
#     elif compounding_freq == CompoundingFreq.SIMPLE:
#         return (((1.0 / discount_factor) - 1.0) / years)
#     elif compounding_freq in {
#             CompoundingFreq.DAILY,
#             CompoundingFreq.MONTHLY,
#             CompoundingFreq.QUARTERLY,
#             CompoundingFreq.SEMIANNUAL,
#             CompoundingFreq.ANNUAL}:
#         periods_per_year = compounding_freq.periods_per_year
#         return periods_per_year * ((1.0 / discount_factor) ** (1.0 / (periods_per_year * years)) - 1.0)
#     else:
#         raise ValueError(f"Invalid compounding_frequency {compounding_freq}")
#
#
# def change_zero_rate_compounding_freq(
#         years,
#         zero_rate,
#         from_freq: CompoundingFreq,
#         to_freq: CompoundingFreq):
#     # First convert to continuous
#     if from_freq == CompoundingFreq.CONTINUOUS:
#         continuous_rate = zero_rate
#     elif from_freq == CompoundingFreq.SIMPLE:
#         continuous_rate = np.log1p(zero_rate)
#     else:
#         n = from_freq.periods_per_year
#         continuous_rate = n * np.log1p(zero_rate / n)
#
#     # Then convert from continuous to target
#     if to_freq == CompoundingFreq.CONTINUOUS:
#         return continuous_rate
#     elif to_freq == CompoundingFreq.SIMPLE:
#         return np.expm1(continuous_rate)
#     else:
#         m = to_freq.periods_per_year
#         return m * np.expm1(continuous_rate / m)
#
#
# def discount_factor_from_zero_rate(
#         years,
#         zero_rate,
#         compounding_freq: CompoundingFreq):
#
#     if compounding_freq == CompoundingFreq.CONTINUOUS:
#         return np.exp(-zero_rate * years)
#     if compounding_freq == CompoundingFreq.SIMPLE:
#         return  1.0 / (1.0 + zero_rate * years)
#     elif compounding_freq in {
#             CompoundingFreq.DAILY,
#             CompoundingFreq.MONTHLY,
#             CompoundingFreq.QUARTERLY,
#             CompoundingFreq.SEMIANNUAL,
#             CompoundingFreq.ANNUAL}:
#         periods_per_year = compounding_freq.periods_per_year
#         return 1.0 / (1.0 + zero_rate / periods_per_year) ** (periods_per_year * years)
#     else:
#         raise ValueError(f"Invalid compounding_frequency {compounding_freq}")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#