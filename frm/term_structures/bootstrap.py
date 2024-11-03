# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
from frm.enums import DayCountBasis
from frm.utils import get_busdaycal
from frm.term_structures.bootstrap_helpers import cash_rates_quote_helper, bootstrap_cash_rates, bootstrap_futures

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

# Cash rates and futures define their own forward and discount curves.
# DF of cash rate definitions to → ZeroCurve
# DF of futures quotes → Zero Curve

# Swap curve bootstrappers
# Boostrap
# (i) forward curve
# (ii) discount curve
# (iii) forward and discount curves
# Based on fixed rates or basis spreads

# Inputs
# Set of quote instruments, quote leg vs reference leg
# Zero curve (to be solved)
# Defined zero curves (if any).
# Adjust pillar points to solve instruments to nil.

curve_date = pd.Timestamp('2024-06-28')
busdaycal = get_busdaycal('AUD')
settlement_delay = 1
day_count_basis = DayCountBasis.ACT_365

# Cash rates
data = {
    'tenor': ['on','tn','1w','3m'],
    'rate': np.array([4.30450, 4.30450, 4.30450, 4.44530])/100
}
cash_rates_df = pd.DataFrame(data)
cash_rates_df = cash_rates_quote_helper(df=cash_rates_df, curve_date=curve_date, day_count_basis=day_count_basis,
                                        settlement_delay=settlement_delay, busdaycal=busdaycal)
cash_rates_df = bootstrap_cash_rates(df=cash_rates_df, curve_date=curve_date)

# Futures
data = {
    'effective_date': pd.DatetimeIndex([
        '2024-09-13', '2024-12-13', '2025-03-14', '2025-06-13', '2025-09-12', '2025-12-12', '2026-03-13', '2026-06-12'
    ]),
    'maturity_date': pd.DatetimeIndex([
        '2024-12-13', '2025-03-13', '2025-06-16', '2025-09-15', '2025-12-12', '2026-03-12', '2026-06-15', '2026-09-14'
    ]),
    'price': [95.4800, 95.4500, 95.5100, 95.6000, 95.7100, 95.8000, 95.8700, 95.9100],
    'convexity_bps': [0, 0, 0, 0, 0, 0, 0, 0]
}
futures_df = pd.DataFrame(data)
futures_df = bootstrap_futures(df=futures_df, curve_date=curve_date, cash_rates_df=cash_rates_df, day_count_basis=day_count_basis)

