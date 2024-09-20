# -*- coding: utf-8 -*-


import os
import pathlib

os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve()) 

from instruments.bond import Bond
from market_data.zero_curve import ZeroCurve

import pandas as pd


#% Construct zero curve for pricing
date = pd.Timestamp(2021,3,31)
df = pd.DataFrame([
    [pd.Timestamp(2021,3,31),1.0],
    [pd.Timestamp(2021,4,1),0.99999],
    [pd.Timestamp(2021,4,4),0.99996],
    [pd.Timestamp(2021,4,11),0.99988],
    [pd.Timestamp(2021,7,5),0.99751],
    [pd.Timestamp(2021,9,15),0.99436],
    [pd.Timestamp(2021,12,21),0.98861],
    [pd.Timestamp(2022,3,21),0.98211],
    [pd.Timestamp(2022,6,15),0.97522],
    [pd.Timestamp(2022,9,21),0.96702],
    [pd.Timestamp(2022,12,20),0.95961],
    [pd.Timestamp(2023,3,20),0.95244],
    [pd.Timestamp(2023,6,20),0.94553],
    [pd.Timestamp(2024,4,4),0.92506],
    [pd.Timestamp(2025,4,7),0.90329],
    [pd.Timestamp(2026,4,5),0.88444],
    [pd.Timestamp(2027,4,4),0.86506],
    [pd.Timestamp(2028,4,4),0.84584],
    [pd.Timestamp(2029,4,4),0.82639],
    [pd.Timestamp(2030,4,4),0.80889],
    [pd.Timestamp(2031,4,5),0.79012],
    [pd.Timestamp(2033,4,4),0.75335],
    [pd.Timestamp(2036,4,7),0.7014],
    [pd.Timestamp(2041,4,8),0.62326],
    [pd.Timestamp(2046,4,4),0.56472],
    [pd.Timestamp(2051,4,4),0.51662],
    [pd.Timestamp(2061,4,4),0.44873],
    [pd.Timestamp(2071,4,4),0.40244],
    [pd.Timestamp(2081,4,6),0.36389]
    ],columns=['date','discount_factor'])
zc = ZeroCurve(date=date,raw_data=df)

bond = Bond(
    effective_date=pd.Timestamp(2021, 3, 15),
    maturity_date=pd.Timestamp(2031, 3, 15),
    fixed_rate=0.05,
    currency='USD',
    dsc_crv=zc
    )

pricing = bond.leg.price()




