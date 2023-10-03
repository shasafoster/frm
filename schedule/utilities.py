# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:10:55 2023

@author: shasa
"""

import pandas as pd

def convert_column_type(df: pd.DataFrame):
    for col in df.columns:
        if df[col].apply(isinstance, args=(float,)).all():
            df[col] = pd.to_numeric(df[col])
            
    return df