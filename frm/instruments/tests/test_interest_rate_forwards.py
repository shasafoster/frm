# -*- coding: utf-8 -*-


import os
import pathlib

os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve())

from instruments.interest_rate_forwards import IRFuture 


import pandas as pd


irfutures1 = [IRFuture(effective_date=pd.Timestamp(2022,6,15),maturity_date=pd.Timestamp(2022,9,15),day_count_basis='ACT/360',price=98.4714),
                IRFuture(effective_date=pd.Timestamp(2022,9,15),maturity_date=pd.Timestamp(2022,12,21),day_count_basis='ACT/360',price=97.8203),
                IRFuture(effective_date=pd.Timestamp(2022,12,21),maturity_date=pd.Timestamp(2023,3,21),day_count_basis='ACT/360',price=97.3464),
                IRFuture(effective_date=pd.Timestamp(2023,3,21),maturity_date=pd.Timestamp(2023,6,15),day_count_basis='ACT/360',price=97.0356),
                IRFuture(effective_date=pd.Timestamp(2023,6,15),maturity_date=pd.Timestamp(2023,9,21),day_count_basis='ACT/360',price=96.8673),
                IRFuture(effective_date=pd.Timestamp(2023,9,21),maturity_date=pd.Timestamp(2023,12,20),day_count_basis='ACT/360',price=96.8777),
                IRFuture(effective_date=pd.Timestamp(2023,12,20),maturity_date=pd.Timestamp(2024,3,20),day_count_basis='ACT/360',price=96.9729),
                IRFuture(effective_date=pd.Timestamp(2024,3,20),maturity_date=pd.Timestamp(2024,6,20),day_count_basis='ACT/360',price=97.0827)]


irfutures2 = [IRFuture(imm_delivery_year=2022, imm_delivery_month=9,day_count_basis='ACT/360',price=98.4714),
             IRFuture(imm_delivery_year=2022, imm_delivery_month=12,day_count_basis='ACT/360',price=97.8203),
             IRFuture(imm_delivery_year=2023, imm_delivery_month=3,day_count_basis='ACT/360',price=97.3464),
             IRFuture(imm_delivery_year=2023, imm_delivery_month=6,day_count_basis='ACT/360',price=97.0356),
             IRFuture(imm_delivery_year=2023, imm_delivery_month=9,day_count_basis='ACT/360',price=96.8673),
             IRFuture(imm_delivery_year=2023, imm_delivery_month=12,day_count_basis='ACT/360',price=96.8777),
             IRFuture(imm_delivery_year=2023, imm_delivery_month=3,day_count_basis='ACT/360',price=96.9729),
             IRFuture(imm_delivery_year=2024, imm_delivery_month=6,day_count_basis='ACT/360',price=97.0827)]