# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

from frm.schedule.schedule import payment_schedule 
import numpy as np
import pandas as pd

#%% 1 period schedule, quarterly interval

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                               end_date=pd.Timestamp('15-Apr-2020'),
                               payment_freq='Q')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')]])
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


#%% 2 period schedule, quarterly interval

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                               end_date=pd.Timestamp('15-Jul-2020'),
                              payment_freq='Q')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')],
                [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Jul-2020')]])
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% 4 period schedule, quarterly interval

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                               end_date=pd.Timestamp('15-Jan-2021'),
                               payment_freq='Q')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')],
                [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Jul-2020')],
                [pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Oct-2020'),pd.Timestamp('15-Oct-2020')],
                [pd.Timestamp('15-Oct-2020'),pd.Timestamp('15-Jan-2021'),pd.Timestamp('15-Jan-2021')]])
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% 1 period schedule, annual interval

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('15-Jan-2021'),
                          payment_freq='A')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Jan-2021'),pd.Timestamp('15-Jan-2021')]])
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% Dates falling on the weekend roll

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('15-May-2020'),
                          payment_freq='M')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('17-Feb-2020'),pd.Timestamp('17-Feb-2020')], # 15-Feb-2020 falls on Saturday, roll to 17th
                [pd.Timestamp('17-Feb-2020'),pd.Timestamp('16-Mar-2020'),pd.Timestamp('16-Mar-2020')], # 15-Mar-2020 falls on Sunday, roll to 16th
                [pd.Timestamp('16-Mar-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')], 
                [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-May-2020'),pd.Timestamp('15-May-2020')]])
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% Input dates don't roll

df_schedule = payment_schedule(start_date=pd.Timestamp('30-Sep-2020'), 
                          end_date=pd.Timestamp('31-Oct-2020'),
                          payment_freq='M')

arr = np.array([[pd.Timestamp('30-Sep-2020'),pd.Timestamp('31-Oct-2020'),pd.Timestamp('31-Oct-2020')]]) # 31-Aug-2020 is Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% With roll_convention='following', dates roll to next month if fall on weekend

df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M',
                          roll_convention='following')

arr = np.array([[pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('2-Mar-2020')], # 29-Feb-2020 falls on Saturday, roll to 2-March-2020
                [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% Test the payment delay 

df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M', 
                          roll_convention='following',
                          payment_delay=1)

arr = np.array([[pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('3-Mar-2020')], # 29-Feb-2020 falls on Saturday, roll to 2-March-2020
                [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('1-Apr-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('1-May-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('1-June-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% Test in advance payments

df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M', 
                          roll_convention='following', 
                          payment_type='in_advance', 
                          payment_delay=1)

arr = np.array([[pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('3-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 2-March-2020
                [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('3-Mar-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('1-Apr-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('1-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

#%% Test Stubs

# The default stub setting is a 'first short' stub
df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('31-Jan-2020'),pd.Timestamp('31-Jan-2020')], 
                [pd.Timestamp('31-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

# The default stub setting is a 'first short' stub (same result as above)
df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M', 
                          stub='first_short')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('31-Jan-2020'),pd.Timestamp('31-Jan-2020')],
                [pd.Timestamp('31-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M', 
                          stub='first_long')

arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


#%% Test fixing dates

df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                          end_date=pd.Timestamp('31-May-2020'),
                          payment_freq='M', 
                          stub='first_long',
                          add_fixing_dates=True) # default of 2 fixing delay added

arr = np.array([[pd.Timestamp('13-Jan-2020'),pd.Timestamp('26-Feb-2020'),pd.Timestamp('15-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                [pd.Timestamp('26-Feb-2020'),pd.Timestamp('27-Mar-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                [pd.Timestamp('27-Mar-2020'),pd.Timestamp('28-Apr-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                [pd.Timestamp('28-Apr-2020'),pd.Timestamp('29-May-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
assert (df_schedule == pd.DataFrame(arr,columns=['fixing_period_start','fixing_period_end','period_start','period_end','payment_date'])).all().all()











