# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))

from frm.schedule.schedule import payment_schedule, generate_date_schedule, Frequency
import numpy as np
import pandas as pd
import pytest

def test_generate_date_schedule_forward_months():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-06-30')
    freq = Frequency.from_code('m')
    direction = 'forward'

    expected_start_dates = [
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'),
        pd.Timestamp('2023-04-01'), pd.Timestamp('2023-05-01'), pd.Timestamp('2023-06-01')
    ]
    expected_end_dates = [
        pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-01'),
        pd.Timestamp('2023-05-01'), pd.Timestamp('2023-06-01'), pd.Timestamp('2023-06-30')
    ]

    start_dates, end_dates = generate_date_schedule(start_date, end_date, freq, direction)
    
    assert start_dates == expected_start_dates
    assert end_dates == expected_end_dates


def test_generate_date_schedule_backward_months():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-06-30')
    freq = Frequency.from_code('m')
    direction = 'backward'

    expected_start_dates = [
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-30'), pd.Timestamp('2023-02-28'), 
        pd.Timestamp('2023-03-30'), pd.Timestamp('2023-04-30'), pd.Timestamp('2023-05-30')
    ]
    expected_end_dates = [
        pd.Timestamp('2023-01-30'), pd.Timestamp('2023-02-28'), pd.Timestamp('2023-03-30'),
        pd.Timestamp('2023-04-30'), pd.Timestamp('2023-05-30'), pd.Timestamp('2023-06-30')
    ]

    start_dates, end_dates = generate_date_schedule(start_date, end_date, freq, direction)
    
    assert start_dates == expected_start_dates
    assert end_dates == expected_end_dates
    

def test_generate_date_schedule_forward_years():
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-01-01')
    freq = Frequency.from_code('a')
    direction = 'forward'

    expected_start_dates = [pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-01')]
    expected_end_dates = [pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-01'), pd.Timestamp('2023-01-01')]

    start_dates, end_dates = generate_date_schedule(start_date, end_date, freq, direction)
    
    assert start_dates == expected_start_dates
    assert end_dates == expected_end_dates


def test_generate_date_schedule_forward_days():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-05')
    freq = Frequency.from_code('d')
    direction = 'forward'

    expected_start_dates = [
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03'),
        pd.Timestamp('2023-01-04')
    ]
    expected_end_dates = [
        pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03'), pd.Timestamp('2023-01-04'),
        pd.Timestamp('2023-01-05')
    ]

    start_dates, end_dates = generate_date_schedule(start_date, end_date, freq, direction)

    assert start_dates == expected_start_dates
    assert end_dates == expected_end_dates

def test_generate_date_schedule_same_start_and_end_date():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-01')
    freq = Frequency.from_code('d')
    direction = 'forward'
    
    with pytest.raises(ValueError, match="'start_date' must be earlier than 'end_date'"):
        generate_date_schedule(start_date, end_date, freq, direction)


def test_generate_date_schedule_invalid_direction():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    freq = Frequency.from_code('m')
    direction = 'sideways'  # invalid direction
    
    with pytest.raises(ValueError, match="Invalid direction 'sideways'. Must be 'forward' or 'backward'."):
        generate_date_schedule(start_date, end_date, freq, direction)
     
        
# def test_payment_schedule_no_date_rolling():

#     # 1 period schedule, quarterly interval
#     df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
#                                    end_date=pd.Timestamp('15-Apr-2020'),
#                                    payment_freq='q')
#     arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')]])
#     assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
#     # 2 period schedule, quarterly interval
#     df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
#                                    end_date=pd.Timestamp('15-Jul-2020'),
#                                   payment_freq='q')
#     arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')],
#                     [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Jul-2020')]])
#     assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
#     # 4 period schedule, quarterly interval
#     df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
#                                    end_date=pd.Timestamp('15-Jan-2021'),
#                                    payment_freq='q')
#     arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')],
#                     [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Jul-2020')],
#                     [pd.Timestamp('15-Jul-2020'),pd.Timestamp('15-Oct-2020'),pd.Timestamp('15-Oct-2020')],
#                     [pd.Timestamp('15-Oct-2020'),pd.Timestamp('15-Jan-2021'),pd.Timestamp('15-Jan-2021')]])
#     assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
#     # 1 period schedule, annual interval
#     df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
#                               end_date=pd.Timestamp('15-Jan-2021'),
#                               payment_freq='a')    
#     arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('15-Jan-2021'),pd.Timestamp('15-Jan-2021')]])
#     assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()



def test_payment_schedule_date_rolling():

    # # Dates falling on the weekend roll
    # df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
    #                           end_date=pd.Timestamp('15-May-2020'),
    #                           payment_freq='M')
    # arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('17-Feb-2020'),pd.Timestamp('17-Feb-2020')], # 15-Feb-2020 falls on Saturday, roll to 17th
    #                 [pd.Timestamp('17-Feb-2020'),pd.Timestamp('16-Mar-2020'),pd.Timestamp('16-Mar-2020')], # 15-Mar-2020 falls on Sunday, roll to 16th
    #                 [pd.Timestamp('16-Mar-2020'),pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-Apr-2020')], 
    #                 [pd.Timestamp('15-Apr-2020'),pd.Timestamp('15-May-2020'),pd.Timestamp('15-May-2020')]])
    # assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


    # Input dates don't roll
    df_schedule = payment_schedule(start_date=pd.Timestamp('30-Sep-2020'), 
                              end_date=pd.Timestamp('31-Oct-2020'),
                              payment_freq='M')
    # 31-Oct-2020 is Saturday, don't roll period_end, only payment_date as input dates are not rolled.
    arr = np.array([[pd.Timestamp('30-Sep-2020'),pd.Timestamp('31-Oct-2020'),pd.Timestamp('30-Oct-2020')]]) 
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()

    # With roll_convention='following', dates roll to next month if fall on weekend
    df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='M',
                              roll_convention='following')
    arr = np.array(
         # 29-Feb-2020 falls on Saturday, roll to 2-March-2020
        [[pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('2-Mar-2020')], 
         [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
         [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
         # 31-May-2020 falls on a Sunday, roll payment_date to Monday 1-June-2020. Don't roll final period_end date as input dates are not rolled
         [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('1-Jun-2020')]]
    ) 
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
    # Test the payment delay 
    df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='M', 
                              roll_convention='following',
                              payment_delay=1)
    arr = np.array([
        [pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('3-Mar-2020')], # 29-Feb-2020 falls on Saturday, roll to 2-March-2020
        [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('1-Apr-2020')], 
        [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('1-May-2020')], 
        # 31-May-2020 falls on a Sunday, roll payment_date to Tuesday 2-June-2020.  Don't roll final period_end date as input dates are not rolled
        [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('2-Jun-2020')]
    ]) 
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
    # Test in advance payments
    df_schedule = payment_schedule(start_date=pd.Timestamp('31-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='M', 
                              roll_convention='following', 
                              payment_type='in_advance', 
                              payment_delay=1)
    arr = np.array([
        # 31-Jan-2020 is a Friday, 1 business day payment_delay rolls to Monday 3-Feb-2020
        [pd.Timestamp('31-Jan-2020'),pd.Timestamp('2-Mar-2020'),pd.Timestamp('3-Feb-2020')], 
        [pd.Timestamp('2-Mar-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('3-Mar-2020')], 
        [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('1-Apr-2020')], 
        [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('1-May-2020')]
    ]) 
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


if __name__ == "__main__":    
    # Test generate_date_schedule()
    test_generate_date_schedule_forward_months()
    test_generate_date_schedule_backward_months()
    test_generate_date_schedule_forward_years()
    test_generate_date_schedule_forward_days()
    test_generate_date_schedule_same_start_and_end_date()
    test_generate_date_schedule_invalid_direction()
    # test_generate_date_schedule_invalid_period_type()
    # test_generate_date_schedule_invalid_nb_periods()
    # test_payment_schedule_no_date_rolling()
    test_payment_schedule_date_rolling()
    

def test_payment_schedule_stubs():
    # The default stub setting is a 'first short' stub
    df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='m')
    arr = np.array(
        [[pd.Timestamp('15-Jan-2020'),pd.Timestamp('31-Jan-2020'),pd.Timestamp('31-Jan-2020')], 
         # EoM (29-Feb-2020) falls on Saturday, so roll to Friday 28-Feb-2020 due to modified following
         [pd.Timestamp('31-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], 
         [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
         [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
         [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('29-May-2020')]
    ]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


    # The default stub setting is a 'first short' stub (same result as above)
    df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='m', 
                              first_stub_type='short')
    
    arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('31-Jan-2020'),pd.Timestamp('31-Jan-2020')],
                    [pd.Timestamp('31-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                    [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                    [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                    [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()
    
    df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
                              end_date=pd.Timestamp('31-May-2020'),
                              payment_freq='M', 
                              first_stub_type='long')
    
    arr = np.array([[pd.Timestamp('15-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
                    [pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
                    [pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
                    [pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
    assert (df_schedule == pd.DataFrame(arr,columns=['period_start','period_end','payment_date'])).all().all()


# #%% Test fixing dates

# df_schedule = payment_schedule(start_date=pd.Timestamp('15-Jan-2020'), 
#                           end_date=pd.Timestamp('31-May-2020'),
#                           payment_freq='M', 
#                           first_stub_type='long',
#                           add_fixing_dates=True) # default of 2 fixing delay added

# arr = np.array([[pd.Timestamp('13-Jan-2020'),pd.Timestamp('26-Feb-2020'),pd.Timestamp('15-Jan-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('28-Feb-2020')], # 29-Feb-2020 falls on Saturday, roll to 28-Feb-2020
#                 [pd.Timestamp('26-Feb-2020'),pd.Timestamp('27-Mar-2020'),pd.Timestamp('28-Feb-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('31-Mar-2020')], 
#                 [pd.Timestamp('27-Mar-2020'),pd.Timestamp('28-Apr-2020'),pd.Timestamp('31-Mar-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('30-Apr-2020')], 
#                 [pd.Timestamp('28-Apr-2020'),pd.Timestamp('29-May-2020'),pd.Timestamp('30-Apr-2020'),pd.Timestamp('31-May-2020'),pd.Timestamp('31-May-2020')]]) # 31-May-2020 falls on a Saturday, don't roll as input dates are not rolled
# assert (df_schedule == pd.DataFrame(arr,columns=['fixing_period_start','fixing_period_end','period_start','period_end','payment_date'])).all().all()


#%%

import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from frm.schedule.schedule import payment_schedule, generate_date_schedule
import numpy as np
import pandas as pd

# Step 1: Read in the test cases defined in the excel spreadsheet 
df = pd.read_excel(current_dir + "\\payment_schedule_test_definitions.xlsx")
df_test_description = df[['test #','test group','description']]
function_parameters = ['start_date', 'end_date', 'payment_freq', 'roll_convention', 'day_roll', 'first_cpn_end_date', 'last_cpn_start_date', 'first_stub_type', 'last_stub_type', 'payment_type', 'payment_delay', 'roll_user_specified_dates']
df_input = df[['test #'] + function_parameters]
df_input = df_input[~df_input.drop(columns=['test #']).isna().all(axis=1)]
df_output = df[['test #','period_start','period_end','payment_date']]

# Step 2: Build test case dictionaries from dataframe
test_cases = []
for test_num in df_test_description['test #'].unique():
    
    test_description = df_test_description[df_test_description['test #'] == test_num].iloc[0].to_dict()
    function_parameters = df_input[df_input['test #'] == test_num].iloc[0].to_dict()
    expected_output = df_output[df_output['test #'] == test_num].reset_index(drop=True)
    
    test_cases.append({
        'test_description': test_description,
        'function_parameters': function_parameters,
        'expected_output': expected_output
    })

for case in test_cases:
    # Call the function with the test case inputs
    function_parameters = case['function_parameters']
    function_parameters.pop('test #')
    df_schedule = payment_schedule(**function_parameters)

    expected_output = case['expected_output'].drop('test #', axis=1)

    # Compare the result with the expected DataFrame
    try:
        pd.testing.assert_frame_equal(df_schedule, expected_output)
        print()
    except AssertionError as e:
        print(f"Test failed for inputs: {case['inputs']}")
        raise e