# -*- coding: utf-8 -*-
import os
if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from frm.schedule.schedule import schedule, generate_date_schedule, Frequency
import pandas as pd
import pytest

def test_schedule():
    # Step 1: Read in the test cases defined in the excel spreadsheet 
    df = pd.read_excel(current_dir + "\\payment_schedule_test_definitions.xlsx", sheet_name='test_cases')
    df_test_description = df[['test_#','test_bucket','description']]
    function_parameters = ['start_date', 'end_date', 'frequency', 'roll_convention', 'day_roll', 'first_cpn_end_date', 'last_cpn_start_date', 'first_stub_type', 'last_stub_type', 'roll_user_specified_dates']
    df_input = df[['test_#'] + function_parameters]
    df_input = df_input[~df_input.drop(columns=['test_#']).isna().all(axis=1)]
    
    df_correct_results = pd.read_excel(current_dir + "\\payment_schedule_test_definitions.xlsx", sheet_name='correct_test_results')
    df_correct_results = df_correct_results[['test_#','period_start','period_end']]
    
    # Step 2: Build test case dictionaries from dataframe
    test_cases = []
    for test_num in df_test_description['test_#'].unique(): 
        test_descriptions = df_test_description[df_test_description['test_#'] == test_num].iloc[0].to_dict()
        function_parameters = df_input[df_input['test_#'] == test_num].iloc[0].to_dict()
        correct_results = df_correct_results[df_correct_results['test_#'] == test_num].reset_index(drop=True)
        
        test_cases.append({
            'test_descriptions': test_descriptions,
            'function_parameters': function_parameters,
            'correct_results': correct_results
        })
    
    for case in test_cases:
        # Call the function with the test case inputs
        function_parameters = case['function_parameters']
        function_parameters.pop('test_#')
        df_schedule = schedule(**function_parameters)
    
        correct_result = case['correct_results'].drop('test_#', axis=1)
    
        # Compare the result with the expected DataFrame
        try:
            pd.testing.assert_frame_equal(df_schedule, correct_result)
            print('Test case', case['test_descriptions']['test_#'], 'passed:', case['test_descriptions']['description'])
        except AssertionError as e:
            print(f"Test failed for inputs: {case['function_parameters']}")
            raise e
        
        
def test_generate_date_schedule_forward_months():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-06-30')
    frequency = Frequency.from_value('m')
    direction = 'forward'

    expected_start_dates = pd.Series([
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'),
        pd.Timestamp('2023-04-03'), pd.Timestamp('2023-05-01'), pd.Timestamp('2023-06-01')
    ])
    expected_end_dates = pd.Series([
        pd.Timestamp('2023-02-01'), pd.Timestamp('2023-03-01'), pd.Timestamp('2023-04-03'),
        pd.Timestamp('2023-05-01'), pd.Timestamp('2023-06-01'), pd.Timestamp('2023-06-30')
    ])

    start_dates, end_dates = generate_date_schedule(start_date, end_date, frequency, direction)
    
    assert (pd.Series(start_dates) == expected_start_dates).all()
    assert (pd.Series(end_dates) == expected_end_dates).all()


def test_generate_date_schedule_backward_months():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-06-30')
    frequency = Frequency.from_value('m')
    direction = 'backward'

    expected_start_dates = pd.Series([
        pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-30'), pd.Timestamp('2023-02-28'), 
        pd.Timestamp('2023-03-30'), pd.Timestamp('2023-04-28'), pd.Timestamp('2023-05-30')
    ])
    expected_end_dates = pd.Series([
        pd.Timestamp('2023-01-30'), pd.Timestamp('2023-02-28'), pd.Timestamp('2023-03-30'),
        pd.Timestamp('2023-04-28'), pd.Timestamp('2023-05-30'), pd.Timestamp('2023-06-30')
    ])

    start_dates, end_dates = generate_date_schedule(start_date, end_date, frequency, direction)
    
    assert (pd.Series(start_dates) == expected_start_dates).all()
    assert (pd.Series(end_dates) == expected_end_dates).all()
    

def test_generate_date_schedule_forward_years():
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2023-01-01')
    frequency = Frequency.from_value('a')
    direction = 'forward'

    expected_start_dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-03')])
    expected_end_dates = pd.Series([pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-03'), pd.Timestamp('2023-01-01')])

    start_dates, end_dates = generate_date_schedule(start_date, end_date, frequency, direction)
    
    assert (pd.Series(start_dates) == expected_start_dates).all()
    assert (pd.Series(end_dates) == expected_end_dates).all()


def test_generate_date_schedule_forward_days():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-05')
    freq = Frequency.from_value('d')
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
    frequency = Frequency.from_value('d')
    direction = 'forward'
    
    with pytest.raises(ValueError, match="'start_date' must be earlier than 'end_date'"):
        generate_date_schedule(start_date, end_date, frequency, direction)


def test_generate_date_schedule_invalid_direction():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    frequency = Frequency.from_value('m')
    direction = 'sideways'  # invalid direction
    
    with pytest.raises(ValueError, match="Invalid direction 'sideways'. Must be 'forward' or 'backward'."):
        generate_date_schedule(start_date, end_date, frequency, direction)
     
        
if __name__ == "__main__":    
    # Test generate_date_schedule()
    test_generate_date_schedule_forward_months()
    test_generate_date_schedule_backward_months()
    test_generate_date_schedule_forward_years()
    test_generate_date_schedule_forward_days()
    test_generate_date_schedule_same_start_and_end_date()
    test_generate_date_schedule_invalid_direction()
    
    # Test schedule()

    