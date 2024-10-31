# -*- coding: utf-8 -*-
import os
from frm.utils import get_schedule, generate_date_schedule, PeriodFrequency, StubType, RollConvention, DayRoll, TimingConvention
import pandas as pd
import pytest    
os.chdir(os.environ.get('PROJECT_DIR_FRM'))

def test_schedule():
    # Step 1: Read in the test cases defined in the excel spreadsheet
    fp = './tests/utils/payment_schedule_test_definitions.xlsx'

    df = pd.read_excel(io=fp, sheet_name='test_cases')
    df_test_description = df[['test_#','test_bucket','description']]
    function_parameters = ['start_date', 'end_date', 'frequency', 'roll_convention', 'day_roll',
                           'first_cpn_end_date', 'last_cpn_start_date', 'first_stub_type', 'last_stub_type',
                           'roll_user_specified_dates', 'add_payment_dates', 'payment_timing', 'payment_delay']

    df_input = df[['test_#'] + function_parameters]
    df_input = df_input[~df_input.drop(columns=['test_#']).isna().all(axis=1)]
    df_correct_results = pd.read_excel(io=fp, sheet_name='correct_test_results')
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
        function_parameters = {k:v for k,v in function_parameters.items() if not pd.isna(v)}

        if 'frequency' in function_parameters.keys():
            function_parameters['frequency'] = PeriodFrequency.from_value(function_parameters['frequency'])
        if 'roll_convention' in function_parameters.keys():
            function_parameters['roll_convention'] = RollConvention.from_value(function_parameters['roll_convention'])
        if 'first_stub_type' in function_parameters.keys():
            function_parameters['first_stub_type'] = StubType.from_value(function_parameters['first_stub_type'])
        if 'last_stub_type' in function_parameters.keys():
            function_parameters['last_stub_type'] = StubType.from_value(function_parameters['last_stub_type'])
        if 'day_roll' in function_parameters.keys():
            function_parameters['day_roll'] = DayRoll.from_value(function_parameters['day_roll'])
        if 'payment_timing' in function_parameters.keys():
            function_parameters['payment_timing'] = TimingConvention.from_value(function_parameters['payment_timing'])

        df_schedule = get_schedule(**function_parameters)

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
    frequency = PeriodFrequency.from_value('monthly')
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
    frequency = PeriodFrequency.from_value('monthly')
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
    frequency = PeriodFrequency.from_value('annual')
    direction = 'forward'

    expected_start_dates = pd.Series([pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-03')])
    expected_end_dates = pd.Series([pd.Timestamp('2021-01-01'), pd.Timestamp('2022-01-03'), pd.Timestamp('2023-01-01')])

    start_dates, end_dates = generate_date_schedule(start_date, end_date, frequency, direction)
    
    assert (pd.Series(start_dates) == expected_start_dates).all()
    assert (pd.Series(end_dates) == expected_end_dates).all()


def test_generate_date_schedule_forward_days():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-01-05')
    freq = PeriodFrequency.from_value('daily')
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
    frequency = PeriodFrequency.from_value('daily')
    direction = 'forward'
    
    with pytest.raises(ValueError, match="'start_date' must be earlier than 'end_date'"):
        generate_date_schedule(start_date, end_date, frequency, direction)


def test_generate_date_schedule_invalid_direction():
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    frequency = PeriodFrequency.from_value('monthly')
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
    test_schedule()

    