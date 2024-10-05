# -*- coding: utf-8 -*-
import os

if __name__ == "__main__":
    os.chdir(os.environ.get('PROJECT_DIR_FRM'))


import re


def test_regex_patterns():

    # Define the regex pattern
    call_put_pattern = r'^(0?[1-9]|[1-4]\d)[_ ]?(delta|Δ)[_ ]?(call|c|put|p)$'
    # Valid patterns: '5_delta_call', '05_delta_call', '5 delta call', '5deltacall', '5_Δ_call', '5 Δ call', '5Δcall', '5Δc'

    # Test cases: (string, expected result)
    test_cases = [
        # Pass
        ('5_delta_call', True),
        ('05_delta_call', True),
        ('5 delta call', True),
        ('5deltacall', True),
        ('5_Δ_call', True),
        ('5 Δ call', True),
        ('5Δcall', True),
        ('5Δc', True),
        ('1_delta_call', True),
        ('09_delta_put', True),
        ('10_delta_call', True),
        ('25_delta_put', True),
        ('49_delta_call', True),
        # Fail
        ('0_delta_call', False),
        ('50_delta_put', False),
        ('99_delta_put', False),
        ('100_delta_call', False),
        ('01_delta_put_extra', False),
        ('delta_call', False),
        ('delta_25_call', False)
    ]

    # Run the test cases
    for test_str, expected in test_cases:
        match = bool(re.match(call_put_pattern, test_str))
        result = 'Pass' if match == expected else 'Fail'
        print(f'Test: {test_str} | Expected: {expected} | Match: {match} | Result: {result}')
        assert result == 'Pass'


    # Define the regex pattern
    atm_delta_neutral_column_pattern = r'^atm[_ ]?(delta|Δ)[_ ]?neutral$'

    # Test cases: (string, expected result)
    test_cases = [
        ('atm_delta_neutral', True),
        ('atm_Δ_neutral', True),
        ('atmΔneutral', True),
        ('atm_Δneutral', True),
        ('atmneutral', False),  # Should not match, "delta" or "Δ" is mandatory
        ('atm delta neutral', True),
        ('atm delta xneutral', False),  # Invalid
        ('atm_neutral', False),  # "delta" or "Δ" missing
        ('atm_Δ_neut', False),  # Invalid
        ('delta_neutral', False)  # Missing "atm"
    ]

    # Run the test cases
    for test_str, expected in test_cases:
        match = bool(re.match(atm_delta_neutral_column_pattern, test_str))
        result = 'Pass' if match == expected else 'Fail'
        print(f'Test: {test_str} | Expected: {expected} | Match: {match} | Result: {result}')
        assert result == 'Pass'

test_regex_patterns()
