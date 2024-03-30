# -*- coding: utf-8 -*-


USD_LIBOR_SWAP_1Y = {
    # General key terms
    'notional_currency': 'USD',
    'city_holidays': 'new_york',
     # Pay fixed leg key terms
    'pay_payment_frequency':'A',
    'pay_day_count_basis':'ACT/360',
     # Receive float leg key terms
    'rec_payment_frequency':'Q',
    'rec_day_count_basis':'ACT/360',
    'rec_float_spread':0.0}

USD_LIBOR_SWAP_ABOVE_1Y = {
    # General key terms
    'notional_currency': 'USD',
    'city_holidays': 'new_york',
     # Pay fixed leg key terms
    'pay_payment_frequency':'S',
    'pay_day_count_basis':'30/360',
     # Receive float leg key terms
    'rec_payment_frequency':'Q',
    'rec_day_count_basis':'ACT/360',
    'rec_float_spread':0.0}

USD_SOFR = {
    # General key terms
    'notional_currency': 'USD',
    'city_holidays': 'new_york',
     # Pay fixed leg key terms
    'pay_payment_frequency':'A',
    'pay_day_count_basis':'ACT/360',
     # Receive float leg key terms
    'rec_payment_frequency':'A',
    'rec_day_count_basis':'ACT/365',
    'rec_float_spread':0.0}

JPY_LIBOR_SWAP = {
    # General key terms
    'notional_currency': 'JPY',
    'city_holidays': 'tokyo',
     # Pay fixed leg key terms
    'pay_payment_frequency':'S',
    'pay_day_count_basis':'ACT/360',
     # Receive float leg key terms
    'rec_payment_frequency':'S',
    'rec_day_count_basis':'ACT/36S',
    'rec_float_spread':0.0}

NZD_BKBM_SWAP = {
    # General key terms
    'notional_currency': 'NZD',
    'city_holidays': 'wellington',
     # Pay fixed leg key terms
    'pay_payment_frequency':'S',
    'pay_day_count_basis':'ACT/365',
     # Receive float leg key terms
    'rec_payment_frequency':'Q',
    'rec_day_count_basis':'ACT/365',
    'rec_float_spread':0.0}

AUD_BBSW_SWAP_3Y_AND_UNDER = {
    # General key terms
    'notional_currency': 'AUD',
    'city_holidays': 'sydney',
     # Pay fixed leg key terms
    'pay_payment_frequency':'Q',
    'pay_day_count_basis':'ACT/365',
     # Receive float leg key terms
    'rec_payment_frequency':'Q',
    'rec_day_count_basis':'ACT/365',
    'rec_float_spread':0.0}

AUD_BBSW_SWAP_ABOVE_3Y = {
    # General key terms
    'notional_currency': 'AUD',
    'city_holidays': 'sydney',
     # Pay fixed leg key terms
    'pay_payment_frequency':'S',
    'pay_day_count_basis':'ACT/365',
     # Receive float leg key terms
    'rec_payment_frequency':'',
    'rec_day_count_basis':'ACT/365',
    'rec_float_spread':0.0}