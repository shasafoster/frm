# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import os
    import pathlib
    os.chdir(pathlib.Path(__file__).parent.parent.parent.parent.resolve())     
    print('__main__ - current working directory:', os.getcwd())

#from frm.frm.market_data.iban_ccys import VALID_CCYS

import os
import holidays
import pandas_market_calendars as mcal
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import time
import pickle
from typing import Literal


def log(log_idx, t1, msg):
    t2 = time.time()
    if __name__ == "__main__":
        print(log_idx, msg, round(t2-t1,2))
    return log_idx + 1, t2


# Define your currency to country/market mapping
ccy_country_mapping = {
    
    # Use market calendars from 'pandas_market_calendars' in 1st instance
    'USD': 'NYSE',  # United States Dollar - New York Stock Exchange
    'EUR': 'EUREX',  # Euro - EUREX
    'JPY': 'JPX',  # Japanese Yen - Tokyo Stock Exchange
    'GBP': 'LSE',  # British Pound - London Stock Exchange
#    'AUD': 'ASX',  # Australian Dollar - Australian Securities Exchange
    'CAD': 'TSX',  # Canadian Dollar - Toronto Stock Exchange
    'CHF': 'SIX',  # Swiss Franc - SIX Swiss Exchange
    'NZD': 'XNZE',  # New Zealand Dollar - New Zealand Stock Exchange
    'BRL': 'BVMF',  # Brazilian Real - B3 - Brasil Bolsa Balcão
    'INR': 'NSE',  # Indian Rupee - National Stock Exchange of India
    'HKD': 'HKEX',  # Hong Kong Dollar - HKEX
    'NOK': 'OSE',  # Norwegian Krone - Olso Stock Exchange
    
    # Otherwise use country holidays from 'holidays'
    'AUD': ('Australia','NSW'),
    'MXN': 'Mexico', # Mexican Peso - Mexican Stock Exchange
    'SGD': 'Singapore',  # Singapore Dollar
    'KRW': 'SouthKorea',  # South Korean Won
    'TRY': 'Turkey',  # Turkish Lira
    # 'CNY': 'China',  # Chinese Yuan
    'SEK': 'Sweden',  # Swedish Krona    
    'RUB': 'Russia',  # Russian Ruble    
    'ZAR': 'SouthAfrica',  # South African Rand
    'DKK': 'Denmark',  # Danish Krone
    'PLN': 'Poland',  # Polish Złoty
    'THB': 'Thailand',  # Thai Baht
    'IDR': 'Indonesia',  # Indonesian Rupiah
    'HUF': 'Hungary',  # Hungarian Forint
    'CZK': 'Czech',  # Czech Koruna
    'ILS': 'Israel',  # Israeli Shekel
    'CLP': 'Chile',  # Chilean Peso
    'PHP': 'Philippines',  # Philippine Peso
    'AED': 'UnitedArabEmirates',  # UAE Dirham
    'COP': 'Colombia',  # Colombian Peso
    'SAR': 'SaudiArabia',  # Saudi Riyal
    'MYR': 'Malaysia',  # Malaysian Ringgit
    'RON': 'Romania',  # Romanian Leu
    'ARS': 'Argentina',  # Argentine Peso
    # Add more currencies here...
}

def get_holidays(ccy,
                 start_date=None, 
                 end_date=None):

    log_idx, t = log(1, time.time(), 'start get_holidays()')
      
    ccy = ccy.upper()
    if start_date == None:
        start_date = dt.today().date() + relativedelta(years=-10)
    if end_date == None:
        end_date = dt.today().date() + relativedelta(years=40)
    
    try:
        with open('./frm/frm/schedule/ccy_holidays_dict.pkl', 'rb') as f:
            ccy_holidays_dict = pickle.load(f)  
            
            if ccy.upper() in ccy_holidays_dict.keys():
                log_idx, t = log(log_idx, t, 'read in pickled holidays')
                return ccy_holidays_dict[ccy]  
            else:
                print(ccy, ' is not in ccy_holidays_dict.pkl')
            
    except FileNotFoundError:

        if ccy.upper() in ccy_country_mapping.keys():
            holiday_index = ccy_country_mapping[ccy.upper()]
            
            if holiday_index in mcal.get_calendar_names():
                # Get calendar for the market
                cal = mcal.get_calendar(holiday_index)
                # Get market schedule
                market_schedule = cal.schedule(start_date=start_date, end_date=end_date)
                valid_days_np  = np.array(market_schedule.index.values.astype('datetime64[D]'), dtype='datetime64[D]')
                all_days = np.arange(valid_days_np[0], valid_days_np[-1] + np.timedelta64(1, 'D'), dtype='datetime64[D]')
                # the holidays are the days that are in all_days but not in valid_days
                holidays_np = np.setdiff1d(all_days, valid_days_np)
                log_idx, t = log(log_idx, t, 'get holidays for ' + ccy + ' from pandas-market-calendars')
                return holidays_np 
            else:
                # If the market is not in the pandas_market_calendars package, get the country holidays
                
                if isinstance(holiday_index, tuple):
                    country, prov = holiday_index[0], holiday_index[1]
                    
                elif isinstance(holiday_index, str):
                    country = holiday_index
                    prov = None
                
                try:
                    country_holidays = holidays.CountryHoliday(country=country, prov=prov, years=list(range(start_date.year, end_date.year+1)))
                except:
                    country_holidays = holidays.CountryHoliday(country=country, prov=prov)
                
                # Convert to numpy datetime64[D] format and return busdaycalendar
                holidays_np = np.array(list(country_holidays.keys()))
                log_idx, t = log(log_idx, t, 'get holidays for ' + ccy + ' from holidays.CountryHoliday()')
                return holidays_np
        else:
            return None


# https://legislation.nsw.gov.au/view/html/inforce/current/act-2008-049#pt.3A
# the first Monday in August (Bank Holiday).
extra_financial_holidays = {
    
    
    }




def get_calendar(ccys,
                 cities=None,
                 start_date=None,
                 end_date=None) -> np.busdaycalendar:
    """
    Create a calendar which has the holidays and business of the currency inputs.

    Parameters
    ----------
    ccys : array of strings, optional
        DESCRIPTION. Array of three letter currency codes, The default is None.

    Returns
    -------
    CustomBusinessDay() object, observing the holiday of the cities
    
    """
    log_idx, t = log(1, time.time(), 'start get_calendar()')
        
    if ccys is None:
        return np.busdaycalendar()
    
    if start_date == None:
        start_date = dt.datetime(2000, 1, 1)
    if end_date == None:
        end_date = dt.datetime(2100, 1, 1)    
    
    if type(ccys) is str: 
        ccys = [ccys]
    elif type(ccys) is list:
        ccys = list(set(ccys))
        ccys = [ccy.upper() for ccy in ccys]
            
    weekmask = getWeekMask(ccys)

    holidays = []
    if ccys is not None:
        holidays += [get_holidays(ccy, start_date, end_date) for ccy in ccys]
    log_idx, t = log(log_idx, t, 'get holidays') 
   
    # Flattan the list of lists
    holidays = [h for holiday_list in holidays for h in holiday_list]  
    log_idx, t = log(log_idx, t, 'flatten holidays') 
    
    if holidays == []:
        return np.busdaycalendar(weekmask=weekmask)
    else:
        return np.busdaycalendar(weekmask=weekmask, holidays=holidays) #.values.astype('datetime64[D]'))
        
    
def getWeekMask(ccys):
    """
    Return the business days of the countries associated with the provided ccys.
    Muslim/Jewish states have have Friday & Saturday as the weekend.
    
    Parameters
    ----------
    ccys : numpy array of strings
        array of three letter currency codes 
        
    Returns
    -------
    The business days consistent across all ccys 
    
    """
    
    if ccys is None:
        return 'Mon Tue Wed Thu Fri'
    else:    
        fri_sat_weekend = set(['AFN',  #Afghanistan
                               'BHD',  #Bahrain
                               'BDT',  #Bangladesh
                               'DZD',  #Algeria
                               'EGP',  #Egypt
                               'ILS',  #Israel
                               'IQD',  #Iraq
                               'JOD',  #Jordan
                               'KWD',  #Kuwait
                               'LYD',  #Libya
                               'MYR',  #Malaysia (some states)
                               'MVR',  #Maldives
                               'OMR',  #Oman
                               'ILS',  #Palestine (depending on the area)
                               'QAR',  #Qatar
                               'SAR',  #Saudi Arabia
                               'SDG',  #Sudan
                               'SYP',  #Syria
                               'YER',  #Yemen
                               ])
    
        bool_arr = [ccy.upper() in fri_sat_weekend for ccy in ccys]
        
        if True in bool_arr and False in bool_arr:
            return 'Mon Tue Wed Thu'
        elif False in bool_arr:
            return 'Mon Tue Wed Thu Fri'
        else:
            return 'Sun Mon Tue Wed Thu'


#%%

if __name__ == "__main__":
    print('main')
    log_idx = 1
    t = time.time()
    
    #cal = get_calendar(['USD','AUD','JPY','NZD','EUR'])
    
    for ccy in ['USD','AUD','JPY','NZD','EUR']:
        h = get_holidays(ccy)
    
    

#%% Pickle all possible calls of ccy_country_mapping 

# ccy_holidays_dict = {}
# for ccy in ccy_country_mapping.keys():
#     try:    
#         ccy_holidays_dict[ccy] = get_holidays(ccy)
#     except:
#         print(ccy)

# with open('ccy_holidays_dict.pkl', 'wb') as f:
#     pickle.dump(ccy_holidays_dict, f)



