import pytest
import importlib

# Auto-generated import tests

@pytest.mark.parametrize('module_path', [
    'frm.__init__',
    'frm.enums.__init__',
    'frm.enums.helper',
    'frm.enums.term_structures',
    'frm.enums.utils',
    'frm.instruments.__init__',
    'frm.instruments.leg',
    'frm.instruments.swap',
    'frm.pricing_engine.__init__',
    'frm.pricing_engine.black76_bachelier',
    'frm.pricing_engine.clewlow_strickland_1_factor',
    'frm.pricing_engine.cosine_method_generic',
    'frm.pricing_engine.garman_kohlhagen',
    'frm.pricing_engine.geometric_brownian_motion',
    'frm.pricing_engine.heston',
    'frm.pricing_engine.hw1f',
    'frm.pricing_engine.in_progress.gbm_speed_exploration',
    'frm.pricing_engine.in_progress.monte_carlo_multiprocessing',
    'frm.pricing_engine.monte_carlo_generic',
    'frm.pricing_engine.option',
    'frm.pricing_engine.sabr',
    'frm.pricing_engine.volatility_generic',
    'frm.term_structures.__init__',
    'frm.term_structures.archive.fx_volatility_surface_old',
    'frm.term_structures.archive.historical_swap_index_fixings_old',
    'frm.term_structures.archive.ois_fixings',
    'frm.term_structures.archive.swap_curve_old',
    'frm.term_structures.bootstrap_helpers',
    'frm.term_structures.fx_helpers',
    'frm.term_structures.fx_volatility_surface',
    'frm.term_structures.historical_swap_index_fixings',
    'frm.term_structures.iban_ccys',
    'frm.term_structures.interest_rate_option_helpers',
    'frm.term_structures.optionlet',
    'frm.term_structures.swap_curve',
    'frm.term_structures.swaption',
    'frm.term_structures.zero_curve',
    'frm.term_structures.zero_curve_helpers',
    'frm.utils.__init__',
    'frm.utils.business_day_calendar',
    'frm.utils.custom_errors_and_warnings',
    'frm.utils.daycount',
    'frm.utils.interpolation',
    'frm.utils.schedule',
    'frm.utils.schedule_old',
    'frm.utils.settings',
    'frm.utils.tenor',
    'frm.utils.utilities',
])
def test_module_imports(module_path):
    """Test that each module can be imported."""
    importlib.import_module(module_path)

@pytest.mark.parametrize('module_path,name', [
    ('frm.enums.term_structures', 'Enum'),
    ('frm.enums.utils', 'Enum'),
    ('frm.enums.utils', 'clean_enum_value'),
    ('frm.enums.utils', 'get_enum_member'),
    ('frm.enums.utils', 'is_valid_enum_value'),
    ('frm.instruments.leg', 'ABC'),
    ('frm.instruments.leg', 'CompoundingFreq'),
    ('frm.instruments.leg', 'CouponSchedule'),
    ('frm.instruments.leg', 'DayCountBasis'),
    ('frm.instruments.leg', 'MarketDataNotAvailableError'),
    ('frm.instruments.leg', 'Optional'),
    ('frm.instruments.leg', 'PayRcv'),
    ('frm.instruments.leg', 'RFRFixingCalcMethod'),
    ('frm.instruments.leg', 'RFRSwapCurve'),
    ('frm.instruments.leg', 'TermRate'),
    ('frm.instruments.leg', 'TermSwapCurve'),
    ('frm.instruments.leg', 'ZeroCurve'),
    ('frm.instruments.leg', 'abstractmethod'),
    ('frm.instruments.leg', 'dataclass'),
    ('frm.instruments.leg', 'discount_factor_from_zero_rate'),
    ('frm.instruments.leg', 'root_scalar'),
    ('frm.instruments.leg', 'year_frac'),
    ('frm.instruments.swap', 'ABC'),
    ('frm.instruments.swap', 'FixedLeg'),
    ('frm.instruments.swap', 'FloatRFRLeg'),
    ('frm.instruments.swap', 'FloatTermLeg'),
    ('frm.instruments.swap', 'Optional'),
    ('frm.instruments.swap', 'Union'),
    ('frm.instruments.swap', 'ZerocouponLeg'),
    ('frm.instruments.swap', 'dataclass'),
    ('frm.pricing_engine.black76_bachelier', 'minimize'),
    ('frm.pricing_engine.black76_bachelier', 'norm'),
    ('frm.pricing_engine.black76_bachelier', 'root_scalar'),
    ('frm.pricing_engine.clewlow_strickland_1_factor', 'MAX_SIMULATIONS_PER_LOOP'),
    ('frm.pricing_engine.clewlow_strickland_1_factor', 'generate_rand_nbs'),
    ('frm.pricing_engine.clewlow_strickland_1_factor', 'interp1d'),
    ('frm.pricing_engine.cosine_method_generic', 'norm'),
    ('frm.pricing_engine.garman_kohlhagen', 'newton'),
    ('frm.pricing_engine.garman_kohlhagen', 'norm'),
    ('frm.pricing_engine.garman_kohlhagen', 'root_scalar'),
    ('frm.pricing_engine.geometric_brownian_motion', 'generate_rand_nbs'),
    ('frm.pricing_engine.heston', 'Tuple'),
    ('frm.pricing_engine.heston', 'garman_kohlhagen_solve_implied_vol'),
    ('frm.pricing_engine.heston', 'garman_kohlhagen_solve_strike_from_delta'),
    ('frm.pricing_engine.heston', 'get_cos_truncation_range'),
    ('frm.pricing_engine.heston', 'njit'),
    ('frm.pricing_engine.heston', 'norm'),
    ('frm.pricing_engine.heston', 'normal_corr'),
    ('frm.pricing_engine.hw1f', 'CompoundingFreq'),
    ('frm.pricing_engine.hw1f', 'Optional'),
    ('frm.pricing_engine.hw1f', 'PrettyTable'),
    ('frm.pricing_engine.hw1f', 'ZeroCurve'),
    ('frm.pricing_engine.hw1f', 'dataclass'),
    ('frm.pricing_engine.hw1f', 'field'),
    ('frm.pricing_engine.hw1f', 'generate_rand_nbs'),
    ('frm.pricing_engine.hw1f', 'norm'),
    ('frm.pricing_engine.in_progress.gbm_speed_exploration', 'jit'),
    ('frm.pricing_engine.in_progress.gbm_speed_exploration', 'prange'),
    ('frm.pricing_engine.in_progress.monte_carlo_multiprocessing', 'ProcessPoolExecutor'),
    ('frm.pricing_engine.monte_carlo_generic', 'MAX_SIMULATIONS_PER_LOOP'),
    ('frm.pricing_engine.sabr', 'NDArray'),
    ('frm.pricing_engine.sabr', 'Optional'),
    ('frm.term_structures.archive.ois_fixings', 'DayCountBasis'),
    ('frm.term_structures.archive.ois_fixings', 'RFRFixingCalcMethod'),
    ('frm.term_structures.archive.ois_fixings', 'dataclass'),
    ('frm.term_structures.bootstrap_helpers', 'DayCountBasis'),
    ('frm.term_structures.bootstrap_helpers', 'Optional'),
    ('frm.term_structures.bootstrap_helpers', 'clean_tenor'),
    ('frm.term_structures.bootstrap_helpers', 'get_busdaycal'),
    ('frm.term_structures.bootstrap_helpers', 'interp1d'),
    ('frm.term_structures.bootstrap_helpers', 'tenor_to_date_offset'),
    ('frm.term_structures.bootstrap_helpers', 'workday'),
    ('frm.term_structures.bootstrap_helpers', 'year_frac'),
    ('frm.term_structures.fx_helpers', 'Union'),
    ('frm.term_structures.fx_helpers', 'clean_tenor'),
    ('frm.term_structures.fx_helpers', 'convert_column_to_consistent_data_type'),
    ('frm.term_structures.fx_helpers', 'tenor_to_date_offset'),
    ('frm.term_structures.fx_volatility_surface', 'CompoundingFreq'),
    ('frm.term_structures.fx_volatility_surface', 'CubicSpline'),
    ('frm.term_structures.fx_volatility_surface', 'DayCountBasis'),
    ('frm.term_structures.fx_volatility_surface', 'FXSmileInterpolationMethod'),
    ('frm.term_structures.fx_volatility_surface', 'InitVar'),
    ('frm.term_structures.fx_volatility_surface', 'InterpolatedUnivariateSpline'),
    ('frm.term_structures.fx_volatility_surface', 'Optional'),
    ('frm.term_structures.fx_volatility_surface', 'ZeroCurve'),
    ('frm.term_structures.fx_volatility_surface', 'check_delta_convention'),
    ('frm.term_structures.fx_volatility_surface', 'clean_vol_quotes_column_names'),
    ('frm.term_structures.fx_volatility_surface', 'dataclass'),
    ('frm.term_structures.fx_volatility_surface', 'field'),
    ('frm.term_structures.fx_volatility_surface', 'flat_forward_interp'),
    ('frm.term_structures.fx_volatility_surface', 'forward_volatility'),
    ('frm.term_structures.fx_volatility_surface', 'fx_forward_curve_helper'),
    ('frm.term_structures.fx_volatility_surface', 'fx_term_structure_helper'),
    ('frm.term_structures.fx_volatility_surface', 'garman_kohlhagen_price'),
    ('frm.term_structures.fx_volatility_surface', 'garman_kohlhagen_solve_implied_vol'),
    ('frm.term_structures.fx_volatility_surface', 'garman_kohlhagen_solve_strike_from_delta'),
    ('frm.term_structures.fx_volatility_surface', 'generate_rand_nbs'),
    ('frm.term_structures.fx_volatility_surface', 'get_busdaycal'),
    ('frm.term_structures.fx_volatility_surface', 'get_delta_smile_quote_details'),
    ('frm.term_structures.fx_volatility_surface', 'heston_calibrate_vanilla_smile'),
    ('frm.term_structures.fx_volatility_surface', 'heston_price_vanilla_european'),
    ('frm.term_structures.fx_volatility_surface', 'heston_simulate'),
    ('frm.term_structures.fx_volatility_surface', 'interp_fx_forward_curve_df'),
    ('frm.term_structures.fx_volatility_surface', 'resolve_fx_curve_dates'),
    ('frm.term_structures.fx_volatility_surface', 'simulate_gbm_path'),
    ('frm.term_structures.fx_volatility_surface', 'solve_call_put_quotes_from_strategy_quotes'),
    ('frm.term_structures.fx_volatility_surface', 'validate_ccy_pair'),
    ('frm.term_structures.fx_volatility_surface', 'workday'),
    ('frm.term_structures.fx_volatility_surface', 'year_frac'),
    ('frm.term_structures.historical_swap_index_fixings', 'ABC'),
    ('frm.term_structures.historical_swap_index_fixings', 'DayCountBasis'),
    ('frm.term_structures.historical_swap_index_fixings', 'RFRFixingCalcMethod'),
    ('frm.term_structures.historical_swap_index_fixings', 'dataclass'),
    ('frm.term_structures.interest_rate_option_helpers', 'DayCountBasis'),
    ('frm.term_structures.interest_rate_option_helpers', 'Optional'),
    ('frm.term_structures.interest_rate_option_helpers', 'PeriodFreq'),
    ('frm.term_structures.interest_rate_option_helpers', 'TermRate'),
    ('frm.term_structures.interest_rate_option_helpers', 'ZeroCurve'),
    ('frm.term_structures.interest_rate_option_helpers', 'clean_tenor'),
    ('frm.term_structures.interest_rate_option_helpers', 'convert_column_to_consistent_data_type'),
    ('frm.term_structures.interest_rate_option_helpers', 'make_schedule'),
    ('frm.term_structures.interest_rate_option_helpers', 'tenor_to_date_offset'),
    ('frm.term_structures.interest_rate_option_helpers', 'year_frac'),
    ('frm.term_structures.optionlet', 'BaseSchedule'),
    ('frm.term_structures.optionlet', 'DayCountBasis'),
    ('frm.term_structures.optionlet', 'List'),
    ('frm.term_structures.optionlet', 'Optional'),
    ('frm.term_structures.optionlet', 'PeriodFreq'),
    ('frm.term_structures.optionlet', 'TermRate'),
    ('frm.term_structures.optionlet', 'Union'),
    ('frm.term_structures.optionlet', 'VOL_N_BOUNDS'),
    ('frm.term_structures.optionlet', 'ZeroCurve'),
    ('frm.term_structures.optionlet', 'bachelier_price'),
    ('frm.term_structures.optionlet', 'bachelier_solve_implied_vol'),
    ('frm.term_structures.optionlet', 'black76_price'),
    ('frm.term_structures.optionlet', 'black76_sln_to_normal_vol'),
    ('frm.term_structures.optionlet', 'black76_sln_to_normal_vol_analytical'),
    ('frm.term_structures.optionlet', 'black76_solve_implied_vol'),
    ('frm.term_structures.optionlet', 'calc_sln_vol_for_strike_from_sabr_params'),
    ('frm.term_structures.optionlet', 'clean_tenor'),
    ('frm.term_structures.optionlet', 'convert_column_to_consistent_data_type'),
    ('frm.term_structures.optionlet', 'dataclass'),
    ('frm.term_structures.optionlet', 'field'),
    ('frm.term_structures.optionlet', 'make_schedule'),
    ('frm.term_structures.optionlet', 'normal_vol_atm_to_black76_sln_atm'),
    ('frm.term_structures.optionlet', 'normal_vol_to_black76_sln'),
    ('frm.term_structures.optionlet', 'solve_alpha_from_sln_vol'),
    ('frm.term_structures.optionlet', 'standardise_relative_quote_col_names'),
    ('frm.term_structures.optionlet', 'tenor_to_date_offset'),
    ('frm.term_structures.optionlet', 'year_frac'),
    ('frm.term_structures.swap_curve', 'CompoundingFreq'),
    ('frm.term_structures.swap_curve', 'PeriodFreq'),
    ('frm.term_structures.swap_curve', 'RFRFixingCalcMethod'),
    ('frm.term_structures.swap_curve', 'RFRFixings'),
    ('frm.term_structures.swap_curve', 'TermFixings'),
    ('frm.term_structures.swap_curve', 'TermRate'),
    ('frm.term_structures.swap_curve', 'ZeroCurve'),
    ('frm.term_structures.swap_curve', 'dataclass'),
    ('frm.term_structures.swap_curve', 'year_frac'),
    ('frm.term_structures.zero_curve', 'CompoundingFreq'),
    ('frm.term_structures.zero_curve', 'DayCountBasis'),
    ('frm.term_structures.zero_curve', 'InitVar'),
    ('frm.term_structures.zero_curve', 'Optional'),
    ('frm.term_structures.zero_curve', 'RFRFixingCalcMethod'),
    ('frm.term_structures.zero_curve', 'TermRate'),
    ('frm.term_structures.zero_curve', 'Union'),
    ('frm.term_structures.zero_curve', 'ZeroCurveExtrapMethod'),
    ('frm.term_structures.zero_curve', 'ZeroCurveInterpMethod'),
    ('frm.term_structures.zero_curve', 'clean_tenor'),
    ('frm.term_structures.zero_curve', 'convert_column_to_consistent_data_type'),
    ('frm.term_structures.zero_curve', 'dataclass'),
    ('frm.term_structures.zero_curve', 'day_count'),
    ('frm.term_structures.zero_curve', 'discount_factor_from_zero_rate'),
    ('frm.term_structures.zero_curve', 'field'),
    ('frm.term_structures.zero_curve', 'relativedelta'),
    ('frm.term_structures.zero_curve', 'splev'),
    ('frm.term_structures.zero_curve', 'splrep'),
    ('frm.term_structures.zero_curve', 'tenor_to_date_offset'),
    ('frm.term_structures.zero_curve', 'year_frac'),
    ('frm.term_structures.zero_curve', 'zero_rate_from_discount_factor'),
    ('frm.term_structures.zero_curve_helpers', 'CompoundingFreq'),
    ('frm.utils.business_day_calendar', 'reduce'),
    ('frm.utils.daycount', 'DayCountBasis'),
    ('frm.utils.schedule', 'DayCountBasis'),
    ('frm.utils.schedule', 'DayRoll'),
    ('frm.utils.schedule', 'ExchangeNotionals'),
    ('frm.utils.schedule', 'List'),
    ('frm.utils.schedule', 'PeriodFreq'),
    ('frm.utils.schedule', 'RollConv'),
    ('frm.utils.schedule', 'Stub'),
    ('frm.utils.schedule', 'TimingConvention'),
    ('frm.utils.schedule', 'Tuple'),
    ('frm.utils.schedule', 'Union'),
    ('frm.utils.schedule', 'dataclass'),
    ('frm.utils.schedule', 'day_count'),
    ('frm.utils.schedule', 'field'),
    ('frm.utils.schedule', 'year_frac'),
    ('frm.utils.settings', 'DayCountBasis'),
    ('frm.utils.tenor', 'DateOffset'),
    ('frm.utils.tenor', 'Optional'),
])
def test_specific_imports(module_path, name):
    """Test that specific names can be imported from modules."""
    module = importlib.import_module(module_path)
    assert hasattr(module, name), f'{module_path} is missing export {name}'