import pandas as pd
import numpy as np
from frm.pricing_engine import HullWhite1Factor
from frm.term_structures import ZeroCurve
from frm.enums import CompoundingFreq, ZeroCurveInterpMethod




def test_discount_factor():

    mean_reversion = 0.03
    volatility = 0.02

    curve_date = pd.Timestamp('2024-04-01')
    years = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    rates = np.array([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    pillar_df = pd.DataFrame(np.array([years, rates]).T, columns=['years', 'zero_rate'])

    zero_curve = ZeroCurve(curve_date=curve_date,
                           pillar_df=pillar_df,
                           compounding_freq=CompoundingFreq.CONTINUOUS,
                           interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR)

    # Initialize model
    hw1f = HullWhite1Factor(
        zero_curve=zero_curve,
        mean_rev_lvl=mean_reversion,
        vol=volatility
    )

    hw1f.setup_theta()

    T = 4.5

    df_analytical = hw1f.zero_curve.get_discount_factors(years=T)
    df_ode1 = hw1f.calc_discount_factor_by_solving_ode_1(t1=0, t2=T)
    df_ode2 = hw1f.calc_discount_factor_by_solving_ode_2(t1=0, t2=T)

    assert np.isclose(df_analytical, df_ode1, atol=1e-10)
    assert np.isclose(df_analytical, df_ode2, atol=1e-10)
    assert np.isclose(df_ode1, df_ode2, atol=1e-10)


def test_zero_coupon_bond_option():
    # Test details copied from tf_quant_finance
    # https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/models/hull_white/zero_coupon_bond_option_test.py

    curve_date = pd.Timestamp('2024-04-01')
    years = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rates = np.array([0.01] * 5)
    pillar_df = pd.DataFrame(np.array([years, rates]).T, columns=['years', 'zero_rate'])

    zero_curve = ZeroCurve(
        curve_date=curve_date,
        pillar_df=pillar_df,
        compounding_freq=CompoundingFreq.CONTINUOUS,
        interp_method=ZeroCurveInterpMethod.CUBIC_SPLINE_ON_CCZR
    )

    hw_model = HullWhite1Factor(
        zero_curve=zero_curve,
        mean_rev_lvl=0.03,
        vol=0.02
    )

    # Test scalar case
    expiry, maturity = 1.0, 5.0
    strike = np.exp(-0.01 * maturity) / np.exp(-0.01 * expiry)

    scalar_price = hw_model.price_zero_coupon_bond_option(
        expiry_years=expiry,
        maturity_years=maturity,
        K=strike,
        cp=1
    )

    # Test vector case
    expiries = np.array([1.0, 2.0])
    maturities = np.array([5.0, 4.0])
    strikes = np.exp(-0.01 * maturities) / np.exp(-0.01 * expiries)

    vector_prices = hw_model.price_zero_coupon_bond_option(
        expiry_years=expiries,
        maturity_years=maturities,
        K=strikes,
        cp=1
    )

    np.testing.assert_almost_equal(scalar_price, 0.02817777, decimal=6)
    np.testing.assert_array_almost_equal(
        vector_prices,
        np.array([0.02817777, 0.02042677]),
        decimal=6
    )

    # TODO
    #  Add a simulation test case for these parameters

if __name__ == "__main__":
    test_zero_coupon_bond_option()