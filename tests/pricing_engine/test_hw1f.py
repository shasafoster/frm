import pandas as pd
import numpy as np
from frm.pricing_engine import HullWhite1Factor
from frm.term_structures import ZeroCurve
from frm.enums import CompoundingFreq, ZeroCurveInterpMethod


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