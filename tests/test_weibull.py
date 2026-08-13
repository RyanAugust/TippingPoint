import numpy as np
import pytest
from tippingpoint.math import weibull_adstock
from tippingpoint import MarketingReturnCurve

def test_weibull_adstock_shape_and_length():
    spend = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    adstocked = weibull_adstock(spend, shape=1.5, scale=3.0, adstock_type="pdf")
    assert len(adstocked) == len(spend)
    assert np.all(adstocked >= 0)

def test_weibull_pdf_delayed_peak():
    # Shape > 1 produces delayed peak impact in Weibull PDF
    spend = np.zeros(20)
    spend[0] = 1000.0  # single pulse at t=0
    adstocked = weibull_adstock(spend, shape=2.5, scale=4.0, adstock_type="pdf")

    # Peak should not be at t=0
    peak_idx = np.argmax(adstocked)
    assert peak_idx > 0

def test_weibull_cdf_monotonic_decay():
    spend = np.zeros(20)
    spend[0] = 1000.0
    adstocked = weibull_adstock(spend, shape=1.0, scale=4.0, adstock_type="cdf")

    # CDF survival decay from single pulse should decay monotonically
    for t in range(len(adstocked) - 1):
        assert adstocked[t] >= adstocked[t + 1]

def test_weibull_zero_and_edge_cases():
    spend = np.array([])
    assert len(weibull_adstock(spend, shape=1.0, scale=1.0)) == 0

    spend = np.array([10.0, 20.0])
    # Negative shape or scale returns unadstocked copy safely
    np.testing.assert_array_equal(weibull_adstock(spend, shape=-1.0, scale=2.0), spend)

def test_model_weibull_adstock_integration():
    model = MarketingReturnCurve(
        beta=10000, alpha=1.2, half_saturation_k=5000,
        adstock_type="weibull_pdf",
        adstock_params={"shape": 2.0, "scale": 5.0}
    )
    timeline = np.array([0, 1000, 2000, 3000, 4000])
    adstocked = model.adstock_spend(timeline)
    assert len(adstocked) == len(timeline)
