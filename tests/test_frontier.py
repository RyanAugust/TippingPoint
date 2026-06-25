import pytest
import numpy as np
from unittest.mock import patch
from tippingpoint import MaximalFrontierCurve


class TestMaximalFrontierCurve:

  def test_initialization(self):
    """Test standard initialization."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, model_type="2-parameter")
    assert curve.cr == 2.0
    assert curve.r_prime == 500.0
    assert curve.r_max is None
    assert curve.model_type == "2-parameter"

    curve_3p = MaximalFrontierCurve(cr=1.5, r_prime=300.0, r_max=10.0, model_type="3-parameter")
    assert curve_3p.cr == 1.5
    assert curve_3p.r_prime == 300.0
    assert curve_3p.r_max == 10.0
    assert curve_3p.model_type == "3-parameter"

  def test_predict_rate(self):
    """Test prediction of the maximal efficiency rate."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, model_type="2-parameter")
    # For spend = 500 (K/spend = 1): rate = 500/500 + 2.0 = 3.0
    assert curve.predict_rate(500.0) == pytest.approx(3.0)

    curve_3p = MaximalFrontierCurve(cr=2.0, r_prime=500.0, r_max=12.0, model_type="3-parameter")
    # At spend = 0 (approximated as tiny), rate should be close to r_max (12.0)
    assert curve_3p.predict_rate(0.0) == pytest.approx(12.0, abs=1e-3)

  def test_predict_total(self):
    """Test prediction of total maximal results."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, model_type="2-parameter")
    # At spend = 500: rate = 3.0, total = 500 * 3.0 = 1500
    assert curve.predict_total(500.0) == pytest.approx(1500.0)

  def test_get_max_spend_2param(self):
    """Test max spend calculations in 2-parameter model."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, model_type="2-parameter")
    # Target rate below critical results (2.0) -> infinite spend possible
    assert curve.get_max_spend(1.5) == np.inf
    # Target rate = 3.0 -> spend = 500 / (3.0 - 2.0) = 500.0
    assert curve.get_max_spend(3.0) == pytest.approx(500.0)

  def test_get_max_spend_3param(self):
    """Test max spend calculations in 3-parameter model."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, r_max=12.0, model_type="3-parameter")
    # Target rate >= r_max -> 0 spend (unreachable efficiency)
    assert curve.get_max_spend(13.0) == 0.0
    # Target rate <= cr -> infinite spend possible
    assert curve.get_max_spend(1.5) == np.inf

    # Mathematical round-trip check
    spend_target = 300.0
    predicted_rate = curve.predict_rate(spend_target)
    calculated_spend = curve.get_max_spend(predicted_rate)
    assert calculated_spend == pytest.approx(spend_target)

  def test_fit_2param_linear(self):
    """Test standard linear fit for 2-parameter model."""
    # Create fake linear work data: Total = 2.0 * Spend + 500.0
    spends = np.linspace(100, 1000, 10)
    totals = 2.0 * spends + 500.0

    curve = MaximalFrontierCurve.fit_2param(spends, totals, is_rate=False, method="linear")
    assert curve.cr == pytest.approx(2.0, rel=1e-3)
    assert curve.r_prime == pytest.approx(500.0, rel=1e-3)
    assert curve.model_type == "2-parameter"

  def test_fit_2param_nonlinear_tinygrad(self):
    """Test Tinygrad optimization fit for 2-parameter model."""
    spends = np.linspace(100, 1000, 10)
    rates = 500.0 / spends + 2.0

    curve = MaximalFrontierCurve.fit_2param(spends, rates, is_rate=True, method="nonlinear", epochs=50)
    assert curve.cr > 0
    assert curve.r_prime > 0

  def test_fit_2param_quantile(self):
    """Test fitting with quantile pinball loss."""
    spends = np.linspace(100, 1000, 10)
    rates = 500.0 / spends + 2.0

    # Fit 90th percentile
    curve = MaximalFrontierCurve.fit_2param(spends, rates, is_rate=True, method="nonlinear", q=0.9, epochs=50)
    assert curve.cr > 0
    assert curve.r_prime > 0

  def test_fit_3param(self):
    """Test fitting 3-parameter model using Tinygrad."""
    spends = np.linspace(100, 1000, 10)
    rates = 2.0 + 500.0 / (spends + 500.0 / (12.0 - 2.0))

    curve = MaximalFrontierCurve.fit_3param(spends, rates, is_rate=True, epochs=50)
    assert curve.cr > 0
    assert curve.r_prime > 0
    assert curve.r_max > curve.cr

  @patch("matplotlib.pyplot.show")
  def test_plot_frontier(self, mock_show):
    """Test that the plotting helper does not throw errors."""
    curve = MaximalFrontierCurve(cr=2.0, r_prime=500.0, model_type="2-parameter")
    try:
      curve.plot_frontier()
      mock_show.assert_called_once()
    except Exception as e:
      pytest.fail(f"plot_frontier raised an exception: {e}")
