import pytest
import numpy as np
from unittest.mock import patch
import warnings

# Import the class from your module (change 'marketing_curve' to your actual file name without .py)
from tippingpoint import MarketingReturnCurve

class TestMarketingReturnCurve:

  def setup_method(self):
    """Setup a baseline S-Curve for use in multiple tests."""
    # Parameters: Max Return=10000, Shape=2.0 (S-Curve), Half-Saturation=500
    self.s_curve = MarketingReturnCurve(beta=10000.0, alpha=2.0, half_saturation_k=500.0, channel_name="Test_S_Curve")
    
    # Parameters: Shape=0.8 (C-Curve, no warm-up phase)
    self.c_curve = MarketingReturnCurve(beta=10000.0, alpha=0.8, half_saturation_k=500.0, channel_name="Test_C_Curve")

  def test_initialization(self):
    """Test that the class initializes parameters correctly."""
    curve = MarketingReturnCurve(beta=100, alpha=1.5, half_saturation_k=50, channel_name="Test")
    assert curve.beta == 100.0
    assert curve.alpha == 1.5
    assert curve.K == 50.0
    assert curve.channel_name == "Test"

  def test_predict_incremental_return(self):
    """Test the mathematical output of the Hill Function (f(x))."""
    # Testing a simple scenario: spend = K. Return should be exactly half of beta.
    # Formula: (beta * K^a) / (K^a + K^a) = beta / 2
    spend = 500.0
    expected_return = 10000.0 / 2.0
    actual_return = self.s_curve.predict_incremental_return(spend)
    assert actual_return == pytest.approx(expected_return, rel=1e-3)

  def test_predict_marginal_return(self):
    """Test the first derivative calculation (mROAS / f'(x))."""
    linear_curve = MarketingReturnCurve(beta=100.0, alpha=1.0, half_saturation_k=50.0) # For a curve with a=1: f(x) = beta*x / (k+x) -> f'(x) = (beta*k) / (k+x)^2
    spend = 50.0
    expected_mroas = 0.5
    actual_mroas = linear_curve.predict_marginal_return(spend) # Expected f'(50) = (100 * 50) / (50 + 50)^2 = 5000 / 10000 = 0.5
    assert actual_mroas == pytest.approx(expected_mroas, rel=1e-3)

  def test_minimal_marginal_cost_point_s_curve(self):
    """Test f''(x) = 0 for an S-Curve (alpha > 1)."""
    expected_inflection = 500.0 * np.sqrt(1.0 / 3.0) # For alpha = 2.0, inflection = K * sqrt((2-1)/(2+1)) = 500 * sqrt(1/3) ≈ 288.675
    actual_inflection = self.s_curve.get_minimal_marginal_cost_point()
    
    assert actual_inflection == pytest.approx(expected_inflection, rel=1e-3)

  def test_minimal_marginal_cost_point_c_curve(self):
    """Test f''(x) = 0 for a C-Curve (alpha <= 1). Should return 0.0."""
    assert self.c_curve.get_minimal_marginal_cost_point() == 0.0

  def test_diminishing_returns_point_valid(self):
    """Test finding the target spend level for a reachable mROAS."""
    target_mroas = 5.0
    spend_cap = self.s_curve.get_diminishing_returns_point(target_mroas)
    
    assert spend_cap is not None
    actual_mroas_at_cap = self.s_curve.predict_marginal_return(spend_cap)
    assert actual_mroas_at_cap == pytest.approx(target_mroas, rel=1e-3)

  def test_diminishing_returns_point_unreachable(self):
    """Test behavior when the target mROAS is higher than the curve's absolute maximum."""
    unreachable_target = 1000.0 # Way too high
    with pytest.warns(UserWarning, match="is mathematically unreachable"):
      spend_cap = self.s_curve.get_diminishing_returns_point(unreachable_target)
    assert spend_cap is None

  def test_tinygrad_optimization_engine(self):
    """Test that the Tinygrad builder can ingest data and output a curve without crashing."""
    np.random.seed(42)
    spends = np.linspace(1000, 10000, 20)
    returns = (50000 * spends**1.5) / (4000**1.5 + spends**1.5) # Fake responses following roughly an S curve
    # Run with very few epochs just to verify the math/graph builds and executes properly
    model = MarketingReturnCurve.from_historical_data( spend_array=spends, return_array=returns, epochs=10, lr=0.1)# Fast execution for test suite
    
    assert isinstance(model, MarketingReturnCurve)
    assert model.beta > 0
    assert model.alpha > 0
    assert model.K > 0

  def test_evaluate_current_budget(self, capsys):
    """Test the business logic printing function."""
    # Inflection point is ~288.
    self.s_curve.evaluate_current_budget(current_spend=100.0, target_mroas=1.0) # Test WARMING UP (under inflection)
    captured = capsys.readouterr()
    assert "WARMING UP" in captured.out
    cap_spend = self.s_curve.get_diminishing_returns_point(target_mroas=2.0) # Test OVER-SATURATED (over spend cap)
    self.s_curve.evaluate_current_budget(current_spend=cap_spend + 5000, target_mroas=2.0)
    captured = capsys.readouterr()
    assert "OVER-SATURATED" in captured.out

  @patch("matplotlib.pyplot.show") # Mock plt.show() so it doesn't pop up a window and halt the tests
  def test_plot_response_curve(self, mock_show):
    """Test that the plotting function executes without throwing exceptions."""
    try:
      self.s_curve.plot_response_curve(target_mroas=5.0, current_spend=600.0)
      mock_show.assert_called_once()
    except Exception as e:
      pytest.fail(f"plot_response_curve raised an exception: {e}")
