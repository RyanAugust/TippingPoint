import numpy as np
import pytest
from tippingpoint import MarketingReturnCurve, evaluate_curve_fit, format_fit_report
from tippingpoint.math import hill_function

def test_evaluate_curve_fit():
  np.random.seed(42)
  spend = np.linspace(100, 10000, 60)
  true_return = hill_function(spend, 1000.0, 2.0, 4000.0)
  noisy_return = true_return + np.random.normal(0, 10, len(spend))

  model = MarketingReturnCurve.fit(spend, noisy_return, method="frequentist")

  metrics = model.evaluate_fit()

  assert "r_squared" in metrics
  assert "adj_r_squared" in metrics
  assert "rmse" in metrics
  assert "mae" in metrics
  assert "mape" in metrics
  assert "aic" in metrics
  assert "bic" in metrics

  # High quality fit should have R^2 > 0.95
  assert metrics["r_squared"] > 0.95
  assert metrics["adj_r_squared"] > 0.95
  assert metrics["rmse"] < 20.0
  assert metrics["mape"] < 15.0

  # Format report
  report_str = format_fit_report(metrics)
  assert "Goodness-of-Fit" in report_str
  assert "R-Squared" in report_str

def test_frequentist_delta_method_uncertainty():
  np.random.seed(42)
  spend = np.linspace(500, 15000, 50)
  returns = hill_function(spend, 2000.0, 1.8, 6000.0) + np.random.normal(0, 15, 50)

  model = MarketingReturnCurve.fit_frequentist(spend, returns, confidence_level=0.95)

  # Scalar prediction with intervals
  pt, low, high = model.predict_incremental_return(6000.0, return_interval=True, confidence_level=0.95)
  assert low < pt < high
  assert abs(pt - 1000.0) < 50.0  # At K=6000, Hill is at 50% capacity (~1000)

  # Array prediction with intervals
  s_eval = np.array([2000.0, 6000.0, 10000.0])
  pt_arr, low_arr, high_arr = model.predict_incremental_return(s_eval, return_interval=True)
  assert len(pt_arr) == 3
  assert np.all(low_arr <= pt_arr)
  assert np.all(pt_arr <= high_arr)

  # Marginal return derivative prediction with intervals
  m_pt, m_low, m_high = model.predict_marginal_return(6000.0, return_interval=True)
  assert m_low <= m_pt <= m_high

def test_bayesian_predictive_intervals():
  np.random.seed(42)
  spend = np.linspace(500, 10000, 30)
  returns = hill_function(spend, 1500.0, 1.5, 4000.0) + np.random.normal(0, 10, 30)

  model = MarketingReturnCurve.fit_bayesian(spend, returns, n_samples=50, chains=2, burn_in=10)

  pt, low, high = model.predict_incremental_return(4000.0, return_interval=True, confidence_level=0.90)
  assert low <= pt <= high

  m_pt, m_low, m_high = model.predict_marginal_return(4000.0, return_interval=True, confidence_level=0.90)
  assert m_low <= m_pt <= m_high

def test_plot_response_curve_with_frequentist_intervals():
  np.random.seed(42)
  spend = np.linspace(500, 10000, 30)
  returns = hill_function(spend, 1500.0, 1.5, 4000.0) + np.random.normal(0, 10, 30)

  model = MarketingReturnCurve.fit_frequentist(spend, returns)
  fig = model.plot_response_curve(show_intervals=True, show=False)
  assert fig is not None
