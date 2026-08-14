import numpy as np
import pytest
from tippingpoint import MarketingReturnCurve
from tippingpoint.fitting.frequentist import fit_frequentist_nls
from tippingpoint.math import hill_function

def test_frequentist_nls_synthetic_fit():
  np.random.seed(42)
  n = 60
  spend = np.linspace(100, 10000, n)
  true_beta = 500.0
  true_alpha = 2.0
  true_k = 3000.0

  searches = hill_function(spend, true_beta, true_alpha, true_k)
  searches_noisy = searches + np.random.normal(0, 5, n)

  res = fit_frequentist_nls(spend, searches_noisy, adstock_type="none")

  assert "beta" in res
  assert "alpha" in res
  assert "K" in res
  assert "standard_errors" in res
  assert "confidence_intervals" in res

  # Fitted parameters should be close to true values
  assert np.isclose(res["beta"], true_beta, rtol=0.2)
  assert np.isclose(res["alpha"], true_alpha, rtol=0.2)
  assert np.isclose(res["K"], true_k, rtol=0.2)

  # Confidence intervals should contain true parameter values or be valid tuples
  ci_beta = res["confidence_intervals"]["beta"]
  assert ci_beta[0] < res["beta"] < ci_beta[1]

def test_marketing_return_curve_fit_frequentist():
  np.random.seed(42)
  spend = np.linspace(500, 15000, 50)
  return_val = hill_function(spend, 1200.0, 1.8, 5000.0) + np.random.normal(0, 10, 50)

  model = MarketingReturnCurve.fit_frequentist(
      spend, return_val, channel_name="Test_Frequentist", confidence_level=0.95
  )

  assert isinstance(model, MarketingReturnCurve)
  assert model.channel_name == "Test_Frequentist"
  assert model.standard_errors is not None
  assert "beta" in model.standard_errors
  assert model.confidence_intervals is not None

  summary = model.summary()
  assert "standard_errors" in summary
  assert "confidence_intervals" in summary

def test_frequentist_fit_with_baseline():
  np.random.seed(42)
  spend = np.linspace(100, 8000, 50)
  true_baseline = 150.0
  return_val = true_baseline + hill_function(spend, 800.0, 2.0, 2500.0) + np.random.normal(0, 5, 50)

  model = MarketingReturnCurve.fit_frequentist(
      spend, return_val, fit_baseline=True, channel_name="Frequentist_Baseline"
  )

  assert model.baseline > 0
  assert "baseline" in model.standard_errors
  assert "baseline" in model.confidence_intervals
