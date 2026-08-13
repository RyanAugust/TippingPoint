import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from tippingpoint import MarketingReturnCurve
from tippingpoint.viz import CurveVisualizer

def test_plot_response_curve_basic():
  model = MarketingReturnCurve(beta=25000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="TestChannel")
  fig = CurveVisualizer.plot_response_curve(model, target_mroas=1.0, current_spend=4000.0)
  assert fig is not None
  assert isinstance(fig, plt.Figure)
  plt.close(fig)

def test_plot_response_curve_with_posterior_samples():
  # Create synthetic posterior samples
  n = 200
  samples = {
    'beta': np.random.normal(25000, 1000, n),
    'alpha': np.random.normal(1.5, 0.1, n),
    'K': np.random.normal(5000, 200, n),
    'theta': np.random.uniform(0.1, 0.4, n)
  }
  model = MarketingReturnCurve(beta=25000.0, alpha=1.5, half_saturation_k=5000.0, theta=0.3, channel_name="BayesianChannel", posterior_samples=samples)

  spend_data = np.linspace(500, 15000, 30)
  return_data = model.predict_incremental_return(spend_data) + np.random.normal(0, 500, 30)

  fig = CurveVisualizer.plot_response_curve(
    model,
    target_mroas=1.2,
    current_spend=7000.0,
    show_intervals=True,
    scatter=(spend_data, return_data)
  )
  assert fig is not None
  plt.close(fig)

def test_plot_response_curve_weibull_scatter():
  model = MarketingReturnCurve(
    beta=50000.0, alpha=1.2, half_saturation_k=10000.0,
    adstock_type="weibull_pdf",
    adstock_params={"shape": 2.0, "scale": 3.0},
    channel_name="WeibullChannel"
  )
  spend = np.array([1000, 2000, 3000, 4000, 5000])
  ret = model.predict_incremental_return(spend)

  fig = model.plot_response_curve(target_mroas=1.0, scatter=(spend, ret), show=False)
  assert fig is not None
  plt.close(fig)

def test_plot_response_curve_c_curve():
  # Alpha <= 1.0 (pure concave, no inflection point)
  model = MarketingReturnCurve(beta=100000.0, alpha=0.8, half_saturation_k=20000.0, channel_name="ConcaveChannel")
  fig = model.plot_response_curve(target_mroas=0.5, show=False)
  assert fig is not None
  plt.close(fig)
