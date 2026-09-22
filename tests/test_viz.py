import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from dxpoint import MarketingReturnCurve
from dxpoint.viz import CurveVisualizer

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

def test_plot_response_curve_with_baseline():
  model = MarketingReturnCurve(
      beta=50000.0, alpha=1.5, half_saturation_k=10000.0,
      baseline=5000.0, channel_name="BaselineChannel"
  )
  spend = np.array([1000, 2000, 3000, 4000, 5000])
  ret = model.predict_incremental_return(spend, include_baseline=True)
  fig = model.plot_response_curve(target_mroas=1.0, scatter=(spend, ret), show=False, include_baseline=True)
  assert fig is not None
  plt.close(fig)


def test_plot_executive_view_basic():
  model = MarketingReturnCurve(beta=60000.0, alpha=1.6, half_saturation_k=15000.0, channel_name="YouTube Video")
  fig = model.plot_executive_view(target_mroas=1.15, current_spend=12000.0, show=False)

  assert fig is not None
  assert isinstance(fig, plt.Figure)
  assert len(fig.axes) == 2  # Two distinct views: Saturation (left) and Marginal (right)

  ax1, ax2 = fig.axes
  # Check panel titles
  assert "Media Saturation" in ax1.get_title(loc='left')
  assert "Marginal Return" in ax2.get_title(loc='left')

  # Check legend in ax2 does not print target mROAS value like (1.15)
  _, labels2 = ax2.get_legend_handles_labels()
  assert "Target Hurdle Rate" in labels2
  for label in labels2:
    assert "(1.15)" not in label

  plt.close(fig)


def test_plot_executive_view_tiny_mroas():
  # Test with small mROAS hurdle rate (e.g. conversions or searches per dollar)
  model = MarketingReturnCurve(beta=5000.0, alpha=1.4, half_saturation_k=10000.0, channel_name="Performance Search")
  tiny_target = 0.002
  fig = model.plot_executive_view(target_mroas=tiny_target, current_spend=8000.0, show=False)

  assert fig is not None
  ax1, ax2 = fig.axes
  _, labels2 = ax2.get_legend_handles_labels()

  # Verify no raw target float appears in legend
  assert "Target Hurdle Rate" in labels2
  for label in labels2:
    assert "0.002" not in label

  plt.close(fig)


def test_plot_executive_view_concave_c_curve():
  # Alpha <= 1.0 (pure concave, no inflection warm-up point)
  model = MarketingReturnCurve(beta=80000.0, alpha=0.85, half_saturation_k=12000.0, channel_name="Paid Search Brand")
  fig = model.plot_executive_view(target_mroas=1.0, current_spend=15000.0, show=False)

  assert fig is not None
  assert len(fig.axes) == 2
  plt.close(fig)


def test_plot_executive_view_with_uncertainty():
  n = 100
  samples = {
      'beta': np.random.normal(50000, 2000, n),
      'alpha': np.random.normal(1.5, 0.08, n),
      'K': np.random.normal(12000, 500, n),
      'theta': np.random.uniform(0.1, 0.3, n)
  }
  model = MarketingReturnCurve(
      beta=50000.0, alpha=1.5, half_saturation_k=12000.0,
      posterior_samples=samples, channel_name="BayesianExecutive"
  )
  spends = np.linspace(1000, 25000, 20)
  returns = model.predict_incremental_return(spends)

  fig = model.plot_executive_view(
      target_mroas=1.2, current_spend=14000.0,
      show_intervals=True, scatter=(spends, returns), show=False
  )
  assert fig is not None
  assert len(fig.axes) == 2
  plt.close(fig)

