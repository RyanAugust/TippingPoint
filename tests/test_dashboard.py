import numpy as np
import pytest
from tippingpoint import MarketingReturnCurve
from tippingpoint.dashboard import create_plotly_plot

def test_create_plotly_plot_basic():
  model = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=4000.0, channel_name="PlotlyChannel")
  fig = create_plotly_plot(model, target_mroas=1.0)
  assert fig is not None
  assert len(fig.data) >= 3  # Return, mROAS, Target mROAS line

def test_create_plotly_plot_with_scatter_and_adstock():
  model = MarketingReturnCurve(
    beta=30000.0, alpha=1.2, half_saturation_k=6000.0,
    theta=0.4,
    channel_name="AdstockedPlotly"
  )
  spends = np.array([1000, 2000, 4000, 6000, 8000])
  returns = model.predict_incremental_return(spends)
  fig = create_plotly_plot(model, target_mroas=1.2, scatter=(spends, returns))
  assert fig is not None
  assert len(fig.data) >= 3
