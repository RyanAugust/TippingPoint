import pytest
import numpy as np
from tippingpoint import MarketingReturnCurve

class TestTippingPoints:

    def setup_method(self):
        # S-Curve: beta=10000, alpha=2.0, K=500
        # Inflection (max efficiency) = K * sqrt((2-1)/(2+1)) = 500 * sqrt(1/3) approx 288.675
        self.model = MarketingReturnCurve(beta=10000.0, alpha=2.0, half_saturation_k=500.0, channel_name="Test")

    def test_precomputed_points(self):
        """Test that tipping points are pre-computed on initialization."""
        assert "max_efficiency_point" in self.model.tipping_points
        assert "max_profit_point" in self.model.tipping_points

        expected_inflection = 500.0 * np.sqrt(1.0 / 3.0)
        assert self.model.max_efficiency_point == pytest.approx(expected_inflection, rel=1e-3)

        # Max profit point should be where f'(x) = 1.0
        if self.model.max_profit_point:
            mroas = self.model.predict_marginal_return(self.model.max_profit_point)
            assert mroas == pytest.approx(1.0, rel=1e-3)

    def test_summary(self):
        """Test the summary method structure."""
        s = self.model.summary()
        assert s["channel"] == "Test"
        assert "parameters" in s
        assert "tipping_points" in s
        assert s["parameters"]["beta"] == 10000.0
        assert s["tipping_points"]["max_efficiency_point"] == self.model.max_efficiency_point

    def test_c_curve_tipping_points(self):
        """Test tipping points for a C-Curve (alpha <= 1)."""
        c_model = MarketingReturnCurve(beta=10000.0, alpha=0.8, half_saturation_k=500.0)
        # For alpha <= 1, inflection is 0
        assert c_model.max_efficiency_point == 0.0
        # Max profit point should still be where mROAS = 1.0 (if reachable)
        if c_model.max_profit_point:
            mroas = c_model.predict_marginal_return(c_model.max_profit_point)
            assert mroas == pytest.approx(1.0, rel=1e-3)
