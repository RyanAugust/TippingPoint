import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from tippingpoint import MarketingReturnCurve

class TestModels:
    def test_adstock_spend(self):
        model = MarketingReturnCurve(beta=10000, alpha=1.5, half_saturation_k=5000, theta=0.5)
        spends = np.array([100.0, 100.0, 100.0])
        adstocked = model.adstock_spend(spends)
        np.testing.assert_allclose(adstocked, [100.0, 150.0, 175.0])

    @patch('builtins.print')
    def test_evaluate_current_budget(self, mock_print):
        model = MarketingReturnCurve(beta=10000, alpha=1.5, half_saturation_k=5000)

        # Test warning up phase
        model.evaluate_current_budget(current_spend=100.0, target_mroas=1.0)
        args, _ = mock_print.call_args
        assert "WARMING UP" in args[0]

        # Test over-saturated phase
        # Ensure max spend exists
        max_spend = model.get_diminishing_returns_point(target_mroas=0.1)
        if max_spend:
            model.evaluate_current_budget(current_spend=max_spend + 10000, target_mroas=0.1)
            args, _ = mock_print.call_args
            assert "OVER-SATURATED" in args[0]

        # Test optimal scaling zone
        min_spend = model.get_minimal_marginal_cost_point()
        model.evaluate_current_budget(current_spend=min_spend + 100.0, target_mroas=0.01)
        args, _ = mock_print.call_args
        assert "OPTIMAL SCALING ZONE" in args[0]

    @patch('matplotlib.pyplot.show')
    def test_plot_response_curve(self, mock_show):
        model = MarketingReturnCurve(beta=10000, alpha=1.5, half_saturation_k=5000)
        fig = model.plot_response_curve(target_mroas=1.0, show=True)
        assert fig is not None
        mock_show.assert_called_once()

    @patch('streamlit.web.cli.main')
    @patch('sys.argv')
    def test_launch_dashboard(self, mock_argv, mock_st_main):
        model = MarketingReturnCurve(beta=10000, alpha=1.5, half_saturation_k=5000)
        model.launch_dashboard()
        mock_st_main.assert_called_once()

    def test_diminishing_returns_unreachable(self):
        model = MarketingReturnCurve(beta=1000, alpha=1.5, half_saturation_k=5000)
        with pytest.warns(UserWarning, match="is mathematically unreachable"):
            max_spend = model.get_diminishing_returns_point(target_mroas=10.0)
            assert max_spend is None

    def test_weibull_adstock_spend(self):
        model_pdf = MarketingReturnCurve(
            beta=10000, alpha=1.2, half_saturation_k=5000,
            adstock_type="weibull_pdf",
            adstock_params={"shape": 2.0, "scale": 3.0}
        )
        spends = np.array([100.0, 100.0, 100.0, 100.0])
        adstocked = model_pdf.adstock_spend(spends)
        assert len(adstocked) == 4
        assert np.all(adstocked >= 0)

        model_cdf = MarketingReturnCurve(
            beta=10000, alpha=1.2, half_saturation_k=5000,
            adstock_type="weibull_cdf",
            adstock_params={"shape": 2.0, "scale": 3.0}
        )
        adstocked_cdf = model_cdf.adstock_spend(spends)
        assert len(adstocked_cdf) == 4

    def test_summary_and_update_loss(self):
        model = MarketingReturnCurve(beta=25000, alpha=1.3, half_saturation_k=4000, theta=0.5, baseline=1200)
        model.update_loss(1234.5)
        s = model.summary()
        assert s["loss"] == 1234.5
        assert s["parameters"]["baseline"] == 1200
        assert s["parameters"]["theta"] == 0.5
        assert "adstock_half_life_days" in s["parameters"]

