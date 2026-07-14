import pytest
import numpy as np
from tippingpoint import MarketingReturnCurve
from tippingpoint.math import days_to_theta, geometric_adstock

class TestAdstock:

    def test_days_to_theta(self):
        """Test conversion of decay half-life in days to decay rate theta."""
        assert days_to_theta(0) == 0.0
        assert days_to_theta(-5) == 0.0
        # Half life of 1 day means 50% decay (theta = 0.5)
        assert days_to_theta(1) == pytest.approx(0.5)
        # Half life of 7 days means theta approx 0.9057
        assert days_to_theta(7) == pytest.approx(0.5 ** (1/7))

    def test_numpy_geometric_adstock(self):
        """Test NumPy geometric adstock decay values."""
        spends = np.array([100.0, 100.0, 100.0])
        # Theta = 0.5: [100, 150, 175]
        adstocked = geometric_adstock(spends, 0.5)
        np.testing.assert_allclose(adstocked, [100.0, 150.0, 175.0])

    def test_mle_fitting_with_fixed_adstock(self):
        """Test MLE gradient fitting with fixed adstock decay."""
        spends = np.array([1000, 2000, 5000, 10000, 15000, 25000])
        returns = np.array([100, 300, 1000, 2500, 5000, 8000])

        # Fit with a 3-day fixed adstock decay
        model = MarketingReturnCurve.from_historical_data(
            spend_array=spends,
            return_array=returns,
            adstock_type="fixed",
            adstock_fixed_days=3.0,
            epochs=100
        )
        expected_theta = 0.5 ** (1.0 / 3.0)
        assert model.theta == pytest.approx(expected_theta, rel=1e-5)

    def test_mle_fitting_with_free_adstock(self):
        """Test MLE gradient fitting with free optimized adstock."""
        spends = np.array([1000, 2000, 5000, 10000, 15000, 25000])
        returns = np.array([100, 300, 1000, 2500, 5000, 8000])

        # Fit with fully free adstock
        model = MarketingReturnCurve.from_historical_data(
            spend_array=spends,
            return_array=returns,
            adstock_type="free",
            epochs=100
        )
        assert 0.0 <= model.theta < 1.0

    def test_mle_fitting_with_bounded_adstock(self):
        """Test MLE gradient fitting with bounded adstock half-life."""
        spends = np.array([1000, 2000, 5000, 10000, 15000, 25000])
        returns = np.array([100, 300, 1000, 2500, 5000, 8000])

        # Bounded between 1 and 7 days half-life
        min_days, max_days = 1.0, 7.0
        model = MarketingReturnCurve.from_historical_data(
            spend_array=spends,
            return_array=returns,
            adstock_type="bounded",
            adstock_bounds=(min_days, max_days),
            epochs=100
        )
        theta_min = 0.5 ** (1.0 / min_days)
        theta_max = 0.5 ** (1.0 / max_days)
        assert theta_min <= model.theta <= theta_max

    def test_bayesian_fitting_with_free_adstock(self):
        """Test MCMC Bayesian fitting with free optimized adstock."""
        spends = np.array([1000, 2000, 5000, 10000, 15000, 25000])
        returns = np.array([100, 300, 1000, 2500, 5000, 8000])

        # Fit with fully free adstock
        model = MarketingReturnCurve.fit_bayesian(
            spend_array=spends,
            return_array=returns,
            adstock_type="free",
            n_samples=50,
            chains=2,
            burn_in=10
        )
        assert 0.0 <= model.theta < 1.0
        assert "theta" in model.posterior_samples
