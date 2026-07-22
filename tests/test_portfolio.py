import pytest
import numpy as np
from tippingpoint import MarketingReturnCurve, PortfolioAllocator

class TestPortfolioAllocator:

    def test_allocation_equal_curves(self):
        m1 = MarketingReturnCurve(beta=10000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch1")
        m2 = MarketingReturnCurve(beta=10000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch2")

        allocator = PortfolioAllocator([m1, m2])
        res = allocator.allocate_budget(total_budget=10000.0)

        assert res["success"]
        # Should split equally
        assert res["allocation"]["Ch1"] == pytest.approx(5000.0, rel=1e-2)
        assert res["allocation"]["Ch2"] == pytest.approx(5000.0, rel=1e-2)
        assert res["total_budget"] == 10000.0

    def test_allocation_one_superior_curve(self):
        m1 = MarketingReturnCurve(beta=50000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Superior")
        m2 = MarketingReturnCurve(beta=1000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Inferior")

        allocator = PortfolioAllocator([m1, m2])
        res = allocator.allocate_budget(total_budget=10000.0)

        assert res["success"]
        # Should heavily favor the superior channel
        assert res["allocation"]["Superior"] > 8000.0
        assert res["allocation"]["Inferior"] < 2000.0

    def test_allocation_with_bounds(self):
        m1 = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch1")
        m2 = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch2")

        allocator = PortfolioAllocator([m1, m2])
        # Ch1 is forced to spend at least 8000
        bounds = {"Ch1": (8000.0, 10000.0)}
        res = allocator.allocate_budget(total_budget=10000.0, channel_bounds=bounds)

        assert res["success"]
        assert res["allocation"]["Ch1"] >= 7999.0
        assert res["allocation"]["Ch2"] <= 2001.0

    def test_invalid_initialization(self):
        with pytest.raises(ValueError):
            PortfolioAllocator([])

        m1 = MarketingReturnCurve(beta=10000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="SameName")
        m2 = MarketingReturnCurve(beta=10000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="SameName")
        with pytest.raises(ValueError):
            PortfolioAllocator([m1, m2])

    def test_allocation_zero_budget(self):
        m1 = MarketingReturnCurve(beta=10000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch1")
        allocator = PortfolioAllocator([m1])
        res = allocator.allocate_budget(total_budget=0.0)
        assert res["overall_roas"] == 0.0

    def test_allocation_tight_bounds(self):
        m1 = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch1")
        m2 = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch2")
        allocator = PortfolioAllocator([m1, m2])
        # Min bounds equal total budget
        bounds = {"Ch1": (5000.0, 10000.0), "Ch2": (5000.0, 10000.0)}
        res = allocator.allocate_budget(total_budget=10000.0, channel_bounds=bounds)
        assert res["allocation"]["Ch1"] == pytest.approx(5000.0, rel=1e-2)
        assert res["allocation"]["Ch2"] == pytest.approx(5000.0, rel=1e-2)

    def test_allocation_impossible_bounds(self):
        m1 = MarketingReturnCurve(beta=20000.0, alpha=1.5, half_saturation_k=5000.0, channel_name="Ch1")
        allocator = PortfolioAllocator([m1])
        # Min bound > total budget
        bounds = {"Ch1": (20000.0, 30000.0)}
        res = allocator.allocate_budget(total_budget=10000.0, channel_bounds=bounds)
        # Should fail gracefully
        assert not res["success"]
        assert res["allocation"]["Ch1"] == 20000.0

