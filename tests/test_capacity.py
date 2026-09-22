import os
import sys
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
  sys.path.insert(0, repo_root)

from dxpoint import (
    MarketingReturnCurve,
    CapacityProjector,
    ProjectedReturnCurve,
    PortfolioAllocator,
)

class TestCapacityProjector:

  def test_direct_overrides(self):
    """Test that explicit m_beta and m_k overrides take absolute precedence."""
    projector = CapacityProjector(m_beta=2.5, m_k=3.0)
    mults = projector.compute_multipliers()
    assert mults.m_beta == 2.5
    assert mults.m_k == 3.0

  def test_rf_multipliers_reach_expansion(self):
    """Test Reach expansion calculations with frequency held at target."""
    projector = CapacityProjector(
        current_reach_penetration=0.15,
        target_reach_penetration=0.60,
        current_frequency=3.0,
        target_frequency=3.0,
        reach_elasticity=0.65,
        spend_reach_elasticity=0.75,
    )
    mults = projector.compute_multipliers()
    expected_ratio = 0.60 / 0.15  # 4.0
    expected_beta = expected_ratio ** 0.65
    expected_k = expected_ratio ** 0.75

    assert mults.m_beta == pytest.approx(expected_beta, rel=1e-3)
    assert mults.m_k == pytest.approx(expected_k, rel=1e-3)
    assert mults.m_beta_rf == pytest.approx(expected_beta, rel=1e-3)
    assert mults.m_k_rf == pytest.approx(expected_k, rel=1e-3)

  def test_rf_frequency_reclamation(self):
    """Test frequency reclamation when current frequency exceeds target."""
    projector = CapacityProjector(
        current_reach_penetration=0.30,
        target_reach_penetration=0.30,  # No reach expansion
        current_frequency=6.0,
        target_frequency=3.0,
        frequency_elasticity=0.50,
        frequency_capacity_factor=0.15,
    )
    mults = projector.compute_multipliers()
    expected_k = (6.0 / 3.0) ** 0.50  # sqrt(2) approx 1.414
    waste_frac = 1.0 - (3.0 / 6.0)     # 0.5
    expected_beta = 1.0 + 0.15 * waste_frac   # 1.075

    assert mults.m_k == pytest.approx(expected_k, rel=1e-3)
    assert mults.m_beta == pytest.approx(expected_beta, rel=1e-3)

  def test_user_defined_funnel_multipliers(self):
    """Test custom user-defined funnel multipliers passed at runtime."""
    custom_funnels = {
        "lower_only": {"m_beta": 1.0, "m_k": 1.0},
        "mid_lower": {"m_beta": 1.25, "m_k": 1.35},
        "full_funnel": {"m_beta": 1.70, "m_k": 2.10},
    }
    projector = CapacityProjector(
        current_funnel="lower_only",
        target_funnel="full_funnel",
        funnel_multipliers=custom_funnels,
    )
    mults = projector.compute_multipliers()
    assert mults.m_beta == pytest.approx(1.70, rel=1e-3)
    assert mults.m_k == pytest.approx(2.10, rel=1e-3)
    assert mults.m_beta_product == pytest.approx(1.70, rel=1e-3)
    assert mults.m_k_product == pytest.approx(2.10, rel=1e-3)

  def test_user_defined_vertical_multipliers(self):
    """Test vertical-specific multiplier lookup configured by the user."""
    vertical_configs = {
        "retail_ecommerce": {
            "lower_only": {"m_beta": 1.0, "m_k": 1.0},
            "full_funnel": {"m_beta": 1.65, "m_k": 1.95},
        },
        "finance_b2b": {
            "lower_only": {"m_beta": 1.0, "m_k": 1.0},
            "full_funnel": {"m_beta": 1.35, "m_k": 1.50},
        }
    }
    projector = CapacityProjector(
        vertical="finance_b2b",
        current_funnel="lower_only",
        target_funnel="full_funnel",
        vertical_multipliers=vertical_configs,
    )
    mults = projector.compute_multipliers()
    assert mults.m_beta == pytest.approx(1.35, rel=1e-3)
    assert mults.m_k == pytest.approx(1.50, rel=1e-3)

  def test_damped_synergy_formulation(self):
    """Test that combining R&F and Product unlocks uses the damped formulation.

    Formula: M = 1.0 + rho * ((M_rf - 1.0) + (M_prod - 1.0))
    """
    custom_funnels = {
        "lower_only": {"m_beta": 1.0, "m_k": 1.0},
        "full_funnel": {"m_beta": 1.50, "m_k": 1.60},
    }
    rho = 0.80
    projector = CapacityProjector(
        current_reach_penetration=0.20,
        target_reach_penetration=0.40,  # 2.0x ratio
        reach_elasticity=1.0,           # Delta beta rf = 2.0 - 1.0 = 1.0
        spend_reach_elasticity=1.0,     # Delta k rf = 2.0 - 1.0 = 1.0
        current_frequency=3.0,
        target_frequency=3.0,
        current_funnel="lower_only",
        target_funnel="full_funnel",    # Delta beta prod = 0.50, Delta k prod = 0.60
        funnel_multipliers=custom_funnels,
        synergy_damping=rho,
    )
    mults = projector.compute_multipliers()

    # Delta beta: 1.0 + 0.50 = 1.50 -> Damped: 1.0 + 0.80 * 1.50 = 2.20
    expected_beta = 1.0 + rho * (1.0 + 0.50)
    # Delta k: 1.0 + 0.60 = 1.60 -> Damped: 1.0 + 0.80 * 1.60 = 2.28
    expected_k = 1.0 + rho * (1.0 + 0.60)

    assert mults.m_beta == pytest.approx(expected_beta, rel=1e-3)
    assert mults.m_k == pytest.approx(expected_k, rel=1e-3)

  def test_damped_formulation_single_driver_undamped(self):
    """When only ONE driver is active, it should not be artificially damped."""
    # Only product active
    p_only = CapacityProjector(m_beta_product=1.50, m_k_product=1.80, synergy_damping=0.75)
    m1 = p_only.compute_multipliers()
    assert m1.m_beta == pytest.approx(1.50, rel=1e-4)
    assert m1.m_k == pytest.approx(1.80, rel=1e-4)

    # Only RF active
    rf_only = CapacityProjector(
        current_reach_penetration=0.25,
        target_reach_penetration=0.50,
        reach_elasticity=1.0,
        spend_reach_elasticity=1.0,
        synergy_damping=0.75
    )
    m2 = rf_only.compute_multipliers()
    assert m2.m_beta == pytest.approx(2.0, rel=1e-4)
    assert m2.m_k == pytest.approx(2.0, rel=1e-4)


class TestProjectedReturnCurve:

  def setup_method(self):
    # S-Curve base model: beta=10000, alpha=2.0, K=500
    self.base_model = MarketingReturnCurve(
        beta=10000.0,
        alpha=2.0,
        half_saturation_k=500.0,
        channel_name="Performance Video"
    )

  def test_curve_scaling_and_inflection(self):
    """Test that ProjectedReturnCurve scales beta and K and shifts inflection proportionally."""
    m_beta = 2.0
    m_k = 3.0
    proj_curve = self.base_model.project_capacity(m_beta=m_beta, m_k=m_k)

    assert isinstance(proj_curve, ProjectedReturnCurve)
    assert isinstance(proj_curve, MarketingReturnCurve)
    assert proj_curve.beta == pytest.approx(20000.0, rel=1e-4)
    assert proj_curve.K == pytest.approx(1500.0, rel=1e-4)
    assert proj_curve.alpha == pytest.approx(2.0, rel=1e-4)

    # Inflection point should scale exactly with m_k
    base_inflection = self.base_model.get_minimal_marginal_cost_point()
    proj_inflection = proj_curve.get_minimal_marginal_cost_point()
    assert proj_inflection == pytest.approx(base_inflection * m_k, rel=1e-3)

  def test_headroom_evaluation(self):
    """Test evaluate_unlocked_headroom calculation and metrics."""
    proj_curve = self.base_model.project_capacity(
        m_beta=2.0,
        m_k=2.5,
        channel_name="Projected Video"
    )
    summary = proj_curve.evaluate_unlocked_headroom(target_mroas=1.0, current_spend=400.0, verbose=True)

    assert "unlocked_headroom" in summary
    assert "current_spend_evaluation" in summary
    assert summary["unlocked_headroom"]["spend_headroom"] > 0
    assert summary["unlocked_headroom"]["return_headroom"] > 0
    assert summary["current_spend_evaluation"]["current_spend"] == 400.0
    assert summary["projected_curve"]["beta"] == 20000.0
    assert summary["projected_curve"]["K"] == 1250.0

  def test_portfolio_allocator_integration(self):
    """Test that ProjectedReturnCurve can be optimized directly inside PortfolioAllocator."""
    search_model = MarketingReturnCurve(beta=15000.0, alpha=1.5, half_saturation_k=800.0, channel_name="Search")
    proj_video = self.base_model.project_capacity(m_beta=1.8, m_k=2.2, channel_name="Video (Full Funnel)")

    allocator = PortfolioAllocator([search_model, proj_video])
    result = allocator.allocate_budget(total_budget=5000.0)

    assert "allocation" in result
    assert "Search" in result["allocation"]
    assert "Video (Full Funnel)" in result["allocation"]
    total_allocated = sum(result["allocation"].values())
    assert total_allocated == pytest.approx(5000.0, rel=1e-3)
    assert result["expected_total_return"] > 0

  def test_plot_curve_comparison(self):
    """Test that plot_curve_comparison generates an uncluttered 2-panel figure cleanly."""
    proj_curve = self.base_model.project_capacity(m_beta=1.5, m_k=1.8)
    fig = proj_curve.plot_curve_comparison(target_mroas=1.0, current_spend=350.0, show=False)
    assert fig is not None
    assert len(fig.axes) == 2  # 2 panels: Saturation (left) and Marginal (right)

    ax1, ax2 = fig.axes
    assert "Media Saturation" in ax1.get_title(loc='left')
    assert "Marginal Return" in ax2.get_title(loc='left')

    # Verify target hurdle rate label is clean without raw numbers in parens
    _, labels2 = ax2.get_legend_handles_labels()
    assert "Target Hurdle Rate" in labels2
    for label in labels2:
      assert "(1.0)" not in label

    plt.close(fig)

  def test_vertical_benchmarks_dict_integration(self):
    """Test using vertical benchmarks passed explicitly as a dictionary."""
    sample_benchmarks = {
        "b2b_software": {
            "lower_only": {"m_beta": 1.00, "m_k": 1.00},
            "full_funnel": {"m_beta": 1.55, "m_k": 1.85},
            "rf_targets": {"target_frequency": 2.2, "target_reach_penetration": 0.40},
        }
    }
    proj = self.base_model.project_capacity(
        vertical="b2b_software",
        vertical_multipliers=sample_benchmarks,
        current_funnel="lower_only",
        target_funnel="full_funnel",
    )
    assert proj.multipliers.m_beta == pytest.approx(1.55, rel=1e-3)
    assert proj.multipliers.m_k == pytest.approx(1.85, rel=1e-3)
    assert proj.beta == pytest.approx(10000.0 * 1.55, rel=1e-3)
    assert proj.K == pytest.approx(500.0 * 1.85, rel=1e-3)
    # Checks that target_frequency from rf_targets was adopted (2.2)
    assert proj.multipliers.rf_details["target_frequency"] == 2.2

  def test_vertical_benchmarks_json_path(self, tmp_path):
    """Test loading vertical benchmarks from a portable JSON file path."""
    import json

    benchmarks_data = {
        "digital_services": {
            "lower_only": {"m_beta": 1.00, "m_k": 1.00},
            "full_funnel": {"m_beta": 1.80, "m_k": 2.35},
            "rf_targets": {"target_frequency": 3.0, "target_reach_penetration": 0.65},
        }
    }
    json_path = tmp_path / "benchmarks.json"
    json_path.write_text(json.dumps(benchmarks_data))

    proj = self.base_model.project_capacity(
        vertical="digital_services",
        vertical_multipliers=str(json_path),
        current_funnel="lower_only",
        target_funnel="full_funnel",
    )
    assert proj.multipliers.m_beta == pytest.approx(1.80, rel=1e-3)
    assert proj.multipliers.m_k == pytest.approx(2.35, rel=1e-3)
    assert proj.multipliers.rf_details["target_frequency"] == 3.0

  def test_benchmarks_not_in_core_package(self):
    """Verify that vertical-specific benchmarks are decoupled from the core library namespace."""
    import dxpoint

    assert not hasattr(dxpoint, "VERTICAL_BENCHMARKS")
