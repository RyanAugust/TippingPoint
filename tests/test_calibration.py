import numpy as np
import pytest
from tippingpoint import MarketingReturnCurve, MultiChannelMMM, PortfolioAllocator
from tippingpoint.math import hill_function
from tippingpoint.validation import format_validation_report, format_multichannel_validation_report

def test_bayesian_with_calibration_experiment():
    np.random.seed(42)
    x = np.linspace(1000, 50000, 20)
    beta_true = 100000
    alpha_true = 1.5
    K_true = 20000
    y = hill_function(x, beta_true, alpha_true, K_true) + np.random.normal(0, 1000, size=x.shape)

    # Add an experiment at spend = 25000 with lift known precisely
    exp_spend = 25000.0
    true_lift_at_exp = hill_function(exp_spend, beta_true, alpha_true, K_true)
    experiments = [
        {"spend": exp_spend, "lift": true_lift_at_exp, "se": 100.0}
    ]

    model = MarketingReturnCurve.fit_bayesian(
        spend_array=x,
        return_array=y,
        calibration_experiments=experiments,
        n_samples=200,
        burn_in=50,
        chains=2
    )

    pred_at_exp = model.predict_incremental_return(exp_spend)
    # With tight experimental SE=100, prediction should be extremely close to experiment lift
    assert abs(pred_at_exp - true_lift_at_exp) < 3000
    assert len(model.calibration_experiments) == 1

def test_model_with_baseline():
    model = MarketingReturnCurve(beta=50000, alpha=1.2, half_saturation_k=10000, baseline=12000)
    assert model.baseline == 12000

    # Incremental return only
    inc_return = model.predict_incremental_return(10000, include_baseline=False)
    # Total return including baseline
    tot_return = model.predict_incremental_return(10000, include_baseline=True)
    assert tot_return == inc_return + 12000

    # Summary includes baseline
    summary = model.summary()
    assert summary["parameters"]["baseline"] == 12000

def test_standalone_incrementality_validation():
    # Setup known curve: beta=1000, alpha=2.0, K=5000
    model = MarketingReturnCurve(beta=1000.0, alpha=2.0, half_saturation_k=5000.0, theta=0.0, channel_name="YouTube")

    # True lifts at 2000, 5000, 10000
    l_2k = hill_function(2000.0, 1000.0, 2.0, 5000.0)   # 137.93
    l_5k = hill_function(5000.0, 1000.0, 2.0, 5000.0)   # 500.0
    l_10k = hill_function(10000.0, 1000.0, 2.0, 5000.0) # 800.0

    exps = [
        {"name": "GeoTest_2k", "spend": 2000.0, "lift": l_2k + 5.0, "se": 10.0},
        {"name": "GeoTest_5k", "spend": 5000.0, "lift": l_5k - 10.0, "ci": (l_5k - 40.0, l_5k + 20.0)},
        {"name": "GeoTest_10k", "spend": 10000.0, "lift": l_10k + 15.0, "se": 20.0}
    ]

    report = model.validate_experiments(exps, verbose=True)

    assert report["channel"] == "YouTube"
    assert report["num_experiments"] == 3
    assert report["verdict"] in ["EXCELLENT", "ALIGNED"]
    assert report["ci_coverage_pct"] == 100.0
    assert report["chi2_reduced"] < 1.5
    assert len(report["experiments"]) == 3

    # Check individual experiment metrics
    exp1 = report["experiments"][0]
    assert exp1["name"] == "GeoTest_2k"
    assert abs(exp1["error"] - (-5.0)) < 1e-3
    assert exp1["in_95_ci"] is True
    assert exp1["z_score"] is not None

    # Test single experiment convenience method
    single_rep = model.validate_experiment(exps[0])
    assert single_rep["num_experiments"] == 1

    # Test formatted report string
    report_str = format_validation_report(report)
    assert "GeoTest_2k" in report_str
    assert "Overall Status" in report_str

def test_attach_experiments_and_validate_no_args():
    model = MarketingReturnCurve(beta=1000.0, alpha=2.0, half_saturation_k=5000.0, channel_name="Paid Search")
    l_5k = hill_function(5000.0, 1000.0, 2.0, 5000.0)

    # Attach via add_experiment and attach_experiments
    model.add_experiment(spend=5000.0, lift=l_5k, se=10.0, name="Search_Q1")
    model.attach_experiments([{"spend": 8000.0, "lift": hill_function(8000.0, 1000.0, 2.0, 5000.0), "se": 15.0, "name": "Search_Q2"}])

    assert len(model.calibration_experiments) == 2

    # Validate without args
    rep = model.validate_experiments()
    assert rep["num_experiments"] == 2
    assert rep["verdict"] == "EXCELLENT"

def test_validation_with_adstock_scaling():
    # Model with adstock theta=0.5 -> effective spend is 2x raw daily spend
    model = MarketingReturnCurve(beta=1000.0, alpha=2.0, half_saturation_k=10000.0, theta=0.5, channel_name="Video")

    # Raw spend 5000 -> effective spend 10000 -> lift = 500
    exp = {"spend": 5000.0, "lift": 500.0, "se": 15.0}

    rep = model.validate_experiments(exp, spend_is_raw=True)
    assert rep["experiments"][0]["effective_spend"] == pytest.approx(10000.0)
    assert rep["experiments"][0]["predicted_lift"] == pytest.approx(500.0)
    assert rep["verdict"] == "EXCELLENT"

def test_multichannel_validation_and_parallel_attachment():
    m1 = MarketingReturnCurve(beta=1000.0, alpha=2.0, half_saturation_k=5000.0, channel_name="Search")
    m2 = MarketingReturnCurve(beta=2000.0, alpha=1.5, half_saturation_k=8000.0, channel_name="Social")

    l_search = hill_function(4000.0, 1000.0, 2.0, 5000.0)
    l_social = hill_function(6000.0, 2000.0, 1.5, 8000.0)

    # 1. Parallel validation via Dict format
    mmm = MultiChannelMMM({"Search": m1, "Social": m2})
    dict_exps = {
        "Search": [{"name": "Search_Q1", "spend": 4000.0, "lift": l_search, "se": 10.0}],
        "Social": {"name": "Social_Q2", "spend": 6000.0, "lift": l_social, "se": 20.0}
    }
    report = mmm.validate_experiments(dict_exps, verbose=True)

    assert report["num_experiments"] == 2
    assert "Search" in report["channels"]
    assert "Social" in report["channels"]
    assert report["verdict"] == "EXCELLENT"
    assert report["ci_coverage_pct"] == 100.0

    # Format multi-channel report
    mc_str = format_multichannel_validation_report(report)
    assert "Search_Q1" in mc_str
    assert "Social_Q2" in mc_str

    # 2. Attach experiments in parallel to MMM
    mmm.attach_experiments(dict_exps)
    rep_attached = mmm.validate_experiments()
    assert rep_attached["num_experiments"] == 2

    # 3. Add single experiment to channel
    mmm.add_experiment(channel="Search", spend=2000.0, lift=hill_function(2000.0, 1000.0, 2.0, 5000.0), se=8.0, name="Search_Small")
    rep_updated = mmm.validate_experiments()
    assert rep_updated["num_experiments"] == 3

def test_portfolio_allocator_parallel_calibration():
    m1 = MarketingReturnCurve(beta=1000.0, alpha=2.0, half_saturation_k=5000.0, channel_name="Search")
    m2 = MarketingReturnCurve(beta=2000.0, alpha=1.5, half_saturation_k=8000.0, channel_name="Social")

    l_search = hill_function(4000.0, 1000.0, 2.0, 5000.0)
    m1.add_experiment(spend=4000.0, lift=l_search, se=10.0, name="Search_Exp")

    allocator = PortfolioAllocator([m1, m2])

    # Check calibration summary before m2 is calibrated
    summary_initial = allocator.get_calibration_summary()
    assert summary_initial["Search"]["verdict"] == "EXCELLENT"
    assert summary_initial["Social"]["verdict"] == "UNTESTED"

    # Attach experiment to Social in parallel via Allocator
    l_social = hill_function(6000.0, 2000.0, 1.5, 8000.0)
    allocator.add_experiment(channel="Social", spend=6000.0, lift=l_social, se=15.0, name="Social_Exp")

    summary_after = allocator.get_calibration_summary()
    assert summary_after["Search"]["verdict"] == "EXCELLENT"
    assert summary_after["Social"]["verdict"] == "EXCELLENT"

    # Validate portfolio across all channels
    p_rep = allocator.validate_experiments()
    assert p_rep["num_experiments"] == 2
    assert p_rep["verdict"] == "EXCELLENT"

def test_validation_error_handling():
    model = MarketingReturnCurve(beta=500.0, alpha=1.0, half_saturation_k=1000.0)

    # Empty list
    with pytest.raises(ValueError, match="No experiments"):
        model.validate_experiments([])

    # Invalid type
    with pytest.raises(TypeError):
        model.validate_experiments(12345)

    # Missing spend key
    with pytest.raises(KeyError, match="missing 'spend'"):
        model.validate_experiments([{"lift": 100}])

    # Missing lift key
    with pytest.raises(KeyError, match="missing 'lift'"):
        model.validate_experiments([{"spend": 100}])

    # Multi-channel missing channel key
    mmm = MultiChannelMMM({"Ch1": model})
    with pytest.raises(KeyError, match="'channel' key"):
        mmm.validate_experiments([{"spend": 100, "lift": 50}])

    # Multi-channel unknown channel
    with pytest.raises(ValueError, match="Channel 'Unknown' not found"):
        mmm.validate_experiments([{"channel": "Unknown", "spend": 100, "lift": 50}])
