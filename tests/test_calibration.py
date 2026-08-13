import numpy as np
import pytest
from tippingpoint import MarketingReturnCurve, MultiChannelMMM
from tippingpoint.math import hill_function

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
