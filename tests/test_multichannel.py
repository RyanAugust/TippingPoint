import numpy as np
import pandas as pd
import pytest
from tippingpoint import MarketingReturnCurve, MultiChannelMMM
from tippingpoint.math import hill_function

@pytest.fixture
def multichannel_data():
    np.random.seed(42)
    T = 50
    spend_search = np.linspace(500, 10000, T)
    spend_social = np.linspace(1000, 20000, T)
    spend_yt = np.random.uniform(500, 8000, T)

    baseline_true = 5000.0
    r_search = hill_function(spend_search, beta=20000, alpha=1.2, K=5000)
    r_social = hill_function(spend_social, beta=35000, alpha=1.5, K=10000)
    r_yt = hill_function(spend_yt, beta=15000, alpha=1.0, K=4000)

    y_total = baseline_true + r_search + r_social + r_yt + np.random.normal(0, 500, T)
    y_total = np.maximum(y_total, 0)

    spend_dict = {
        "Search": spend_search,
        "Social": spend_social,
        "YouTube": spend_yt
    }
    return spend_dict, y_total

def test_multichannel_gradient_fit(multichannel_data):
    spend_dict, y_total = multichannel_data
    mmm = MultiChannelMMM.fit_gradient_descent(
        spend_data=spend_dict,
        return_array=y_total,
        epochs=500,
        lr=0.05,
        fit_baseline=True
    )

    assert "Search" in mmm.channels
    assert "Social" in mmm.channels
    assert "YouTube" in mmm.channels
    assert mmm.baseline > 0
    assert mmm.channels["Search"].beta > 0

    # Decomposition
    contribs = mmm.predict_channel_contributions({"Search": 5000, "Social": 10000, "YouTube": 4000})
    assert "Baseline" in contribs
    assert "Search" in contribs
    assert contribs["Baseline"] == mmm.baseline

    # PortfolioAllocator integration
    allocator = mmm.get_allocator()
    assert len(allocator.models) == 3
    result = allocator.allocate_budget(total_budget=30000)
    assert "allocation" in result



def test_multichannel_bayesian_fit(multichannel_data):
    spend_dict, y_total = multichannel_data
    mmm = MultiChannelMMM.fit_bayesian(
        spend_data=spend_dict,
        return_array=y_total,
        n_samples=100,
        chains=2,
        burn_in=50,
        fit_baseline=True
    )

    assert mmm.baseline > 0
    assert len(mmm.channels) == 3
    for name, model in mmm.channels.items():
        assert model.posterior_samples is not None
        assert 'beta' in model.posterior_samples

    total_pred = mmm.predict_total_return({"Search": 5000, "Social": 10000, "YouTube": 4000})
    assert total_pred > mmm.baseline

def test_multichannel_dataframe_and_array_inputs(multichannel_data):
    import pandas as pd
    spend_dict, y_total = multichannel_data
    df_spend = pd.DataFrame(spend_dict)

    # Test DataFrame input
    mmm_df = MultiChannelMMM.fit_gradient_descent(
        spend_data=df_spend,
        return_array=y_total,
        epochs=100,
        adstock_types="none"
    )
    assert len(mmm_df.channels) == 3

    # Test 2D numpy array input
    mat_spend = df_spend.values
    mmm_mat = MultiChannelMMM.fit_gradient_descent(
        spend_data=mat_spend,
        return_array=y_total,
        channel_names=["ChA", "ChB", "ChC"],
        epochs=100
    )
    assert "ChA" in mmm_mat.channels
    assert "ChB" in mmm_mat.channels
    assert "ChC" in mmm_mat.channels

def test_multichannel_adstock_and_calibration(multichannel_data):
    spend_dict, y_total = multichannel_data

    # Bounded & fixed adstock
    adstock_types = {"Search": "bounded", "Social": "fixed", "YouTube": "free"}
    adstock_bounds = {"Search": (1.0, 7.0)}
    adstock_fixed_days = {"Social": 4.0}

    calib_experiments = [
        {"channel": "Search", "spend": 5000.0, "lift": 6000.0, "ci": (4500.0, 7500.0)},
        {"channel": "Social", "spend": 8000.0, "lift": 10000.0, "se": 400.0}
    ]

    mmm = MultiChannelMMM.fit_bayesian(
        spend_data=spend_dict,
        return_array=y_total,
        n_samples=50,
        chains=2,
        burn_in=20,
        fit_baseline=True,
        adstock_types=adstock_types,
        adstock_bounds=adstock_bounds,
        adstock_fixed_days=adstock_fixed_days,
        calibration_experiments=calib_experiments
    )

    summary = mmm.summary()
    assert "baseline" in summary
    assert "channels" in summary
    assert "Search" in summary["channels"]

def test_multichannel_init_variants():
    m1 = MarketingReturnCurve(beta=10000, alpha=1.2, half_saturation_k=3000, channel_name="M1")
    m2 = MarketingReturnCurve(beta=20000, alpha=1.4, half_saturation_k=5000, channel_name="M2")

    mmm_list = MultiChannelMMM([m1, m2], baseline=1000.0)
    assert len(mmm_list.channels) == 2
    assert mmm_list.baseline == 1000.0

    with pytest.raises(ValueError):
        MultiChannelMMM(channels="invalid_type")

def test_multichannel_historical_decomposition(multichannel_data):
    spend_dict, y_total = multichannel_data
    mmm = MultiChannelMMM.fit_gradient_descent(
        spend_data=spend_dict,
        return_array=y_total,
        epochs=100,
        fit_baseline=True
    )

    # Test time-series array predictions
    pred_ts = mmm.predict_total_return(spend_dict)
    assert len(pred_ts) == len(y_total)
    assert np.all(pred_ts > 0)

    contrib_ts = mmm.predict_channel_contributions(spend_dict)
    assert "Baseline" in contrib_ts
    assert len(contrib_ts["Baseline"]) == len(y_total)
    assert "Search" in contrib_ts

    # Test full historical attribution table
    decomp = mmm.decompose_historical_contributions(spend_dict, return_array=y_total)
    assert "contributions_df" in decomp
    assert "summary_table" in decomp
    assert "total_predicted" in decomp

    summary_df = decomp["summary_table"]
    assert "Channel" in summary_df.columns
    assert "Share of Spend (%)" in summary_df.columns
    assert "Share of Return (%)" in summary_df.columns
    assert "ROI" in summary_df.columns
    assert len(summary_df) == 4  # Baseline + 3 channels

def test_multichannel_geo_hierarchical_bayesian():
    np.random.seed(42)
    T = 20
    geos = ["US_West", "US_East"]
    geo_data = []

    for g in geos:
      s_search = np.linspace(1000, 5000, T)
      s_social = np.linspace(2000, 8000, T)
      ret = 3000 + hill_function(s_search, 10000, 1.2, 3000) + hill_function(s_social, 15000, 1.3, 5000)
      for t in range(T):
        geo_data.append({
          "geo": g,
          "week": t,
          "Search": s_search[t],
          "Social": s_social[t],
          "revenue": ret[t]
        })

    df_geo = pd.DataFrame(geo_data)
    y_geo = df_geo["revenue"].values

    mmm_geo = MultiChannelMMM.fit_bayesian(
        spend_data=df_geo,
        return_array=y_geo,
        n_samples=30,
        chains=2,
        burn_in=10,
        hierarchical=True
    )

    assert len(mmm_geo.channels) == 2
    assert mmm_geo.posterior_samples is not None
    assert "diagnostics" in mmm_geo.posterior_samples
    assert mmm_geo.posterior_samples["diagnostics"]["is_geo"] is True

def test_multichannel_unified_fit(multichannel_data):
    spend_dict, y_total = multichannel_data
    # Gradient method
    mmm1 = MultiChannelMMM.fit(spend_dict, y_total, method="gradient", epochs=50)
    assert len(mmm1.channels) == 3

    # Bayesian method
    mmm2 = MultiChannelMMM.fit(spend_dict, y_total, method="bayesian", n_samples=20, chains=1, burn_in=5)
    assert len(mmm2.channels) == 3

    # Invalid method
    with pytest.raises(ValueError, match="Unknown multi-channel fitting method"):
        MultiChannelMMM.fit(spend_dict, y_total, method="invalid_engine")



