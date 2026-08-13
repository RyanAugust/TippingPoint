import numpy as np
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
    mmm = MultiChannelMMM.from_historical_data(
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
