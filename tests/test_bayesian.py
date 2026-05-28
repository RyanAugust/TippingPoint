import numpy as np
import pytest
from tippingpoint.curve import MarketingReturnCurve

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    x = np.linspace(1000, 50000, 20)
    beta_true = 100000
    alpha_true = 1.5
    K_true = 20000
    y_true = (beta_true * (x ** alpha_true)) / (K_true ** alpha_true + x ** alpha_true)
    y = y_true + np.random.normal(0, 1000, size=x.shape)
    y = np.maximum(y, 0)
    return x, y

def test_fit_bayesian_basic(synthetic_data):
    x, y = synthetic_data
    model = MarketingReturnCurve.fit_bayesian(x, y, n_samples=500, burn_in=100, chains=2)
    
    assert model.beta > 0
    assert model.alpha > 0
    assert model.K > 0
    assert model.posterior_samples is not None
    assert 'beta' in model.posterior_samples
    assert len(model.posterior_samples['beta']) == 1000 # 500 * 2

def test_fit_bayesian_with_priors(synthetic_data):
    x, y = synthetic_data
    priors = {
        'beta': (np.log(100000), 0.1),
        'alpha': (np.log(1.5), 0.1),
        'K': (np.log(20000), 0.1)
    }
    model = MarketingReturnCurve.fit_bayesian(x, y, priors=priors, n_samples=200, burn_in=50, chains=1)
    
    # Check if results are close to true values due to tight priors
    assert 90000 < model.beta < 110000
    assert 1.3 < model.alpha < 1.7
    assert 18000 < model.K < 22000

def test_predict_with_samples(synthetic_data):
    x, y = synthetic_data
    model = MarketingReturnCurve.fit_bayesian(x, y, n_samples=100, burn_in=50, chains=1)
    
    # Incremental return
    preds = model.predict_incremental_return([1000, 2000], use_samples=True)
    assert preds.shape == (100, 2)
    
    # Marginal return
    m_preds = model.predict_marginal_return([1000, 2000], use_samples=True)
    assert m_preds.shape == (100, 2)

def test_plot_with_samples(synthetic_data):
    # Just ensure it doesn't crash
    x, y = synthetic_data
    model = MarketingReturnCurve.fit_bayesian(x, y, n_samples=50, burn_in=10, chains=1)
    # Mock plt.show to avoid blocking
    import matplotlib.pyplot as plt
    plt.show = lambda: None
    model.plot_response_curve()
