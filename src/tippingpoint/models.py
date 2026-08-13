import numpy as np
import warnings
from .math import hill_function, hill_first_derivative, get_inflection_point
from .fitting.bayesian import fit_bayesian_mcmc
from .fitting.gradient import fit_mle_gradient
from .viz import CurveVisualizer

class MarketingReturnCurve:
  """A marketing intelligence tool to determine inflection points of a media response curve.

  Based on the Hill Function (Google Meridian methodology), this tool identifies
  the Minimal Marginal Cost Point (peak efficiency) and the Point of Diminishing
  Returns (profitability floor).
  """

  def __init__(self, beta, alpha, half_saturation_k, theta=0.0, channel_name="Generic", posterior_samples=None, baseline=0.0, adstock_type="geometric", adstock_params=None):
    self.beta = float(beta)
    self.alpha = float(alpha)
    self.K = float(half_saturation_k)
    self.theta = float(theta)
    self.baseline = float(baseline)
    self.channel_name = channel_name
    self.posterior_samples = posterior_samples
    self.adstock_type = adstock_type
    self.adstock_params = adstock_params or {}
    self.loss = 0.0
    self.tipping_points = {}
    self.calculate_tipping_points()

  @classmethod
  def fit_bayesian(cls, spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None, fit_baseline=False):
    beta, alpha, K, theta, samples = fit_bayesian_mcmc(
        spend_array, return_array, channel_name, priors, n_samples, chains, burn_in,
        adstock_type=adstock_type, adstock_bounds=adstock_bounds, adstock_fixed_days=adstock_fixed_days,
        calibration_experiments=calibration_experiments, fit_baseline=fit_baseline
    )
    print(f"[{channel_name}] Bayesian fit complete. Samples: {len(samples['beta'])}")
    baseline_val = float(np.mean(samples['baseline'])) if fit_baseline and 'baseline' in samples else 0.0
    return cls(beta, alpha, K, theta, channel_name, posterior_samples=samples, baseline=baseline_val)

  @classmethod
  def from_historical_data(cls, spend_array, return_array, channel_name="Generic", epochs=5000, lr=0.05, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, fit_baseline=False):
    res = fit_mle_gradient(
        spend_array, return_array, epochs, lr,
        adstock_type=adstock_type, adstock_bounds=adstock_bounds, adstock_fixed_days=adstock_fixed_days,
        fit_baseline=fit_baseline
    )
    if fit_baseline and len(res) == 6:
      beta, alpha, K, theta, loss, baseline_val = res
    else:
      beta, alpha, K, theta, loss = res[:5]
      baseline_val = 0.0

    print(f"[{channel_name}] Curve fit complete. Loss: {loss:.4f} (Theta: {theta:.4f})")
    model = cls(beta, alpha, K, theta, channel_name, baseline=baseline_val)
    model.update_loss(loss)
    return model

  def adstock_spend(self, spend_timeline):
    """Applies the model's fitted adstock decay (geometric or Weibull) to a timeline of spends."""
    if self.adstock_type in ["weibull_pdf", "weibull_cdf"]:
      from .math import weibull_adstock
      shape = self.adstock_params.get("shape", 1.5)
      scale = self.adstock_params.get("scale", 7.0)
      w_type = "pdf" if self.adstock_type == "weibull_pdf" else "cdf"
      return weibull_adstock(spend_timeline, shape=shape, scale=scale, adstock_type=w_type)
    from .math import geometric_adstock
    return geometric_adstock(spend_timeline, self.theta)

  def update_loss(self, loss: float) -> None:
    self.loss = float(loss)

  def calculate_tipping_points(self):
    """Pre-computes and caches key strategic inflection points."""
    self.tipping_points = {
      "max_efficiency_point": self.get_minimal_marginal_cost_point(),
      "max_profit_point": self.get_diminishing_returns_point(target_mroas=1.0, warn_unreachable=False)
    }

  @property
  def max_efficiency_point(self):
    return self.tipping_points.get("max_efficiency_point")

  @property
  def max_profit_point(self):
    return self.tipping_points.get("max_profit_point")

  def summary(self):
    half_life = 0.0
    if 0.0 < self.theta < 1.0:
      half_life = float(-np.log(2) / np.log(self.theta))
    elif self.theta >= 1.0:
      half_life = float('inf')
    return {
      "channel": self.channel_name,
      "parameters": {
        "beta": self.beta,
        "alpha": self.alpha,
        "K": self.K,
        "theta": self.theta,
        "baseline": self.baseline,
        "adstock_type": self.adstock_type,
        "adstock_params": self.adstock_params,
        "adstock_half_life_days": half_life
      },
      "tipping_points": self.tipping_points,
      "current_mroas_at_max_profit": self.predict_marginal_return(self.max_profit_point) if self.max_profit_point is not None else None
    }

  def predict_incremental_return(self, spend, use_samples=False, include_baseline=False):
    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      ret = hill_function(spend, beta, alpha, K)
    else:
      ret = hill_function(spend, self.beta, self.alpha, self.K)
    if include_baseline:
      ret = ret + self.baseline
    return ret

  def predict_marginal_return(self, spend, use_samples=False):
    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      return hill_first_derivative(spend, beta, alpha, K)
    return hill_first_derivative(spend, self.beta, self.alpha, self.K)

  def get_minimal_marginal_cost_point(self):
    return get_inflection_point(self.alpha, self.K)

  def get_diminishing_returns_point(self, target_mroas=1.0, tol=1e-5, max_iter=100, warn_unreachable=True):
    if target_mroas <= 0:
      if warn_unreachable:
        warnings.warn(f"Target mROAS ({target_mroas}) must be strictly positive.")
      return None

    if self.alpha > 1.0:
      inflection = self.get_minimal_marginal_cost_point()
      max_mroas = self.predict_marginal_return(inflection)
      if target_mroas >= max_mroas:
        if warn_unreachable:
          warnings.warn(f"Target mROAS ({target_mroas}) is mathematically unreachable.\nMax possible mROAS is {max_mroas:.2f}.")
        return None
      lower_bound = inflection
    elif self.alpha == 1.0:
      max_mroas = self.beta / self.K
      if target_mroas >= max_mroas:
        if warn_unreachable:
          warnings.warn(f"Target mROAS ({target_mroas}) is mathematically unreachable.\nMax possible mROAS is {max_mroas:.2f}.")
        return None
      lower_bound = 0.0
    else:
      lower_bound = 0.0

    # Exponential bracket expansion to find upper bound
    upper_bound = max(lower_bound + self.K, self.K, 1.0)
    while self.predict_marginal_return(upper_bound) > target_mroas:
      upper_bound *= 2.0
      if upper_bound > 1e15:
        if warn_unreachable:
          warnings.warn("Could not find an upper bound for the target mROAS.")
        return None

    # Binary search (Bisection)
    for _ in range(max_iter):
      midpoint = (lower_bound + upper_bound) / 2.0
      mroas_at_mid = self.predict_marginal_return(midpoint)
      if abs(mroas_at_mid - target_mroas) < tol:
        return float(midpoint)
      if mroas_at_mid > target_mroas:
        lower_bound = midpoint
      else:
        upper_bound = midpoint

    return float((lower_bound + upper_bound) / 2.0)

  def evaluate_current_budget(self, current_spend, target_mroas=1.0):
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    mroas = self.predict_marginal_return(current_spend)
    print(f"--- Budget Evaluation: {self.channel_name} ---")
    print(f"Current Spend: ${current_spend:,.2f} | Current mROAS: {mroas:.2f}")
    if min_spend > 0 and current_spend < min_spend:
      print(f"Status: WARMING UP (Inefficient)\nRecommendation: Increase spend to at least ${min_spend:,.2f} to reach peak acquisition efficiency.")
    elif max_spend is not None and current_spend > max_spend:
      print(f"Status: OVER-SATURATED (Unprofitable Marginal Growth)\n Recommendation: Scale back spend to ${max_spend:,.2f} to maintain target unit economics.")
    else:
      print("Status: OPTIMAL SCALING ZONE.\nRecommendation: You are operating within the highly efficient growth window.")

  def plot_response_curve(self, target_mroas=1.0, current_spend=None, show_intervals=True, scatter=None, show=True):
    fig = CurveVisualizer.plot_response_curve(self, target_mroas, current_spend, show_intervals, scatter)
    if show:
      import matplotlib.pyplot as plt
      plt.show()
    return fig

  def launch_dashboard(self):
    """Launches the interactive dashboard for this specific model instance."""
    import streamlit.web.cli as stcli
    import sys
    import os
    import tempfile
    import pickle

    # To pass THIS model instance to the dashboard, we'll use a temporary pickle file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
      pickle.dump(self, tmp)
      tmp_path = tmp.name

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")

    # We set an environment variable so the dashboard knows to load the specific model
    os.environ["TIPPINGPOINT_MODEL_PATH"] = tmp_path

    sys.argv = ["streamlit", "run", dashboard_path]
    stcli.main()


