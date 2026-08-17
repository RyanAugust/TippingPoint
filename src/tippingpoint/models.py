import numpy as np
import warnings
from .math import hill_function, hill_first_derivative, get_inflection_point
from .fitting.bayesian import fit_bayesian_mcmc
from .fitting.gradient import fit_mle_gradient
from .fitting.frequentist import fit_frequentist_nls
from .viz import CurveVisualizer

class MarketingReturnCurve:
  """A marketing intelligence tool to determine inflection points of a media response curve.

  Based on the Hill Function (Google Meridian methodology), this tool identifies
  the Minimal Marginal Cost Point (peak efficiency) and the Point of Diminishing
  Returns (profitability floor).
  """

  def __init__(
      self,
      beta,
      alpha,
      half_saturation_k,
      theta=0.0,
      channel_name="Generic",
      posterior_samples=None,
      baseline=0.0,
      adstock_type="geometric",
      adstock_params=None,
      standard_errors=None,
      confidence_intervals=None,
      covariance_matrix=None,
      train_spend=None,
      train_return=None
  ):
    self.beta = float(beta)
    self.alpha = float(alpha)
    self.K = float(half_saturation_k)
    self.theta = float(theta)
    self.baseline = float(baseline)
    self.channel_name = channel_name
    self.posterior_samples = posterior_samples
    self.adstock_type = adstock_type
    self.adstock_params = adstock_params or {}
    self.standard_errors = standard_errors
    self.confidence_intervals = confidence_intervals
    self.covariance_matrix = covariance_matrix
    self._train_spend = np.asarray(train_spend, dtype=float) if train_spend is not None else None
    self._train_return = np.asarray(train_return, dtype=float) if train_return is not None else None
    self.loss = 0.0
    self.tipping_points = {}
    self.calculate_tipping_points()

  @classmethod
  def fit(
      cls,
      spend_array,
      return_array,
      channel_name="Generic",
      method="auto",
      adstock_type="none",
      adstock_bounds=None,
      adstock_fixed_days=None,
      fit_baseline=False,
      confidence_level=0.95,
      priors=None,
      n_samples=2000,
      chains=4,
      burn_in=1000,
      calibration_experiments=None,
      epochs=5000,
      lr=0.05
  ):
    """Unified entry point for fitting saturation and adstock curves to historical media data.

    Supported methods:
    - 'auto': Selects 'bayesian' if calibration_experiments/priors are given, else 'frequentist'.
    - 'frequentist' (or 'nls'): Non-Linear Least Squares with parameter standard errors and confidence intervals.
    - 'gradient_descent' (or 'gradient', 'mle'): Gradient descent optimization via Tinygrad.
    - 'bayesian' (or 'mcmc'): Metropolis-Hastings MCMC with posterior sampling and experimental calibration.
    """
    method_norm = method.lower() if isinstance(method, str) else "auto"

    if method_norm == "auto":
      if calibration_experiments is not None or priors is not None:
        method_norm = "bayesian"
      else:
        method_norm = "frequentist"

    if method_norm in ["frequentist", "nls"]:
      return cls.fit_frequentist(
          spend_array=spend_array,
          return_array=return_array,
          channel_name=channel_name,
          adstock_type=adstock_type,
          adstock_bounds=adstock_bounds,
          adstock_fixed_days=adstock_fixed_days,
          fit_baseline=fit_baseline,
          confidence_level=confidence_level
      )
    elif method_norm in ["gradient", "gradient_descent", "mle"]:
      return cls.fit_gradient_descent(
          spend_array=spend_array,
          return_array=return_array,
          channel_name=channel_name,
          epochs=epochs,
          lr=lr,
          adstock_type=adstock_type,
          adstock_bounds=adstock_bounds,
          adstock_fixed_days=adstock_fixed_days,
          fit_baseline=fit_baseline
      )
    elif method_norm in ["bayesian", "mcmc"]:
      return cls.fit_bayesian(
          spend_array=spend_array,
          return_array=return_array,
          channel_name=channel_name,
          priors=priors,
          n_samples=n_samples,
          chains=chains,
          burn_in=burn_in,
          adstock_type=adstock_type,
          adstock_bounds=adstock_bounds,
          adstock_fixed_days=adstock_fixed_days,
          calibration_experiments=calibration_experiments,
          fit_baseline=fit_baseline
      )
    else:
      raise ValueError(
          f"Unknown fitting method: '{method}'. Supported methods are: 'auto', 'frequentist', 'gradient_descent', 'bayesian'."
      )

  @classmethod
  def fit_bayesian(cls, spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None, fit_baseline=False):
    beta, alpha, K, theta, samples = fit_bayesian_mcmc(
        spend_array, return_array, channel_name, priors, n_samples, chains, burn_in,
        adstock_type=adstock_type, adstock_bounds=adstock_bounds, adstock_fixed_days=adstock_fixed_days,
        calibration_experiments=calibration_experiments, fit_baseline=fit_baseline
    )
    print(f"[{channel_name}] Bayesian fit complete. Samples: {len(samples['beta'])}")
    baseline_val = float(np.mean(samples['baseline'])) if fit_baseline and 'baseline' in samples else 0.0
    return cls(
        beta=beta,
        alpha=alpha,
        half_saturation_k=K,
        theta=theta,
        channel_name=channel_name,
        posterior_samples=samples,
        baseline=baseline_val,
        train_spend=spend_array,
        train_return=return_array
    )

  @classmethod
  def fit_frequentist(
      cls,
      spend_array,
      return_array,
      channel_name="Generic",
      adstock_type="none",
      adstock_bounds=None,
      adstock_fixed_days=None,
      fit_baseline=False,
      confidence_level=0.95
  ):
    """Fits a Hill Curve to historical data using Frequentist Non-Linear Least Squares (NLS)."""
    res = fit_frequentist_nls(
        spend_array,
        return_array,
        channel_name=channel_name,
        adstock_type=adstock_type,
        adstock_bounds=adstock_bounds,
        adstock_fixed_days=adstock_fixed_days,
        fit_baseline=fit_baseline,
        confidence_level=confidence_level
    )
    print(f"[{channel_name}] Frequentist NLS fit complete. Loss: {res['loss']:.4f} (Theta: {res['theta']:.4f})")
    model = cls(
        beta=res["beta"],
        alpha=res["alpha"],
        half_saturation_k=res["K"],
        theta=res["theta"],
        channel_name=channel_name,
        baseline=res["baseline"],
        standard_errors=res["standard_errors"],
        confidence_intervals=res["confidence_intervals"],
        covariance_matrix=res["covariance_matrix"],
        train_spend=spend_array,
        train_return=return_array
    )
    model.update_loss(res["loss"])
    return model

  @classmethod
  def fit_gradient_descent(cls, spend_array, return_array, channel_name="Generic", epochs=5000, lr=0.05, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, fit_baseline=False):
    """Fits a Hill Curve to historical data using Gradient Descent (Tinygrad Adam)."""
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
    model = cls(
        beta=beta,
        alpha=alpha,
        half_saturation_k=K,
        theta=theta,
        channel_name=channel_name,
        baseline=baseline_val,
        train_spend=spend_array,
        train_return=return_array
    )
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
    res = {
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
    if hasattr(self, "loss"):
      res["loss"] = self.loss
    if self.standard_errors is not None:
      res["standard_errors"] = self.standard_errors
    if self.confidence_intervals is not None:
      res["confidence_intervals"] = self.confidence_intervals
    return res

  def evaluate_fit(self, spend_array=None, return_array=None, verbose=False):
    """Evaluates statistical goodness-of-fit metrics (R², Adj R², RMSE, MAE, MAPE, AIC, BIC)."""
    from .evaluation import evaluate_curve_fit
    if spend_array is None or return_array is None:
      if self._train_spend is None or self._train_return is None:
        raise ValueError("spend_array and return_array must be provided (no cached training data).")
      x, y = self._train_spend, self._train_return
    else:
      x, y = spend_array, return_array
    return evaluate_curve_fit(self, x, y, verbose=verbose)

  def get_optimal_scaling_window(self, target_mroas=1.0):
    """Returns the tuple (min_spend, max_spend) defining the optimal scaling zone."""
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    return (min_spend, max_spend)

  def _predict_delta_method(self, spend, is_derivative=False, confidence_level=0.95, include_baseline=False):
    """Computes predictions and confidence intervals via the Delta Method on the parameter covariance matrix."""
    spend_arr = np.asanyarray(spend, dtype=float)
    is_scalar = spend_arr.ndim == 0

    if is_derivative:
      y_pred = hill_first_derivative(spend_arr, self.beta, self.alpha, self.K)
    else:
      y_pred = hill_function(spend_arr, self.beta, self.alpha, self.K)
      if include_baseline:
        y_pred = y_pred + self.baseline

    if self.covariance_matrix is None:
      if is_scalar:
        return float(y_pred), float(y_pred), float(y_pred)
      return y_pred, y_pred.copy(), y_pred.copy()

    param_names = list(self.covariance_matrix.keys())
    p_num = len(param_names)
    cov_mat = np.zeros((p_num, p_num))
    for i, n1 in enumerate(param_names):
      for j, n2 in enumerate(param_names):
        cov_mat[i, j] = self.covariance_matrix[n1].get(n2, 0.0)

    p_current = {
        "beta": self.beta,
        "alpha": self.alpha,
        "K": self.K,
        "baseline": self.baseline,
        "theta": self.theta
    }

    def eval_func(p_dict):
      b = p_dict.get("beta", self.beta)
      a = p_dict.get("alpha", self.alpha)
      k = p_dict.get("K", self.K)
      base = p_dict.get("baseline", self.baseline) if include_baseline else 0.0
      if is_derivative:
        return hill_first_derivative(spend_arr, b, a, k)
      else:
        return base + hill_function(spend_arr, b, a, k)

    eps = 1e-5
    spend_flat = np.atleast_1d(spend_arr)
    n_pts = len(spend_flat)
    G = np.zeros((n_pts, p_num))

    for j, name in enumerate(param_names):
      p_plus = dict(p_current)
      p_minus = dict(p_current)
      h = eps * (abs(p_current[name]) + 1e-4)
      p_plus[name] += h
      p_minus[name] -= h
      val_plus = np.atleast_1d(eval_func(p_plus))
      val_minus = np.atleast_1d(eval_func(p_minus))
      G[:, j] = (val_plus - val_minus) / (2.0 * h)

    var_pred = np.sum((G @ cov_mat) * G, axis=1)
    se_pred = np.sqrt(np.maximum(0.0, var_pred))

    from scipy.stats import norm
    alpha_ci = 1.0 - confidence_level
    z_crit = float(norm.ppf(1.0 - alpha_ci / 2.0))

    y_low = np.atleast_1d(y_pred) - z_crit * se_pred
    y_high = np.atleast_1d(y_pred) + z_crit * se_pred

    if not is_derivative:
      y_low = np.maximum(0.0, y_low)

    if is_scalar:
      return float(y_pred), float(y_low[0]), float(y_high[0])
    return y_pred, y_low.reshape(spend_arr.shape), y_high.reshape(spend_arr.shape)

  def predict_incremental_return(self, spend, return_interval=False, confidence_level=0.95, use_samples=False, include_baseline=False):
    spend_arr = np.asanyarray(spend, dtype=float)
    if return_interval:
      if self.posterior_samples:
        ret_dist = np.nan_to_num(self.predict_incremental_return(spend_arr, use_samples=True, include_baseline=include_baseline), nan=0.0)
        alpha_ci = (1.0 - confidence_level) / 2.0
        axis = 0 if spend_arr.ndim > 0 else None
        point = np.mean(ret_dist, axis=axis)
        low = np.percentile(ret_dist, alpha_ci * 100.0, axis=axis)
        high = np.percentile(ret_dist, (1.0 - alpha_ci) * 100.0, axis=axis)
        if spend_arr.ndim == 0:
          return float(point), float(low), float(high)
        return point, low, high
      elif self.covariance_matrix is not None:
        return self._predict_delta_method(spend_arr, is_derivative=False, confidence_level=confidence_level, include_baseline=include_baseline)
      else:
        pt = self.predict_incremental_return(spend_arr, use_samples=False, include_baseline=include_baseline)
        if spend_arr.ndim == 0:
          return float(pt), float(pt), float(pt)
        return pt, pt.copy(), pt.copy()

    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      ret = hill_function(spend_arr, beta, alpha, K)
      if include_baseline and 'baseline' in self.posterior_samples:
        base = self.posterior_samples['baseline'][:, np.newaxis]
        ret = ret + base
    else:
      ret = hill_function(spend_arr, self.beta, self.alpha, self.K)
      if include_baseline:
        ret = ret + self.baseline
    return ret

  def predict_marginal_return(self, spend, return_interval=False, confidence_level=0.95, use_samples=False):
    spend_arr = np.asanyarray(spend, dtype=float)
    if return_interval:
      if self.posterior_samples:
        mroas_dist = np.nan_to_num(self.predict_marginal_return(spend_arr, use_samples=True), nan=0.0)
        alpha_ci = (1.0 - confidence_level) / 2.0
        axis = 0 if spend_arr.ndim > 0 else None
        point = np.mean(mroas_dist, axis=axis)
        low = np.percentile(mroas_dist, alpha_ci * 100.0, axis=axis)
        high = np.percentile(mroas_dist, (1.0 - alpha_ci) * 100.0, axis=axis)
        if spend_arr.ndim == 0:
          return float(point), float(low), float(high)
        return point, low, high
      elif self.covariance_matrix is not None:
        return self._predict_delta_method(spend_arr, is_derivative=True, confidence_level=confidence_level)
      else:
        pt = self.predict_marginal_return(spend_arr, use_samples=False)
        if spend_arr.ndim == 0:
          return float(pt), float(pt), float(pt)
        return pt, pt.copy(), pt.copy()

    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      return hill_first_derivative(spend_arr, beta, alpha, K)
    return hill_first_derivative(spend_arr, self.beta, self.alpha, self.K)

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

  def validate_experiments(self, experiments, spend_is_raw=True, verbose=False):
    """Validates this response curve against one or more incrementality experiments.

    Args:
      experiments: Single experiment dict or list of dicts containing 'spend', 'lift',
                   and optional 'se' (standard error) or 'ci' (confidence interval).
      spend_is_raw: If True and model has theta > 0, converts raw test spend to effective
                    adstocked spend via S_eff = S_raw / (1 - theta).
      verbose: If True, prints a formatted validation report to stdout.

    Returns:
      dict: Detailed validation metrics including errors, Z-scores, CI coverage, chi2, and verdict.
    """
    from .validation import validate_curve_experiments
    return validate_curve_experiments(self, experiments, spend_is_raw=spend_is_raw, verbose=verbose)

  def validate_experiment(self, experiment, spend_is_raw=True, verbose=False):
    """Convenience alias for validating a single incrementality experiment."""
    return self.validate_experiments(experiment, spend_is_raw=spend_is_raw, verbose=verbose)

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
