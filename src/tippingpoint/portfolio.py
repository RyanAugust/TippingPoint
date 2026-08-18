import numpy as np
from scipy.optimize import minimize

class PortfolioAllocator:
  """Optimizes budget allocation across multiple MarketingReturnCurve models."""

  def __init__(self, models):
    if not models:
      raise ValueError("At least one model must be provided.")
    self.models = models
    self.channel_names = [m.channel_name for m in models]

    # Ensure channel names are unique
    if len(set(self.channel_names)) != len(self.channel_names):
      raise ValueError("All models must have unique channel_names.")
    self.channels = {m.channel_name: m for m in models}

  def attach_experiments(self, experiments):
    """Associates incrementality experiments in parallel across portfolio channels.

    Args:
      experiments: Dict mapping {channel_name: [exps] | exp} or flat list of dicts with 'channel' key.
    """
    from .validation import normalize_multichannel_experiments
    mapping = normalize_multichannel_experiments(self, experiments)
    for ch_name, exp_list in mapping.items():
      if ch_name in self.channels:
        self.channels[ch_name].attach_experiments(exp_list)
    return self

  def add_experiment(self, channel, spend, lift, se=None, ci=None, name=None):
    """Convenience method to associate an incrementality test with a specific portfolio channel."""
    if channel not in self.channels:
      raise ValueError(f"Channel '{channel}' not found in portfolio channels: {self.channel_names}")
    exp = {"channel": channel, "spend": float(spend), "lift": float(lift)}
    if se is not None: exp["se"] = float(se)
    if ci is not None: exp["ci"] = (float(ci[0]), float(ci[1]))
    if name is not None: exp["name"] = str(name)
    return self.attach_experiments(exp)

  def validate_experiments(self, experiments=None, spend_is_raw=True, verbose=False):
    """Validates all portfolio channels against associated or provided incrementality experiments in parallel.

    Args:
      experiments: Optional dictionary or list of experiments. If None, uses experiments attached to channel curves.
      spend_is_raw: If True, converts raw daily spend to effective adstock where theta > 0.
      verbose: If True, prints a formatted validation report.

    Returns:
      dict: Multi-channel validation summary.
    """
    from .validation import validate_multichannel_experiments
    return validate_multichannel_experiments(self, experiments=experiments, spend_is_raw=spend_is_raw, verbose=verbose)

  def get_calibration_summary(self):
    """Returns a high-level summary of calibration alignment across all portfolio channels."""
    summary = {}
    for ch_name, model in self.channels.items():
      exps = getattr(model, "calibration_experiments", None)
      if exps:
        val = model.validate_experiments(verbose=False)
        summary[ch_name] = {
            "num_experiments": val["num_experiments"],
            "verdict": val["verdict"],
            "mape": val["mape"],
            "ci_coverage_pct": val["ci_coverage_pct"]
        }
      else:
        summary[ch_name] = {
            "num_experiments": 0,
            "verdict": "UNTESTED",
            "mape": None,
            "ci_coverage_pct": None
        }
    return summary

  def allocate_budget(self, total_budget, channel_bounds=None):
    """
    Finds the optimal spend distribution to maximize total return.

    Args:
      total_budget (float): Total budget to allocate.
      channel_bounds (dict, optional): Dictionary of (min_spend, max_spend) bounds
                       keyed by channel_name.

    Returns:
      dict: The optimal allocation, marginal ROAS, and expected return.
    """
    n = len(self.models)

    # Determine bounds for each channel
    bounds = []
    for model in self.models:
      b = (0.0, float(total_budget))
      if channel_bounds and model.channel_name in channel_bounds:
        provided_b = channel_bounds[model.channel_name]
        lb = float(provided_b[0])
        ub = min(float(provided_b[1]), float(total_budget))
        if lb > ub:
          ub = lb
        b = (lb, ub)
      bounds.append(b)

    def objective(spends):
      total_return = 0.0
      for i, model in enumerate(self.models):
        total_return += model.predict_incremental_return(spends[i])
      return -total_return

    def constraint(spends):
      return np.sum(spends) - total_budget

    cons = {'type': 'eq', 'fun': constraint}

    best_res = None
    best_return = float('inf')  # We are minimizing negative return

    # Start points: proportional, channel inflection points, and Dirichlet/random starts
    start_points = []

    # 1. Proportional start respecting lower bounds
    x0_prop = np.array([b[0] for b in bounds], dtype=float)
    rem_budget = total_budget - np.sum(x0_prop)
    if rem_budget > 0:
      x0_prop += rem_budget / n
    start_points.append(x0_prop)

    # 2. Inflection point anchored starts for S-curves
    for i, model in enumerate(self.models):
      x0_inf = np.array([b[0] for b in bounds], dtype=float)
      inf_pt = model.get_minimal_marginal_cost_point()
      if inf_pt > 0 and inf_pt <= bounds[i][1]:
        x0_inf[i] = max(x0_inf[i], inf_pt)
      rem = total_budget - np.sum(x0_inf)
      if rem > 0:
        x0_inf += rem / n
      for j in range(n):
        x0_inf[j] = np.clip(x0_inf[j], bounds[j][0], bounds[j][1])
      if np.abs(np.sum(x0_inf) - total_budget) < 1.0:
        start_points.append(x0_inf)

    # 3. Random Dirichlet starts
    for _ in range(10):
      raw_weights = np.random.exponential(scale=1.0, size=n)
      x_rand = np.array([b[0] for b in bounds], dtype=float)
      rem = total_budget - np.sum(x_rand)
      if rem > 0:
        x_rand += (raw_weights / np.sum(raw_weights)) * rem
      for i in range(n):
        x_rand[i] = np.clip(x_rand[i], bounds[i][0], bounds[i][1])
      start_points.append(x_rand)

    for x0 in start_points:
      res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': False, 'ftol': 1e-7, 'maxiter': 500}
      )

      # Strict budget check
      if np.abs(np.sum(res.x) - total_budget) < 1e-2 and res.fun < best_return:
        best_return = res.fun
        best_res = res

    if best_res is None:
      best_res = res

    is_success = bool(best_res.success and (np.abs(np.sum(best_res.x) - total_budget) < 1e-2))
    allocation = {self.models[i].channel_name: float(best_res.x[i]) for i in range(n)}
    mroas = {self.models[i].channel_name: float(self.models[i].predict_marginal_return(best_res.x[i])) for i in range(n)}
    channel_returns = {self.models[i].channel_name: float(self.models[i].predict_incremental_return(best_res.x[i])) for i in range(n)}
    expected_return = -float(best_res.fun)

    return {
      "total_budget": total_budget,
      "expected_total_return": expected_return,
      "overall_roas": expected_return / total_budget if total_budget > 0 else 0.0,
      "allocation": allocation,
      "marginal_roas": mroas,
      "marginal_roas_at_allocation": mroas,
      "channel_returns": channel_returns,
      "success": is_success,
      "message": best_res.message if is_success else "Optimization failed to satisfy constraints (e.g. impossible bounds)."
    }


