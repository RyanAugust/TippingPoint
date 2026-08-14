import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist
from tippingpoint.math import geometric_adstock, hill_function

def fit_frequentist_nls(
    spend_array,
    return_array,
    channel_name="Generic",
    adstock_type="none",
    adstock_bounds=None,
    adstock_fixed_days=None,
    fit_baseline=False,
    confidence_level=0.95
):
  """Fits a Hill Curve to historical data using Frequentist Non-Linear Least Squares (NLS).

  Computes point estimates, residual variance, parameter covariance matrix,
  standard errors, and Frequentist confidence intervals.
  """
  x = np.array(spend_array, dtype=float)
  y = np.array(return_array, dtype=float)
  n_obs = len(x)

  max_x = float(np.max(x)) if np.any(x > 0) else 1.0
  max_x = 1.0 if max_x <= 0 else max_x
  max_y = float(np.max(y)) if np.any(y > 0) else 1.0
  max_y = 1.0 if max_y <= 0 else max_y

  x_scaled = x / max_x
  y_scaled = y / max_y
  median_x_scaled = float(np.median(x_scaled[x_scaled > 0])) if np.any(x_scaled > 0) else 0.5

  # Determine theta constraints
  fixed_theta = 0.0
  theta_min, theta_max = 0.0, 0.999
  if adstock_type == "fixed":
    if adstock_fixed_days is not None and adstock_fixed_days > 0:
      fixed_theta = 0.5 ** (1.0 / adstock_fixed_days)
  elif adstock_type == "bounded":
    if adstock_bounds is not None:
      min_days, max_days = adstock_bounds
      theta_min = 0.5 ** (1.0 / min_days) if min_days > 0 else 0.0
      theta_max = 0.5 ** (1.0 / max_days) if max_days > 0 else 0.0
      if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min

  has_adstock_param = adstock_type in ["free", "bounded"]

  # Parameter vector: [log_beta, log_k, log_alpha, (log_baseline if fit), (theta_transformed if adstock)]
  # Initial guesses in scaled space
  init_params = [
      np.log(1.2),                        # log_beta
      np.log(median_x_scaled + 1e-4),     # log_k
      0.0                                 # log_alpha (alpha=1.0)
  ]
  bounds = [
      (np.log(1e-4), np.log(100.0)),      # log_beta
      (np.log(1e-5), np.log(50.0)),       # log_k
      (np.log(0.1), np.log(10.0))         # log_alpha (alpha in [0.1, 10.0])
  ]

  base_idx = None
  if fit_baseline:
    base_idx = len(init_params)
    init_params.append(np.log(0.05))
    bounds.append((np.log(1e-6), np.log(5.0)))

  theta_idx = None
  if has_adstock_param:
    theta_idx = len(init_params)
    init_params.append(0.0)  # sigmoid transform -> mid value
    bounds.append((-10.0, 10.0))

  def unpack_params(p):
    beta_scaled = np.exp(p[0])
    k_scaled = np.exp(p[1])
    alpha = np.exp(p[2])
    baseline_scaled = np.exp(p[base_idx]) if fit_baseline else 0.0

    if adstock_type == "free":
      sig = 1.0 / (1.0 + np.exp(-np.clip(p[theta_idx], -30, 30)))
      theta = float(0.999 * sig)
    elif adstock_type == "bounded":
      sig = 1.0 / (1.0 + np.exp(-np.clip(p[theta_idx], -30, 30)))
      theta = float(theta_min + (theta_max - theta_min) * sig)
    elif adstock_type == "fixed":
      theta = fixed_theta
    else:
      theta = 0.0

    return beta_scaled, k_scaled, alpha, baseline_scaled, theta

  def objective(p):
    beta_s, k_s, alpha, base_s, theta = unpack_params(p)
    x_ad = geometric_adstock(x_scaled, theta) if theta > 0 else x_scaled
    y_pred_s = base_s + hill_function(x_ad, beta_s, alpha, k_s)
    res = y_scaled - y_pred_s
    return np.sum(res ** 2)

  # Run L-BFGS-B optimization
  res_opt = minimize(objective, init_params, bounds=bounds, method="L-BFGS-B")

  p_opt = res_opt.x
  beta_s, k_s, alpha, base_s, theta = unpack_params(p_opt)

  # Rescale to unscaled physical space
  beta = float(beta_s * max_y)
  K = float(k_s * max_x)
  alpha = float(alpha)
  baseline = float(base_s * max_y)
  theta = float(theta)

  # Compute unscaled residuals and MSE
  x_adstocked = geometric_adstock(x, theta) if theta > 0 else x
  y_pred = baseline + hill_function(x_adstocked, beta, alpha, K)
  residuals = y - y_pred
  ssr = float(np.sum(residuals ** 2))

  param_names = ["beta", "alpha", "K"]
  if fit_baseline:
    param_names.append("baseline")
  if has_adstock_param:
    param_names.append("theta")

  p_num = len(param_names)
  dof = max(1, n_obs - p_num)
  mse = ssr / dof

  # Compute Jacobian numerical derivative in unscaled physical space for covariance matrix
  def predict_from_unscaled(p_vec):
    b_val, a_val, k_val = p_vec[0], p_vec[1], p_vec[2]
    base_val = p_vec[3] if fit_baseline else 0.0
    th_val = p_vec[4] if (has_adstock_param and len(p_vec) > 4) else (fixed_theta if adstock_type == "fixed" else 0.0)

    x_ad = geometric_adstock(x, th_val) if th_val > 0 else x
    return base_val + hill_function(x_ad, b_val, a_val, k_val)

  p_unscaled = [beta, alpha, K]
  if fit_baseline:
    p_unscaled.append(baseline)
  if has_adstock_param:
    p_unscaled.append(theta)
  p_unscaled = np.array(p_unscaled, dtype=float)

  J = np.zeros((n_obs, p_num))
  eps = 1e-5
  for j in range(p_num):
    p_plus = p_unscaled.copy()
    p_plus[j] = p_plus[j] + eps * (abs(p_plus[j]) + 1e-4)
    p_minus = p_unscaled.copy()
    p_minus[j] = p_minus[j] - eps * (abs(p_minus[j]) + 1e-4)

    h = p_plus[j] - p_minus[j]
    J[:, j] = (predict_from_unscaled(p_plus) - predict_from_unscaled(p_minus)) / h

  # Covariance matrix: cov = mse * inv(J^T * J)
  JTJ = J.T @ J
  try:
    cov = mse * np.linalg.inv(JTJ)
  except np.linalg.LinAlgError:
    cov = mse * np.linalg.pinv(JTJ)

  std_errs = np.sqrt(np.maximum(0.0, np.diag(cov)))

  # Student-t critical value for confidence intervals
  alpha_ci = 1.0 - confidence_level
  t_crit = float(t_dist.ppf(1.0 - alpha_ci / 2.0, df=dof))

  confidence_intervals = {}
  standard_errors = {}
  cov_matrix = {}

  for j, name in enumerate(param_names):
    se = float(std_errs[j])
    val = float(p_unscaled[j])
    ci_low = val - t_crit * se
    ci_high = val + t_crit * se
    standard_errors[name] = se
    confidence_intervals[name] = (float(ci_low), float(ci_high))

  for j1, n1 in enumerate(param_names):
    cov_matrix[n1] = {}
    for j2, n2 in enumerate(param_names):
      cov_matrix[n1][n2] = float(cov[j1, j2])

  return {
      "beta": beta,
      "alpha": alpha,
      "K": K,
      "theta": theta,
      "baseline": baseline,
      "loss": ssr,
      "mse": mse,
      "standard_errors": standard_errors,
      "confidence_intervals": confidence_intervals,
      "covariance_matrix": cov_matrix,
      "dof": dof,
      "confidence_level": confidence_level
  }
