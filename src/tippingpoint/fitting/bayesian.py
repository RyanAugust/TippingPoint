import numpy as np
from tippingpoint.math import geometric_adstock, hill_function

def fit_bayesian_mcmc(spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None, fit_baseline=False):
  """Fits a Hill Curve using Bayesian MCMC (Metropolis-Hastings in transformed space) with optional adstock, baseline, and experimental calibration."""
  x = np.array(spend_array, dtype=float)
  y = np.array(return_array, dtype=float)

  max_y = float(np.max(y)) if np.any(y > 0) else 1.0
  if max_y <= 0:
    max_y = 1.0
  median_x = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
  if median_x <= 0:
    median_x = 1.0

  # Default Priors (LogNormal)
  if priors is None:
    priors = {
      'beta': (np.log(max_y * 1.2), 0.5),
      'alpha': (0.0, 0.5),
      'K': (np.log(median_x), 0.5)
    }

  # Adstock setup
  fixed_theta = 0.0
  theta_min, theta_max = 0.0, 0.999

  if adstock_type == "fixed":
    fixed_theta = 0.5 ** (1.0 / adstock_fixed_days) if adstock_fixed_days is not None and adstock_fixed_days > 0 else 0.0
  elif adstock_type == "bounded":
    if adstock_bounds is not None:
      min_days, max_days = adstock_bounds
      theta_min = 0.5 ** (1.0 / min_days) if min_days > 0 else 0.0
      theta_max = 0.5 ** (1.0 / max_days) if max_days > 0 else 0.0
      if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min

  has_adstock_param = adstock_type in ["free", "bounded"]
  # Parameters: [beta, alpha, K, sigma, (baseline if fit), (theta if free/bounded)]
  num_params = 4 + (1 if fit_baseline else 0) + (1 if has_adstock_param else 0)

  base_idx = 4 if fit_baseline else None
  theta_idx = (4 + (1 if fit_baseline else 0)) if has_adstock_param else None

  def params_from_transformed(psi):
    beta = float(np.exp(psi[0]))
    alpha = float(np.exp(psi[1]))
    k = float(np.exp(psi[2]))
    sigma = float(np.exp(psi[3]))
    baseline = float(np.exp(psi[base_idx])) if fit_baseline else 0.0

    if adstock_type == "free":
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[theta_idx], -30, 30)))
      theta = float(0.999 * sig)
    elif adstock_type == "bounded":
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[theta_idx], -30, 30)))
      theta = float(theta_min + (theta_max - theta_min) * sig)
    elif adstock_type == "fixed":
      theta = fixed_theta
    else:
      theta = 0.0
    return beta, alpha, k, sigma, baseline, theta

  def log_prior(psi):
    lp = 0.0
    for idx, name in enumerate(['beta', 'alpha', 'K']):
      mu, s = priors[name]
      lp += -0.5 * ((psi[idx] - mu) / s) ** 2

    # Half-normal prior on sigma with Jacobian adjustment
    sigma_scale = max_y * 0.1
    sigma = np.exp(psi[3])
    lp += -0.5 * (sigma / sigma_scale) ** 2 + psi[3]

    if fit_baseline:
      base_scale = max_y * 0.2
      base_val = np.exp(psi[base_idx])
      lp += -0.5 * (base_val / base_scale) ** 2 + psi[base_idx]

    # Uniform prior on theta with sigmoid Jacobian adjustment
    if has_adstock_param:
      lp += -np.logaddexp(0.0, psi[theta_idx]) - np.logaddexp(0.0, -psi[theta_idx])
    return lp

  def log_likelihood(beta, alpha, k, sigma, baseline, theta):
    if sigma <= 0 or beta <= 0 or alpha <= 0 or k <= 0:
      return -np.inf

    if theta > 0:
      x_adstocked = geometric_adstock(x, theta)
    else:
      x_adstocked = x

    y_pred = baseline + hill_function(x_adstocked, beta, alpha, k)
    residuals = (y - y_pred) / sigma
    ll = -0.5 * np.sum(residuals ** 2) - len(y) * np.log(sigma)

    # Experimental lift calibration likelihood penalty
    if calibration_experiments:
      for exp in calibration_experiments:
        exp_spend = float(exp["spend"])
        exp_lift = float(exp["lift"])
        exp_se = float(exp.get("se", 0.0))
        if exp_se <= 0 and "ci" in exp:
          ci_low, ci_high = exp["ci"]
          exp_se = (float(ci_high) - float(ci_low)) / 3.92
        if exp_se > 0:
          pred_lift = hill_function(exp_spend, beta, alpha, k)
          ll += -0.5 * ((pred_lift - exp_lift) / exp_se) ** 2

    return ll

  def log_posterior(psi):
    beta, alpha, k, sigma, baseline, theta = params_from_transformed(psi)
    return log_likelihood(beta, alpha, k, sigma, baseline, theta) + log_prior(psi)

  # Initialize chains
  init_sigma = max(float(np.std(y) * 0.1), 1e-4)
  init_list = [
    priors['beta'][0],
    priors['alpha'][0],
    priors['K'][0],
    np.log(init_sigma)
  ]
  if fit_baseline:
    init_list.append(np.log(max_y * 0.05))
  if has_adstock_param:
    init_list.append(0.0)

  init_psi = np.array(init_list)

  all_samples = []
  total_accepted = 0
  total_proposals = 0

  for _ in range(chains):
    curr_psi = init_psi + np.random.normal(0, 0.05, size=num_params)
    curr_log_post = log_posterior(curr_psi)
    step_size = np.full(num_params, 0.02)

    chain_samples = []
    window_accepted = 0
    adapt_window = 10

    for i in range(n_samples + burn_in):
      proposal_psi = curr_psi + np.random.normal(0, step_size)
      prop_log_post = log_posterior(proposal_psi)

      accepted = False
      if prop_log_post > curr_log_post:
        accepted = True
      elif not np.isnan(prop_log_post):
        log_u = np.log(np.random.rand())
        if log_u < (prop_log_post - curr_log_post):
          accepted = True

      if accepted:
        curr_psi = proposal_psi
        curr_log_post = prop_log_post
        window_accepted += 1
        if i >= burn_in:
          total_accepted += 1

      if i >= burn_in:
        total_proposals += 1
        beta_i, alpha_i, k_i, sigma_i, base_i, theta_i = params_from_transformed(curr_psi)
        chain_samples.append([beta_i, alpha_i, k_i, sigma_i, base_i, theta_i])

      # Adaptive step size during burn-in
      if i < burn_in and (i + 1) % adapt_window == 0:
        acc_rate = window_accepted / adapt_window
        if acc_rate > 0.35:
          step_size *= 1.2
        elif acc_rate < 0.20:
          step_size *= 0.8
        step_size = np.clip(step_size, 0.0005, 0.5)
        window_accepted = 0

    all_samples.append(np.array(chain_samples))

  def compute_rhat(chain_array):
    m, n = chain_array.shape
    if m < 2 or n < 2:
      return 1.0
    chain_means = np.mean(chain_array, axis=1)
    overall_mean = np.mean(chain_means)
    b = (n / (m - 1)) * np.sum((chain_means - overall_mean) ** 2)
    chain_vars = np.var(chain_array, axis=1, ddof=1)
    w = np.mean(chain_vars)
    if w == 0:
      return 1.0
    var_hat = ((n - 1) / n) * w + (1 / n) * b
    return float(np.sqrt(max(var_hat / w, 1.0)))

  posterior = np.vstack(all_samples)
  chains_tensor = np.array(all_samples)

  r_hats = {
    'beta': compute_rhat(chains_tensor[:, :, 0]),
    'alpha': compute_rhat(chains_tensor[:, :, 1]),
    'K': compute_rhat(chains_tensor[:, :, 2]),
    'sigma': compute_rhat(chains_tensor[:, :, 3]),
    'baseline': compute_rhat(chains_tensor[:, :, 4]),
    'theta': compute_rhat(chains_tensor[:, :, 5]),
  }

  overall_acc_rate = float(total_accepted / max(total_proposals, 1))

  samples_dict = {
    'beta': posterior[:, 0],
    'alpha': posterior[:, 1],
    'K': posterior[:, 2],
    'sigma': posterior[:, 3],
    'baseline': posterior[:, 4],
    'theta': posterior[:, 5],
    'diagnostics': {
      'acceptance_rate': overall_acc_rate,
      'r_hat': r_hats
    }
  }

  beta_mean = float(np.mean(samples_dict['beta']))
  alpha_mean = float(np.mean(samples_dict['alpha']))
  K_mean = float(np.mean(samples_dict['K']))
  theta_mean = float(np.mean(samples_dict['theta']))

  return beta_mean, alpha_mean, K_mean, theta_mean, samples_dict

def fit_multichannel_bayesian_mcmc(spend_data, return_array, channel_names=None, n_samples=2000, chains=4, burn_in=1000, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None):
  """Fits a joint Multi-Channel Marketing Mix Model using Bayesian MCMC with optional experimental calibration."""
  from tippingpoint.models import MarketingReturnCurve

  if isinstance(spend_data, dict):
    channels = list(spend_data.keys())
    spend_dict = {c: np.array(spend_data[c], dtype=float) for c in channels}
  elif hasattr(spend_data, 'values') and hasattr(spend_data, 'columns'):
    channels = list(spend_data.columns)
    spend_dict = {c: np.array(spend_data[c].values, dtype=float) for c in channels}
  else:
    spend_mat = np.array(spend_data, dtype=float)
    if channel_names is None:
      channels = [f"Channel_{i+1}" for i in range(spend_mat.shape[1])]
    else:
      channels = list(channel_names)
    spend_dict = {channels[i]: spend_mat[:, i] for i in range(len(channels))}

  M = len(channels)
  y = np.array(return_array, dtype=float)
  max_y = float(np.max(y)) if np.any(y > 0) else 1.0
  if max_y <= 0:
    max_y = 1.0

  if adstock_types is None:
    adstock_types_dict = {c: "none" for c in channels}
  elif isinstance(adstock_types, str):
    adstock_types_dict = {c: adstock_types for c in channels}
  else:
    adstock_types_dict = adstock_types

  # Adstock setup per channel
  fixed_thetas = {}
  bounds_dict = {}
  adstock_param_channels = []

  for c in channels:
    ad_type = adstock_types_dict.get(c, "none")
    if ad_type == "fixed":
      days = adstock_fixed_days.get(c, 3.0) if isinstance(adstock_fixed_days, dict) else (adstock_fixed_days or 3.0)
      fixed_thetas[c] = 0.5 ** (1.0 / days) if days > 0 else 0.0
    elif ad_type == "bounded":
      b = adstock_bounds.get(c, (1.0, 14.0)) if isinstance(adstock_bounds, dict) else (adstock_bounds or (1.0, 14.0))
      t_min = 0.5 ** (1.0 / b[0]) if b[0] > 0 else 0.0
      t_max = 0.5 ** (1.0 / b[1]) if b[1] > 0 else 0.0
      if t_min > t_max:
        t_min, t_max = t_max, t_min
      bounds_dict[c] = (t_min, t_max)
      adstock_param_channels.append(c)
    elif ad_type == "free":
      bounds_dict[c] = (0.0, 0.999)
      adstock_param_channels.append(c)
    else:
      fixed_thetas[c] = 0.0

  # Parameter layout:
  # [beta_0...M-1, alpha_0...M-1, K_0...M-1, sigma, (baseline if fit), (theta for free/bounded channels)]
  priors_dict = {}
  init_params = []
  for c in channels:
    s_arr = spend_dict[c]
    med_x = float(np.median(s_arr[s_arr > 0])) if np.any(s_arr > 0) else 1.0
    priors_dict[c] = {
      'beta': (np.log((max_y * 1.2) / max(M, 1)), 0.5),
      'alpha': (0.0, 0.5),
      'K': (np.log(med_x), 0.5)
    }

  for c in channels:
    init_params.extend([priors_dict[c]['beta'][0], priors_dict[c]['alpha'][0], priors_dict[c]['K'][0]])

  init_sigma = max(float(np.std(y) * 0.1), 1e-4)
  init_params.append(np.log(init_sigma))

  if fit_baseline:
    init_params.append(np.log(max_y * 0.1))

  for _ in adstock_param_channels:
    init_params.append(0.0)

  num_params = len(init_params)
  sigma_idx = 3 * M
  base_idx = (3 * M + 1) if fit_baseline else None
  theta_start_idx = (3 * M + (2 if fit_baseline else 1))

  def params_from_transformed(psi):
    betas = {}
    alphas = {}
    ks = {}
    for i, c in enumerate(channels):
      betas[c] = float(np.exp(psi[3 * i]))
      alphas[c] = float(np.exp(psi[3 * i + 1]))
      ks[c] = float(np.exp(psi[3 * i + 2]))

    sigma = float(np.exp(psi[sigma_idx]))
    baseline = float(np.exp(psi[base_idx])) if fit_baseline else 0.0

    thetas = {}
    for i, c in enumerate(adstock_param_channels):
      idx = theta_start_idx + i
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[idx], -30, 30)))
      t_min, t_max = bounds_dict[c]
      thetas[c] = float(t_min + (t_max - t_min) * sig)

    for c in channels:
      if c not in thetas:
        thetas[c] = fixed_thetas[c]

    return betas, alphas, ks, sigma, baseline, thetas

  def log_prior(psi):
    lp = 0.0
    for i, c in enumerate(channels):
      mu_b, s_b = priors_dict[c]['beta']
      mu_a, s_a = priors_dict[c]['alpha']
      mu_k, s_k = priors_dict[c]['K']
      lp += -0.5 * ((psi[3 * i] - mu_b) / s_b) ** 2
      lp += -0.5 * ((psi[3 * i + 1] - mu_a) / s_a) ** 2
      lp += -0.5 * ((psi[3 * i + 2] - mu_k) / s_k) ** 2

    sigma = np.exp(psi[sigma_idx])
    lp += -0.5 * (sigma / (max_y * 0.1)) ** 2 + psi[sigma_idx]

    if fit_baseline:
      base_val = np.exp(psi[base_idx])
      lp += -0.5 * (base_val / (max_y * 0.2)) ** 2 + psi[base_idx]

    for i, _ in enumerate(adstock_param_channels):
      idx = theta_start_idx + i
      lp += -np.logaddexp(0.0, psi[idx]) - np.logaddexp(0.0, -psi[idx])
    return lp

  def log_likelihood(betas, alphas, ks, sigma, baseline, thetas):
    if sigma <= 0:
      return -np.inf
    for c in channels:
      if betas[c] <= 0 or alphas[c] <= 0 or ks[c] <= 0:
        return -np.inf

    y_pred = np.full_like(y, baseline)
    for c in channels:
      s_arr = spend_dict[c]
      th = thetas[c]
      s_ad = geometric_adstock(s_arr, th) if th > 0 else s_arr
      y_pred = y_pred + hill_function(s_ad, betas[c], alphas[c], ks[c])

    residuals = (y - y_pred) / sigma
    ll = -0.5 * np.sum(residuals ** 2) - len(y) * np.log(sigma)

    if calibration_experiments:
      for exp in calibration_experiments:
        c = exp.get("channel")
        if c in channels:
          exp_spend = float(exp["spend"])
          exp_lift = float(exp["lift"])
          exp_se = float(exp.get("se", 0.0))
          if exp_se <= 0 and "ci" in exp:
            ci_l, ci_h = exp["ci"]
            exp_se = (float(ci_h) - float(ci_l)) / 3.92
          if exp_se > 0:
            pred_lift = hill_function(exp_spend, betas[c], alphas[c], ks[c])
            ll += -0.5 * ((pred_lift - exp_lift) / exp_se) ** 2

    return ll

  def log_posterior(psi):
    betas, alphas, ks, sigma, baseline, thetas = params_from_transformed(psi)
    return log_likelihood(betas, alphas, ks, sigma, baseline, thetas) + log_prior(psi)

  all_samples = []
  total_accepted = 0
  total_proposals = 0

  for _ in range(chains):
    curr_psi = np.array(init_params) + np.random.normal(0, 0.05, size=num_params)
    curr_log_post = log_posterior(curr_psi)
    step_size = np.full(num_params, 0.02)

    chain_samples = []
    window_accepted = 0
    adapt_window = 10

    for i in range(n_samples + burn_in):
      prop_psi = curr_psi + np.random.normal(0, step_size)
      prop_log_post = log_posterior(prop_psi)

      accepted = False
      if prop_log_post > curr_log_post:
        accepted = True
      elif not np.isnan(prop_log_post):
        log_u = np.log(np.random.rand())
        if log_u < (prop_log_post - curr_log_post):
          accepted = True

      if accepted:
        curr_psi = prop_psi
        curr_log_post = prop_log_post
        window_accepted += 1
        if i >= burn_in:
          total_accepted += 1

      if i >= burn_in:
        total_proposals += 1
        betas_i, alphas_i, ks_i, sigma_i, base_i, thetas_i = params_from_transformed(curr_psi)
        row = []
        for c in channels:
          row.extend([betas_i[c], alphas_i[c], ks_i[c], thetas_i[c]])
        row.extend([sigma_i, base_i])
        chain_samples.append(row)

      if i < burn_in and (i + 1) % adapt_window == 0:
        acc_rate = window_accepted / adapt_window
        if acc_rate > 0.35:
          step_size *= 1.2
        elif acc_rate < 0.20:
          step_size *= 0.8
        step_size = np.clip(step_size, 0.0005, 0.5)
        window_accepted = 0

    all_samples.append(np.array(chain_samples))

  posterior = np.vstack(all_samples)
  models_dict = {}
  channel_samples_dict = {}

  for idx, c in enumerate(channels):
    b_samples = posterior[:, 4 * idx]
    a_samples = posterior[:, 4 * idx + 1]
    k_samples = posterior[:, 4 * idx + 2]
    th_samples = posterior[:, 4 * idx + 3]
    samples_c = {
      'beta': b_samples,
      'alpha': a_samples,
      'K': k_samples,
      'theta': th_samples
    }
    model = MarketingReturnCurve(
      beta=float(np.mean(b_samples)),
      alpha=float(np.mean(a_samples)),
      half_saturation_k=float(np.mean(k_samples)),
      theta=float(np.mean(th_samples)),
      channel_name=c,
      posterior_samples=samples_c
    )
    models_dict[c] = model
    channel_samples_dict[c] = samples_c

  baseline_mean = float(np.mean(posterior[:, -1])) if fit_baseline else 0.0

  full_samples = {
    'channels': channel_samples_dict,
    'baseline': posterior[:, -1] if fit_baseline else np.zeros(len(posterior)),
    'sigma': posterior[:, -2],
    'diagnostics': {
      'acceptance_rate': float(total_accepted / max(total_proposals, 1))
    }
  }

  return models_dict, baseline_mean, full_samples
