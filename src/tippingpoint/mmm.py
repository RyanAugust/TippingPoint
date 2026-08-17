import numpy as np
import pandas as pd
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
from tippingpoint.math import geometric_adstock, weibull_adstock, hill_function, hill_first_derivative
from tippingpoint.fitting.gradient import tinygrad_geometric_adstock
from tippingpoint.models import MarketingReturnCurve


def _parse_spend_input(spend_data, channel_names=None):
  """Helper to standardize spend input into a dict of {channel_name: 1D np.ndarray} or {geo: {channel: array}}."""
  geos = None
  if isinstance(spend_data, pd.DataFrame):
    if 'geo' in spend_data.columns or 'region' in spend_data.columns:
      geo_col = 'geo' if 'geo' in spend_data.columns else 'region'
      geos = list(spend_data[geo_col].unique())
      channels = [c for c in spend_data.columns if c not in [geo_col, 'date', 'period', 'week', 'return', 'revenue', 'target']]
      geo_spend_dict = {}
      for g in geos:
        sub_df = spend_data[spend_data[geo_col] == g]
        geo_spend_dict[g] = {c: np.array(sub_df[c].values, dtype=float) for c in channels}
      return geo_spend_dict, channels, geos
    else:
      channels = list(spend_data.columns)
      spend_dict = {c: np.array(spend_data[c].values, dtype=float) for c in channels}
      return spend_dict, channels, None
  elif isinstance(spend_data, dict):
    # Check if nested dict for geos: {geo_name: {channel_name: array}}
    first_val = next(iter(spend_data.values()))
    if isinstance(first_val, dict):
      geos = list(spend_data.keys())
      channels = list(first_val.keys())
      geo_spend_dict = {g: {c: np.array(spend_data[g][c], dtype=float) for c in channels} for g in geos}
      return geo_spend_dict, channels, geos
    else:
      channels = list(spend_data.keys())
      spend_dict = {c: np.array(spend_data[c], dtype=float) for c in channels}
      return spend_dict, channels, None
  else:
    spend_mat = np.array(spend_data, dtype=float)
    if channel_names is None:
      channels = [f"Channel_{i+1}" for i in range(spend_mat.shape[1])]
    else:
      channels = list(channel_names)
    spend_dict = {channels[i]: spend_mat[:, i] for i in range(len(channels))}
    return spend_dict, channels, None


def fit_multichannel_gradient(spend_data, return_array, channel_names=None, epochs=5000, lr=0.05,
                              fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None):
  """Fits a joint Multi-Channel Marketing Mix Model using Gradient Descent (Tinygrad Adam)."""
  parsed, channels, _ = _parse_spend_input(spend_data, channel_names)
  if isinstance(parsed, dict) and any(isinstance(v, dict) for v in parsed.values()):
    # If geo data, aggregate spend across geos for global gradient fit
    spend_dict = {c: sum(parsed[g][c] for g in parsed) for c in channels}
  else:
    spend_dict = parsed

  M = len(channels)
  return_arr = np.array(return_array, dtype=float)
  max_y = float(np.max(return_arr)) if np.any(return_arr > 0) else 1.0
  if max_y <= 0:
    max_y = 1.0
  y_scaled = Tensor(return_arr / max_y, dtype=dtypes.float32)
  y_scaled.requires_grad = False

  max_x_dict = {}
  x_scaled_tensors = {}
  optimizable_params = []
  channel_param_tensors = {}

  if adstock_types is None:
    adstock_types_dict = {c: "none" for c in channels}
  elif isinstance(adstock_types, str):
    adstock_types_dict = {c: adstock_types for c in channels}
  else:
    adstock_types_dict = adstock_types

  for c in channels:
    s_arr = spend_dict[c]
    max_x = float(np.max(s_arr)) if np.any(s_arr > 0) else 1.0
    if max_x <= 0:
      max_x = 1.0
    max_x_dict[c] = max_x
    s_scaled = s_arr / max_x
    med_x = float(np.median(s_scaled[s_scaled > 0])) if np.any(s_scaled > 0) else 0.5

    t_x = Tensor(s_scaled, dtype=dtypes.float32)
    t_x.requires_grad = False
    x_scaled_tensors[c] = t_x

    log_beta = Tensor([np.log(1.0 / max(M, 1))], dtype=dtypes.float32)
    log_beta.requires_grad = True
    log_k = Tensor([np.log(med_x + 1e-5)], dtype=dtypes.float32)
    log_k.requires_grad = True
    log_alpha = Tensor([0.0], dtype=dtypes.float32)
    log_alpha.requires_grad = True

    params_c = {'beta': log_beta, 'k': log_k, 'alpha': log_alpha}
    optimizable_params.extend([log_beta, log_k, log_alpha])

    ad_type = adstock_types_dict.get(c, "none")
    if ad_type == "free":
      w = Tensor([0.0], dtype=dtypes.float32)
      w.requires_grad = True
      params_c['adstock_w'] = w
      optimizable_params.append(w)
    elif ad_type == "bounded":
      b = adstock_bounds.get(c, (1.0, 14.0)) if isinstance(adstock_bounds, dict) else (adstock_bounds or (1.0, 14.0))
      theta_min = 0.5 ** (1.0 / b[0]) if b[0] > 0 else 0.0
      theta_max = 0.5 ** (1.0 / b[1]) if b[1] > 0 else 0.0
      if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min
      w = Tensor([0.0], dtype=dtypes.float32)
      w.requires_grad = True
      params_c['adstock_w'] = w
      params_c['theta_bounds'] = (theta_min, theta_max)
      optimizable_params.append(w)
    elif ad_type == "fixed":
      days = adstock_fixed_days.get(c, 3.0) if isinstance(adstock_fixed_days, dict) else (adstock_fixed_days or 3.0)
      th = 0.5 ** (1.0 / days) if days > 0 else 0.0
      params_c['theta_fixed'] = Tensor([th], dtype=dtypes.float32)

    channel_param_tensors[c] = params_c

  log_baseline = None
  if fit_baseline:
    log_baseline = Tensor([np.log(0.1)], dtype=dtypes.float32)
    log_baseline.requires_grad = True
    optimizable_params.append(log_baseline)

  optimizer = Adam(optimizable_params, lr=lr)

  Tensor.training = True
  prev_loss = float('inf')
  with Tensor.train():
    for epoch in range(epochs):
      optimizer.zero_grad()
      y_pred = log_baseline.exp() if fit_baseline else Tensor([0.0], dtype=dtypes.float32)
      for c in channels:
        p = channel_param_tensors[c]
        x_c = x_scaled_tensors[c]
        ad_type = adstock_types_dict.get(c, "none")
        if ad_type == "free":
          th = p['adstock_w'].sigmoid() * 0.999
          x_ad = tinygrad_geometric_adstock(x_c, th)
        elif ad_type == "bounded":
          t_min, t_max = p['theta_bounds']
          th = t_min + (t_max - t_min) * p['adstock_w'].sigmoid()
          x_ad = tinygrad_geometric_adstock(x_c, th)
        elif ad_type == "fixed":
          x_ad = tinygrad_geometric_adstock(x_c, p['theta_fixed'])
        else:
          x_ad = x_c

        ratio = (x_ad + 1e-5) / p['k'].exp()
        ratio_alpha = ratio ** p['alpha'].exp()
        y_pred = y_pred + (p['beta'].exp() * ratio_alpha) / (1.0 + ratio_alpha)

      loss = ((y_pred - y_scaled) ** 2).mean()
      loss.backward()
      optimizer.step()

      if epochs >= 500 and epoch % 100 == 0:
        curr_loss = loss.numpy().item()
        if abs(prev_loss - curr_loss) < 1e-8:
          break
        prev_loss = curr_loss

  Tensor.training = False

  baseline_val = float(log_baseline.exp().numpy().item() * max_y) if fit_baseline else 0.0
  final_loss = float(loss.numpy().item() * (max_y ** 2))

  models_dict = {}
  for c in channels:
    p = channel_param_tensors[c]
    beta_c = float(p['beta'].exp().numpy().item() * max_y)
    alpha_c = float(p['alpha'].exp().numpy().item())
    k_c = float(p['k'].exp().numpy().item() * max_x_dict[c])
    ad_type = adstock_types_dict.get(c, "none")
    if ad_type == "free":
      theta_c = float((p['adstock_w'].sigmoid() * 0.999).numpy().item())
    elif ad_type == "bounded":
      t_min, t_max = p['theta_bounds']
      theta_c = float((t_min + (t_max - t_min) * p['adstock_w'].sigmoid()).numpy().item())
    elif ad_type == "fixed":
      theta_c = float(p['theta_fixed'].numpy().item())
    else:
      theta_c = 0.0

    model = MarketingReturnCurve(beta=beta_c, alpha=alpha_c, half_saturation_k=k_c, theta=theta_c, channel_name=c, baseline=0.0)
    model.update_loss(final_loss)
    models_dict[c] = model

  return models_dict, baseline_val, final_loss


def fit_multichannel_hierarchical_bayesian(spend_data, return_array, channel_names=None, n_samples=2000,
                                           chains=4, burn_in=1000, fit_baseline=True, hierarchical=True,
                                           adstock_types=None, adstock_bounds=None, adstock_fixed_days=None,
                                           calibration_experiments=None):
  """Fits a Meridian-lite Hierarchical Bayesian Marketing Mix Model.

  Features:
    1. Hierarchical shrinkage (partial pooling) across channels for capacity (beta),
       S-curve steepness (alpha), half-saturation (K), and carryover decay (theta).
    2. Optional Geo-level hierarchical partial pooling when geo/regional data is provided.
    3. Joint simultaneous MCMC estimation of carryover adstock, Hill saturation, baseline,
       and channel coefficients in transformed unconstrained parameter space.
    4. Experimental lift calibration seamlessly integrated into the joint log-likelihood.
    5. Convergence diagnostics including Gelman-Rubin R-hat and acceptance rates.
  """
  parsed, channels, geos = _parse_spend_input(spend_data, channel_names)
  is_geo = geos is not None and len(geos) > 1

  M = len(channels)
  G = len(geos) if is_geo else 1

  if is_geo:
    # return_array can be a dict {geo: array} or 2D array (G, T)
    if isinstance(return_array, dict):
      y_geo_dict = {g: np.array(return_array[g], dtype=float) for g in geos}
    else:
      y_arr = np.array(return_array, dtype=float)
      if y_arr.ndim == 2:
        y_geo_dict = {geos[i]: y_arr[i, :] for i in range(G)}
      else:
        # 1D flattened array corresponding to geo stacked
        T = len(parsed[geos[0]][channels[0]])
        y_geo_dict = {geos[i]: y_arr[i * T:(i + 1) * T] for i in range(G)}
    max_y = max(float(np.max(y_geo_dict[g])) for g in geos)
    geo_spend_dict = parsed
    # Aggregated spend for priors
    spend_dict = {c: sum(geo_spend_dict[g][c] for g in geos) for c in channels}
  else:
    spend_dict = parsed
    y_total = np.array(return_array, dtype=float)
    max_y = float(np.max(y_total)) if np.any(y_total > 0) else 1.0

  if max_y <= 0:
    max_y = 1.0

  if adstock_types is None:
    adstock_types_dict = {c: "none" for c in channels}
  elif isinstance(adstock_types, str):
    adstock_types_dict = {c: adstock_types for c in channels}
  else:
    adstock_types_dict = adstock_types

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

  ref_k_dict = {}
  for c in channels:
    s_arr = spend_dict[c]
    med_x = float(np.median(s_arr[s_arr > 0])) if np.any(s_arr > 0) else 1.0
    ref_k_dict[c] = med_x

  # Parameter indices in unconstrained space:
  # Channels: 3 * M parameters [log_beta_c, log_alpha_c, log_rel_k_c] where K_c = ref_k_c * exp(log_rel_k_c)
  # Adstock: len(adstock_param_channels) parameters [logit_theta_c]
  # Noise: log_sigma
  # Baseline: log_baseline (if fit_baseline)
  # Hierarchical hyperpriors (if hierarchical):
  #   [mu_beta, log_sigma_beta, mu_alpha, log_sigma_alpha, mu_k, log_sigma_k]
  #   if adstock params: [mu_theta, log_sigma_theta]
  # Geo random effects (if is_geo):
  #   G * M parameters [delta_geo_m] and log_sigma_geo

  init_params = []
  for c in channels:
    init_params.extend([np.log((max_y * 1.2) / max(M, 1)), 0.0, 0.0])

  for _ in adstock_param_channels:
    init_params.append(0.0)

  init_sigma = max(float(np.std(return_array) * 0.1), 1e-4)
  init_params.append(np.log(init_sigma))

  if fit_baseline:
    init_params.append(np.log(max_y * 0.1))

  hier_start_idx = len(init_params)
  if hierarchical:
    # mu_beta, log_sigma_beta
    init_params.extend([np.log((max_y * 1.2) / max(M, 1)), np.log(0.5)])
    # mu_alpha, log_sigma_alpha
    init_params.extend([0.0, np.log(0.3)])
    # mu_k, log_sigma_k
    init_params.extend([0.0, np.log(0.5)])
    if adstock_param_channels:
      # mu_theta, log_sigma_theta
      init_params.extend([0.0, np.log(0.5)])

  geo_start_idx = len(init_params)
  if is_geo:
    # G * M geo multipliers and log_sigma_geo
    init_params.extend([0.0] * (G * M))
    init_params.append(np.log(0.2))

  num_params = len(init_params)

  # Parameter index markers
  sigma_idx = 3 * M + len(adstock_param_channels)
  base_idx = (sigma_idx + 1) if fit_baseline else None

  def params_from_transformed(psi):
    betas = {}
    alphas = {}
    ks = {}
    for i, c in enumerate(channels):
      betas[c] = float(np.exp(psi[3 * i]))
      alphas[c] = float(np.exp(psi[3 * i + 1]))
      ks[c] = float(ref_k_dict[c] * np.exp(psi[3 * i + 2]))

    thetas = {}
    for i, c in enumerate(adstock_param_channels):
      idx = 3 * M + i
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[idx], -30, 30)))
      t_min, t_max = bounds_dict[c]
      thetas[c] = float(t_min + (t_max - t_min) * sig)

    for c in channels:
      if c not in thetas:
        thetas[c] = fixed_thetas[c]

    sigma = float(np.exp(psi[sigma_idx]))
    baseline = float(np.exp(psi[base_idx])) if fit_baseline else 0.0

    return betas, alphas, ks, thetas, sigma, baseline

  def log_prior(psi):
    lp = 0.0

    if hierarchical:
      # Extract hyperparameters
      mu_b = psi[hier_start_idx]
      s_b = np.exp(psi[hier_start_idx + 1])
      mu_a = psi[hier_start_idx + 2]
      s_a = np.exp(psi[hier_start_idx + 3])
      mu_k = psi[hier_start_idx + 4]
      s_k = np.exp(psi[hier_start_idx + 5])

      # Hyperpriors
      lp += -0.5 * ((mu_b - np.log((max_y * 1.2) / max(M, 1))) / 1.0) ** 2
      lp += -0.5 * (s_b / 0.5) ** 2 + psi[hier_start_idx + 1]
      lp += -0.5 * (mu_a / 0.5) ** 2
      lp += -0.5 * (s_a / 0.3) ** 2 + psi[hier_start_idx + 3]
      lp += -0.5 * (mu_k / 0.5) ** 2
      lp += -0.5 * (s_k / 0.5) ** 2 + psi[hier_start_idx + 5]

      if adstock_param_channels:
        mu_th = psi[hier_start_idx + 6]
        s_th = np.exp(psi[hier_start_idx + 7])
        lp += -0.5 * (mu_th / 1.0) ** 2
        lp += -0.5 * (s_th / 0.5) ** 2 + psi[hier_start_idx + 7]

      # Channel priors conditional on hyperparameters (Hierarchical partial pooling)
      for i, c in enumerate(channels):
        lp += -0.5 * ((psi[3 * i] - mu_b) / max(s_b, 1e-4)) ** 2 - np.log(max(s_b, 1e-4))
        lp += -0.5 * ((psi[3 * i + 1] - mu_a) / max(s_a, 1e-4)) ** 2 - np.log(max(s_a, 1e-4))
        lp += -0.5 * ((psi[3 * i + 2] - mu_k) / max(s_k, 1e-4)) ** 2 - np.log(max(s_k, 1e-4))

      for i, _ in enumerate(adstock_param_channels):
        idx = 3 * M + i
        if adstock_param_channels:
          mu_th = psi[hier_start_idx + 6]
          s_th = np.exp(psi[hier_start_idx + 7])
          lp += -0.5 * ((psi[idx] - mu_th) / max(s_th, 1e-4)) ** 2 - np.log(max(s_th, 1e-4))
        else:
          lp += -np.logaddexp(0.0, psi[idx]) - np.logaddexp(0.0, -psi[idx])
    else:
      # Independent unpooled priors
      for i, c in enumerate(channels):
        lp += -0.5 * ((psi[3 * i] - np.log((max_y * 1.2) / max(M, 1))) / 0.7) ** 2
        lp += -0.5 * (psi[3 * i + 1] / 0.5) ** 2
        lp += -0.5 * (psi[3 * i + 2] / 0.7) ** 2

      for i, _ in enumerate(adstock_param_channels):
        idx = 3 * M + i
        lp += -np.logaddexp(0.0, psi[idx]) - np.logaddexp(0.0, -psi[idx])

    # Observation noise prior
    sigma = np.exp(psi[sigma_idx])
    lp += -0.5 * (sigma / (max_y * 0.15)) ** 2 + psi[sigma_idx]

    # Baseline prior
    if fit_baseline:
      base_val = np.exp(psi[base_idx])
      lp += -0.5 * (base_val / (max_y * 0.25)) ** 2 + psi[base_idx]

    # Geo random effects priors
    if is_geo:
      s_geo = np.exp(psi[-1])
      lp += -0.5 * (s_geo / 0.3) ** 2 + psi[-1]
      for g_idx in range(G * M):
        val = psi[geo_start_idx + g_idx]
        lp += -0.5 * (val / max(s_geo, 1e-4)) ** 2 - np.log(max(s_geo, 1e-4))

    return lp

  def log_likelihood(psi):
    betas, alphas, ks, thetas, sigma, baseline = params_from_transformed(psi)
    if sigma <= 0:
      return -np.inf

    ll = 0.0

    if is_geo:
      s_geo = np.exp(psi[-1]) if is_geo else 1.0
      for g_i, g in enumerate(geos):
        y_g = y_geo_dict[g]
        y_pred = np.full_like(y_g, baseline / G)
        for m_i, c in enumerate(channels):
          delta_gm = psi[geo_start_idx + g_i * M + m_i]
          beta_gm = betas[c] * np.exp(delta_gm) / G
          s_arr = geo_spend_dict[g][c]
          th = thetas[c]
          s_ad = geometric_adstock(s_arr, th) if th > 0 else s_arr
          y_pred = y_pred + hill_function(s_ad, beta_gm, alphas[c], ks[c])

        res = (y_g - y_pred) / sigma
        ll += -0.5 * np.sum(res ** 2) - len(y_g) * np.log(sigma)
    else:
      y_pred = np.full_like(y_total, baseline)
      for c in channels:
        s_arr = spend_dict[c]
        th = thetas[c]
        s_ad = geometric_adstock(s_arr, th) if th > 0 else s_arr
        y_pred = y_pred + hill_function(s_ad, betas[c], alphas[c], ks[c])

      res = (y_total - y_pred) / sigma
      ll += -0.5 * np.sum(res ** 2) - len(y_total) * np.log(sigma)

    # Experimental lift calibration
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
    lp = log_prior(psi)
    if not np.isfinite(lp):
      return -np.inf
    ll = log_likelihood(psi)
    if not np.isfinite(ll):
      return -np.inf
    return ll + lp

  all_samples = []
  total_accepted = 0
  total_proposals = 0

  for _ in range(chains):
    curr_psi = np.array(init_params) + np.random.normal(0, 0.03, size=num_params)
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
        betas_i, alphas_i, ks_i, thetas_i, sigma_i, base_i = params_from_transformed(curr_psi)
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

  # Gelman-Rubin R-hat calculation
  r_hat_dict = {}
  if chains >= 2 and n_samples > 10:
    for idx, c in enumerate(channels):
      for p_offset, p_name in enumerate(['beta', 'alpha', 'K', 'theta']):
        col = 4 * idx + p_offset
        chain_means = [np.mean(chain[:, col]) for chain in all_samples]
        chain_vars = [np.var(chain[:, col], ddof=1) for chain in all_samples]
        N_s = len(all_samples[0])
        M_c = len(all_samples)
        B = (N_s / (M_c - 1)) * np.sum((chain_means - np.mean(chain_means)) ** 2)
        W = np.mean(chain_vars)
        var_plus = ((N_s - 1) / N_s) * W + (1.0 / N_s) * B
        r_hat = np.sqrt(var_plus / W) if W > 0 else 1.0
        r_hat_dict[f"{c}_{p_name}"] = float(np.round(r_hat, 3))

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
      'acceptance_rate': float(total_accepted / max(total_proposals, 1)),
      'r_hat': r_hat_dict,
      'hierarchical': hierarchical,
      'is_geo': is_geo
    }
  }

  return models_dict, baseline_mean, full_samples


# Backward compatible alias
fit_multichannel_bayesian_mcmc = fit_multichannel_hierarchical_bayesian


class MultiChannelMMM:
  """Meridian-Lite Hierarchical Bayesian Marketing Mix Model.

  Jointly estimates:
    Y_t = Baseline + sum_{m=1}^M Hill_m(Adstock_m(S_{m, 1:t})) + eps_t

  Key Capabilities:
    - Hierarchical Bayesian Partial Pooling across channels & geos.
    - Joint simultaneous estimation of Carryover Adstock (theta), Hill Saturation (alpha, K),
      and Channel Return Coefficients (beta).
    - Prevents cross-channel double-counting and omitted variable bias.
    - Experimental calibration (lift studies, geo experiments) integration.
    - Historical contribution decomposition, Share of Return, ROI with 90% credible intervals.
    - Direct integration with PortfolioAllocator for global budget optimization.
  """

  def __init__(self, channels, baseline=0.0, loss=0.0, posterior_samples=None):
    """
    Args:
      channels (dict or list): Dict mapping channel_name -> MarketingReturnCurve or list of models.
      baseline (float): Shared organic / baseline non-media return.
      loss (float): Final fitting loss.
      posterior_samples (dict, optional): Joint MCMC posterior samples.
    """
    if isinstance(channels, list):
      self.channels = {m.channel_name: m for m in channels}
    elif isinstance(channels, dict):
      self.channels = channels
    else:
      raise ValueError("channels must be a list of MarketingReturnCurve or a dict of {name: model}.")
    self.baseline = float(baseline)
    self.loss = float(loss)
    self.posterior_samples = posterior_samples

  @classmethod
  def fit(
      cls,
      spend_data,
      return_array,
      channel_names=None,
      method="auto",
      epochs=5000,
      lr=0.05,
      fit_baseline=True,
      adstock_types=None,
      adstock_bounds=None,
      adstock_fixed_days=None,
      n_samples=2000,
      chains=4,
      burn_in=1000,
      hierarchical=True,
      calibration_experiments=None
  ):
    """Unified entry point for fitting multi-channel marketing mix models."""
    method_norm = method.lower() if isinstance(method, str) else "auto"
    if method_norm == "auto":
      if calibration_experiments is not None:
        method_norm = "bayesian"
      else:
        method_norm = "gradient"

    if method_norm in ["gradient", "gradient_descent", "mle"]:
      return cls.fit_gradient_descent(
          spend_data=spend_data,
          return_array=return_array,
          channel_names=channel_names,
          epochs=epochs,
          lr=lr,
          fit_baseline=fit_baseline,
          adstock_types=adstock_types,
          adstock_bounds=adstock_bounds,
          adstock_fixed_days=adstock_fixed_days
      )
    elif method_norm in ["bayesian", "hierarchical_bayesian", "mcmc"]:
      return cls.fit_bayesian(
          spend_data=spend_data,
          return_array=return_array,
          channel_names=channel_names,
          n_samples=n_samples,
          chains=chains,
          burn_in=burn_in,
          fit_baseline=fit_baseline,
          hierarchical=hierarchical,
          adstock_types=adstock_types,
          adstock_bounds=adstock_bounds,
          adstock_fixed_days=adstock_fixed_days,
          calibration_experiments=calibration_experiments
      )
    else:
      raise ValueError(f"Unknown multi-channel fitting method: '{method}'")

  @classmethod
  def fit_gradient_descent(cls, spend_data, return_array, channel_names=None, epochs=5000, lr=0.05,
                           fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None):
    """Fits a joint multi-channel MMM using Gradient Descent (MLE / Tinygrad Adam)."""
    models_dict, baseline, loss = fit_multichannel_gradient(
      spend_data=spend_data, return_array=return_array, channel_names=channel_names,
      epochs=epochs, lr=lr, fit_baseline=fit_baseline,
      adstock_types=adstock_types, adstock_bounds=adstock_bounds,
      adstock_fixed_days=adstock_fixed_days
    )
    return cls(channels=models_dict, baseline=baseline, loss=loss)

  @classmethod
  def fit_bayesian(cls, spend_data, return_array, channel_names=None, n_samples=2000, chains=4, burn_in=1000,
                   fit_baseline=True, hierarchical=True, adstock_types=None, adstock_bounds=None,
                   adstock_fixed_days=None, calibration_experiments=None):
    """Fits a Meridian-lite Hierarchical Bayesian Marketing Mix Model."""
    models_dict, baseline, samples = fit_multichannel_hierarchical_bayesian(
      spend_data=spend_data, return_array=return_array, channel_names=channel_names,
      n_samples=n_samples, chains=chains, burn_in=burn_in, fit_baseline=fit_baseline,
      hierarchical=hierarchical, adstock_types=adstock_types, adstock_bounds=adstock_bounds,
      adstock_fixed_days=adstock_fixed_days,
      calibration_experiments=calibration_experiments
    )
    return cls(channels=models_dict, baseline=baseline, posterior_samples=samples)

  # Alias for explicit clarity
  fit_hierarchical_bayesian = fit_bayesian

  def predict_total_return(self, spend_dict, use_samples=False):
    """Predicts total response (baseline + all channel responses) given a dictionary of channel spends.

    Supports both single-period scalars and multi-period 1D numpy arrays.
    """
    first_val = next(iter(spend_dict.values()))
    is_array = hasattr(first_val, '__len__') and not isinstance(first_val, (str, bytes))

    if is_array:
      T = len(first_val)
      total = np.full(T, self.baseline)
      for cname, spend in spend_dict.items():
        if cname in self.channels:
          model = self.channels[cname]
          s_ad = model.adstock_spend(spend)
          total += model.predict_incremental_return(s_ad, use_samples=use_samples)
      return total
    else:
      total = self.baseline
      for cname, spend in spend_dict.items():
        if cname in self.channels:
          total += self.channels[cname].predict_incremental_return(spend, use_samples=use_samples)
      return total

  def predict_channel_contributions(self, spend_dict, use_samples=False):
    """Decomposes response into individual channel incremental contributions and baseline.

    Supports both single-period scalar spend queries and multi-period time-series arrays.
    """
    first_val = next(iter(spend_dict.values()))
    is_array = hasattr(first_val, '__len__') and not isinstance(first_val, (str, bytes))

    if is_array:
      T = len(first_val)
      contributions = {"Baseline": np.full(T, self.baseline)}
      for cname, spend in spend_dict.items():
        if cname in self.channels:
          model = self.channels[cname]
          s_ad = model.adstock_spend(spend)
          contributions[cname] = model.predict_incremental_return(s_ad, use_samples=use_samples)
      return contributions
    else:
      contributions = {"Baseline": self.baseline}
      for cname, spend in spend_dict.items():
        if cname in self.channels:
          contributions[cname] = self.channels[cname].predict_incremental_return(spend, use_samples=use_samples)
      return contributions

  def decompose_historical_contributions(self, spend_data, return_array=None):
    """Computes comprehensive historical attribution, share of spend vs return, and channel ROI.

    Returns:
      dict containing:
        - 'contributions_df': pd.DataFrame with time-series breakdown of Baseline and all channels.
        - 'summary_table': pd.DataFrame with Channel, Total Spend, Total Contribution,
                           Share of Spend (%), Share of Return (%), ROI, and current mROAS.
        - 'total_predicted': np.ndarray of total model predictions.
    """
    parsed, channels, _ = _parse_spend_input(spend_data)
    if isinstance(parsed, dict) and any(isinstance(v, dict) for v in parsed.values()):
      spend_dict = {c: sum(parsed[g][c] for g in parsed) for c in channels}
    else:
      spend_dict = parsed

    contribs = self.predict_channel_contributions(spend_dict)
    contribs_df = pd.DataFrame(contribs)
    total_predicted = contribs_df.sum(axis=1).values

    summary_rows = []
    total_spend_all = sum(float(np.sum(spend_dict[c])) for c in channels)
    total_return_all = float(np.sum(total_predicted))

    # Baseline row
    total_base = float(np.sum(contribs_df["Baseline"]))
    summary_rows.append({
      "Channel": "Baseline (Organic)",
      "Total Spend": 0.0,
      "Total Contribution": total_base,
      "Share of Spend (%)": 0.0,
      "Share of Return (%)": (total_base / max(total_return_all, 1e-6)) * 100.0,
      "ROI": np.nan,
      "Current mROAS": np.nan
    })

    for c in channels:
      c_spend = float(np.sum(spend_dict[c]))
      c_contrib = float(np.sum(contribs_df[c]))
      roi = (c_contrib / c_spend) if c_spend > 0 else 0.0
      last_spend = float(spend_dict[c][-1]) if len(spend_dict[c]) > 0 else 0.0
      mroas = self.channels[c].predict_marginal_return(last_spend)

      summary_rows.append({
        "Channel": c,
        "Total Spend": c_spend,
        "Total Contribution": c_contrib,
        "Share of Spend (%)": (c_spend / max(total_spend_all, 1e-6)) * 100.0,
        "Share of Return (%)": (c_contrib / max(total_return_all, 1e-6)) * 100.0,
        "ROI": roi,
        "Current mROAS": mroas
      })

    summary_table = pd.DataFrame(summary_rows)

    return {
      "contributions_df": contribs_df,
      "summary_table": summary_table,
      "total_predicted": total_predicted,
      "actual_return": np.array(return_array, dtype=float) if return_array is not None else None
    }

  def get_allocator(self):
    """Returns a PortfolioAllocator configured with all fitted channel models."""
    from tippingpoint.portfolio import PortfolioAllocator
    return PortfolioAllocator(list(self.channels.values()))

  def validate_experiments(self, experiments, spend_is_raw=True, verbose=False):
    """Validates multi-channel MMM curves against a collection of channel-specific incrementality experiments.

    Args:
      experiments: List of experiment dicts, each specifying a 'channel' key.
      spend_is_raw: If True, scales raw daily test spend by (1 - theta) to evaluate against effective adstock.
      verbose: If True, prints a multi-channel validation summary to stdout.

    Returns:
      dict: Multi-channel validation summary with per-channel breakdown and global metrics.
    """
    from .validation import validate_multichannel_experiments
    return validate_multichannel_experiments(self, experiments, spend_is_raw=spend_is_raw, verbose=verbose)

  def summary(self):
    """Returns a dictionary summarizing all channel curves, baseline, and MCMC diagnostics."""
    res = {
      "baseline": self.baseline,
      "loss": self.loss,
      "channels": {cname: m.summary() for cname, m in self.channels.items()}
    }
    if self.posterior_samples and 'diagnostics' in self.posterior_samples:
      res["diagnostics"] = self.posterior_samples['diagnostics']
    return res

