import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
from tippingpoint.math import geometric_adstock, hill_function
from tippingpoint.fitting.gradient import tinygrad_geometric_adstock
from tippingpoint.models import MarketingReturnCurve

def fit_multichannel_gradient(spend_data, return_array, channel_names=None, epochs=5000, lr=0.05, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None):
  """Fits a joint Multi-Channel Marketing Mix Model using Gradient Descent (Tinygrad Adam)."""
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

def fit_multichannel_bayesian_mcmc(spend_data, return_array, channel_names=None, n_samples=2000, chains=4, burn_in=1000, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None):
  """Fits a joint Multi-Channel Marketing Mix Model using Bayesian MCMC with optional experimental calibration."""
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


class MultiChannelMMM:
  """Joint multi-channel marketing mix model decomposing total returns across channels.

  Fits simultaneously:
    Y_t = Baseline + sum_{m=1}^M Hill_m(Adstock_m(S_{m, 1:t})) + eps_t

  Prevents omitted variable bias and cross-channel double-counting.
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
  def from_historical_data(cls, spend_data, return_array, channel_names=None, epochs=5000, lr=0.05, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None):
    """Fits a joint multi-channel MMM using Gradient Descent (MLE / Tinygrad Adam)."""
    models_dict, baseline, loss = fit_multichannel_gradient(
      spend_data=spend_data, return_array=return_array, channel_names=channel_names,
      epochs=epochs, lr=lr, fit_baseline=fit_baseline,
      adstock_types=adstock_types, adstock_bounds=adstock_bounds,
      adstock_fixed_days=adstock_fixed_days
    )
    return cls(channels=models_dict, baseline=baseline, loss=loss)

  @classmethod
  def fit_bayesian(cls, spend_data, return_array, channel_names=None, n_samples=2000, chains=4, burn_in=1000, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None, calibration_experiments=None):
    """Fits a joint multi-channel MMM using Bayesian MCMC with optional experimental calibration."""
    models_dict, baseline, samples = fit_multichannel_bayesian_mcmc(
      spend_data=spend_data, return_array=return_array, channel_names=channel_names,
      n_samples=n_samples, chains=chains, burn_in=burn_in, fit_baseline=fit_baseline,
      adstock_types=adstock_types, adstock_bounds=adstock_bounds,
      adstock_fixed_days=adstock_fixed_days,
      calibration_experiments=calibration_experiments
    )
    return cls(channels=models_dict, baseline=baseline, posterior_samples=samples)

  def predict_total_return(self, spend_dict):
    """Predicts total response (baseline + all channel responses) given a dictionary of channel spends."""
    total = self.baseline
    for cname, spend in spend_dict.items():
      if cname in self.channels:
        total += self.channels[cname].predict_incremental_return(spend)
    return total

  def predict_channel_contributions(self, spend_dict):
    """Decomposes response into individual channel incremental contributions and baseline."""
    contributions = {"Baseline": self.baseline}
    for cname, spend in spend_dict.items():
      if cname in self.channels:
        contributions[cname] = self.channels[cname].predict_incremental_return(spend)
    return contributions

  def get_allocator(self):
    """Returns a PortfolioAllocator configured with all fitted channel models."""
    from tippingpoint.portfolio import PortfolioAllocator
    return PortfolioAllocator(list(self.channels.values()))

  def summary(self):
    return {
      "baseline": self.baseline,
      "loss": self.loss,
      "channels": {cname: m.summary() for cname, m in self.channels.items()}
    }
