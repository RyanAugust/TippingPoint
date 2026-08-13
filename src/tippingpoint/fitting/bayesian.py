import numpy as np
from tippingpoint.math import geometric_adstock, hill_function

def fit_bayesian_mcmc(spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None):
  """Fits a Hill Curve using Bayesian MCMC (Metropolis-Hastings in transformed space) with optional adstock."""
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

  num_params = 5 if adstock_type in ["free", "bounded"] else 4

  def params_from_transformed(psi):
    beta = float(np.exp(psi[0]))
    alpha = float(np.exp(psi[1]))
    k = float(np.exp(psi[2]))
    sigma = float(np.exp(psi[3]))
    if adstock_type == "free":
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[4], -30, 30)))
      theta = float(0.999 * sig)
    elif adstock_type == "bounded":
      sig = 1.0 / (1.0 + np.exp(-np.clip(psi[4], -30, 30)))
      theta = float(theta_min + (theta_max - theta_min) * sig)
    elif adstock_type == "fixed":
      theta = fixed_theta
    else:
      theta = 0.0
    return beta, alpha, k, sigma, theta

  def log_prior(psi):
    lp = 0.0
    for idx, name in enumerate(['beta', 'alpha', 'K']):
      mu, s = priors[name]
      lp += -0.5 * ((psi[idx] - mu) / s) ** 2

    # Half-normal prior on sigma with Jacobian adjustment
    sigma_scale = max_y * 0.1
    sigma = np.exp(psi[3])
    lp += -0.5 * (sigma / sigma_scale) ** 2 + psi[3]

    # Uniform prior on theta with sigmoid Jacobian adjustment
    if adstock_type in ["free", "bounded"]:
      lp += -np.logaddexp(0.0, psi[4]) - np.logaddexp(0.0, -psi[4])
    return lp

  def log_likelihood(beta, alpha, k, sigma, theta):
    if sigma <= 0 or beta <= 0 or alpha <= 0 or k <= 0:
      return -np.inf

    if theta > 0:
      x_adstocked = geometric_adstock(x, theta)
    else:
      x_adstocked = x

    y_pred = hill_function(x_adstocked, beta, alpha, k)
    residuals = (y - y_pred) / sigma
    return -0.5 * np.sum(residuals ** 2) - len(y) * np.log(sigma)

  def log_posterior(psi):
    beta, alpha, k, sigma, theta = params_from_transformed(psi)
    return log_likelihood(beta, alpha, k, sigma, theta) + log_prior(psi)

  # Initialize chains
  init_sigma = max(float(np.std(y) * 0.1), 1e-4)
  init_psi = np.array([
    priors['beta'][0],
    priors['alpha'][0],
    priors['K'][0],
    np.log(init_sigma)
  ] + ([0.0] if num_params == 5 else []))

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
        beta_i, alpha_i, k_i, sigma_i, theta_i = params_from_transformed(curr_psi)
        chain_samples.append([beta_i, alpha_i, k_i, sigma_i, theta_i])

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
    'theta': compute_rhat(chains_tensor[:, :, 4]),
  }

  overall_acc_rate = float(total_accepted / max(total_proposals, 1))

  samples_dict = {
    'beta': posterior[:, 0],
    'alpha': posterior[:, 1],
    'K': posterior[:, 2],
    'sigma': posterior[:, 3],
    'theta': posterior[:, 4],
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
