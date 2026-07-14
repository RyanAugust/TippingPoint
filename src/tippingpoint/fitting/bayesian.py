import numpy as np
from tippingpoint.math import geometric_adstock

def fit_bayesian_mcmc(spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None):
  """Fits a Hill Curve using Bayesian MCMC (Metropolis-Hastings) with optional adstock."""
  x = np.array(spend_array, dtype=float) + 1e-5
  y = np.array(return_array, dtype=float)

  # Default Priors (LogNormal)
  if priors is None:
    max_y = np.max(y)
    median_x = np.median(x[x > 1e-4]) if np.any(x > 1e-4) else 1.0
    priors = {
      'beta': (np.log(max_y * 1.2), 0.5),
      'alpha': (0.0, 0.5),
      'K': (np.log(median_x), 0.5)
    }

  # Adstock setup
  fixed_theta = 0.0
  theta_min, theta_max = 0.0, 0.999

  if adstock_type == "fixed":
    fixed_theta = 0.5 ** (1.0 / adstock_fixed_days) if adstock_fixed_days > 0 else 0.0
  elif adstock_type == "bounded":
    if adstock_bounds is not None:
      min_days, max_days = adstock_bounds
      theta_min = 0.5 ** (1.0 / min_days) if min_days > 0 else 0.0
      theta_max = 0.5 ** (1.0 / max_days) if max_days > 0 else 0.0

  def log_likelihood(beta, alpha, K, sigma, theta=0.0):
    if beta <= 0 or alpha <= 0 or K <= 0 or sigma <= 0: return -np.inf

    # Calculate theta or apply adstock
    if adstock_type == "none":
      x_adstocked = x
    elif adstock_type == "fixed":
      x_adstocked = geometric_adstock(x, fixed_theta)
    elif adstock_type == "free":
      if theta <= 0.0 or theta >= 0.999: return -np.inf
      x_adstocked = geometric_adstock(x, theta)
    elif adstock_type == "bounded":
      if theta < theta_min or theta > theta_max: return -np.inf
      x_adstocked = geometric_adstock(x, theta)

    y_pred = (beta * (x_adstocked ** alpha)) / (K ** alpha + x_adstocked ** alpha)
    return -0.5 * np.sum(((y - y_pred) / sigma) ** 2) - len(y) * np.log(sigma)

  def log_prior(beta, alpha, K, sigma, theta=0.0):
    lp = 0
    for name, val in [('beta', beta), ('alpha', alpha), ('K', K)]:
      mu, s = priors[name]
      lp += -0.5 * ((np.log(val) - mu) / s) ** 2 - np.log(val)
    lp += -0.5 * (sigma / (np.max(y) * 0.1)) ** 2
    return lp

  def log_posterior(params):
    if len(params) == 5:
      beta, alpha, K, sigma, theta = params
      return log_likelihood(beta, alpha, K, sigma, theta) + log_prior(beta, alpha, K, sigma, theta)
    else:
      beta, alpha, K, sigma = params
      return log_likelihood(beta, alpha, K, sigma) + log_prior(beta, alpha, K, sigma)

  # Simple Metropolis-Hastings Sampler
  all_samples = []
  for _ in range(chains):
    # Initialize parameters
    if adstock_type in ["free", "bounded"]:
      # Initialize theta in the middle of its bounds
      init_theta = 0.5 * (theta_min + theta_max)
      current_params = np.array([
        np.exp(priors['beta'][0]),
        np.exp(priors['alpha'][0]),
        np.exp(priors['K'][0]),
        np.std(y) * 0.1,
        init_theta
      ])
    else:
      current_params = np.array([
        np.exp(priors['beta'][0]),
        np.exp(priors['alpha'][0]),
        np.exp(priors['K'][0]),
        np.std(y) * 0.1
      ])

    current_log_post = log_posterior(current_params)
    samples = []
    step_size = current_params * 0.05
    # Fix zero step size for theta if it initialized to 0
    if len(step_size) == 5 and step_size[4] == 0:
      step_size[4] = 0.05

    for i in range(n_samples + burn_in):
      proposal = current_params + np.random.normal(0, step_size)
      proposal_log_post = log_posterior(proposal)
      if proposal_log_post > current_log_post or np.random.rand() < np.exp(proposal_log_post - current_log_post):
        current_params = proposal
        current_log_post = proposal_log_post
      if i >= burn_in:
        samples.append(current_params.copy())
    all_samples.append(np.array(samples))

  posterior = np.vstack(all_samples)

  if adstock_type in ["free", "bounded"]:
    samples_dict = {
      'beta': posterior[:, 0],
      'alpha': posterior[:, 1],
      'K': posterior[:, 2],
      'sigma': posterior[:, 3],
      'theta': posterior[:, 4]
    }
    theta_mean = np.mean(samples_dict['theta'])
  else:
    samples_dict = {
      'beta': posterior[:, 0],
      'alpha': posterior[:, 1],
      'K': posterior[:, 2],
      'sigma': posterior[:, 3],
      'theta': np.full_like(posterior[:, 0], fixed_theta)
    }
    theta_mean = fixed_theta

  beta_mean = np.mean(samples_dict['beta'])
  alpha_mean = np.mean(samples_dict['alpha'])
  K_mean = np.mean(samples_dict['K'])

  return beta_mean, alpha_mean, K_mean, theta_mean, samples_dict
