import numpy as np

def fit_bayesian_mcmc(spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000):
  """Fits a Hill Curve using Bayesian MCMC (Metropolis-Hastings)."""
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

  def log_likelihood(beta, alpha, K, sigma):
    if beta <= 0 or alpha <= 0 or K <= 0 or sigma <= 0: return -np.inf
    y_pred = (beta * (x ** alpha)) / (K ** alpha + x ** alpha)
    return -0.5 * np.sum(((y - y_pred) / sigma) ** 2) - len(y) * np.log(sigma)

  def log_prior(beta, alpha, K, sigma):
    lp = 0
    for name, val in [('beta', beta), ('alpha', alpha), ('K', K)]:
      mu, s = priors[name]
      lp += -0.5 * ((np.log(val) - mu) / s) ** 2 - np.log(val)
    lp += -0.5 * (sigma / (np.max(y) * 0.1)) ** 2
    return lp

  def log_posterior(params):
    beta, alpha, K, sigma = params
    return log_likelihood(beta, alpha, K, sigma) + log_prior(beta, alpha, K, sigma)

  all_samples = []
  for _ in range(chains):
    current_params = np.array([
      np.exp(priors['beta'][0]),
      np.exp(priors['alpha'][0]),
      np.exp(priors['K'][0]),
      np.std(y) * 0.1
    ])
    current_log_post = log_posterior(current_params)
    samples = []
    step_size = current_params * 0.05
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
  samples_dict = {
    'beta': posterior[:, 0],
    'alpha': posterior[:, 1],
    'K': posterior[:, 2],
    'sigma': posterior[:, 3]
  }

  beta_mean = np.mean(samples_dict['beta'])
  alpha_mean = np.mean(samples_dict['alpha'])
  K_mean = np.mean(samples_dict['K'])

  return beta_mean, alpha_mean, K_mean, samples_dict
