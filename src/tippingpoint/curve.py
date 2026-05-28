import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
import matplotlib.pyplot as plt

import warnings

class MarketingReturnCurve:
  """A marketing intelligence tool to determine inflection points of a media response curve.

  Based on the Hill Function (Google Meridian methodology), this tool identifies
  the Minimal Marginal Cost Point (peak efficiency) and the Point of Diminishing
  Returns (profitability floor).

  Attributes:
    beta (float): The asymptote (maximum possible return/capacity).
    alpha (float): The shape parameter (>1 for S-shape, <=1 for C-shape).
    K (float): The half-saturation point (spend where half of beta is reached).
    channel_name (str): Name of the media channel.
    posterior_samples (dict, optional): MCMC samples for alpha, beta, K, and sigma.
  """

  def __init__(self, beta, alpha, half_saturation_k, channel_name="Generic", posterior_samples=None):
    """Initializes the MarketingReturnCurve with specific parameters.

    Args:
      beta (float): Maximum possible capacity.
      alpha (float): Shape parameter.
      half_saturation_k (float): Half-saturation spend amount.
      channel_name (str): Label for the channel. Defaults to "Generic".
      posterior_samples (dict, optional): Dictionary of parameter samples.
    """
    self.beta = float(beta)
    self.alpha = float(alpha)
    self.K = float(half_saturation_k)
    self.channel_name = channel_name
    self.posterior_samples = posterior_samples

  @classmethod
  def fit_bayesian(cls, spend_array, return_array, channel_name="Generic", priors=None, n_samples=2000, chains=4, burn_in=1000):
    """Fits a Hill Curve using Bayesian MCMC (Metropolis-Hastings).

    Args:
      spend_array (array-like): Historical spend data.
      return_array (array-like): Historical return/KPI data.
      channel_name (str): Label for the channel. Defaults to "Generic".
      priors (dict, optional): LogNormal priors for 'beta', 'alpha', 'K'.
        Format: {'param': (mu, sigma)}.
      n_samples (int): Number of samples per chain. Defaults to 2000.
      chains (int): Number of MCMC chains. Defaults to 4.
      burn_in (int): Number of initial samples to discard. Defaults to 1000.

    Returns:
      MarketingReturnCurve: An instance of the curve fitted with posterior means.
    """
    x = np.array(spend_array, dtype=float) + 1e-5
    y = np.array(return_array, dtype=float)

    # Default Priors (LogNormal)
    if priors is None:
      max_y = np.max(y)
      median_x = np.median(x[x > 1e-4]) if np.any(x > 1e-4) else 1.0
      priors = {
        'beta': (np.log(max_y * 1.2), 0.5),
        'alpha': (0.0, 0.5), # Centered at 1.0 (log(1)=0)
        'K': (np.log(median_x), 0.5)
      }

    def log_likelihood(beta, alpha, K, sigma):
      if beta <= 0 or alpha <= 0 or K <= 0 or sigma <= 0: return -np.inf
      y_pred = (beta * (x ** alpha)) / (K ** alpha + x ** alpha)
      return -0.5 * np.sum(((y - y_pred) / sigma) ** 2) - len(y) * np.log(sigma)

    def log_prior(beta, alpha, K, sigma):
      lp = 0
      # LogNormal priors for beta, alpha, K
      for name, val in [('beta', beta), ('alpha', alpha), ('K', K)]:
        mu, s = priors[name]
        lp += -0.5 * ((np.log(val) - mu) / s) ** 2 - np.log(val)
      # Half-Normal prior for sigma
      lp += -0.5 * (sigma / (np.max(y) * 0.1)) ** 2
      return lp

    def log_posterior(params):
      beta, alpha, K, sigma = params
      return log_likelihood(beta, alpha, K, sigma) + log_prior(beta, alpha, K, sigma)

    # Simple Metropolis-Hastings Sampler
    all_samples = []
    for _ in range(chains):
      # Initialize
      current_params = np.array([
        np.exp(priors['beta'][0]),
        np.exp(priors['alpha'][0]),
        np.exp(priors['K'][0]),
        np.std(y) * 0.1
      ])
      current_log_post = log_posterior(current_params)

      samples = []
      # Adaptive step size (simplified)
      step_size = current_params * 0.05

      for i in range(n_samples + burn_in):
        proposal = current_params + np.random.normal(0, step_size)
        proposal_log_post = log_posterior(proposal)

        if proposal_log_post > current_log_post or np.random.rand() < np.exp(proposal_log_post - current_log_post):
          current_params = proposal
          current_log_post = proposal_log_post

        if i >= burn_in:
          samples.append(current_params.copy())

        # Small adaptation during burn-in
        if i < burn_in and i % 100 == 0 and i > 0:
          # This is a very crude adaptation
          pass

      all_samples.append(np.array(samples))

    posterior = np.vstack(all_samples)
    samples_dict = {
      'beta': posterior[:, 0],
      'alpha': posterior[:, 1],
      'K': posterior[:, 2],
      'sigma': posterior[:, 3]
    }

    # Point estimates (posterior mean)
    beta_mean = np.mean(samples_dict['beta'])
    alpha_mean = np.mean(samples_dict['alpha'])
    K_mean = np.mean(samples_dict['K'])

    print(f"[{channel_name}] Bayesian fit complete. Samples: {len(posterior)}")
    return cls(beta_mean, alpha_mean, K_mean, channel_name, posterior_samples=samples_dict)

  @classmethod
  def from_historical_data(cls, spend_array, return_array, channel_name="Generic", epochs=5000, lr=0.05):
    """Fits a Hill Curve to historical data using MLE (Adam optimizer).

    Args:
      spend_array (array-like): Historical spend data.
      return_array (array-like): Historical return/KPI data.
      channel_name (str): Label for the channel. Defaults to "Generic".
      epochs (int): Number of optimization epochs. Defaults to 5000.
      lr (float): Learning rate for the optimizer. Defaults to 0.05.

    Returns:
      MarketingReturnCurve: An instance of the curve fitted with optimized parameters.
    """
    max_y = np.max(return_array)
    median_x = np.median(spend_array[spend_array > 0]) if np.any(spend_array > 0) else 1.0

    Tensor.traning = True
    x = Tensor(spend_array, dtype=dtypes.float32)
    x.requires_grad = False
    y = Tensor(return_array, dtype=dtypes.float32)
    y.requires_grad = False

    log_beta = Tensor([np.log(max_y * 1.5)], dtype=dtypes.float32)
    log_beta.requires_grad = True
    log_k = Tensor([np.log(median_x + 1e-5)], dtype=dtypes.float32)
    log_k.requires_grad = True
    log_alpha = Tensor([0.5], dtype=dtypes.float32)
    log_alpha.requires_grad = True
    optimizer = Adam([log_beta, log_k, log_alpha], lr=lr)

    Tensor.traning = True
    with Tensor.train():
      for _ in range(epochs):
        optimizer.zero_grad()
        beta = log_beta.exp()
        k = log_k.exp()
        alpha = log_alpha.exp()
        x_safe = x + 1e-5 # Add tiny epsilon to x to prevent 0^alpha resulting in NaNs
        y_pred = (beta * (x_safe ** alpha)) / (k ** alpha + x_safe ** alpha) # Hill Function
        loss = ((y_pred - y) ** 2).mean() # Loss Function (Mean Squared Error)
        loss.backward()
        optimizer.step()
    Tensor.traning = False
    final_loss = loss.numpy().item()
    print(f"[{channel_name}] Curve fit complete. Loss: {final_loss:.4f}")
    return cls(log_beta.exp().numpy().item(), log_alpha.exp().numpy().item(), log_k.exp().numpy().item(), channel_name)

  def predict_incremental_return(self, spend, use_samples=False):
    """Calculates the total incremental return for a given spend.

    Args:
      spend (float or array-like): The spend amount(s) to evaluate.
      use_samples (bool): If True, returns a distribution using posterior samples.
        Defaults to False.

    Returns:
      float or numpy.ndarray: The predicted incremental return(s).
    """
    spend = np.array(spend, dtype=float) + 1e-5
    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      return (beta * (spend ** alpha)) / (K ** alpha + spend ** alpha)
    return (self.beta * (spend ** self.alpha)) / (self.K ** self.alpha + spend ** self.alpha)

  def predict_marginal_return(self, spend, use_samples=False):
    """Calculates the first derivative (Marginal ROAS) at a given spend.

    Args:
      spend (float or array-like): The spend amount(s) to evaluate.
      use_samples (bool): If True, returns a distribution using posterior samples.
        Defaults to False.

    Returns:
      float or numpy.ndarray: The predicted marginal return(s).
    """
    spend = np.array(spend, dtype=float) + 1e-5
    if use_samples and self.posterior_samples:
      beta = self.posterior_samples['beta'][:, np.newaxis]
      alpha = self.posterior_samples['alpha'][:, np.newaxis]
      K = self.posterior_samples['K'][:, np.newaxis]
      numerator = beta * alpha * (K ** alpha) * (spend ** (alpha - 1))
      denominator = (K ** alpha + spend ** alpha) ** 2
      return numerator / denominator
    numerator = self.beta * self.alpha * (self.K ** self.alpha) * (spend ** (self.alpha - 1))
    denominator = (self.K ** self.alpha + spend ** self.alpha) ** 2
    return numerator / denominator

  def get_minimal_marginal_cost_point(self):
    """Identifies the inflection point where marginal return peaks.

    This corresponds to the spend level where efficiency is maximized (f''(x) = 0).

    Returns:
      float: The spend amount at the inflection point.
    """
    if self.alpha <= 1: return 0.0 # If alpha <= 1, it's a C-Curve. Diminishing returns occur instantly at Spend = 0.
    inflection_point = self.K * (((self.alpha - 1) / (self.alpha + 1)) ** (1 / self.alpha)) # Closed-form solution for the inflection point of a Hill function)
    return inflection_point

  def get_diminishing_returns_point(self, target_mroas=1.0, tol=1e-5, max_iter=100):
    """Solves for the spend level where Marginal ROAS hits a specific target.

    Args:
      target_mroas (float): The minimum acceptable marginal return. Defaults to 1.0.
      tol (float): Convergence tolerance for the bisection search. Defaults to 1e-5.
      max_iter (int): Maximum iterations for the search. Defaults to 100.

    Returns:
      float or None: The spend amount at the diminishing returns point, or None if unreachable.
    """
    inflection = max(self.get_minimal_marginal_cost_point(), 1e-5)
    max_mroas = self.predict_marginal_return(inflection)
    if target_mroas >= max_mroas:
      warnings.warn(f"Target mROAS ({target_mroas}) is mathematically unreachable.\nMax possible mROAS is {max_mroas:.2f}.")
      return None
    lower_bound = inflection
    upper_bound = inflection + self.K
    while self.predict_marginal_return(upper_bound) > target_mroas:
      upper_bound += self.K
      if upper_bound > self.K * 1000: # Safety break
        warnings.warn("Could not find an upper bound for the target mROAS.")
        return None
    for _ in range(max_iter): # Native Bisection Search
      midpoint = (lower_bound + upper_bound) / 2.0
      mroas_at_mid = self.predict_marginal_return(midpoint)
      if abs(mroas_at_mid - target_mroas) < tol: return midpoint # If we are within the acceptable tolerance, return the spend amount
      # Because mROAS is strictly decreasing after the inflection point:
      if mroas_at_mid > target_mroas: lower_bound = midpoint # mROAS is too high, we need to spend MORE to drive it down to the target
      else: upper_bound = midpoint # mROAS is too low, we need to spend LESS to bring it back up to the target
    return (lower_bound + upper_bound) / 2.0 # Return the best approximation if max iterations are reached

  def evaluate_current_budget(self, current_spend, target_mroas=1.0):
    """Provides a strategic evaluation of the current budget allocation.

    Prints recommendations based on the relationship between current spend,
    the peak efficiency point, and the diminishing returns point.

    Args:
      current_spend (float): The current amount being spent.
      target_mroas (float): The target marginal return floor. Defaults to 1.0.
    """
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas)
    mroas = self.predict_marginal_return(current_spend)
    print(f"--- Budget Evaluation: {self.channel_name} ---")
    print(f"Current Spend: ${current_spend:,.2f} | Current mROAS: {mroas:.2f}")
    if current_spend < min_spend: print("Status: WARMING UP (Inefficient)\nRecommendation: Increase spend to at least ${min_spend:,.2f} to reach peak acquisition efficiency.")
    elif max_spend is not None and current_spend > max_spend: print("Status: OVER-SATURATED (Unprofitable Marginal Growth)\n Recommendation: Scale back spend to ${max_spend:,.2f} to maintain target unit economics.")
    else: print("Status: OPTIMAL SCALING ZONE.\nRecommendation: You are operating within the highly efficient growth window.")

  def plot_response_curve(self, target_mroas=1.0, current_spend=None, show_intervals=True):
    """Generates a visualization of the media response and marginal return curves.

    Args:
      target_mroas (float): The target marginal return floor. Defaults to 1.0.
      current_spend (float, optional): The current spend to mark on the chart.
      show_intervals (bool): If True and posterior samples exist, plots
        the 90% credible interval. Defaults to True.
    """
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas)

    plot_limit = max_spend * 1.5 if max_spend else min_spend * 4
    plot_limit = max(plot_limit, current_spend * 1.2 if current_spend else 0)

    x_vals = np.linspace(0, plot_limit, 500)

    if show_intervals and self.posterior_samples:
      y_returns_dist = self.predict_incremental_return(x_vals, use_samples=True)
      y_return = np.mean(y_returns_dist, axis=0)
      y_return_low = np.percentile(y_returns_dist, 5, axis=0)
      y_return_high = np.percentile(y_returns_dist, 95, axis=0)

      y_mroas_dist = self.predict_marginal_return(x_vals, use_samples=True)
      y_mroas = np.mean(y_mroas_dist, axis=0)
      y_mroas_low = np.percentile(y_mroas_dist, 5, axis=0)
      y_mroas_high = np.percentile(y_mroas_dist, 95, axis=0)
    else:
      y_return = self.predict_incremental_return(x_vals)
      y_mroas = self.predict_marginal_return(x_vals)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Primary Axis: Response Curve
    ax1.plot(x_vals, y_return, color='#2CA02C', linewidth=3, label="Incremental Return")
    if show_intervals and self.posterior_samples:
      ax1.fill_between(x_vals, y_return_low, y_return_high, color='#2CA02C', alpha=0.2, label="90% Credible Interval")

    ax1.set_xlabel('Spend ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Incremental Return', color='#2CA02C', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2CA02C')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Secondary Axis: Marginal Return
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_mroas, color='#1F77B4', linestyle='--', linewidth=2, label="Marginal ROAS")
    if show_intervals and self.posterior_samples:
      ax2.fill_between(x_vals, y_mroas_low, y_mroas_high, color='#1F77B4', alpha=0.1)

    ax2.set_ylabel('Marginal ROAS (mROAS)', color='#1F77B4', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#1F77B4')
    ax2.axhline(target_mroas, color='gray', linestyle=':', label="Target mROAS Floor")
    # Mark Inflection Points
    ax2.plot(min_spend, self.predict_marginal_return(min_spend), marker='*', color='gold', markersize=15, markeredgecolor='black', label="Minimal Marginal Cost (Peak Efficiency)")
    if max_spend:
      ax2.plot(max_spend, target_mroas, marker='X', color='red', markersize=12, label="Point of Diminishing Returns")
      ax1.axvspan(min_spend, max_spend, color='green', alpha=0.1, label='Optimal Scaling Zone') # Shade the "Optimal Scaling Zone"
    if current_spend: ax1.axvline(current_spend, color='purple', linestyle='-.', label="Current Spend")
    # Combine Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.title(f'Response Curve Analysis: {self.channel_name}\n$\\alpha={self.alpha:.2f}, K={self.K:,.0f}, \\beta={self.beta:,.0f}$', fontsize=14)
    plt.tight_layout()
    plt.show()
