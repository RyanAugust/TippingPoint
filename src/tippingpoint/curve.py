import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
import matplotlib.pyplot as plt

import warnings

class MarketingReturnCurve:
  """A marketing intelligence tool to determine the inflection points of a media
  response curve based on the Hill Function (Google Meridian methodology). """

  def __init__(self, beta, alpha, half_saturation_k, channel_name="Generic"):
    self.beta = float(beta)
    self.alpha = float(alpha)
    self.K = float(half_saturation_k)
    self.channel_name = channel_name

  @classmethod
  def from_historical_data(cls, spend_array, return_array, channel_name="Generic", epochs=5000, lr=0.05):
    """Phase 1: PyTorch Optimization Engine.
    Fits a Hill Curve to historical data using Adam optimizer. """
    max_y = np.max(return_array)
    median_x = np.median(spend_array[spend_array > 0]) if np.any(spend_array > 0) else 1.0

    Tensor.traning = True
    x = Tensor(spend_array, dtype=dtypes.float32, requires_grad=False)
    y = Tensor(return_array, dtype=dtypes.float32, requires_grad=False)

    log_beta = Tensor([np.log(max_y * 1.5)], dtype=dtypes.float32, requires_grad=True)
    log_k = Tensor([np.log(median_x + 1e-5)], dtype=dtypes.float32, requires_grad=True)
    log_alpha = Tensor([0.5], dtype=dtypes.float32, requires_grad=True)
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

  def predict_incremental_return(self, spend):
    """f(x): The baseline Hill Function calculation."""
    spend = np.array(spend, dtype=float) + 1e-5
    return (self.beta * (spend ** self.alpha)) / (self.K ** self.alpha + spend ** self.alpha)

  def predict_marginal_return(self, spend):
    """f'(x): The first derivative (Marginal ROAS / mCPA inverse)."""
    spend = np.array(spend, dtype=float) + 1e-5
    numerator = self.beta * self.alpha * (self.K ** self.alpha) * (spend ** (self.alpha - 1))
    denominator = (self.K ** self.alpha + spend ** self.alpha) ** 2
    return numerator / denominator

  def get_minimal_marginal_cost_point(self):
    """Solves for f''(x) = 0.
    The inflection point where marginal return peaks (acquisition cost is at its lowest). """
    if self.alpha <= 1: return 0.0 # If alpha <= 1, it's a C-Curve. Diminishing returns occur instantly at Spend = 0.
    inflection_point = self.K * (((self.alpha - 1) / (self.alpha + 1)) ** (1 / self.alpha)) # Closed-form solution for the inflection point of a Hill function)
    return inflection_point

  def get_diminishing_returns_point(self, target_mroas=1.0, tol=1e-5, max_iter=100):
    """Solves for x where f'(x) = target_mroas, specifically after the inflection point.
    This establishes the hard budget cap based on unit economics.
    (Uses a native bisection method, removing the need for SciPy). """
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
    """Translates mathematical points into strategic marketing intelligence."""
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas)
    mroas = self.predict_marginal_return(current_spend)
    print(f"--- Budget Evaluation: {self.channel_name} ---")
    print(f"Current Spend: ${current_spend:,.2f} | Current mROAS: {mroas:.2f}")
    if current_spend < min_spend: print("Status: WARMING UP (Inefficient)\nRecommendation: Increase spend to at least ${min_spend:,.2f} to reach peak acquisition efficiency.")
    elif max_spend is not None and current_spend > max_spend: print("Status: OVER-SATURATED (Unprofitable Marginal Growth)\n Recommendation: Scale back spend to ${max_spend:,.2f} to maintain target unit economics.")
    else: print("Status: OPTIMAL SCALING ZONE.\nRecommendation: You are operating within the highly efficient growth window.")

  def plot_response_curve(self, target_mroas=1.0, current_spend=None):
    """Generates an executive-friendly dual-axis chart mapping the optimal scaling zone."""
    min_spend = self.get_minimal_marginal_cost_point()
    max_spend = self.get_diminishing_returns_point(target_mroas)

    plot_limit = max_spend * 1.5 if max_spend else min_spend * 4
    plot_limit = max(plot_limit, current_spend * 1.2 if current_spend else 0)

    x_vals = np.linspace(0, plot_limit, 500)
    y_return = self.predict_incremental_return(x_vals)
    y_mroas = self.predict_marginal_return(x_vals)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Primary Axis: Response Curve
    ax1.plot(x_vals, y_return, color='#2CA02C', linewidth=3, label="Incremental Return")
    ax1.set_xlabel('Spend ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Incremental Return', color='#2CA02C', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2CA02C')
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Secondary Axis: Marginal Return
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_mroas, color='#1F77B4', linestyle='--', linewidth=2, label="Marginal ROAS")
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
