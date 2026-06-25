import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class MaximalFrontierCurve:
  """A marketing equivalent of cycling power-duration models.

  Models the maximal frontier of marketing efficiency (e.g., ROAS or Conversions
  per Dollar) as a function of spend, inspired by sports science critical power modeling.
  Supports 2-parameter, 3-parameter/Morton, Exponential (CP Exp), Ward-Smith, and Omni Domain models.

  Attributes:
    cr (float): Critical Results (asymptotic minimum efficiency / baseline rate).
    r_prime (float, optional): Results Capacity (reserve of high-efficiency results).
    r_max (float, optional): Maximum Results rate (efficiency at near-zero spend).
    tau (float, optional): Decay scale parameter (for CP Exp and Ward-Smith).
    r_end (float, optional): Ultimate fatigue efficiency floor (for Omni Domain).
    tau_1 (float, optional): Short-range decay constant (for Omni Domain).
    tau_2 (float, optional): Long-range decay constant (for Omni Domain).
    model_type (str): The specific model style.
  """

  def __init__(self, cr, r_prime=None, r_max=None, tau=None, r_end=None, tau_1=None, tau_2=None, model_type="2-parameter"):
    self.cr = float(cr)
    self.r_prime = float(r_prime) if r_prime is not None else None
    self.r_max = float(r_max) if r_max is not None else None
    self.tau = float(tau) if tau is not None else None
    self.r_end = float(r_end) if r_end is not None else None
    self.tau_1 = float(tau_1) if tau_1 is not None else None
    self.tau_2 = float(tau_2) if tau_2 is not None else None
    self.model_type = model_type.lower()

  def predict_rate(self, spend):
    """Predicts the maximal results rate (efficiency, e.g. ROAS) for a given spend.

    Args:
      spend (float or array-like): The spend amount(s) to evaluate.

    Returns:
      float or numpy.ndarray: The predicted maximal rate(s).
    """
    spend = np.array(spend, dtype=float) + 1e-5
    if self.model_type == "2-parameter":
      return self.r_prime / spend + self.cr
    elif self.model_type in ("3-parameter", "morton"):
      return self.cr + self.r_prime / (spend + self.r_prime / (self.r_max - self.cr + 1e-8))
    elif self.model_type in ("exponential", "cp-exp"):
      return self.cr + (self.r_max - self.cr) * np.exp(-spend / self.tau)
    elif self.model_type == "ward-smith":
      x = spend / self.tau
      # Use Taylor series expansion for x close to 0 to avoid division-by-zero numerical noise
      term = np.where(x < 1e-4, 1.0 - x / 2.0 + (x**2) / 6.0, (1.0 - np.exp(-x)) / x)
      return self.cr + (self.r_max - self.cr) * term
    elif self.model_type == "omni-domain":
      return (self.r_max - self.cr) * np.exp(-spend / self.tau_1) + (self.cr - self.r_end) * np.exp(-spend / self.tau_2) + self.r_end
    else:
      raise ValueError(f"Unknown model type: {self.model_type}")

  def predict_total(self, spend):
    """Predicts the maximal total results for a given spend.

    Args:
      spend (float or array-like): The spend amount(s) to evaluate.

    Returns:
      float or numpy.ndarray: The predicted maximal total results.
    """
    return spend * self.predict_rate(spend)

  def get_max_spend(self, target_rate):
    """Calculates the maximum spend possible to maintain a target efficiency rate.

    Args:
      target_rate (float or array-like): The target efficiency rate.

    Returns:
      float or numpy.ndarray: The maximum spend, or np.inf if the target is below critical results.
    """
    target_rate = np.array(target_rate, dtype=float)
    is_scalar = target_rate.ndim == 0
    target_rate = np.atleast_1d(target_rate)

    if self.model_type == "2-parameter":
      val = np.where(target_rate <= self.cr, np.inf, self.r_prime / (target_rate - self.cr))
    elif self.model_type in ("3-parameter", "morton"):
      val = self.r_prime / (target_rate - self.cr) - self.r_prime / (self.r_max - self.cr + 1e-8)
      val = np.where(target_rate >= self.r_max, 0.0, val)
      val = np.where(target_rate <= self.cr, np.inf, val)
    elif self.model_type in ("exponential", "cp-exp"):
      val = self.tau * np.log((self.r_max - self.cr) / (target_rate - self.cr + 1e-8))
      val = np.where(target_rate >= self.r_max, 0.0, val)
      val = np.where(target_rate <= self.cr, np.inf, val)
    else: # Numerical cases (Ward-Smith, Omni Domain) where closed form isn't possible
      val = np.zeros_like(target_rate)
      for i, t_rate in enumerate(target_rate):
        if t_rate <= self.cr:
          val[i] = np.inf
        elif t_rate >= self.r_max:
          val[i] = 0.0
        else:
          # Bisection search
          lower = 0.0
          upper = 1e6
          while self.predict_rate(upper) > t_rate:
            upper *= 10.0
            if upper > 1e12:
              break
          for _ in range(50):
            mid = (lower + upper) / 2.0
            if self.predict_rate(mid) > t_rate:
              lower = mid
            else:
              upper = mid
          val[i] = (lower + upper) / 2.0

    return val[0] if is_scalar else val

  @classmethod
  def fit(cls, spend, results, model_type="2-parameter", is_rate=False, q=None, epochs=3000, lr=0.01):
    """Fits any of the supported marketing performance frontier models.

    Args:
      spend (array-like): Spend data.
      results (array-like): Results data (total or rate).
      model_type (str): "2-parameter", "3-parameter" / "morton", "exponential" / "cp-exp", "ward-smith", or "omni-domain".
      is_rate (bool): True if results is already a rate (e.g., ROAS), False if total results.
      q (float, optional): Quantile to fit (0-1). If provided, fits the quantile frontier using Pinball loss.
      epochs (int): Training iterations.
      lr (float): Learning rate for optimization.

    Returns:
      MaximalFrontierCurve: The fitted frontier model.
    """
    m_type = model_type.lower()
    if m_type == "2-parameter":
      return cls.fit_2param(spend, results, is_rate=is_rate, q=q, epochs=epochs, lr=lr)
    elif m_type in ("3-parameter", "morton"):
      return cls.fit_3param(spend, results, is_rate=is_rate, q=q, epochs=epochs, lr=lr)
    elif m_type in ("exponential", "cp-exp"):
      return cls.fit_exponential(spend, results, is_rate=is_rate, q=q, epochs=epochs, lr=lr)
    elif m_type == "ward-smith":
      return cls.fit_ward_smith(spend, results, is_rate=is_rate, q=q, epochs=epochs, lr=lr)
    elif m_type == "omni-domain":
      return cls.fit_omni_domain(spend, results, is_rate=is_rate, q=q, epochs=epochs, lr=lr)
    else:
      raise ValueError(f"Unknown model type: {model_type}. Select from: '2-parameter', '3-parameter', 'morton', 'cp-exp', 'ward-smith', 'omni-domain'")

  @classmethod
  def fit_2param(cls, spend, results, is_rate=False, method="linear", q=None, epochs=2000, lr=0.01):
    spend = np.array(spend, dtype=float)
    results = np.array(results, dtype=float)
    mask = spend > 1e-5
    spend = spend[mask]
    results = results[mask]

    if is_rate:
      rate = results
    else:
      rate = results / spend

    if method == "linear" and q is None:
      if is_rate:
        total = spend * rate
      else:
        total = results
      slope, intercept = np.polyfit(spend, total, 1)
      return cls(cr=slope, r_prime=intercept, model_type="2-parameter")

    Tensor.training = True
    log_cr = Tensor([np.log(max(np.median(rate), 1e-5))], dtype=dtypes.float32)
    log_cr.requires_grad = True
    log_r_prime = Tensor([np.log(max(np.median(spend) * np.median(rate), 1e-5))], dtype=dtypes.float32)
    log_r_prime.requires_grad = True

    optimizer = Adam([log_cr, log_r_prime], lr=lr)
    x_tensor = Tensor(spend, dtype=dtypes.float32)
    y_tensor = Tensor(rate, dtype=dtypes.float32)

    for _ in range(epochs):
      optimizer.zero_grad()
      cr = log_cr.exp()
      r_prime = log_r_prime.exp()
      y_pred = r_prime / x_tensor + cr

      if q is not None:
        diff = y_pred - y_tensor
        loss = (diff * q).maximum(diff * (q - 1)).mean()
      else:
        loss = ((y_pred - y_tensor) ** 2).mean()

      loss.backward()
      optimizer.step()

    Tensor.training = False
    return cls(cr=log_cr.exp().numpy().item(), r_prime=log_r_prime.exp().numpy().item(), model_type="2-parameter")

  @classmethod
  def fit_3param(cls, spend, results, is_rate=False, q=None, epochs=3000, lr=0.01):
    spend = np.array(spend, dtype=float)
    results = np.array(results, dtype=float)
    mask = spend > 1e-5
    spend = spend[mask]
    results = results[mask]

    if is_rate:
      rate = results
    else:
      rate = results / spend

    Tensor.training = True

    init_cr = np.min(rate) * 0.9
    init_r_max = np.max(rate) * 1.1
    init_r_prime = np.median(spend) * (init_r_max - init_cr) * 0.1

    log_cr = Tensor([np.log(max(init_cr, 1e-5))], dtype=dtypes.float32)
    log_cr.requires_grad = True
    log_r_prime = Tensor([np.log(max(init_r_prime, 1e-5))], dtype=dtypes.float32)
    log_r_prime.requires_grad = True
    log_gap = Tensor([np.log(max(init_r_max - init_cr, 1e-5))], dtype=dtypes.float32)
    log_gap.requires_grad = True

    optimizer = Adam([log_cr, log_r_prime, log_gap], lr=lr)
    x_tensor = Tensor(spend, dtype=dtypes.float32)
    y_tensor = Tensor(rate, dtype=dtypes.float32)

    for _ in range(epochs):
      optimizer.zero_grad()
      cr = log_cr.exp()
      r_prime = log_r_prime.exp()
      y_pred = cr + r_prime / (x_tensor + r_prime / (log_gap.exp() + 1e-8))

      if q is not None:
        diff = y_pred - y_tensor
        loss = (diff * q).maximum(diff * (q - 1)).mean()
      else:
        loss = ((y_pred - y_tensor) ** 2).mean()

      loss.backward()
      optimizer.step()

    Tensor.training = False
    final_cr = log_cr.exp().numpy().item()
    final_r_prime = log_r_prime.exp().numpy().item()
    final_r_max = final_cr + log_gap.exp().numpy().item()

    return cls(cr=final_cr, r_prime=final_r_prime, r_max=final_r_max, model_type="3-parameter")

  @classmethod
  def fit_exponential(cls, spend, results, is_rate=False, q=None, epochs=3000, lr=0.01):
    spend = np.array(spend, dtype=float)
    results = np.array(results, dtype=float)
    mask = spend > 1e-5
    spend = spend[mask]
    results = results[mask]

    if is_rate:
      rate = results
    else:
      rate = results / spend

    Tensor.training = True

    init_cr = np.min(rate) * 0.9
    init_r_max = np.max(rate) * 1.1
    init_tau = np.median(spend)

    log_cr = Tensor([np.log(max(init_cr, 1e-5))], dtype=dtypes.float32)
    log_cr.requires_grad = True
    log_gap = Tensor([np.log(max(init_r_max - init_cr, 1e-5))], dtype=dtypes.float32)
    log_gap.requires_grad = True
    log_tau = Tensor([np.log(max(init_tau, 1e-5))], dtype=dtypes.float32)
    log_tau.requires_grad = True

    optimizer = Adam([log_cr, log_gap, log_tau], lr=lr)
    x_tensor = Tensor(spend, dtype=dtypes.float32)
    y_tensor = Tensor(rate, dtype=dtypes.float32)

    for _ in range(epochs):
      optimizer.zero_grad()
      cr = log_cr.exp()
      tau = log_tau.exp()
      y_pred = cr + log_gap.exp() * (-x_tensor / tau).exp()

      if q is not None:
        diff = y_pred - y_tensor
        loss = (diff * q).maximum(diff * (q - 1)).mean()
      else:
        loss = ((y_pred - y_tensor) ** 2).mean()

      loss.backward()
      optimizer.step()

    Tensor.training = False
    final_cr = log_cr.exp().numpy().item()
    final_r_max = final_cr + log_gap.exp().numpy().item()
    final_tau = log_tau.exp().numpy().item()

    return cls(cr=final_cr, r_max=final_r_max, tau=final_tau, model_type="exponential")

  @classmethod
  def fit_ward_smith(cls, spend, results, is_rate=False, q=None, epochs=3000, lr=0.01):
    spend = np.array(spend, dtype=float)
    results = np.array(results, dtype=float)
    mask = spend > 1e-5
    spend = spend[mask]
    results = results[mask]

    if is_rate:
      rate = results
    else:
      rate = results / spend

    Tensor.training = True

    init_cr = np.min(rate) * 0.9
    init_r_max = np.max(rate) * 1.1
    init_tau = np.median(spend)

    log_cr = Tensor([np.log(max(init_cr, 1e-5))], dtype=dtypes.float32)
    log_cr.requires_grad = True
    log_gap = Tensor([np.log(max(init_r_max - init_cr, 1e-5))], dtype=dtypes.float32)
    log_gap.requires_grad = True
    log_tau = Tensor([np.log(max(init_tau, 1e-5))], dtype=dtypes.float32)
    log_tau.requires_grad = True

    optimizer = Adam([log_cr, log_gap, log_tau], lr=lr)
    x_tensor = Tensor(spend, dtype=dtypes.float32)
    y_tensor = Tensor(rate, dtype=dtypes.float32)

    for _ in range(epochs):
      optimizer.zero_grad()
      cr = log_cr.exp()
      tau = log_tau.exp()
      denom = x_tensor / tau + 1e-8
      y_pred = cr + log_gap.exp() * (1.0 - (-x_tensor / tau).exp()) / denom

      if q is not None:
        diff = y_pred - y_tensor
        loss = (diff * q).maximum(diff * (q - 1)).mean()
      else:
        loss = ((y_pred - y_tensor) ** 2).mean()

      loss.backward()
      optimizer.step()

    Tensor.training = False
    final_cr = log_cr.exp().numpy().item()
    final_r_max = final_cr + log_gap.exp().numpy().item()
    final_tau = log_tau.exp().numpy().item()

    return cls(cr=final_cr, r_max=final_r_max, tau=final_tau, model_type="ward-smith")

  @classmethod
  def fit_omni_domain(cls, spend, results, is_rate=False, q=None, epochs=3000, lr=0.01):
    spend = np.array(spend, dtype=float)
    results = np.array(results, dtype=float)
    mask = spend > 1e-5
    spend = spend[mask]
    results = results[mask]

    if is_rate:
      rate = results
    else:
      rate = results / spend

    Tensor.training = True

    init_cr = np.median(rate)
    init_r_end = np.min(rate) * 0.9
    init_r_max = np.max(rate) * 1.1
    init_tau1 = np.median(spend) * 0.1
    init_tau2 = np.median(spend) * 2.0

    log_r_end = Tensor([np.log(max(init_r_end, 1e-5))], dtype=dtypes.float32)
    log_r_end.requires_grad = True
    log_gap_cr = Tensor([np.log(max(init_cr - init_r_end, 1e-5))], dtype=dtypes.float32)
    log_gap_cr.requires_grad = True
    log_gap_max = Tensor([np.log(max(init_r_max - init_cr, 1e-5))], dtype=dtypes.float32)
    log_gap_max.requires_grad = True
    log_tau1 = Tensor([np.log(max(init_tau1, 1e-5))], dtype=dtypes.float32)
    log_tau1.requires_grad = True
    log_gap_tau = Tensor([np.log(max(init_tau2 - init_tau1, 1e-5))], dtype=dtypes.float32)
    log_gap_tau.requires_grad = True

    optimizer = Adam([log_r_end, log_gap_cr, log_gap_max, log_tau1, log_gap_tau], lr=lr)

    x_tensor = Tensor(spend, dtype=dtypes.float32)
    y_tensor = Tensor(rate, dtype=dtypes.float32)

    for _ in range(epochs):
      optimizer.zero_grad()
      r_end = log_r_end.exp()
      cr = r_end + log_gap_cr.exp()
      tau_1 = log_tau1.exp()
      tau_2 = tau_1 + log_gap_tau.exp()

      y_pred = log_gap_max.exp() * (-x_tensor / tau_1).exp() + log_gap_cr.exp() * (-x_tensor / tau_2).exp() + r_end

      if q is not None:
        diff = y_pred - y_tensor
        loss = (diff * q).maximum(diff * (q - 1)).mean()
      else:
        loss = ((y_pred - y_tensor) ** 2).mean()

      loss.backward()
      optimizer.step()

    Tensor.training = False
    final_r_end = log_r_end.exp().numpy().item()
    final_cr = final_r_end + log_gap_cr.exp().numpy().item()
    final_r_max = final_cr + log_gap_max.exp().numpy().item()
    final_tau_1 = log_tau1.exp().numpy().item()
    final_tau_2 = final_tau_1 + log_gap_tau.exp().numpy().item()

    return cls(cr=final_cr, r_max=final_r_max, r_end=final_r_end, tau_1=final_tau_1, tau_2=final_tau_2, model_type="omni-domain")

  def plot_frontier(self, spend=None, results=None, is_rate=False):
    """Plots the fitted maximal frontier curve alongside the observed performance data."""
    # Google Brand Colors
    G_BLUE = '#4285F4'
    G_RED = '#EA4335'
    G_YELLOW = '#FBBC04'
    G_GREEN = '#34A853'
    G_GRAY = '#5F6368'
    G_LIGHT_GRAY = '#F8F9FA'

    if spend is not None:
      spend = np.array(spend, dtype=float)
      max_x = np.max(spend) * 1.2
    else:
      max_x = (self.r_prime * 5 if self.r_prime is not None else (self.tau * 5 if self.tau is not None else self.tau_2 * 3))

    x_vals = np.linspace(max_x * 0.01, max_x, 500)
    y_vals = self.predict_rate(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    ax1.set_facecolor('white')

    # Plot frontier curve
    ax1.plot(x_vals, y_vals, color=G_BLUE, linewidth=3.5, label="Maximal Frontier", zorder=3)

    # Plot raw data points if provided
    if spend is not None and results is not None:
      results = np.array(results, dtype=float)
      if is_rate:
        rate = results
      else:
        rate = results / (spend + 1e-5)
      ax1.scatter(spend, rate, color=G_GRAY, alpha=0.4, s=30, edgecolors='white', linewidth=0.8, label="Observed Performance", zorder=2)

    # Labels and grid
    ax1.set_xlabel('Spend', fontsize=11, color=G_GRAY, fontweight='500', labelpad=10)
    ax1.set_ylabel('Efficiency Rate', color=G_BLUE, fontsize=11, fontweight='500', labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=10, colors=G_GRAY)

    # Format axes
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:g}k' if x >= 1000 else f'${x:g}'))

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(G_LIGHT_GRAY)
    ax1.spines['bottom'].set_color(G_LIGHT_GRAY)
    ax1.grid(True, linestyle='-', alpha=0.1, color=G_GRAY)

    # Horizontal asymptote line for CR (or r_end for Omni Domain)
    asymptote_val = self.r_end if self.model_type == "omni-domain" else self.cr
    asymptote_label = "Asymptotic Floor" if self.model_type == "omni-domain" else "Critical Results"
    ax1.axhline(asymptote_val, color=G_RED, linestyle='--', linewidth=1.5, alpha=0.7, label=f"{asymptote_label} ({asymptote_val:.2f})")

    # Legend
    ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=1.0, fontsize=10)

    # Title
    plt.title(f'Marketing Efficiency Frontier Analysis ({self.model_type.upper()})', loc='left', fontsize=16, fontweight='bold', pad=25, color='#202124')

    # Subtitle with parameters dynamically formatted based on model
    if self.model_type == "2-parameter":
      sub_text = f"Parameters: CR={self.cr:.2f}, R'={self.r_prime:,.0f}"
    elif self.model_type in ("3-parameter", "morton"):
      sub_text = f"Parameters: CR={self.cr:.2f}, R'={self.r_prime:,.0f}, R_max={self.r_max:.2f}"
    elif self.model_type in ("exponential", "cp-exp"):
      sub_text = f"Parameters: CR={self.cr:.2f}, R_max={self.r_max:.2f}, τ={self.tau:,.0f}"
    elif self.model_type == "ward-smith":
      sub_text = f"Parameters: CR={self.cr:.2f}, R_max={self.r_max:.2f}, τ={self.tau:,.0f}"
    elif self.model_type == "omni-domain":
      sub_text = f"Parameters: R_end={self.r_end:.2f}, CR={self.cr:.2f}, R_max={self.r_max:.2f}, τ₁={self.tau_1:,.0f}, τ₂={self.tau_2:,.0f}"
    else:
      sub_text = ""

    fig.text(0.125, 0.91, sub_text, fontsize=10, color=G_GRAY)

    plt.tight_layout()
    plt.show()
