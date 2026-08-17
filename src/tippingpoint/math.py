import numpy as np
from scipy.signal import lfilter

def days_to_theta(days):
  """Converts half-life in days/periods to geometric decay rate theta.

  Formula: theta = 0.5 ** (1 / days)
  """
  if days <= 0:
    return 0.0
  return float(0.5 ** (1.0 / days))

def geometric_adstock(spend, theta):
  """Applies geometric adstock decay to a spend array using a vectorized recursive filter.

  Formula: S_t_adstocked = S_t + theta * S_{t-1_adstocked}
  """
  spend = np.asanyarray(spend, dtype=float)
  if spend.size == 0:
    return np.array([], dtype=float)
  if theta <= 0.0:
    return spend.copy()
  if theta >= 1.0:
    return np.cumsum(spend)
  return lfilter([1.0], [1.0, -float(theta)], spend)

def hill_function(spend, beta, alpha, K):
  """Calculates the Hill Function value: f(x) = (beta * (x/K)^alpha) / (1 + (x/K)^alpha)."""
  spend = np.asanyarray(spend, dtype=float)
  is_scalar = spend.ndim == 0 and not isinstance(beta, np.ndarray)

  spend_safe = np.maximum(spend, 0.0)
  ratio = spend_safe / K

  with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    ratio_alpha = ratio ** alpha
    val = np.where(np.isinf(ratio_alpha), 1.0, ratio_alpha / (1.0 + ratio_alpha))
    result = beta * np.where(spend_safe <= 0.0, 0.0, val)

  return float(result) if is_scalar else result

def hill_first_derivative(spend, beta, alpha, K):
  """Calculates the first derivative of the Hill Function (Marginal ROAS)."""
  spend = np.asanyarray(spend, dtype=float)
  is_scalar = spend.ndim == 0 and not isinstance(beta, np.ndarray)

  spend_safe = np.maximum(spend, 0.0)
  ratio = spend_safe / K

  with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
    ratio_alpha = ratio ** alpha
    ratio_alpha_minus_1 = ratio ** (alpha - 1.0)
    denom = (1.0 + ratio_alpha) ** 2
    val = np.where(np.isinf(denom), 0.0, ratio_alpha_minus_1 / denom)
    result = (beta * alpha / K) * val
    if np.any(spend < 0):
      result = np.where(spend < 0, 0.0, result)

  return float(result) if is_scalar else result

def get_inflection_point(alpha, K):
  """Calculates the inflection point where marginal return peaks (f''(x) = 0)."""
  if alpha <= 1.0:
    return 0.0
  return float(K * (((alpha - 1.0) / (alpha + 1.0)) ** (1.0 / alpha)))

def weibull_adstock(spend, shape, scale, adstock_type="pdf", max_lag=None):
  """Applies Weibull adstock transformation (PDF or CDF decay).

  Args:
    spend (array-like): 1D array of media spend over time.
    shape (float): Weibull shape parameter (k > 0). If k > 1, peak effect is delayed.
    scale (float): Weibull scale parameter (lambda > 0), controls decay duration.
    adstock_type (str): 'pdf' (peaked/delayed decay) or 'cdf' (cumulative retention).
    max_lag (int, optional): Maximum lag window. Defaults to full length of spend.

  Returns:
    np.ndarray: Adstocked effective spend array.
  """
  spend = np.array(spend, dtype=float)
  N = len(spend)
  if N == 0:
    return np.array([], dtype=float)
  if shape <= 0 or scale <= 0:
    return spend.copy()

  L = min(max_lag, N) if max_lag is not None else N
  lags = np.arange(L, dtype=float)

  if adstock_type.lower() == "pdf":
    # Weibull PDF over discrete lags: (shape / scale) * (lag / scale)^(shape - 1) * exp(-(lag / scale)^shape)
    x_val = (lags + 1.0) / scale
    weights = (shape / scale) * (x_val ** (shape - 1.0)) * np.exp(-(x_val ** shape))
  else:
    # Weibull CDF survival decay
    x_val = lags / scale
    weights = np.exp(-(x_val ** shape))

  weight_sum = np.sum(weights)
  if weight_sum > 0:
    weights = weights / weight_sum
  else:
    weights = np.zeros(L, dtype=float)
    weights[0] = 1.0

  # Vectorized 1D convolution
  adstocked = np.convolve(spend, weights, mode='full')[:N]
  return adstocked
