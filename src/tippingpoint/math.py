import numpy as np

def days_to_theta(days):
  """Converts half-life in days/periods to geometric decay rate theta.

  Formula: theta = 0.5 ** (1 / days)
  """
  if days <= 0:
    return 0.0
  return float(0.5 ** (1.0 / days))

def geometric_adstock(spend, theta):
  """Applies geometric adstock decay to a spend array.

  Formula: S_t_adstocked = S_t + theta * S_{t-1_adstocked}
  """
  spend = np.array(spend, dtype=float)
  adstocked = np.zeros_like(spend)
  current = 0.0
  for t in range(len(spend)):
    current = spend[t] + theta * current
    adstocked[t] = current
  return adstocked

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
