import numpy as np

def days_to_theta(days):
  """Converts half-life in days/periods to geometric decay rate theta.

  Formula: theta = 0.5 ** (1 / days)
  """
  if days <= 0: return 0.0
  return 0.5 ** (1.0 / days)

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
  """Calculates the Hill Function value: f(x) = (beta * x^alpha) / (K^alpha + x^alpha)."""
  spend = np.array(spend, dtype=float) + 1e-5
  return (beta * (spend ** alpha)) / (K ** alpha + spend ** alpha)

def hill_first_derivative(spend, beta, alpha, K):
  """Calculates the first derivative of the Hill Function (Marginal ROAS)."""
  spend = np.array(spend, dtype=float) + 1e-5
  numerator = beta * alpha * (K ** alpha) * (spend ** (alpha - 1))
  denominator = (K ** alpha + spend ** alpha) ** 2
  return numerator / denominator

def get_inflection_point(alpha, K):
  """Calculates the inflection point where marginal return peaks (f''(x) = 0)."""
  if alpha <= 1: return 0.0
  return K * (((alpha - 1) / (alpha + 1)) ** (1 / alpha))
