import numpy as np

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
