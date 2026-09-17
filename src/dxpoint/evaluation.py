import numpy as np

def evaluate_curve_fit(model, spend_array, return_array, verbose=False):
  """Evaluates statistical goodness-of-fit metrics for a fitted MarketingReturnCurve.

  Calculates R-squared, Adjusted R-squared, RMSE, MAE, MAPE, AIC, and BIC.

  Args:
    model: Fitted MarketingReturnCurve instance.
    spend_array: Array of spend observations.
    return_array: Array of observed return / response values.
    verbose: If True, prints a formatted table of metrics.

  Returns:
    dict: Statistical fit metrics.
  """
  x = np.asarray(spend_array, dtype=float)
  y = np.asarray(return_array, dtype=float)
  N = len(y)
  if N == 0:
    raise ValueError("Empty return_array provided for fit evaluation.")

  # Apply adstock transformation
  x_adstocked = model.adstock_spend(x)
  y_pred = model.predict_incremental_return(x_adstocked, include_baseline=True)

  residuals = y - y_pred
  ssr = float(np.sum(residuals ** 2))
  y_mean = float(np.mean(y))
  sst = float(np.sum((y - y_mean) ** 2))

  if sst > 1e-12:
    r_squared = float(1.0 - (ssr / sst))
  else:
    r_squared = 1.0 if ssr < 1e-12 else 0.0

  # Count fitted parameters: beta, alpha, K (+ baseline if > 0, + theta if > 0)
  p = 3
  if getattr(model, "baseline", 0.0) > 0.0:
    p += 1
  if getattr(model, "theta", 0.0) > 0.0:
    p += 1

  # Adjusted R^2
  if N > p + 1 and sst > 1e-12:
    adj_r_squared = float(1.0 - ((1.0 - r_squared) * (N - 1) / (N - p - 1)))
  else:
    adj_r_squared = r_squared

  rmse = float(np.sqrt(ssr / N))
  mae = float(np.mean(np.abs(residuals)))

  non_zero_mask = (y != 0.0)
  if np.any(non_zero_mask):
    mape = float(np.mean(np.abs(residuals[non_zero_mask] / y[non_zero_mask])) * 100.0)
  else:
    mape = 0.0

  # Information Criteria
  mse_safe = max(ssr / N, 1e-12)
  aic = float(N * np.log(mse_safe) + 2 * p)
  if N > p + 1:
    aicc = float(aic + (2 * p * (p + 1)) / (N - p - 1))
  else:
    aicc = aic
  bic = float(N * np.log(mse_safe) + p * np.log(N))

  result = {
      "channel": model.channel_name,
      "num_observations": N,
      "num_parameters": p,
      "r_squared": r_squared,
      "adj_r_squared": adj_r_squared,
      "rmse": rmse,
      "mae": mae,
      "mape": mape,
      "aic": aic,
      "aicc": aicc,
      "bic": bic,
      "ssr": ssr,
      "sst": sst,
      "residual_mean": float(np.mean(residuals)),
      "residual_std": float(np.std(residuals))
  }

  if verbose:
    print(format_fit_report(result))

  return result


def format_fit_report(metrics):
  """Formats a fit metrics dictionary into a readable summary table."""
  lines = []
  lines.append(f"=== Goodness-of-Fit Evaluation: {metrics['channel']} ===")
  lines.append(f"Observations (N):     {metrics['num_observations']}")
  lines.append(f"Fitted Parameters (p): {metrics['num_parameters']}")
  lines.append(f"R-Squared (R²):       {metrics['r_squared']:.4f}")
  lines.append(f"Adjusted R²:          {metrics['adj_r_squared']:.4f}")
  lines.append(f"RMSE:                 {metrics['rmse']:,.2f}")
  lines.append(f"MAE:                  {metrics['mae']:,.2f}")
  lines.append(f"MAPE:                 {metrics['mape']:.2f}%")
  lines.append(f"AIC / AICc:           {metrics['aic']:.2f} / {metrics['aicc']:.2f}")
  lines.append(f"BIC:                  {metrics['bic']:.2f}")
  lines.append(f"Residual Std Dev:     {metrics['residual_std']:,.2f}")
  return "\n".join(lines)
