import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes

def tinygrad_geometric_adstock(spend, theta):
  """Applies geometric adstock decay in Tinygrad (vectorized Toeplitz weights)."""
  N = spend.shape[0]
  grid = Tensor.arange(N)
  diff = grid.unsqueeze(1) - grid.unsqueeze(0)
  mask = (diff >= 0).cast(dtypes.float32)
  diff_safe = diff * mask
  weights = (theta ** diff_safe) * mask
  return weights.matmul(spend)

def fit_mle_gradient(spend_array, return_array, epochs=5000, lr=0.05, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None, fit_baseline=False):
  """Fits a Hill Curve to historical data using MLE (Adam optimizer), with optional adstock and baseline."""
  spend_arr = np.array(spend_array, dtype=float)
  return_arr = np.array(return_array, dtype=float)

  max_x = float(np.max(spend_arr)) if np.any(spend_arr > 0) else 1.0
  if max_x <= 0:
    max_x = 1.0
  max_y = float(np.max(return_arr)) if np.any(return_arr > 0) else 1.0
  if max_y <= 0:
    max_y = 1.0

  spend_scaled = spend_arr / max_x
  return_scaled = return_arr / max_y
  median_x_scaled = float(np.median(spend_scaled[spend_scaled > 0])) if np.any(spend_scaled > 0) else 0.5

  Tensor.training = True
  x = Tensor(spend_scaled, dtype=dtypes.float32)
  x.requires_grad = False
  y = Tensor(return_scaled, dtype=dtypes.float32)
  y.requires_grad = False

  log_beta = Tensor([np.log(1.2)], dtype=dtypes.float32)
  log_beta.requires_grad = True
  log_k = Tensor([np.log(median_x_scaled + 1e-5)], dtype=dtypes.float32)
  log_k.requires_grad = True
  log_alpha = Tensor([0.0], dtype=dtypes.float32)
  log_alpha.requires_grad = True

  optimizable_params = [log_beta, log_k, log_alpha]
  log_baseline = None
  if fit_baseline:
    log_baseline = Tensor([np.log(0.1)], dtype=dtypes.float32)
    log_baseline.requires_grad = True
    optimizable_params.append(log_baseline)

  theta_tensor = None
  theta_min, theta_max = 0.0, 0.999

  if adstock_type == "none":
    pass
  elif adstock_type == "fixed":
    if adstock_fixed_days is not None and adstock_fixed_days > 0:
      theta_val = 0.5 ** (1.0 / adstock_fixed_days)
    else:
      theta_val = 0.0
    theta_tensor = Tensor([theta_val], dtype=dtypes.float32)
  elif adstock_type == "free":
    adstock_w = Tensor([0.0], dtype=dtypes.float32)
    adstock_w.requires_grad = True
    optimizable_params.append(adstock_w)
  elif adstock_type == "bounded":
    if adstock_bounds is not None:
      min_days, max_days = adstock_bounds
      theta_min = 0.5 ** (1.0 / min_days) if min_days > 0 else 0.0
      theta_max = 0.5 ** (1.0 / max_days) if max_days > 0 else 0.0
      if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min
    adstock_w = Tensor([0.0], dtype=dtypes.float32)
    adstock_w.requires_grad = True
    optimizable_params.append(adstock_w)

  optimizer = Adam(optimizable_params, lr=lr)

  Tensor.training = True
  prev_loss = float('inf')
  with Tensor.train():
    for epoch in range(epochs):
      optimizer.zero_grad()
      beta = log_beta.exp()
      k = log_k.exp()
      alpha = log_alpha.exp()
      base = log_baseline.exp() if fit_baseline else Tensor([0.0], dtype=dtypes.float32)

      # Apply adstock transformation
      if adstock_type == "none":
        x_adstocked = x
      elif adstock_type == "fixed":
        x_adstocked = tinygrad_geometric_adstock(x, theta_tensor)
      elif adstock_type == "free":
        theta = adstock_w.sigmoid() * 0.999
        x_adstocked = tinygrad_geometric_adstock(x, theta)
      elif adstock_type == "bounded":
        theta = theta_min + (theta_max - theta_min) * adstock_w.sigmoid()
        x_adstocked = tinygrad_geometric_adstock(x, theta)

      ratio = (x_adstocked + 1e-5) / k
      ratio_alpha = ratio ** alpha
      y_pred = base + (beta * ratio_alpha) / (1.0 + ratio_alpha)
      loss = ((y_pred - y) ** 2).mean()
      loss.backward()
      optimizer.step()

      if epochs >= 500 and epoch % 100 == 0:
        curr_loss = loss.numpy().item()
        if abs(prev_loss - curr_loss) < 1e-8:
          break
        prev_loss = curr_loss
  Tensor.training = False

  beta_val = float(log_beta.exp().numpy().item() * max_y)
  alpha_val = float(log_alpha.exp().numpy().item())
  k_val = float(log_k.exp().numpy().item() * max_x)
  final_loss = float(loss.numpy().item() * (max_y ** 2))
  baseline_val = float(log_baseline.exp().numpy().item() * max_y) if fit_baseline else 0.0

  if adstock_type == "none":
    theta_val = 0.0
  elif adstock_type == "fixed":
    theta_val = float(theta_tensor.numpy().item())
  elif adstock_type == "free":
    theta_val = float((adstock_w.sigmoid() * 0.999).numpy().item())
  elif adstock_type == "bounded":
    theta_val = float((theta_min + (theta_max - theta_min) * adstock_w.sigmoid()).numpy().item())

  if fit_baseline:
    return beta_val, alpha_val, k_val, theta_val, final_loss, baseline_val
  return beta_val, alpha_val, k_val, theta_val, final_loss

