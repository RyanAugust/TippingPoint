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

def fit_mle_gradient(spend_array, return_array, epochs=5000, lr=0.05, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None):
  """Fits a Hill Curve to historical data using MLE (Adam optimizer), with optional adstock."""
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

  optimizable_params = [log_beta, log_k, log_alpha]
  theta_tensor = None
  theta_min, theta_max = 0.0, 0.999

  if adstock_type == "none":
    pass
  elif adstock_type == "fixed":
    if adstock_fixed_days is not None:
      theta_val = 0.5 ** (1.0 / adstock_fixed_days) if adstock_fixed_days > 0 else 0.0
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
    adstock_w = Tensor([0.0], dtype=dtypes.float32)
    adstock_w.requires_grad = True
    optimizable_params.append(adstock_w)

  optimizer = Adam(optimizable_params, lr=lr)

  Tensor.traning = True
  with Tensor.train():
    for _ in range(epochs):
      optimizer.zero_grad()
      beta = log_beta.exp()
      k = log_k.exp()
      alpha = log_alpha.exp()

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

      x_safe = x_adstocked + 1e-5
      y_pred = (beta * (x_safe ** alpha)) / (k ** alpha + x_safe ** alpha)
      loss = ((y_pred - y) ** 2).mean()
      loss.backward()
      optimizer.step()
  Tensor.traning = False

  beta_val = log_beta.exp().numpy().item()
  alpha_val = log_alpha.exp().numpy().item()
  k_val = log_k.exp().numpy().item()
  final_loss = loss.numpy().item()

  if adstock_type == "none":
    theta_val = 0.0
  elif adstock_type == "fixed":
    theta_val = theta_tensor.numpy().item()
  elif adstock_type == "free":
    theta_val = (adstock_w.sigmoid() * 0.999).numpy().item()
  elif adstock_type == "bounded":
    theta_val = (theta_min + (theta_max - theta_min) * adstock_w.sigmoid()).numpy().item()

  return beta_val, alpha_val, k_val, theta_val, final_loss
