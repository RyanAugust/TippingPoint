import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import dtypes

def fit_mle_gradient(spend_array, return_array, epochs=5000, lr=0.05):
  """Fits a Hill Curve to historical data using MLE (Adam optimizer)."""
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
      x_safe = x + 1e-5
      y_pred = (beta * (x_safe ** alpha)) / (k ** alpha + x_safe ** alpha)
      loss = ((y_pred - y) ** 2).mean()
      loss.backward()
      optimizer.step()
  Tensor.traning = False

  beta_val = log_beta.exp().numpy().item()
  alpha_val = log_alpha.exp().numpy().item()
  k_val = log_k.exp().numpy().item()
  final_loss = loss.numpy().item()

  return beta_val, alpha_val, k_val, final_loss
