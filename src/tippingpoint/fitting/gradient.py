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

def fit_multichannel_gradient(spend_data, return_array, channel_names=None, epochs=5000, lr=0.05, fit_baseline=True, adstock_types=None, adstock_bounds=None, adstock_fixed_days=None):
  """Fits a joint Multi-Channel Marketing Mix Model using Gradient Descent (Tinygrad Adam)."""
  from tippingpoint.models import MarketingReturnCurve

  if isinstance(spend_data, dict):
    channels = list(spend_data.keys())
    spend_dict = {c: np.array(spend_data[c], dtype=float) for c in channels}
  elif hasattr(spend_data, 'values') and hasattr(spend_data, 'columns'):
    channels = list(spend_data.columns)
    spend_dict = {c: np.array(spend_data[c].values, dtype=float) for c in channels}
  else:
    spend_mat = np.array(spend_data, dtype=float)
    if channel_names is None:
      channels = [f"Channel_{i+1}" for i in range(spend_mat.shape[1])]
    else:
      channels = list(channel_names)
    spend_dict = {channels[i]: spend_mat[:, i] for i in range(len(channels))}

  M = len(channels)
  return_arr = np.array(return_array, dtype=float)
  max_y = float(np.max(return_arr)) if np.any(return_arr > 0) else 1.0
  if max_y <= 0:
    max_y = 1.0
  y_scaled = Tensor(return_arr / max_y, dtype=dtypes.float32)
  y_scaled.requires_grad = False

  max_x_dict = {}
  x_scaled_tensors = {}
  optimizable_params = []
  channel_param_tensors = {}

  if adstock_types is None:
    adstock_types_dict = {c: "none" for c in channels}
  elif isinstance(adstock_types, str):
    adstock_types_dict = {c: adstock_types for c in channels}
  else:
    adstock_types_dict = adstock_types

  for c in channels:
    s_arr = spend_dict[c]
    max_x = float(np.max(s_arr)) if np.any(s_arr > 0) else 1.0
    if max_x <= 0:
      max_x = 1.0
    max_x_dict[c] = max_x
    s_scaled = s_arr / max_x
    med_x = float(np.median(s_scaled[s_scaled > 0])) if np.any(s_scaled > 0) else 0.5

    t_x = Tensor(s_scaled, dtype=dtypes.float32)
    t_x.requires_grad = False
    x_scaled_tensors[c] = t_x

    log_beta = Tensor([np.log(1.0 / max(M, 1))], dtype=dtypes.float32)
    log_beta.requires_grad = True
    log_k = Tensor([np.log(med_x + 1e-5)], dtype=dtypes.float32)
    log_k.requires_grad = True
    log_alpha = Tensor([0.0], dtype=dtypes.float32)
    log_alpha.requires_grad = True

    params_c = {'beta': log_beta, 'k': log_k, 'alpha': log_alpha}
    optimizable_params.extend([log_beta, log_k, log_alpha])

    ad_type = adstock_types_dict.get(c, "none")
    if ad_type == "free":
      w = Tensor([0.0], dtype=dtypes.float32)
      w.requires_grad = True
      params_c['adstock_w'] = w
      optimizable_params.append(w)
    elif ad_type == "bounded":
      b = adstock_bounds.get(c, (1.0, 14.0)) if isinstance(adstock_bounds, dict) else (adstock_bounds or (1.0, 14.0))
      theta_min = 0.5 ** (1.0 / b[0]) if b[0] > 0 else 0.0
      theta_max = 0.5 ** (1.0 / b[1]) if b[1] > 0 else 0.0
      if theta_min > theta_max:
        theta_min, theta_max = theta_max, theta_min
      w = Tensor([0.0], dtype=dtypes.float32)
      w.requires_grad = True
      params_c['adstock_w'] = w
      params_c['theta_bounds'] = (theta_min, theta_max)
      optimizable_params.append(w)
    elif ad_type == "fixed":
      days = adstock_fixed_days.get(c, 3.0) if isinstance(adstock_fixed_days, dict) else (adstock_fixed_days or 3.0)
      th = 0.5 ** (1.0 / days) if days > 0 else 0.0
      params_c['theta_fixed'] = Tensor([th], dtype=dtypes.float32)

    channel_param_tensors[c] = params_c

  log_baseline = None
  if fit_baseline:
    log_baseline = Tensor([np.log(0.1)], dtype=dtypes.float32)
    log_baseline.requires_grad = True
    optimizable_params.append(log_baseline)

  optimizer = Adam(optimizable_params, lr=lr)

  Tensor.training = True
  prev_loss = float('inf')
  with Tensor.train():
    for epoch in range(epochs):
      optimizer.zero_grad()
      y_pred = log_baseline.exp() if fit_baseline else Tensor([0.0], dtype=dtypes.float32)
      for c in channels:
        p = channel_param_tensors[c]
        x_c = x_scaled_tensors[c]
        ad_type = adstock_types_dict.get(c, "none")
        if ad_type == "free":
          th = p['adstock_w'].sigmoid() * 0.999
          x_ad = tinygrad_geometric_adstock(x_c, th)
        elif ad_type == "bounded":
          t_min, t_max = p['theta_bounds']
          th = t_min + (t_max - t_min) * p['adstock_w'].sigmoid()
          x_ad = tinygrad_geometric_adstock(x_c, th)
        elif ad_type == "fixed":
          x_ad = tinygrad_geometric_adstock(x_c, p['theta_fixed'])
        else:
          x_ad = x_c

        ratio = (x_ad + 1e-5) / p['k'].exp()
        ratio_alpha = ratio ** p['alpha'].exp()
        y_pred = y_pred + (p['beta'].exp() * ratio_alpha) / (1.0 + ratio_alpha)

      loss = ((y_pred - y_scaled) ** 2).mean()
      loss.backward()
      optimizer.step()

      if epochs >= 500 and epoch % 100 == 0:
        curr_loss = loss.numpy().item()
        if abs(prev_loss - curr_loss) < 1e-8:
          break
        prev_loss = curr_loss

  Tensor.training = False

  baseline_val = float(log_baseline.exp().numpy().item() * max_y) if fit_baseline else 0.0
  final_loss = float(loss.numpy().item() * (max_y ** 2))

  models_dict = {}
  for c in channels:
    p = channel_param_tensors[c]
    beta_c = float(p['beta'].exp().numpy().item() * max_y)
    alpha_c = float(p['alpha'].exp().numpy().item())
    k_c = float(p['k'].exp().numpy().item() * max_x_dict[c])
    ad_type = adstock_types_dict.get(c, "none")
    if ad_type == "free":
      theta_c = float((p['adstock_w'].sigmoid() * 0.999).numpy().item())
    elif ad_type == "bounded":
      t_min, t_max = p['theta_bounds']
      theta_c = float((t_min + (t_max - t_min) * p['adstock_w'].sigmoid()).numpy().item())
    elif ad_type == "fixed":
      theta_c = float(p['theta_fixed'].numpy().item())
    else:
      theta_c = 0.0

    model = MarketingReturnCurve(beta=beta_c, alpha=alpha_c, half_saturation_k=k_c, theta=theta_c, channel_name=c, baseline=0.0)
    model.update_loss(final_loss)
    models_dict[c] = model

  return models_dict, baseline_val, final_loss
