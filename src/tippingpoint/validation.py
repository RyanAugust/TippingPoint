import numpy as np
from scipy import stats

def validate_curve_experiments(model, experiments, spend_is_raw=True, verbose=False):
  """Evaluates a fitted MarketingReturnCurve against one or more incrementality experiments.

  Args:
    model: An instance of MarketingReturnCurve.
    experiments: A dictionary or list of dictionaries containing experimental results.
      Supported keys per experiment:
        - 'spend' (or 'raw_spend', 'adstocked_spend'): Media spend level tested.
        - 'lift' (or 'incremental_return', 'conversions'): Incremental response measured.
        - 'se' (or 'std_error'): Standard error of the measured lift (optional).
        - 'ci' (or 'confidence_interval'): (lower, upper) 95% confidence interval (optional).
        - 'name' (or 'channel', 'test_name'): Descriptive label for the test (optional).
    spend_is_raw: If True and the curve has theta > 0, converts raw daily test spend to
      steady-state effective adstocked spend (S_eff = S_raw / (1 - theta)).
    verbose: If True, prints a formatted validation report to stdout.

  Returns:
    dict: Detailed evaluation metrics including per-experiment errors, Z-scores,
          confidence interval coverage, and aggregate goodness-of-fit statistics.
  """
  if isinstance(experiments, dict):
    exp_list = [experiments]
  elif isinstance(experiments, (list, tuple)):
    exp_list = list(experiments)
  else:
    raise TypeError(f"experiments must be a dict or list of dicts, got {type(experiments)}")

  if not exp_list:
    raise ValueError("No experiments provided to validate.")

  per_exp_results = []

  for idx, exp in enumerate(exp_list):
    name = exp.get("name") or exp.get("test_name") or exp.get("channel") or f"Experiment_{idx+1}"

    # Extract spend
    if "adstocked_spend" in exp:
      eval_spend = float(exp["adstocked_spend"])
      raw_spend_val = eval_spend * (1.0 - model.theta) if (0.0 < model.theta < 1.0) else eval_spend
    elif "raw_spend" in exp:
      raw_spend_val = float(exp["raw_spend"])
      eval_spend = raw_spend_val / (1.0 - model.theta) if (spend_is_raw and 0.0 < model.theta < 1.0) else raw_spend_val
    elif "spend" in exp:
      raw_spend_val = float(exp["spend"])
      eval_spend = raw_spend_val / (1.0 - model.theta) if (spend_is_raw and 0.0 < model.theta < 1.0) else raw_spend_val
    else:
      raise KeyError(f"Experiment {name} missing 'spend', 'raw_spend', or 'adstocked_spend'.")

    # Extract measured lift
    if "lift" in exp:
      y_true = float(exp["lift"])
    elif "incremental_return" in exp:
      y_true = float(exp["incremental_return"])
    elif "conversions" in exp:
      y_true = float(exp["conversions"])
    else:
      raise KeyError(f"Experiment {name} missing 'lift', 'incremental_return', or 'conversions'.")

    # Model predicted incremental return (excluding organic baseline)
    y_pred = float(model.predict_incremental_return(eval_spend, include_baseline=False))

    error = y_pred - y_true
    abs_error = abs(error)
    pct_error = (error / y_true * 100.0) if y_true != 0.0 else 0.0

    # Extract standard error & confidence interval
    se = exp.get("se") or exp.get("std_error")
    ci = exp.get("ci") or exp.get("confidence_interval")

    ci_lower, ci_upper = None, None
    if ci is not None:
      ci_lower = float(ci[0])
      ci_upper = float(ci[1])
      if se is None:
        se = (ci_upper - ci_lower) / (2.0 * 1.96)
    elif se is not None:
      se = float(se)
      ci_lower = y_true - 1.96 * se
      ci_upper = y_true + 1.96 * se

    if se is not None and se > 0:
      z_score = float(error / se)
      p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_score))))
      in_ci = bool(ci_lower <= y_pred <= ci_upper)
    else:
      z_score = None
      p_value = None
      in_ci = None

    per_exp_results.append({
        "name": name,
        "spend": raw_spend_val,
        "effective_spend": eval_spend,
        "measured_lift": y_true,
        "predicted_lift": y_pred,
        "error": error,
        "abs_error": abs_error,
        "pct_error": pct_error,
        "std_error": se,
        "ci": (ci_lower, ci_upper) if ci_lower is not None else None,
        "z_score": z_score,
        "p_value": p_value,
        "in_95_ci": in_ci
    })

  n_total = len(per_exp_results)
  mae = float(np.mean([r["abs_error"] for r in per_exp_results]))
  mape = float(np.mean([abs(r["pct_error"]) for r in per_exp_results]))

  valid_z = [r["z_score"] for r in per_exp_results if r["z_score"] is not None]
  if valid_z:
    chi2_stat = float(np.sum([z**2 for z in valid_z]))
    dof = len(valid_z)
    chi2_reduced = float(chi2_stat / dof)
    omnibus_p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=dof))
    n_in_ci = sum(1 for r in per_exp_results if r["in_95_ci"] is True)
    coverage_pct = float((n_in_ci / dof) * 100.0)
  else:
    chi2_stat = None
    dof = 0
    chi2_reduced = None
    omnibus_p_value = None
    coverage_pct = None

  # Determine validation verdict
  if chi2_reduced is not None:
    if coverage_pct >= 90.0 and chi2_reduced <= 1.5:
      verdict = "EXCELLENT"
    elif coverage_pct >= 66.7 and chi2_reduced <= 3.84:
      verdict = "ALIGNED"
    else:
      verdict = "MISALIGNED"
  else:
    if mape <= 10.0:
      verdict = "EXCELLENT"
    elif mape <= 25.0:
      verdict = "ALIGNED"
    else:
      verdict = "MISALIGNED"

  result = {
      "channel": model.channel_name,
      "num_experiments": n_total,
      "verdict": verdict,
      "mae": mae,
      "mape": mape,
      "chi2": chi2_stat,
      "chi2_reduced": chi2_reduced,
      "omnibus_p_value": omnibus_p_value,
      "ci_coverage_pct": coverage_pct,
      "experiments": per_exp_results
  }

  if verbose:
    print(format_validation_report(result))

  return result


def validate_multichannel_experiments(mmm_model, experiments, spend_is_raw=True, verbose=False):
  """Evaluates a multi-channel MMM model across experiments across different channels.

  Args:
    mmm_model: MultiChannelMMM instance.
    experiments: List of experiment dictionaries, each containing a 'channel' key.
    spend_is_raw: Whether spends are raw daily spend.
    verbose: If True, prints a summary report.

  Returns:
    dict: Multi-channel validation summary with per-channel breakdown and global metrics.
  """
  if isinstance(experiments, dict):
    exp_list = [experiments]
  else:
    exp_list = list(experiments)

  channel_exp_map = {}
  for exp in exp_list:
    ch = exp.get("channel")
    if not ch:
      raise KeyError("Each experiment in multi-channel validation must specify a 'channel' key.")
    if ch not in mmm_model.channels:
      raise ValueError(f"Channel '{ch}' not found in MMM model channels: {list(mmm_model.channels.keys())}")
    channel_exp_map.setdefault(ch, []).append(exp)

  channel_reports = {}
  all_exp_results = []

  for ch, exps in channel_exp_map.items():
    ch_curve = mmm_model.channels[ch]
    rep = validate_curve_experiments(ch_curve, exps, spend_is_raw=spend_is_raw, verbose=False)
    channel_reports[ch] = rep
    all_exp_results.extend(rep["experiments"])

  n_total = len(all_exp_results)
  mae = float(np.mean([r["abs_error"] for r in all_exp_results]))
  mape = float(np.mean([abs(r["pct_error"]) for r in all_exp_results]))

  valid_z = [r["z_score"] for r in all_exp_results if r["z_score"] is not None]
  if valid_z:
    chi2_stat = float(np.sum([z**2 for z in valid_z]))
    dof = len(valid_z)
    chi2_reduced = float(chi2_stat / dof)
    omnibus_p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=dof))
    coverage_pct = float((sum(1 for r in all_exp_results if r["in_95_ci"] is True) / dof) * 100.0)
  else:
    chi2_stat = None
    chi2_reduced = None
    omnibus_p_value = None
    coverage_pct = None

  if chi2_reduced is not None:
    if coverage_pct >= 90.0 and chi2_reduced <= 1.5:
      verdict = "EXCELLENT"
    elif coverage_pct >= 66.7 and chi2_reduced <= 3.84:
      verdict = "ALIGNED"
    else:
      verdict = "MISALIGNED"
  else:
    if mape <= 10.0:
      verdict = "EXCELLENT"
    elif mape <= 25.0:
      verdict = "ALIGNED"
    else:
      verdict = "MISALIGNED"

  result = {
      "num_experiments": n_total,
      "verdict": verdict,
      "mae": mae,
      "mape": mape,
      "chi2": chi2_stat,
      "chi2_reduced": chi2_reduced,
      "omnibus_p_value": omnibus_p_value,
      "ci_coverage_pct": coverage_pct,
      "channels": channel_reports,
      "all_experiments": all_exp_results
  }

  if verbose:
    print(format_multichannel_validation_report(result))

  return result


def format_validation_report(report):
  """Formats a validation dictionary into a clean text summary."""
  lines = []
  lines.append(f"=== Incrementality Validation Report: {report['channel']} ===")
  lines.append(f"Overall Status:       {report['verdict']}")
  lines.append(f"Evaluated Tests:      {report['num_experiments']}")
  lines.append(f"Mean Absolute Error:  {report['mae']:,.2f}")
  lines.append(f"Mean Abs Pct Error:   {report['mape']:.2f}%")

  if report["ci_coverage_pct"] is not None:
    lines.append(f"95% CI Coverage:      {report['ci_coverage_pct']:.1f}%")
  if report["chi2_reduced"] is not None:
    lines.append(f"Reduced Chi-Square:   {report['chi2_reduced']:.3f} (p = {report['omnibus_p_value']:.4f})")

  lines.append("\n--- Test Details ---")
  header = f"{'Test Name':<18} | {'Spend':<10} | {'Measured':<10} | {'Predicted':<10} | {'Error':<9} | {'% Dev':<8} | {'Z-Score':<8} | {'In 95% CI'}"
  lines.append(header)
  lines.append("-" * len(header))

  for e in report["experiments"]:
    z_str = f"{e['z_score']:+.2f}" if e['z_score'] is not None else "N/A"
    in_ci_str = "YES" if e['in_95_ci'] is True else ("NO" if e['in_95_ci'] is False else "N/A")
    lines.append(
        f"{e['name']:<18} | ${e['spend']:<9,.0f} | {e['measured_lift']:<10,.1f} | {e['predicted_lift']:<10,.1f} | {e['error']:<+9,.1f} | {e['pct_error']:<+7.1f}% | {z_str:<8} | {in_ci_str}"
    )

  return "\n".join(lines)


def format_multichannel_validation_report(report):
  """Formats a multi-channel validation dictionary into a clean text summary."""
  lines = []
  lines.append("=== Multi-Channel Incrementality Validation Report ===")
  lines.append(f"Overall Status:       {report['verdict']}")
  lines.append(f"Total Tests:          {report['num_experiments']}")
  lines.append(f"Mean Absolute Error:  {report['mae']:,.2f}")
  lines.append(f"Mean Abs Pct Error:   {report['mape']:.2f}%")
  if report["ci_coverage_pct"] is not None:
    lines.append(f"95% CI Coverage:      {report['ci_coverage_pct']:.1f}%")
  if report["chi2_reduced"] is not None:
    lines.append(f"Reduced Chi-Square:   {report['chi2_reduced']:.3f} (p = {report['omnibus_p_value']:.4f})")

  for ch, ch_rep in report["channels"].items():
    lines.append(f"\n[{ch}] Status: {ch_rep['verdict']} | MAE: {ch_rep['mae']:,.1f} | MAPE: {ch_rep['mape']:.1f}%")
    for e in ch_rep["experiments"]:
      z_str = f"Z={e['z_score']:+.2f}" if e['z_score'] is not None else ""
      in_ci = "In 95% CI" if e['in_95_ci'] else "Outside 95% CI"
      lines.append(f"  - {e['name']}: Spend ${e['spend']:,.0f} -> Measured: {e['measured_lift']:,.1f}, Predicted: {e['predicted_lift']:,.1f} (Error: {e['error']:+,.1f}, {in_ci} {z_str})")

  return "\n".join(lines)
