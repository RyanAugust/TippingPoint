import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class CurveVisualizer:
  """Handles visualization for media response curves."""

  # Google Brand Colors
  G_BLUE = '#4285F4'
  G_RED = '#EA4335'
  G_YELLOW = '#FBBC04'
  G_GREEN = '#34A853'
  G_GRAY = '#5F6368'
  G_LIGHT_GRAY = '#F8F9FA'

  @classmethod
  def plot_response_curve(cls, model, target_mroas=1.0, current_spend=None, show_intervals=True, scatter=None, include_baseline=False):
    """Generates a visualization of the media response and marginal return curves."""
    min_spend = model.get_minimal_marginal_cost_point()
    max_spend = model.get_diminishing_returns_point(target_mroas)

    # Determine plot limits
    max_x = max_spend * 1.5 if max_spend else min_spend * 4
    if current_spend: max_x = max(max_x, current_spend * 1.2)
    if scatter is not None: max_x = max(max_x, np.max(scatter[0]) * 1.1)

    if max_spend and max_x > 100 * max_spend:
      max_x = max_spend * 3.0

    x_vals = np.linspace(0, max_x, 500)
    has_intervals = False
    interval_label = "Uncertainty Interval"

    if show_intervals and model.posterior_samples:
      y_return, y_return_low, y_return_high = model.predict_incremental_return(
          x_vals, return_interval=True, confidence_level=0.90, include_baseline=include_baseline
      )
      y_mroas, y_mroas_low, y_mroas_high = model.predict_marginal_return(x_vals, return_interval=True, confidence_level=0.90)
      has_intervals = True
      interval_label = "90% Credible Interval"
    elif show_intervals and model.covariance_matrix is not None:
      y_return, y_return_low, y_return_high = model.predict_incremental_return(
          x_vals, return_interval=True, confidence_level=0.95, include_baseline=include_baseline
      )
      y_mroas, y_mroas_low, y_mroas_high = model.predict_marginal_return(x_vals, return_interval=True, confidence_level=0.95)
      has_intervals = True
      interval_label = "95% Confidence Interval"
    else:
      y_return = model.predict_incremental_return(x_vals, include_baseline=include_baseline)
      y_mroas = model.predict_marginal_return(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    ax1.set_facecolor('white')

    # Primary Axis: Response Curve
    curve_label = "Total Return" if include_baseline else "Incremental Return"
    y_axis_label = "Total Return ($)" if include_baseline else "Incremental Return ($)"
    ax1.plot(x_vals, y_return, color=cls.G_BLUE, linewidth=3.5, label=curve_label, zorder=3)
    if has_intervals:
      ax1.fill_between(x_vals, y_return_low, y_return_high, color=cls.G_BLUE, alpha=0.15, label=interval_label, zorder=2)

    ax1.set_xlabel('Spend', fontsize=11, color=cls.G_GRAY, fontweight='500', labelpad=10)
    ax1.set_ylabel(y_axis_label, color=cls.G_BLUE, fontsize=11, fontweight='500', labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=10, colors=cls.G_GRAY)

    # Secondary Axis: Marginal Return
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_mroas, color=cls.G_GRAY, linestyle=(0, (5, 2)), linewidth=1.5, label="Marginal ROAS", alpha=0.6, zorder=1)
    if has_intervals:
      ax2.fill_between(x_vals, y_mroas_low, y_mroas_high, color=cls.G_GRAY, alpha=0.05, zorder=0)

    ax2.set_ylabel('Marginal ROAS (mROAS)', color=cls.G_GRAY, fontsize=11, fontweight='500', labelpad=10)
    ax2.tick_params(axis='y', labelcolor=cls.G_GRAY, labelsize=10)
    ax2.axhline(target_mroas, color=cls.G_RED, linestyle=':', linewidth=1, alpha=0.5, label="Target mROAS")

    # Optimal Scaling Zone
    if max_spend and max_spend > min_spend:
      ax1.axvspan(min_spend, max_spend, color=cls.G_GREEN, alpha=0.08, label='Optimal Scaling Zone', zorder=0)
      # Use blended transform (x in data coords, y in axes fraction) for robust placement
      ax1.text((min_spend + max_spend) / 2.0, 0.03, 'OPTIMAL ZONE',
               transform=ax1.get_xaxis_transform(),
               horizontalalignment='center', verticalalignment='bottom',
               fontsize=9, color=cls.G_GREEN, fontweight='bold', alpha=0.7)

    # Current Spend marker
    if current_spend:
      ax1.axvline(current_spend, color=cls.G_RED, linestyle='--', linewidth=1.5, alpha=0.8, label=f"Current Spend (${current_spend:,.0f})", zorder=4)
      curr_ret = model.predict_incremental_return(current_spend, include_baseline=include_baseline)
      ax1.scatter(current_spend, curr_ret, color=cls.G_RED, s=60, edgecolors='white', linewidth=1.5, zorder=5)

    # Scatter data
    if scatter is not None:
      scatter_spend, scatter_return = scatter
      scatter_spend_adstocked = model.adstock_spend(scatter_spend)
      has_adstock = (model.theta > 0) or (model.adstock_type and model.adstock_type != "none")
      ax1.scatter(scatter_spend_adstocked, scatter_return, color=cls.G_BLUE, alpha=0.3, s=40, edgecolors='white', linewidth=0.8,
                  label="Historical Data (Adstocked)" if has_adstock else "Historical Data", zorder=1)

    # Markers for key points
    if min_spend > 0:
      ax2.scatter(min_spend, model.predict_marginal_return(min_spend), marker='o', color=cls.G_YELLOW, s=100, edgecolors=cls.G_GRAY, linewidth=1, label="Peak Efficiency", zorder=6)

    # Formatting with smart scaling ($M, $k, $)
    def format_spend(x, p):
      if abs(x) >= 1e6:
        return f'${x*1e-6:g}M'
      elif abs(x) >= 1e3:
        return f'${x*1e-3:g}k'
      else:
        return f'${x:g}'

    def format_return(x, p):
      if abs(x) >= 1e6:
        return f'{x*1e-6:g}M'
      elif abs(x) >= 1e3:
        return f'{x*1e-3:g}k'
      else:
        return f'{x:g}'

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_spend))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_return))
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    # Hide spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(cls.G_LIGHT_GRAY)
    ax1.spines['bottom'].set_color(cls.G_LIGHT_GRAY)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax1.grid(True, linestyle='-', alpha=0.1, color=cls.G_GRAY)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, facecolor='white', framealpha=1.0, fontsize=10)

    # Title
    plt.title(f'Media Response Analysis: {model.channel_name}', loc='left', fontsize=16, fontweight='bold', pad=25, color='#202124')

    # Subtitle with parameters
    fig.text(0.125, 0.91, f'Hill Curve Parameters: α={model.alpha:.2f}, K={model.K:,.0f}, β={model.beta:,.0f}',
             fontsize=10, color=cls.G_GRAY)

    plt.tight_layout()
    return fig

  @classmethod
  def plot_executive_view(
      cls,
      model,
      target_mroas=1.0,
      current_spend=None,
      show_intervals=True,
      scatter=None,
      include_baseline=False,
      figsize=(16, 7),
  ):
    """Generates a simplified, uncluttered 2-panel presentation-ready view for executives.

    Panel 1 (Left): Saturation curve showing expected return vs spend.
    Panel 2 (Right): Marginal return curve showing efficiency (mROAS) vs spend.

    Clearly annotates current investment, peak efficiency, diminishing returns,
    and the optimal scaling zone directly on the chart without repeating them in the legend.
    """
    min_spend = model.get_minimal_marginal_cost_point() or 0.0
    max_spend = model.get_diminishing_returns_point(target_mroas, warn_unreachable=False)

    # Determine plot x limits
    max_x = max_spend * 1.4 if max_spend else (min_spend * 3.5 if min_spend > 0 else model.K * 2.5)
    if current_spend:
      max_x = max(max_x, current_spend * 1.25)
    if scatter is not None:
      max_x = max(max_x, float(np.max(scatter[0])) * 1.1)

    if max_spend and max_x > 50 * max_spend:
      max_x = max_spend * 2.5

    x_vals = np.linspace(0, max_x, 500)
    has_intervals = False
    interval_label = "Uncertainty Interval"

    if show_intervals and model.posterior_samples:
      y_return, y_return_low, y_return_high = model.predict_incremental_return(
          x_vals, return_interval=True, confidence_level=0.90, include_baseline=include_baseline
      )
      y_mroas, y_mroas_low, y_mroas_high = model.predict_marginal_return(
          x_vals, return_interval=True, confidence_level=0.90
      )
      has_intervals = True
      interval_label = "90% Credible Interval"
    elif show_intervals and model.covariance_matrix is not None:
      y_return, y_return_low, y_return_high = model.predict_incremental_return(
          x_vals, return_interval=True, confidence_level=0.95, include_baseline=include_baseline
      )
      y_mroas, y_mroas_low, y_mroas_high = model.predict_marginal_return(
          x_vals, return_interval=True, confidence_level=0.95
      )
      has_intervals = True
      interval_label = "95% Confidence Interval"
    else:
      y_return = model.predict_incremental_return(x_vals, include_baseline=include_baseline)
      y_mroas = model.predict_marginal_return(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Google Sans', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Formatting helper with clean executive scaling ($M, $k, $)
    def format_currency(x, p=None):
      val = float(x)
      abs_val = abs(val)
      if abs_val >= 1e6:
        formatted = f'{val*1e-6:.2f}'.rstrip('0').rstrip('.')
        return f'${formatted}M'
      elif abs_val >= 1e3:
        formatted = f'{val*1e-3:.1f}'.rstrip('0').rstrip('.')
        return f'${formatted}k'
      else:
        return f'${val:,.0f}'

    def format_num(x, p=None):
      val = float(x)
      abs_val = abs(val)
      if abs_val >= 1e6:
        formatted = f'{val*1e-6:.2f}'.rstrip('0').rstrip('.')
        return f'{formatted}M'
      elif abs_val >= 1e3:
        formatted = f'{val*1e-3:.1f}'.rstrip('0').rstrip('.')
        return f'{formatted}k'
      elif abs_val >= 1:
        return f'{val:.2f}'
      elif abs_val >= 0.01:
        return f'{val:.2f}'
      else:
        return f'{val:.4g}'

    # --------------------------------------------------------------------------
    # PANEL 1: Saturation / Total Return Curve
    # --------------------------------------------------------------------------
    curve_label = "Total Return" if include_baseline else "Incremental Return"
    y_axis_label = "Total Return ($)" if include_baseline else "Incremental Return ($)"

    ax1.plot(x_vals, y_return, color=cls.G_BLUE, linewidth=3.5, label=curve_label, zorder=3)
    if has_intervals:
      ax1.fill_between(x_vals, y_return_low, y_return_high, color=cls.G_BLUE, alpha=0.15, label=interval_label, zorder=2)

    if scatter is not None:
      scatter_spend, scatter_return = scatter
      scatter_spend_adstocked = model.adstock_spend(scatter_spend)
      ax1.scatter(scatter_spend_adstocked, scatter_return, color=cls.G_BLUE, alpha=0.25, s=35,
                  edgecolors='white', linewidth=0.8, label="Historical Data", zorder=1)

    # Optimal Scaling Zone on Panel 1
    if max_spend and max_spend > min_spend:
      ax1.axvspan(min_spend, max_spend, color=cls.G_GREEN, alpha=0.12, label='Optimal Scaling Zone', zorder=0)
      ax1.text((min_spend + max_spend) / 2.0, 0.03, 'OPTIMAL SCALING ZONE',
               transform=ax1.get_xaxis_transform(),
               horizontalalignment='center', verticalalignment='bottom',
               fontsize=9, color=cls.G_GREEN, fontweight='bold', alpha=0.85)

    # Markers and annotations on Panel 1
    # 1. Peak Efficiency
    ret_min = 0.0
    if min_spend > 0:
      ret_min = model.predict_incremental_return(min_spend, include_baseline=include_baseline)
      ax1.scatter(min_spend, ret_min, color=cls.G_YELLOW, s=110, edgecolors='#202124', linewidth=1.5, zorder=6)
      ax1.annotate(
          f"Peak Efficiency\n{format_currency(min_spend)}",
          xy=(min_spend, ret_min),
          xytext=(0, 22), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8E1', edgecolor=cls.G_YELLOW, alpha=0.9),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1)
      )

    # 2. Diminishing Returns / Hurdle Limit
    ret_max = 0.0
    if max_spend and max_spend > 0:
      ret_max = model.predict_incremental_return(max_spend, include_baseline=include_baseline)
      ax1.scatter(max_spend, ret_max, color=cls.G_GREEN, s=110, edgecolors='#202124', linewidth=1.5, zorder=6)
      ax1.annotate(
          f"Diminishing Returns\n{format_currency(max_spend)}",
          xy=(max_spend, ret_max),
          xytext=(0, -28), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=cls.G_GREEN, alpha=0.9),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1)
      )

    # 3. Current Investment
    curr_ret = 0.0
    if current_spend:
      curr_ret = model.predict_incremental_return(current_spend, include_baseline=include_baseline)
      ax1.axvline(current_spend, color=cls.G_RED, linestyle='--', linewidth=1.8, alpha=0.85, zorder=4)
      ax1.scatter(current_spend, curr_ret, color=cls.G_RED, s=120, edgecolors='white', linewidth=2, zorder=7)
      ax1.annotate(
          f"Current Spend: {format_currency(current_spend)}\nReturn: {format_currency(curr_ret)}",
          xy=(current_spend, curr_ret),
          xytext=(15, 25), textcoords="offset points",
          ha='left', fontsize=9, fontweight='bold', color=cls.G_RED,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE8E6', edgecolor=cls.G_RED, alpha=0.95),
          arrowprops=dict(arrowstyle='->', color=cls.G_RED, lw=1.5)
      )

    finite_ret = y_return[np.isfinite(y_return)]
    max_y1 = float(np.max(finite_ret)) if len(finite_ret) > 0 else 1000.0
    if has_intervals:
      finite_ret_high = y_return_high[np.isfinite(y_return_high)]
      if len(finite_ret_high) > 0:
        max_y1 = max(max_y1, float(np.max(finite_ret_high)))
    if current_spend and np.isfinite(curr_ret):
      max_y1 = max(max_y1, float(curr_ret))
    if max_spend and max_spend > 0 and np.isfinite(ret_max):
      max_y1 = max(max_y1, float(ret_max))

    ax1.set_title("1. Media Saturation & Return Curve", fontsize=13, fontweight='bold', color='#202124', pad=12, loc='left')
    ax1.set_xlabel("Spend", fontsize=11, color=cls.G_GRAY, fontweight='500', labelpad=8)
    ax1.set_ylabel(y_axis_label, fontsize=11, color='#202124', fontweight='500', labelpad=8)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y1 * 1.15)
    ax1.grid(True, linestyle='-', alpha=0.15, color=cls.G_GRAY)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(cls.G_LIGHT_GRAY)
    ax1.spines['bottom'].set_color(cls.G_LIGHT_GRAY)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)

    # --------------------------------------------------------------------------
    # PANEL 2: Marginal Return Curve (mROAS)
    # --------------------------------------------------------------------------
    ax2.plot(x_vals, y_mroas, color='#1A73E8', linewidth=3.0, label="Marginal Return (mROAS)", zorder=3)
    if has_intervals:
      ax2.fill_between(x_vals, y_mroas_low, y_mroas_high, color='#1A73E8', alpha=0.12, zorder=2)

    # Hurdle line without printing raw numeric value in legend
    ax2.axhline(target_mroas, color=cls.G_RED, linestyle='--', linewidth=1.8, alpha=0.8,
                label="Target Hurdle Rate", zorder=2)

    # Optimal Scaling Zone on Panel 2
    if max_spend and max_spend > min_spend:
      ax2.axvspan(min_spend, max_spend, color=cls.G_GREEN, alpha=0.12, label='Optimal Scaling Zone', zorder=0)
      ax2.text((min_spend + max_spend) / 2.0, 0.03, 'OPTIMAL SCALING ZONE',
               transform=ax2.get_xaxis_transform(),
               horizontalalignment='center', verticalalignment='bottom',
               fontsize=9, color=cls.G_GREEN, fontweight='bold', alpha=0.85)

    # Calculate y2 ceiling for generous annotation headroom
    finite_mroas = y_mroas[np.isfinite(y_mroas)]
    if len(finite_mroas) > 0:
      # If alpha < 1.0, the derivative at x=0 approaches infinity; filter out the boundary singularity
      if model.alpha < 1.0 and len(finite_mroas) > 10:
        max_y2 = float(np.percentile(finite_mroas[1:], 95)) * 1.6
      else:
        max_y2 = float(np.max(finite_mroas))
    else:
      max_y2 = float(target_mroas) * 2.0 if target_mroas else 5.0

    if has_intervals:
      finite_high = y_mroas_high[np.isfinite(y_mroas_high)]
      if len(finite_high) > 0:
        val_high = float(np.percentile(finite_high[1:], 95) * 1.6 if model.alpha < 1.0 and len(finite_high) > 10 else np.max(finite_high))
        max_y2 = max(max_y2, val_high)

    if np.isfinite(target_mroas):
      max_y2 = max(max_y2, float(target_mroas) * 1.2)

    # Markers on Panel 2
    if min_spend > 0:
      mroas_min = model.predict_marginal_return(min_spend)
      if np.isfinite(mroas_min):
        max_y2 = max(max_y2, float(mroas_min))
      ax2.scatter(min_spend, mroas_min, color=cls.G_YELLOW, s=110, edgecolors='#202124', linewidth=1.5, zorder=6)
      ax2.annotate(
          f"Peak Efficiency\n{format_num(mroas_min)} mROAS",
          xy=(min_spend, mroas_min),
          xytext=(0, 20), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8E1', edgecolor=cls.G_YELLOW, alpha=0.9),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1)
      )

    if max_spend and max_spend > 0:
      ax2.scatter(max_spend, target_mroas, color=cls.G_GREEN, s=110, edgecolors='#202124', linewidth=1.5, zorder=6)
      ax2.annotate(
          f"Hurdle Floor\n{format_currency(max_spend)}",
          xy=(max_spend, target_mroas),
          xytext=(0, -26), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='#E6F4EA', edgecolor=cls.G_GREEN, alpha=0.9),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1)
      )

    if current_spend:
      curr_mroas = model.predict_marginal_return(current_spend)
      if np.isfinite(curr_mroas):
        max_y2 = max(max_y2, float(curr_mroas) * 1.15)
      ax2.axvline(current_spend, color=cls.G_RED, linestyle='--', linewidth=1.8, alpha=0.85, zorder=4)
      ax2.scatter(current_spend, curr_mroas, color=cls.G_RED, s=120, edgecolors='white', linewidth=2, zorder=7)
      ax2.annotate(
          f"Current mROAS: {format_num(curr_mroas)}",
          xy=(current_spend, curr_mroas),
          xytext=(15, 20), textcoords="offset points",
          ha='left', fontsize=9, fontweight='bold', color=cls.G_RED,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE8E6', edgecolor=cls.G_RED, alpha=0.95),
          arrowprops=dict(arrowstyle='->', color=cls.G_RED, lw=1.5)
      )

    if not np.isfinite(max_y2) or max_y2 <= 0:
      max_y2 = 5.0

    ax2.set_title("2. Marginal Return & Scaling Efficiency", fontsize=13, fontweight='bold', color='#202124', pad=12, loc='left')
    ax2.set_xlabel("Spend", fontsize=11, color=cls.G_GRAY, fontweight='500', labelpad=8)
    ax2.set_ylabel("Marginal ROAS (mROAS)", fontsize=11, color='#202124', fontweight='500', labelpad=8)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_num))
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, max_y2 * 1.25)
    ax2.grid(True, linestyle='-', alpha=0.15, color=cls.G_GRAY)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(cls.G_LIGHT_GRAY)
    ax2.spines['bottom'].set_color(cls.G_LIGHT_GRAY)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)

    # --------------------------------------------------------------------------
    # Header & Executive Status Callout
    # --------------------------------------------------------------------------
    channel_name = model.channel_name or "Channel"
    fig.suptitle(f"Executive Decision Brief: {channel_name}", fontsize=15, fontweight='bold', color='#202124', y=0.98)

    # Subtitle with clear executive recommendation
    if current_spend is not None and max_spend is not None:
      if min_spend > 0 and current_spend < min_spend:
        status_text = f"Status: Warm-Up Zone (Under-Invested) • Scaling to {format_currency(min_spend)} will increase marginal return efficiency"
      elif current_spend <= max_spend:
        headroom = max_spend - current_spend
        status_text = f"Status: In Optimal Scaling Zone • Headroom of +{format_currency(headroom)} available before reaching diminishing returns"
      else:
        over = current_spend - max_spend
        status_text = f"Status: Diminishing Returns (Over-Saturated) • Spend exceeds hurdle rate threshold by {format_currency(over)}"
    else:
      status_text = "Strategic response analysis showing saturation capacity, marginal efficiency, and optimal scaling thresholds"

    fig.text(0.5, 0.93, status_text, fontsize=10.5, ha='center', color=cls.G_GRAY, style='italic', parse_math=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


