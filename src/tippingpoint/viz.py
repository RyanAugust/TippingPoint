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
  def plot_response_curve(cls, model, target_mroas=1.0, current_spend=None, show_intervals=True, scatter=None):
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
    if show_intervals and model.posterior_samples:
      y_returns_dist = model.predict_incremental_return(x_vals, use_samples=True)
      y_return = np.mean(y_returns_dist, axis=0)
      y_return_low = np.percentile(y_returns_dist, 5, axis=0)
      y_return_high = np.percentile(y_returns_dist, 95, axis=0)

      y_mroas_dist = model.predict_marginal_return(x_vals, use_samples=True)
      y_mroas = np.mean(y_mroas_dist, axis=0)
      y_mroas_low = np.percentile(y_mroas_dist, 5, axis=0)
      y_mroas_high = np.percentile(y_mroas_dist, 95, axis=0)
    else:
      y_return = model.predict_incremental_return(x_vals)
      y_mroas = model.predict_marginal_return(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    ax1.set_facecolor('white')

    # Primary Axis: Response Curve
    ax1.plot(x_vals, y_return, color=cls.G_BLUE, linewidth=3.5, label="Incremental Return", zorder=3)
    if show_intervals and model.posterior_samples:
      ax1.fill_between(x_vals, y_return_low, y_return_high, color=cls.G_BLUE, alpha=0.15, label="90% Credible Interval", zorder=2)

    ax1.set_xlabel('Spend', fontsize=11, color=cls.G_GRAY, fontweight='500', labelpad=10)
    ax1.set_ylabel('Incremental Return', color=cls.G_BLUE, fontsize=11, fontweight='500', labelpad=10)
    ax1.tick_params(axis='both', which='major', labelsize=10, colors=cls.G_GRAY)

    # Secondary Axis: Marginal Return
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_mroas, color=cls.G_GRAY, linestyle=(0, (5, 2)), linewidth=1.5, label="Marginal ROAS", alpha=0.6, zorder=1)
    if show_intervals and model.posterior_samples:
      ax2.fill_between(x_vals, y_mroas_low, y_mroas_high, color=cls.G_GRAY, alpha=0.05, zorder=0)

    ax2.set_ylabel('Marginal ROAS (mROAS)', color=cls.G_GRAY, fontsize=11, fontweight='500', labelpad=10)
    ax2.tick_params(axis='y', labelcolor=cls.G_GRAY, labelsize=10)
    ax2.axhline(target_mroas, color=cls.G_RED, linestyle=':', linewidth=1, alpha=0.5, label=f"Target mROAS ({target_mroas})")

    # Optimal Scaling Zone
    if max_spend and max_spend > min_spend:
      ax1.axvspan(min_spend, max_spend, color=cls.G_GREEN, alpha=0.08, label='Optimal Scaling Zone', zorder=0)
      ax1.text((min_spend + max_spend)/2, plt.ylim()[1]*0.02, 'OPTIMAL ZONE',
               horizontalalignment='center', fontsize=9, color=cls.G_GREEN, fontweight='bold', alpha=0.6)

    # Current Spend marker
    if current_spend:
      ax1.axvline(current_spend, color=cls.G_RED, linestyle='--', linewidth=1.5, alpha=0.8, label=f"Current Spend (${current_spend:,.0f})", zorder=4)
      ax1.scatter(current_spend, model.predict_incremental_return(current_spend), color=cls.G_RED, s=60, edgecolors='white', linewidth=1.5, zorder=5)

    # Scatter data
    if scatter is not None:
      scatter_spend, scatter_return = scatter
      if model.theta > 0:
        from .math import geometric_adstock
        scatter_spend = geometric_adstock(scatter_spend, model.theta)
      ax1.scatter(scatter_spend, scatter_return, color=cls.G_BLUE, alpha=0.3, s=40, edgecolors='white', linewidth=0.8, label="Historical Data (Adstocked)" if model.theta > 0 else "Historical Data", zorder=1)

    # Markers for key points
    if min_spend > 0:
      ax2.scatter(min_spend, model.predict_marginal_return(min_spend), marker='o', color=cls.G_YELLOW, s=100, edgecolors=cls.G_GRAY, linewidth=1, label="Peak Efficiency", zorder=6)

    # Formatting
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:g}k' if x >= 1000 else f'${x:g}'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x/1000:g}k' if x >= 1000 else f'{x:g}'))

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
