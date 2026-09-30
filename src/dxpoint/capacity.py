from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from .models import MarketingReturnCurve

@dataclass
class CapacityMultipliers:
  """Container for distilled capacity and spend scaling multipliers.

  Attributes:
    m_beta: Total combined capacity multiplier on beta (Beta_proj = m_beta * Beta_curr).
    m_k: Total combined spend dilation multiplier on K (K_proj = m_k * K_curr).
    m_beta_rf: Multiplier on beta derived from Reach & Frequency penetration.
    m_k_rf: Multiplier on K derived from Reach & Frequency penetration.
    m_beta_product: Multiplier on beta derived from Product/Funnel usage.
    m_k_product: Multiplier on K derived from Product/Funnel usage.
    synergy_damping: Damping factor rho applied when combining multiple drivers.
    rf_details: Dictionary containing granular R&F metrics and calculations.
    product_details: Dictionary containing granular product/funnel parameters.
  """
  m_beta: float
  m_k: float
  m_beta_rf: float = 1.0
  m_k_rf: float = 1.0
  m_beta_product: float = 1.0
  m_k_product: float = 1.0
  synergy_damping: float = 0.85
  rf_details: Dict[str, Any] = field(default_factory=dict)
  product_details: Dict[str, Any] = field(default_factory=dict)

  def summary(self) -> Dict[str, Any]:
    return {
        "m_beta_total": self.m_beta,
        "m_k_total": self.m_k,
        "rf_component": {
            "m_beta": self.m_beta_rf,
            "m_k": self.m_k_rf,
            "details": self.rf_details
        },
        "product_component": {
            "m_beta": self.m_beta_product,
            "m_k": self.m_k_product,
            "details": self.product_details
        },
        "synergy_damping": self.synergy_damping
    }


class CapacityProjector:
  """Auxiliary capacity prediction engine for MarketingReturnCurve models.

  Calculates user-defined or analytically derived multipliers (M_beta, M_K)
  based on:
  1. Reach & Frequency (R&F) penetration headroom and frequency normalization.
  2. Product usage and funnel architecture transitions (e.g. lower-funnel only
     vs full-funnel).

  Synthesizes multi-driver unlocks using a damped interaction formulation:
    M = 1.0 + rho_synergy * ((M_rf - 1.0) + (M_prod - 1.0))
  """

  def __init__(
      self,
      # Direct multiplier overrides
      m_beta: Optional[float] = None,
      m_k: Optional[float] = None,
      # Reach & Frequency parameters
      tam_size: Optional[float] = None,
      current_reach: Optional[float] = None,
      target_reach: Optional[float] = None,
      current_reach_penetration: Optional[float] = None,
      target_reach_penetration: Optional[float] = None,
      current_frequency: Optional[float] = None,
      target_frequency: Optional[float] = None,
      reach_elasticity: float = 0.65,
      frequency_elasticity: float = 0.50,
      spend_reach_elasticity: float = 0.75,
      frequency_capacity_factor: float = 0.15,
      # Product / Funnel parameters
      m_beta_product: Optional[float] = None,
      m_k_product: Optional[float] = None,
      current_funnel: Optional[str] = None,
      target_funnel: Optional[str] = None,
      funnel_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
      vertical: Optional[str] = None,
      vertical_multipliers: Optional[Union[Dict[str, Any], str]] = None,
      # Synergy synthesis parameters
      synergy_damping: float = 0.85
  ):
    self.m_beta_override = float(m_beta) if m_beta is not None else None
    self.m_k_override = float(m_k) if m_k is not None else None

    # Resolve vertical configuration and defaults
    self.vertical = vertical.lower() if vertical else None
    raw_vm = None
    if isinstance(vertical_multipliers, (str, bytes)) or (hasattr(vertical_multipliers, "__fspath__")):
      import json
      with open(vertical_multipliers, "r", encoding="utf-8") as f:
        raw_vm = json.load(f)
    elif vertical_multipliers is not None:
      raw_vm = vertical_multipliers

    if raw_vm is not None:
      self.vertical_multipliers = self._normalize_vertical_multipliers(raw_vm)
      self._raw_vertical_config = raw_vm
    else:
      self.vertical_multipliers = None
      self._raw_vertical_config = None

    v_defaults = (
        self._raw_vertical_config.get(self.vertical, {}).get("rf_targets", {})
        if (self._raw_vertical_config and self.vertical and self.vertical in self._raw_vertical_config) else {}
    )

    # R&F configuration
    self.tam_size = float(tam_size) if tam_size is not None else None
    self.current_reach = float(current_reach) if current_reach is not None else None
    self.target_reach = float(target_reach) if target_reach is not None else None
    self.current_reach_penetration = float(current_reach_penetration) if current_reach_penetration is not None else None

    if target_reach_penetration is not None:
      self.target_reach_penetration = float(target_reach_penetration)
    elif self.target_reach is None and "target_reach_penetration" in v_defaults:
      self.target_reach_penetration = float(v_defaults["target_reach_penetration"])
    else:
      self.target_reach_penetration = None

    self.current_frequency = float(current_frequency) if current_frequency is not None else None

    if target_frequency is not None:
      self.target_frequency = float(target_frequency)
    elif "target_frequency" in v_defaults:
      self.target_frequency = float(v_defaults["target_frequency"])
    else:
      self.target_frequency = 3.0

    self.reach_elasticity = float(reach_elasticity)
    self.frequency_elasticity = float(frequency_elasticity)
    self.spend_reach_elasticity = float(spend_reach_elasticity)
    self.frequency_capacity_factor = float(frequency_capacity_factor)

    # Product / Funnel configuration
    self.m_beta_product = float(m_beta_product) if m_beta_product is not None else None
    self.m_k_product = float(m_k_product) if m_k_product is not None else None
    self.current_funnel = current_funnel.lower() if current_funnel else None
    self.target_funnel = target_funnel.lower() if target_funnel else None
    self.funnel_multipliers = self._normalize_funnel_multipliers(funnel_multipliers)

    # Damping parameter
    if not (0.0 < synergy_damping <= 1.0):
      raise ValueError(f"synergy_damping must be in (0, 1], got {synergy_damping}")
    self.synergy_damping = float(synergy_damping)

  @staticmethod
  def _normalize_funnel_multipliers(fm: Optional[Dict[str, Dict[str, float]]]) -> Optional[Dict[str, Dict[str, float]]]:
    if not fm:
      return None
    return {k.lower(): {pk.lower(): float(pv) for pk, pv in v.items()} for k, v in fm.items()}

  @staticmethod
  def _normalize_vertical_multipliers(vm: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    if not vm:
      return None
    normalized = {}
    for vk, v in vm.items():
      normalized[vk.lower()] = {}
      for fk, fv in v.items():
        if isinstance(fv, dict) and fk != "rf_targets":
          normalized[vk.lower()][fk.lower()] = {pk.lower(): float(pv) for pk, pv in fv.items()}
    return normalized

  def _compute_rf_multipliers(self) -> Tuple[float, float, Dict[str, Any]]:
    """Calculates M_beta_rf and M_k_rf from Reach & Frequency metrics."""
    # Resolve reach penetration
    rho_curr = None
    rho_target = None

    if self.current_reach_penetration is not None:
      rho_curr = self.current_reach_penetration
    elif self.current_reach is not None and self.tam_size is not None and self.tam_size > 0:
      rho_curr = self.current_reach / self.tam_size

    if self.target_reach_penetration is not None:
      rho_target = self.target_reach_penetration
    elif self.target_reach is not None and self.tam_size is not None and self.tam_size > 0:
      rho_target = self.target_reach / self.tam_size

    details: Dict[str, Any] = {
        "rho_curr": rho_curr,
        "rho_target": rho_target,
        "current_frequency": self.current_frequency,
        "target_frequency": self.target_frequency,
    }

    m_beta_rf = 1.0
    m_k_rf = 1.0

    # 1. Reach headroom
    if rho_curr is not None and rho_target is not None and rho_curr > 0:
      if rho_target > rho_curr:
        reach_expansion_ratio = rho_target / rho_curr
        m_beta_reach = reach_expansion_ratio ** self.reach_elasticity
        m_k_reach = reach_expansion_ratio ** self.spend_reach_elasticity
        details["reach_expansion_ratio"] = reach_expansion_ratio
        details["m_beta_reach"] = m_beta_reach
        details["m_k_reach"] = m_k_reach
        m_beta_rf *= m_beta_reach
        m_k_rf *= m_k_reach
      else:
        details["reach_expansion_ratio"] = 1.0
        details["m_beta_reach"] = 1.0
        details["m_k_reach"] = 1.0

    # 2. Frequency normalization / waste reclamation
    if self.current_frequency is not None and self.current_frequency > 0:
      f_ratio = max(1.0, self.current_frequency / self.target_frequency)
      m_k_freq = f_ratio ** self.frequency_elasticity

      # Mild capacity lift from reclaimed frequency budget
      if self.current_frequency > self.target_frequency:
        waste_frac = 1.0 - (self.target_frequency / self.current_frequency)
        m_beta_freq = 1.0 + self.frequency_capacity_factor * waste_frac
      else:
        m_beta_freq = 1.0

      details["freq_ratio"] = f_ratio
      details["m_beta_freq"] = m_beta_freq
      details["m_k_freq"] = m_k_freq
      m_beta_rf *= m_beta_freq
      m_k_rf *= m_k_freq

    details["m_beta_rf"] = m_beta_rf
    details["m_k_rf"] = m_k_rf
    return m_beta_rf, m_k_rf, details

  def _compute_product_multipliers(self) -> Tuple[float, float, Dict[str, Any]]:
    """Calculates M_beta_product and M_k_product from user-defined vertical or funnel configurations."""
    details: Dict[str, Any] = {
        "current_funnel": self.current_funnel,
        "target_funnel": self.target_funnel,
        "vertical": self.vertical,
    }

    # Direct product multipliers take highest precedence if specified
    if self.m_beta_product is not None and self.m_k_product is not None:
      details["source"] = "direct_product_multipliers"
      return self.m_beta_product, self.m_k_product, details

    # Look up funnel multipliers from vertical dictionary if present
    active_funnel_map = self.funnel_multipliers
    if self.vertical and self.vertical_multipliers and self.vertical in self.vertical_multipliers:
      active_funnel_map = self.vertical_multipliers[self.vertical]
      details["source"] = f"vertical_multipliers[{self.vertical}]"

    if active_funnel_map and self.current_funnel and self.target_funnel:
      if self.current_funnel not in active_funnel_map:
        raise ValueError(
            f"current_funnel '{self.current_funnel}' not found in active funnel multipliers: "
            f"{list(active_funnel_map.keys())}"
        )
      if self.target_funnel not in active_funnel_map:
        raise ValueError(
            f"target_funnel '{self.target_funnel}' not found in active funnel multipliers: "
            f"{list(active_funnel_map.keys())}"
        )

      curr_entry = active_funnel_map[self.current_funnel]
      target_entry = active_funnel_map[self.target_funnel]

      b_curr = curr_entry.get("m_beta", curr_entry.get("beta", 1.0))
      k_curr = curr_entry.get("m_k", curr_entry.get("k", 1.0))
      b_target = target_entry.get("m_beta", target_entry.get("beta", 1.0))
      k_target = target_entry.get("m_k", target_entry.get("k", 1.0))

      m_beta_p = b_target / b_curr if b_curr > 0 else 1.0
      m_k_p = k_target / k_curr if k_curr > 0 else 1.0

      details["source"] = "funnel_transition"
      details["base_entry"] = curr_entry
      details["target_entry"] = target_entry
      details["m_beta_product"] = m_beta_p
      details["m_k_product"] = m_k_p
      return m_beta_p, m_k_p, details

    # Fallback to single product multiplier if only one was provided
    m_beta_p = self.m_beta_product if self.m_beta_product is not None else 1.0
    m_k_p = self.m_k_product if self.m_k_product is not None else 1.0
    details["source"] = "default_or_partial"
    return m_beta_p, m_k_p, details

  def compute_multipliers(self) -> CapacityMultipliers:
    """Computes final combined multipliers using the user-requested damped formulation."""
    # Direct overall overrides take top precedence
    if self.m_beta_override is not None and self.m_k_override is not None:
      return CapacityMultipliers(
          m_beta=self.m_beta_override,
          m_k=self.m_k_override,
          m_beta_rf=1.0,
          m_k_rf=1.0,
          m_beta_product=1.0,
          m_k_product=1.0,
          synergy_damping=self.synergy_damping,
          rf_details={"override": True},
          product_details={"override": True}
      )

    m_beta_rf, m_k_rf, rf_details = self._compute_rf_multipliers()
    m_beta_prod, m_k_prod, prod_details = self._compute_product_multipliers()

    delta_beta_rf = max(0.0, m_beta_rf - 1.0)
    delta_beta_prod = max(0.0, m_beta_prod - 1.0)
    delta_k_rf = max(0.0, m_k_rf - 1.0)
    delta_k_prod = max(0.0, m_k_prod - 1.0)

    # Damped formulation:
    # If both drivers are actively expanding, apply damping factor rho_synergy to their sum
    # If only one driver is expanding, don't artificially damp it.
    if delta_beta_rf > 0 and delta_beta_prod > 0:
      m_beta_combined = 1.0 + self.synergy_damping * (delta_beta_rf + delta_beta_prod)
    else:
      m_beta_combined = 1.0 + delta_beta_rf + delta_beta_prod

    if delta_k_rf > 0 and delta_k_prod > 0:
      m_k_combined = 1.0 + self.synergy_damping * (delta_k_rf + delta_k_prod)
    else:
      m_k_combined = 1.0 + delta_k_rf + delta_k_prod

    # Apply any specific single-axis overrides if provided
    if self.m_beta_override is not None:
      m_beta_combined = self.m_beta_override
    if self.m_k_override is not None:
      m_k_combined = self.m_k_override

    return CapacityMultipliers(
        m_beta=float(m_beta_combined),
        m_k=float(m_k_combined),
        m_beta_rf=float(m_beta_rf),
        m_k_rf=float(m_k_rf),
        m_beta_product=float(m_beta_prod),
        m_k_product=float(m_k_prod),
        synergy_damping=self.synergy_damping,
        rf_details=rf_details,
        product_details=prod_details
    )

  def project(self, curve: MarketingReturnCurve, channel_name: Optional[str] = None) -> "ProjectedReturnCurve":
    """Applies distilled multipliers to a MarketingReturnCurve, returning a ProjectedReturnCurve."""
    mults = self.compute_multipliers()
    name = channel_name or f"{curve.channel_name} (Projected)"
    return ProjectedReturnCurve(
        base_curve=curve,
        multipliers=mults,
        channel_name=name
    )


class ProjectedReturnCurve(MarketingReturnCurve):
  """A MarketingReturnCurve augmented with capacity and spend expansion multipliers.

  Maintains full functional compatibility with MarketingReturnCurve and PortfolioAllocator,
  while retaining references to the empirical base curve and providing strategic headroom
  analytics.
  """

  def __init__(
      self,
      base_curve: MarketingReturnCurve,
      multipliers: Union[CapacityMultipliers, Tuple[float, float]],
      channel_name: Optional[str] = None
  ):
    if isinstance(multipliers, tuple):
      mult_obj = CapacityMultipliers(m_beta=float(multipliers[0]), m_k=float(multipliers[1]))
    else:
      mult_obj = multipliers

    self.base_curve = base_curve
    self.multipliers = mult_obj
    self.is_projected = True

    proj_beta = float(base_curve.beta * mult_obj.m_beta)
    proj_k = float(base_curve.K * mult_obj.m_k)
    name = channel_name or f"{base_curve.channel_name} (Projected)"

    super().__init__(
        beta=proj_beta,
        alpha=base_curve.alpha,
        half_saturation_k=proj_k,
        theta=base_curve.theta,
        channel_name=name,
        baseline=base_curve.baseline,
        adstock_type=base_curve.adstock_type,
        adstock_params=base_curve.adstock_params,
        train_spend=base_curve._train_spend,
        train_return=base_curve._train_return,
        calibration_experiments=base_curve.calibration_experiments
    )

  def evaluate_unlocked_headroom(
      self,
      target_mroas: float = 1.0,
      current_spend: Optional[float] = None,
      verbose: bool = True
  ) -> Dict[str, Any]:
    """Evaluates the incremental spend capacity and return headroom unlocked by the projection.

    Args:
      target_mroas: Hurdle rate for diminishing returns (stop scaling point).
      current_spend: Optional current spend level to evaluate headroom against.
      verbose: If True, prints a formatted executive summary.

    Returns:
      Dict with comparative metrics between the base curve and projected curve.
    """
    base_inflection = self.base_curve.get_minimal_marginal_cost_point()
    proj_inflection = self.get_minimal_marginal_cost_point()

    base_max_spend = self.base_curve.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    proj_max_spend = self.get_diminishing_returns_point(target_mroas, warn_unreachable=False)

    base_return_at_max = (
        float(self.base_curve.predict_incremental_return(base_max_spend))
        if base_max_spend is not None else 0.0
    )
    proj_return_at_max = (
        float(self.predict_incremental_return(proj_max_spend))
        if proj_max_spend is not None else 0.0
    )

    spend_headroom = (
        proj_max_spend - base_max_spend
        if (proj_max_spend is not None and base_max_spend is not None) else None
    )
    spend_headroom_pct = (
        (spend_headroom / base_max_spend) * 100.0
        if (spend_headroom is not None and base_max_spend > 0) else None
    )

    return_headroom = (
        proj_return_at_max - base_return_at_max
        if (proj_max_spend is not None and base_max_spend is not None) else None
    )
    return_headroom_pct = (
        (return_headroom / base_return_at_max) * 100.0
        if (return_headroom is not None and base_return_at_max > 0) else None
    )

    summary = {
        "channel": self.channel_name,
        "target_mroas": target_mroas,
        "multipliers": {
            "m_beta": self.multipliers.m_beta,
            "m_k": self.multipliers.m_k,
        },
        "base_curve": {
            "beta": self.base_curve.beta,
            "K": self.base_curve.K,
            "inflection_spend": base_inflection,
            "diminishing_returns_spend": base_max_spend,
            "return_at_diminishing_returns": base_return_at_max,
        },
        "projected_curve": {
            "beta": self.beta,
            "K": self.K,
            "inflection_spend": proj_inflection,
            "diminishing_returns_spend": proj_max_spend,
            "return_at_diminishing_returns": proj_return_at_max,
        },
        "unlocked_headroom": {
            "spend_headroom": spend_headroom,
            "spend_headroom_pct": spend_headroom_pct,
            "return_headroom": return_headroom,
            "return_headroom_pct": return_headroom_pct,
        }
    }

    if current_spend is not None:
      current_spend = float(current_spend)
      summary["current_spend_evaluation"] = {
          "current_spend": current_spend,
          "base_mroas": float(self.base_curve.predict_marginal_return(current_spend)),
          "projected_mroas": float(self.predict_marginal_return(current_spend)),
          "spend_headroom_from_current": (
              proj_max_spend - current_spend if proj_max_spend is not None else None
          )
      }

    if verbose:
      self._print_headroom_report(summary, current_spend)

    return summary

  def _print_headroom_report(self, summary: Dict[str, Any], current_spend: Optional[float]):
    b = summary["base_curve"]
    p = summary["projected_curve"]
    u = summary["unlocked_headroom"]
    m = summary["multipliers"]

    print("\n=======================================================")
    print(f"       CAPACITY HEADROOM ANALYSIS: {self.channel_name}")
    print("=======================================================")
    print(f"Multipliers Applied: Beta x{m['m_beta']:.2f} | K x{m['m_k']:.2f}")
    print(f"Hurdle Rate (Target mROAS): {summary['target_mroas']:.2f}\n")

    print("--- Curve Scaling Boundaries ---")
    b_max_str = f"${b['diminishing_returns_spend']:,.2f}" if b['diminishing_returns_spend'] else "N/A"
    p_max_str = f"${p['diminishing_returns_spend']:,.2f}" if p['diminishing_returns_spend'] else "N/A"
    print(f"Peak Efficiency Spend (Inflection):  ${b['inflection_spend']:,.2f}  -->  ${p['inflection_spend']:,.2f}")
    print(f"Max Profitable Spend (Stop Scaling): {b_max_str}  -->  {p_max_str}")

    if u["spend_headroom"] is not None:
      sign = "+" if u["spend_headroom"] >= 0 else ""
      print("\n--- Unlocked Growth Potential ---")
      print(f"Additional Profitable Budget:        {sign}${u['spend_headroom']:,.2f} ({sign}{u['spend_headroom_pct']:.1f}%)")
      print(f"Additional Incremental Return:       {sign}${u['return_headroom']:,.2f} ({sign}{u['return_headroom_pct']:.1f}%)")

    if current_spend is not None and "current_spend_evaluation" in summary:
      ce = summary["current_spend_evaluation"]
      print(f"\n--- Current Spend Assessment (${current_spend:,.2f}) ---")
      print(f"Base mROAS: {ce['base_mroas']:.2f}  |  Projected mROAS: {ce['projected_mroas']:.2f}")
      if ce["spend_headroom_from_current"] is not None:
        print(f"Remaining Headroom to Projected Ceiling: ${ce['spend_headroom_from_current']:,.2f}")
    print("=======================================================\n")

  def plot_curve_comparison(
      self,
      target_mroas: float = 1.0,
      current_spend: Optional[float] = None,
      show: bool = True,
      figsize: Tuple[int, int] = (16, 7),
  ):
    """Generates an uncluttered 2-panel comparison presenting the shift from base to projected curve.

    Panel 1 (Left): Saturation curve showing historical base curve vs. projected curve
                    with highlighted incremental return unlock.
    Panel 2 (Right): Marginal return curve showing efficiency (mROAS) comparison and
                     hurdle rate threshold.

    Emphasizes the new projected curve as the primary focus while keeping the older
    base curve muted for reference.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    base_max = self.base_curve.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    proj_max = self.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    base_min = self.base_curve.get_minimal_marginal_cost_point() or 0.0
    proj_min = self.get_minimal_marginal_cost_point() or 0.0

    max_x = proj_max * 1.35 if proj_max else (base_max * 2.0 if base_max else self.K * 2.5)
    if current_spend:
      max_x = max(max_x, current_spend * 1.25)

    if proj_max and max_x > 50 * proj_max:
      max_x = proj_max * 2.5

    x_vals = np.linspace(0, max_x, 500)

    y_base_return = self.base_curve.predict_incremental_return(x_vals)
    y_proj_return = self.predict_incremental_return(x_vals)

    y_base_mroas = self.base_curve.predict_marginal_return(x_vals)
    y_proj_mroas = self.predict_marginal_return(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Google Sans', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    # Color palette
    BASE_COLOR = '#80868B'     # Muted gray for older reference curve
    PROJ_COLOR = '#7C3AED'     # Vibrant Violet/Purple for new projected curve focus
    TARGET_COLOR = '#EA4335'   # Coral Red for current spend and hurdle
    GRAY = '#5F6368'
    LIGHT_GRAY = '#F8F9FA'

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
    # PANEL 1: Pure Saturation / Return Curves (Base vs Projected)
    # --------------------------------------------------------------------------
    # 1. Base (Older) curve: muted, lower opacity, reference
    ax1.plot(x_vals, y_base_return, color=BASE_COLOR, linewidth=2.0, linestyle='--',
             alpha=0.55, label="Historical Baseline (Reference)", zorder=2)

    # 2. Projected (Newer) curve: primary focus, bold, vibrant
    ax1.plot(x_vals, y_proj_return, color=PROJ_COLOR, linewidth=3.5, linestyle='-',
             label=f"Projected Curve (β ×{self.multipliers.m_beta:.2f}, K ×{self.multipliers.m_k:.2f})", zorder=4)

    # 3. Unlocked Headroom Area
    ax1.fill_between(x_vals, y_base_return, y_proj_return, color=PROJ_COLOR, alpha=0.12,
                     label="Unlocked Return Capacity", zorder=1)

    # Mark base stop scaling (reference)
    base_ret_max = 0.0
    if base_max and base_max > 0:
      base_ret_max = float(self.base_curve.predict_incremental_return(base_max))
      ax1.scatter(base_max, base_ret_max, color=BASE_COLOR, s=70, marker='o',
                  edgecolors='#5F6368', linewidth=1.0, alpha=0.7,
                  label=f"Base Cap ({format_currency(base_max)})", zorder=3)

    # Mark projected stop scaling (focus)
    proj_ret_max = 0.0
    if proj_max and proj_max > 0:
      proj_ret_max = float(self.predict_incremental_return(proj_max))
      ax1.scatter(proj_max, proj_ret_max, color=PROJ_COLOR, s=120, marker='o',
                  edgecolors='#202124', linewidth=1.5,
                  label=f"Projected Cap ({format_currency(proj_max)})", zorder=5)
      ax1.annotate(
          f"Projected Cap: {format_currency(proj_max)}\nReturn: {format_currency(proj_ret_max)}",
          xy=(proj_max, proj_ret_max),
          xytext=(0, -28), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E8FF', edgecolor=PROJ_COLOR, alpha=1.0, zorder=10),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1, zorder=10),
          zorder=10
      )

    # Current spend marker if provided
    curr_proj_ret = 0.0
    if current_spend:
      curr_proj_ret = float(self.predict_incremental_return(current_spend))
      ax1.axvline(current_spend, color=TARGET_COLOR, linestyle='--', linewidth=1.8, alpha=0.85, zorder=4)
      ax1.scatter(current_spend, curr_proj_ret, color=TARGET_COLOR, s=120, edgecolors='white', linewidth=2, zorder=6)
      ax1.annotate(
          f"Current Spend: {format_currency(current_spend)}\nProjected Return: {format_currency(curr_proj_ret)}",
          xy=(current_spend, curr_proj_ret),
          xytext=(15, 25), textcoords="offset points",
          ha='left', fontsize=9, fontweight='bold', color=TARGET_COLOR,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE8E6', edgecolor=TARGET_COLOR, alpha=1.0, zorder=10),
          arrowprops=dict(arrowstyle='->', color=TARGET_COLOR, lw=1.5, zorder=10),
          zorder=10
      )

    # Calculate y1 limits
    finite_ret = y_proj_return[np.isfinite(y_proj_return)]
    max_y1 = float(np.max(finite_ret)) if len(finite_ret) > 0 else 1000.0
    if current_spend and np.isfinite(curr_proj_ret):
      max_y1 = max(max_y1, float(curr_proj_ret))
    if proj_max and proj_max > 0 and np.isfinite(proj_ret_max):
      max_y1 = max(max_y1, float(proj_ret_max))

    ax1.set_title("1. Media Saturation & Capacity Unlock", fontsize=13, fontweight='bold', color='#202124', pad=12, loc='left')
    ax1.set_xlabel("Media Spend", fontsize=11, color=GRAY, fontweight='500', labelpad=8)
    ax1.set_ylabel("Incremental Return ($)", fontsize=11, color='#202124', fontweight='500', labelpad=8)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, max_y1 * 1.18)
    ax1.grid(True, linestyle='-', alpha=0.15, color=GRAY)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(LIGHT_GRAY)
    ax1.spines['bottom'].set_color(LIGHT_GRAY)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)

    # --------------------------------------------------------------------------
    # PANEL 2: Marginal Return (mROAS) Comparison
    # --------------------------------------------------------------------------
    # 1. Base (Older) marginal curve: muted reference
    ax2.plot(x_vals, y_base_mroas, color=BASE_COLOR, linestyle=':', linewidth=2.0, alpha=0.55,
             label="Base mROAS (Reference)", zorder=2)

    # 2. Projected (Newer) marginal curve: primary focus
    ax2.plot(x_vals, y_proj_mroas, color=PROJ_COLOR, linestyle='-', linewidth=3.0,
             label="Projected mROAS (New Curve)", zorder=4)

    # 3. Hurdle line without printing raw numeric value in legend
    ax2.axhline(target_mroas, color=TARGET_COLOR, linestyle='--', linewidth=1.6, alpha=0.8,
                label="Target Hurdle Rate", zorder=3)

    # Projected Optimal Zone (no legend entry)
    if proj_max and proj_max > proj_min:
      ax2.axvspan(proj_min, proj_max, color=PROJ_COLOR, alpha=0.08, zorder=0)
      ax2.text((proj_min + proj_max) / 2.0, 0.03, 'PROJECTED OPTIMAL ZONE',
               transform=ax2.get_xaxis_transform(),
               horizontalalignment='center', verticalalignment='bottom',
               fontsize=9, color=PROJ_COLOR, fontweight='bold', alpha=0.85)

    # Markers on Panel 2
    # Base stop scaling
    if base_max and base_max > 0:
      ax2.scatter(base_max, target_mroas, color=BASE_COLOR, s=70, marker='o',
                  edgecolors='#5F6368', linewidth=1.0, alpha=0.7,
                  label=f"Base Cap ({format_currency(base_max)})", zorder=3)

    # Projected stop scaling
    if proj_max and proj_max > 0:
      ax2.scatter(proj_max, target_mroas, color=PROJ_COLOR, s=120, marker='o',
                  edgecolors='#202124', linewidth=1.5,
                  label=f"Projected Cap ({format_currency(proj_max)})", zorder=5)
      ax2.annotate(
          f"Hurdle Floor\n{format_currency(proj_max)}",
          xy=(proj_max, target_mroas),
          xytext=(0, -26), textcoords="offset points",
          ha='center', fontsize=9, fontweight='bold', color='#202124',
          bbox=dict(boxstyle='round,pad=0.25', facecolor='#F3E8FF', edgecolor=PROJ_COLOR, alpha=1.0, zorder=10),
          arrowprops=dict(arrowstyle='->', color='#202124', lw=1, zorder=10),
          zorder=10
      )

    # Current spend marker on Panel 2 (no legend entry)
    curr_proj_mroas = 0.0
    if current_spend:
      curr_proj_mroas = float(self.predict_marginal_return(current_spend))
      curr_base_mroas = float(self.base_curve.predict_marginal_return(current_spend))
      ax2.axvline(current_spend, color=TARGET_COLOR, linestyle='--', linewidth=1.8, alpha=0.85, zorder=4)
      ax2.scatter(current_spend, curr_proj_mroas, color=TARGET_COLOR, s=120, edgecolors='white', linewidth=2, zorder=6)
      ax2.annotate(
          f"Projected mROAS: {format_num(curr_proj_mroas)}\n(vs Base: {format_num(curr_base_mroas)})",
          xy=(current_spend, curr_proj_mroas),
          xytext=(15, 20), textcoords="offset points",
          ha='left', fontsize=9, fontweight='bold', color=TARGET_COLOR,
          bbox=dict(boxstyle='round,pad=0.3', facecolor='#FCE8E6', edgecolor=TARGET_COLOR, alpha=1.0, zorder=10),
          arrowprops=dict(arrowstyle='->', color=TARGET_COLOR, lw=1.5, zorder=10),
          zorder=10
      )

    # Calculate y2 ceiling for generous annotation headroom
    finite_proj = y_proj_mroas[np.isfinite(y_proj_mroas)]
    if len(finite_proj) > 0:
      if self.alpha < 1.0 and len(finite_proj) > 10:
        max_y2 = float(np.percentile(finite_proj[1:], 95)) * 1.6
      else:
        max_y2 = float(np.max(finite_proj))
    else:
      max_y2 = float(target_mroas) * 2.0 if target_mroas else 5.0

    if np.isfinite(target_mroas):
      max_y2 = max(max_y2, float(target_mroas) * 1.2)
    if current_spend:
      if np.isfinite(curr_proj_mroas):
        max_y2 = max(max_y2, float(curr_proj_mroas) * 1.15)

    if not np.isfinite(max_y2) or max_y2 <= 0:
      max_y2 = 5.0

    ax2.set_title("2. Marginal Return & Efficiency (mROAS)", fontsize=13, fontweight='bold', color='#202124', pad=12, loc='left')
    ax2.set_xlabel("Media Spend", fontsize=11, color=GRAY, fontweight='500', labelpad=8)
    ax2.set_ylabel("Marginal ROAS (mROAS)", fontsize=11, color='#202124', fontweight='500', labelpad=8)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_currency))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_num))
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, max_y2 * 1.25)
    ax2.grid(True, linestyle='-', alpha=0.15, color=GRAY)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(LIGHT_GRAY)
    ax2.spines['bottom'].set_color(LIGHT_GRAY)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)

    # --------------------------------------------------------------------------
    # Header & Executive Takeaway Subtitle
    # --------------------------------------------------------------------------
    fig.suptitle(f"Capacity Projection: {self.channel_name}", fontsize=15, fontweight='bold', color='#202124', y=0.98)

    mult_desc = f"Capacity Expansion: β ×{self.multipliers.m_beta:.2f} (Ceiling), K ×{self.multipliers.m_k:.2f} (Dilation) • Damping ρ={self.multipliers.synergy_damping:.2f}"
    if base_max and proj_max and proj_max > base_max:
      delta_spend = proj_max - base_max
      delta_ret = proj_ret_max - base_ret_max
      status_text = f"{mult_desc} • Unlocks +{format_currency(delta_spend)} Scaling Headroom (+{format_currency(delta_ret)} Incremental Return)"
    else:
      status_text = mult_desc

    fig.text(0.5, 0.93, status_text, fontsize=10.5, ha='center', color=GRAY, style='italic', parse_math=False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if show:
      plt.show()
    return fig
