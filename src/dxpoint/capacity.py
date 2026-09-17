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
      show: bool = True
  ):
    """Generates an overlaid visualization comparing the base curve vs the projected curve."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    base_max = self.base_curve.get_diminishing_returns_point(target_mroas, warn_unreachable=False)
    proj_max = self.get_diminishing_returns_point(target_mroas, warn_unreachable=False)

    max_x = proj_max * 1.35 if proj_max else (base_max * 2.0 if base_max else self.K * 2.5)
    if current_spend:
      max_x = max(max_x, current_spend * 1.3)

    x_vals = np.linspace(0, max_x, 500)

    y_base_return = self.base_curve.predict_incremental_return(x_vals)
    y_proj_return = self.predict_incremental_return(x_vals)

    y_base_mroas = self.base_curve.predict_marginal_return(x_vals)
    y_proj_mroas = self.predict_marginal_return(x_vals)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Roboto', 'Open Sans', 'Arial', 'DejaVu Sans']

    fig, ax1 = plt.subplots(figsize=(13, 7.5), facecolor='white')
    ax1.set_facecolor('white')

    # Color palette
    BASE_COLOR = '#1A73E8'     # Google Blue
    PROJ_COLOR = '#9334E6'     # Purple / Violet
    TARGET_COLOR = '#EA4335'   # Coral Red
    GRAY = '#5F6368'

    # Primary Axis: Return Curves
    ax1.plot(x_vals, y_base_return, color=BASE_COLOR, linewidth=2.5, linestyle='-',
             label="Current Observed Return", zorder=3)
    ax1.plot(x_vals, y_proj_return, color=PROJ_COLOR, linewidth=3.5, linestyle='-',
             label=f"Projected Return (β x{self.multipliers.m_beta:.2f}, K x{self.multipliers.m_k:.2f})", zorder=4)

    # Shaded Headroom Delta
    ax1.fill_between(x_vals, y_base_return, y_proj_return, color=PROJ_COLOR, alpha=0.10,
                     label="Unlocked Incremental Return", zorder=2)

    # Secondary Axis: Marginal Return
    ax2 = ax1.twinx()
    ax2.plot(x_vals, y_base_mroas, color=BASE_COLOR, linestyle=':', linewidth=1.5, alpha=0.6,
             label="Base mROAS", zorder=1)
    ax2.plot(x_vals, y_proj_mroas, color=PROJ_COLOR, linestyle='--', linewidth=2.0, alpha=0.7,
             label="Projected mROAS", zorder=1)
    ax2.axhline(target_mroas, color=TARGET_COLOR, linestyle='-', linewidth=1.2, alpha=0.6,
                label=f"Target mROAS ({target_mroas:.2f})")

    # Mark diminishing returns points
    if base_max:
      ax1.axvline(base_max, color=BASE_COLOR, linestyle=':', linewidth=1.2, alpha=0.7)
      ax1.scatter(base_max, self.base_curve.predict_incremental_return(base_max),
                  color=BASE_COLOR, s=80, marker='o', edgecolors='white', linewidth=1.5,
                  label=f"Current Stop Scaling (${base_max:,.0f})", zorder=5)

    if proj_max:
      ax1.axvline(proj_max, color=PROJ_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
      ax1.scatter(proj_max, self.predict_incremental_return(proj_max),
                  color=PROJ_COLOR, s=110, marker='s', edgecolors='white', linewidth=1.5,
                  label=f"Projected Stop Scaling (${proj_max:,.0f})", zorder=5)

    # Current spend marker if provided
    if current_spend:
      ax1.axvline(current_spend, color=TARGET_COLOR, linestyle='-', linewidth=1.8, alpha=0.9,
                  label=f"Current Spend (${current_spend:,.0f})", zorder=6)

    # Labels and scales
    def fmt(x, p):
      if abs(x) >= 1e6: return f'${x*1e-6:g}M'
      elif abs(x) >= 1e3: return f'${x*1e-3:g}k'
      else: return f'${x:g}'

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax1.set_xlabel('Media Spend', fontsize=11, color=GRAY, fontweight='500', labelpad=10)
    ax1.set_ylabel('Incremental Return', fontsize=11, color=PROJ_COLOR, fontweight='500', labelpad=10)
    ax2.set_ylabel('Marginal ROAS (mROAS)', fontsize=11, color=GRAY, fontweight='500', labelpad=10)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    # Spines & Grid
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.grid(True, linestyle='-', alpha=0.1, color=GRAY)

    # Combine Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True,
               facecolor='white', framealpha=0.95, fontsize=9.5)

    plt.title(f'Capacity Projection Analysis: {self.channel_name}', loc='left',
              fontsize=15, fontweight='bold', pad=22, color='#202124')
    fig.text(0.125, 0.91,
             f"Base (β={self.base_curve.beta:,.0f}, K={self.base_curve.K:,.0f})  -->  "
             f"Projected (β={self.beta:,.0f}, K={self.K:,.0f})  |  Synergy Damping={self.multipliers.synergy_damping:.2f}",
             fontsize=9.5, color=GRAY)

    plt.tight_layout()
    if show:
      plt.show()
    return fig
