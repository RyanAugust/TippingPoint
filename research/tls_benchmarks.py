"""Empirical TLS Sector Benchmarks for YouTube Full-Funnel Capacity Projection.

Based on Google-commissioned Nielsen MMM meta-analyses, Google Meridian causal modeling,
Google internal multi-touch conversion path panels, and TLS DM&A Learning Agendas.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import json

TLS_VERTICAL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "tech_b2b": {
        "lower_only": {"m_beta": 1.00, "m_k": 1.00},
        "mid_lower": {"m_beta": 1.25, "m_k": 1.35},
        "upper_lower": {"m_beta": 1.38, "m_k": 1.55},
        "full_funnel": {"m_beta": 1.55, "m_k": 1.85},
        "rf_targets": {"target_frequency": 2.2, "target_reach_penetration": 0.60},
    },
    "apps_platforms": {
        "lower_only": {"m_beta": 1.00, "m_k": 1.00},
        "mid_lower": {"m_beta": 1.35, "m_k": 1.50},
        "upper_lower": {"m_beta": 1.55, "m_k": 1.85},
        "full_funnel": {"m_beta": 1.80, "m_k": 2.35},
        "rf_targets": {"target_frequency": 3.0, "target_reach_penetration": 0.45},
    },
    "home_consumer_services": {
        "lower_only": {"m_beta": 1.00, "m_k": 1.00},
        "mid_lower": {"m_beta": 1.28, "m_k": 1.40},
        "upper_lower": {"m_beta": 1.45, "m_k": 1.70},
        "full_funnel": {"m_beta": 1.65, "m_k": 2.10},
        "rf_targets": {"target_frequency": 2.7, "target_reach_penetration": 0.50},
    },
    "education_careers": {
        "lower_only": {"m_beta": 1.00, "m_k": 1.00},
        "mid_lower": {"m_beta": 1.26, "m_k": 1.38},
        "upper_lower": {"m_beta": 1.42, "m_k": 1.65},
        "full_funnel": {"m_beta": 1.60, "m_k": 1.95},
        "rf_targets": {"target_frequency": 2.8, "target_reach_penetration": 0.40},
    },
    "landmark_local_services": {
        "lower_only": {"m_beta": 1.00, "m_k": 1.00},
        "mid_lower": {"m_beta": 1.20, "m_k": 1.30},
        "upper_lower": {"m_beta": 1.32, "m_k": 1.50},
        "full_funnel": {"m_beta": 1.45, "m_k": 1.75},
        "rf_targets": {"target_frequency": 2.5, "target_reach_penetration": 0.42},
    },
}

def get_tls_benchmarks(vertical: Optional[str] = None) -> Dict[str, Any]:
  """Returns TLS vertical benchmarks, optionally filtered by a specific vertical key."""
  if vertical:
    v_norm = vertical.lower()
    if v_norm not in TLS_VERTICAL_BENCHMARKS:
      raise ValueError(f"Unknown TLS vertical '{vertical}'. Available: {list(TLS_VERTICAL_BENCHMARKS.keys())}")
    return TLS_VERTICAL_BENCHMARKS[v_norm]
  return TLS_VERTICAL_BENCHMARKS

def load_benchmarks_from_json(json_path: Optional[str] = None) -> Dict[str, Any]:
  """Loads vertical benchmarks from a JSON file."""
  path = Path(json_path) if json_path else Path(__file__).parent / "tls_benchmarks.json"
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)
