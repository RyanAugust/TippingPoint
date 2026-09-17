# Tipping Point Research: Full-Funnel & Reach/Frequency Benchmarks

This directory contains research benchmarks, empirical proofs, and vertical multiplier profiles designed for use with **Tipping Point's** [`CapacityProjector`](../src/tippingpoint/capacity.py).

> [!NOTE]
> This directory is excluded from PyPI package distributions (`wheel` and `sdist`). It serves as an organizational repository for industry/vertical research, internal calibration data, and portable JSON configs that teams can distribute independently.

---

## 1. Empirical Sources

The benchmarks provided in this repository are distilled from:
1. **Nielsen MMM Meta-Analysis (Google-Commissioned)**: Evaluated 53,153 US campaigns across 104 weeks measuring AI-powered video formats (VRC, VVC, Demand Gen, PMax), alongside longitudinal brand-level studies across 21 unique enterprise advertisers.
2. **Google Meridian Full-Funnel Causal Modeling**: Two-stage structural causal models showing that upper-funnel brand equity (e.g. Branded Google Query Volume) expands downstream conversion velocity and observable media impact by up to $1.5\times$.
3. **Google Internal Multi-Touch Panels**: 56,000 active advertisers analyzing video touchpoint assists, and TLS Sector DM&A Learning Agendas (Titan overlap studies showing $+75\%$ conversion rate lift for video + search vs. search alone).

---

## 2. TLS Sector Multipliers Summary

| Sub-Vertical Key | Sub-Vertical Name | $M_\beta$ (Capacity Lift) | $M_K$ (Spend Scale) | Optimal Weekly Freq ($f_{\text{target}}$) | Target Reach Pen ($\rho_{\text{target}}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tech_b2b` | Tech B2B (Enterprise SaaS / IT) | **$1.55\times$** | **$1.85\times$** | **$2.0 - 2.5$** | **$55\% - 70\%$** (Qualified TAM) |
| `apps_platforms` | Apps & Platforms (Consumer Mobile) | **$1.80\times$** | **$2.35\times$** | **$2.8 - 3.5$** | **$35\% - 50\%$** (Category Mobile) |
| `home_consumer_services` | Home & Consumer Services (Auto / Home) | **$1.65\times$** | **$2.10\times$** | **$2.5 - 3.0$** | **$40\% - 55\%$** (Households) |
| `education_careers` | Education & Careers (EdTech / Degree) | **$1.60\times$** | **$1.95\times$** | **$2.5 - 3.2$** | **$35\% - 45\%$** (Career Switchers) |
| `landmark_local_services` | Landmark & Local Services (Legal / Lead) | **$1.45\times$** | **$1.75\times$** | **$2.2 - 2.8$** | **$35\% - 50\%$** (Serviced DMAs) |

---

## 3. How to Use These Benchmarks with Tipping Point

### Option A: Import Directly from Python
```python
from tippingpoint import MarketingReturnCurve
from research.tls_benchmarks import TLS_VERTICAL_BENCHMARKS

# Fit current historical curve
model = MarketingReturnCurve.fit(spends, returns, channel_name="B2B Demand Gen")

# Project capacity using Tech B2B multipliers and optimal frequency
projected = model.project_capacity(
    vertical="tech_b2b",
    vertical_multipliers=TLS_VERTICAL_BENCHMARKS,
    current_funnel="lower_only",
    target_funnel="full_funnel"
)

projected.evaluate_unlocked_headroom(target_mroas=1.2)
```

### Option B: Load from JSON (For Cross-Team / Organization Distribution)
If your organization maintains uniform multipliers in a centralized file:

```python
from tippingpoint import MarketingReturnCurve

# Fit current curve
model = MarketingReturnCurve.fit(spends, returns, channel_name="App Acquisition")

# Pass JSON path directly into project_capacity
projected = model.project_capacity(
    vertical="apps_platforms",
    vertical_multipliers="research/tls_benchmarks.json",
    current_funnel="lower_only",
    target_funnel="full_funnel"
)

projected.plot_curve_comparison(target_mroas=1.0)
```
