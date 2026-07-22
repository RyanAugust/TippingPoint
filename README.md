# Tipping Point
**Author:** [Ryan Duecker](ryanduecker@google.com)

[![PyPI Downloads](https://img.shields.io/pypi/dm/tippingpt.svg?label=PyPI%20downloads)](
https://pypi.org/project/tippingpt/)

A lightweight, high-performance marketing intelligence module that uses machine learning and calculus to determine the exact inflection points of a media response curve.

Growth marketers and media buyers constantly ask two questions: *"When are we out of the inefficient learning phase?"* and *"When should we stop scaling spend?"* By fitting historical performance data to a continuous mathematical curve, this tool identifies the **Minimal Marginal Cost Point** (where efficiency peaks) and the **Point of Diminishing Returns** (where scaling is no longer profitable), defining your exact **Optimal Scaling Zone**.

Tipping Point scales from single-channel analysis to a full scenario planning engine. Using the included `PortfolioAllocator`, advertisers can instantly calculate the exact budget distribution that maximizes total return across multiple channels.

## 🧠 Methodology

This project leverages the mathematical foundations of modern Marketing Mix Modeling (MMM)—specifically the techniques popularized by [Google’s Meridian](https://github.com/google/meridian).

### 1. Media Saturation (The Hill Function)
Instead of basic linear or logarithmic approximations, this module natively models media saturation using the **Hill Function**.

$$Return = \frac{\beta \cdot Spend^\alpha}{K^\alpha + Spend^\alpha}$$

*   **$\beta$ (Beta):** The asymptote (maximum possible return/capacity).
*   **$\alpha$ (Alpha):** The shape parameter. S-shaped ($\alpha > 1$) or C-shaped ($\alpha \le 1$).
*   **$K$ (Half-Saturation):** The spend amount at which you achieve half of the maximum return.

### 2. Geometric Adstock (Lagged Effects)
Media impact decays over time. The module implements Meridian-style **Geometric Adstock**, transforming raw spends into effective memory-adjusted spends:

$$ S_{t\_adstocked} = S_t + \theta \cdot S_{t-1\_adstocked} $$

Tipping Point supports 4 adstock optimization modes: `none`, `free` (fully optimized $\theta$), `bounded` (constrained half-life), and `fixed` (explicit decay days).

### 3. The Calculus Engine
Using exact calculus, the module provides strategic recommendations:
*   **Marginal ROAS ($f'(x)$):** The efficiency of the *next* dollar spent.
*   **Peak Efficiency ($f''(x) = 0$):** The inflection point. Spend *at least* this much to exit the warm-up phase.
*   **Stop Scaling Point ($f'(x) = Target\_mROAS$):** The exact spend level where efficiency drops below your baseline unit economics.

## 🚀 Installation & Prerequisites

This module uses **tinygrad** for ultra-lightweight GPU-accelerated gradient descent, **scipy** for portfolio optimization, and **plotly/streamlit** for visualization.

```bash
pip install tippingpt
```

## 💻 Usage

### 1. Fitting from Historical Data
Pass raw `Spend` and `Return` arrays directly into the module.

```python
import numpy as np
from tippingpoint import MarketingReturnCurve

spends = np.array([1200, 5000, 15000, 25000, 40000])
returns = np.array([200, 1500, 12000, 22000, 28000])

# Fit with Gradient Descent (MLE) & bounded adstock (1-14 days half-life)
model = MarketingReturnCurve.from_historical_data(
    spend_array=spends,
    return_array=returns,
    channel_name="YouTube",
    epochs=1000,
    adstock_type="bounded",
    adstock_bounds=(1.0, 14.0)
)
```

### 2. Extracting Intelligence & Inflection Points

```python
# Evaluate headroom based on current spend and a target return floor
model.evaluate_current_budget(current_spend=12000, target_mroas=1.5)
```
**Example Output:**
```text
--- Budget Evaluation: Paid Social ---
Current Spend: $12,000.00 | Current mROAS: 2.10
Status: OPTIMAL SCALING ZONE
Recommendation: You are operating within the highly efficient growth window.
```

### 3. Portfolio Optimization (Scenario Planning)
Instantly calculate optimal budget allocation across multiple fitted channels.

```python
from tippingpoint import PortfolioAllocator

# Initialize the Allocator with your fitted channels
allocator = PortfolioAllocator([model_search, model_youtube, model_tv])

# Run a scenario analysis for a $1,000,000 budget
scenario = allocator.allocate_budget(
    total_budget=1000000,
    channel_bounds={"Television": (50000, 200000)} # Optional min/max constraints
)

print(scenario["allocation"])
print(f"Expected Return: ${scenario['expected_total_return']:,.2f}")
```

### 4. Interactive Multi-Channel Dashboard
Explore your models and run cross-channel portfolio optimization interactively using the built-in Streamlit dashboard.

**Stage 1: Channel Configuration:** Dynamically fit, configure, and stack multiple channels. Features conversion value multipliers and interactive Adstock carryover timelines.
**Stage 2: Portfolio Optimization:** Set global budgets and constraints. Generates optimal scale mix (stacked area) plots and cross-channel saturation overlays.

```bash
# Launch the dashboard
tipp dashboard
```

## 🛠 Integrating with existing MMMs (Meridian)
If you already run Google Meridian, you can extract the posterior mean parameters and initialize the class without refitting:

```python
model = MarketingReturnCurve(beta=100000, alpha=1.8, half_saturation_k=20000, theta=0.75)
```
