# Tipping Point
**Author:** Ryan Duecker (ryanduecker@google.com)

A lightweight, high-performance marketing intelligence module that uses machine learning and calculus to determine the exact inflection points of a media response curve. 

Growth marketers and media buyers constantly ask two questions: *"When are we out of the inefficient learning phase?"* and *"When should we stop scaling spend?"* By fitting historical performance data to a continuous mathematical curve, this tool identifies the **Minimal Marginal Cost Point** (where efficiency peaks) and the **Point of Diminishing Returns** (where scaling is no longer profitable), defining your exact **Optimal Scaling Zone**.

## 🧠 Methodology

This project leverages the mathematical foundations of modern Marketing Mix Modeling (MMM)—specifically the techniques popularized by [Google’s Meridian](https://github.com/google/meridian). 

Instead of basic linear or logarithmic approximations, this module natively models media saturation using the **Hill Function**.

$$Return = \frac{\beta \cdot Spend^\alpha}{K^\alpha + Spend^\alpha}$$

*   **$\beta$ (Beta):** The asymptote (maximum possible return/capacity).
*   **$\alpha$ (Alpha):** The shape parameter. If $\alpha > 1$, the curve is S-shaped (featuring a warm-up phase). If $\alpha \le 1$, it strictly exhibits diminishing returns from the start (C-shaped).
*   **$K$ (Half-Saturation):** The spend amount at which you achieve half of the maximum return.

### The Calculus Engine
Once the Hill Curve parameters are found, the module uses exact calculus to provide strategic recommendations:
*   **Marginal ROAS ($f'(x)$):** Represents the efficiency of the *next* dollar spent.
*   **Minimal Marginal Cost ($f''(x) = 0$):** The inflection point of the S-Curve. This is the absolute peak of Marginal ROAS. You should spend *at least* this much to exit the inefficient warm-up phase.
*   **Point of Diminishing Returns ($f'(x) = Target\_mROAS$):** The exact spend level where the efficiency drops below your acceptable baseline unit economics (e.g., Target CPA or Target ROAS constraint).

## 🚀 Installation & Prerequisites

This module uses **tinygrad** for ultra-lightweight GPU-accelerated gradient descent, alongside standard scientific libraries.

```bash
pip install tinygrad numpy scipy matplotlib
```

## 💻 Usage

### 1. Fitting from Historical Data
You can pass raw `Spend` and `Return` (Revenue, Conversions, etc.) arrays directly into the module. The PyTorch/Tinygrad backend will automatically find the optimal $\beta$, $\alpha$, and $K$ parameters.

```python
import numpy as np
from marketing_curve import MarketingReturnCurve

# 1. Provide your historical marketing data (Spend vs. Revenue/Conversions)
spends = np.array([1200, 5000, 15000, 25000, 40000])
returns = np.array([200, 1500, 12000, 22000, 28000])

# 2. Fit the Curve
model = MarketingReturnCurve.from_historical_data(
    spend_array=spends,
    return_array=returns,
    channel_name="Paid Social",
    epochs=3000,
    lr=0.05
)
```

### 2. Extracting Intelligence & Inflection Points
Once the curve is fitted (or if you manually provide parameters from an existing Meridian model), you can extract actionable business intelligence.

```python
target_mroas = 1.5      # We need at least $1.50 back on the marginal dollar
current_spend = 12000   # Our current daily/weekly budget

# Get precise inflection points
optimal_floor = model.get_minimal_marginal_cost_point()
spend_cap = model.get_diminishing_returns_point(target_mroas)

print(f"Start Scaling At: ${optimal_floor:,.2f}")
print(f"Stop Scaling At: ${spend_cap:,.2f}")

# Get a text-based evaluation of your current strategy
model.evaluate_current_budget(current_spend, target_mroas)
```
**Example Output:**
```text
--- Budget Evaluation: Paid Social ---
Current Spend: $12,000.00 | Current mROAS: 2.10
Status: OPTIMAL SCALING ZONE
Recommendation: You are operating within the highly efficient growth window.
```

### 3. Visualization
Generate an executive-ready, dual-axis chart mapping the Incremental Return curve against the Marginal ROAS curve, explicitly highlighting the Optimal Scaling Zone.

```python
model.plot_response_curve(target_mroas=1.5, current_spend=12000)
```

## 🛠 Integrating with existing MMMs (Meridian)
If you already run Google Meridian, you do not need to use the `from_historical_data` method. You can simply extract the posterior mean parameters for a specific channel directly from your Meridian output and initialize the class:

```python
# Assuming you extracted beta, alpha, and K from your Meridian posterior
model = MarketingReturnCurve(beta=100000, alpha=1.8, half_saturation_k=20000)
```
