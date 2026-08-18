# Tipping Point
**Author:** [Ryan Duecker](ryanduecker@google.com)

[![PyPI Downloads](https://img.shields.io/pypi/dm/tippingpt.svg?label=PyPI%20downloads)](https://pypi.org/project/tippingpt/)

A lightweight, marketing intelligence module that assists in identifying media response curves and determining the inflection points.

Growth marketers and media buyers ask two fundamental questions:
1) *"When are we out of the inefficient learning phase?"*
2) *"When should we stop scaling spend?"*

By fitting performance data to continuous saturation curves, Tipping Point identifies the **Minimal Marginal Cost Point** (the inflection point where acquisition cost is lowest) and the **Point of Diminishing Returns** (where marginal ROAS hits your profitability hurdle rate), defining your exact **Optimal Scaling Zone**.

Tipping Point focuses primarily on **single-channel curve fitting** and **cross-channel portfolio planning**, keeping single-channel workflows fast, lightweight, and accessible without requiring heavy econometric setup.

---

## Core Methodology

Tipping Point leverages the mathematical foundations of modern response modeling—specifically the Hill saturation and adstock formulations popularized by [Google’s Meridian](https://github.com/google/meridian).

### 1. Media Saturation (The Hill Function)
Instead of basic linear or logarithmic approximations, this module natively models media saturation using the **Hill Function**.

$$Return = \beta_0 + \frac{\beta \cdot Spend_{adstocked}^\alpha}{K^\alpha + Spend_{adstocked}^\alpha}$$

*   **$\beta$ (Beta - Capacity):** Maximum incremental return capacity.
*   **$\alpha$ (Alpha - Shape):** The learning curve. $\alpha > 1$ produces an **S-curve** (initial warm-up phase where frequency builds momentum); $\alpha \le 1$ produces a **C-curve** (immediate concave diminishing returns).
*   **$K$ (Half-Saturation):** The spend level required to achieve 50% of maximum incremental capacity.
*   **$\beta_0$ (Baseline Demand):** Optional organic, non-media baseline return.

### 2. Adstock (Lagged Effects & Memory)
Advertising impacts persist beyond the day of exposure. Tipping Point supports multiple memory decay models:

*   **Geometric Adstock:** Exponential memory decay parameterized by retention rate $\theta \in [0, 1)$:
    $$S_{t\_adstocked} = S_t + \theta \cdot S_{t-1\_adstocked}$$
*   **Weibull Adstock:** Flexible delayed response curves using **Weibull PDF** (lagged peak effect) or **Weibull CDF** (flexible S/C decay) with shape $k$ and scale $\lambda$.

During single-channel training, adstock can be set to `none`, `fixed` (explicit half-life), `bounded` (constrained half-life window), or `free` (unconstrained optimization).

### 3. Margin-Focused Calculus & Tipping Points
Using marginal rates of change rather than historical blended averages, the module calculates:
*   **Marginal ROAS ($f'(x)$):** The efficiency of the *next* dollar spent.
*   **Peak Efficiency Point ($f''(x) = 0$):** The inflection point. Spend at least this much to exit the warm-up phase.
*   **Stop Scaling Point ($f'(x) = \text{Target\_mROAS}$):** The exact spend level where marginal return drops below your baseline unit economics.
*   **Optimal Scaling Zone:** The high-velocity growth window between the Peak Efficiency Point and the Stop Scaling Point.

---

## Installation

```bash
pip install tippingpt
```

This module uses **tinygrad** for ultra-lightweight GPU-accelerated gradient descent, **scipy** for portfolio optimization, and **plotly/streamlit** for interactive visualization. Bayesian MCMC estimation is built-in with adaptive burn-in tuning.

---

## Single-Channel Usage

### 1. Fitting Curves from Historical Data
Pass raw `Spend` and `Return` arrays directly into the module. You can fit using **Gradient Descent (MLE)**, **Frequentist Non-Linear Least Squares (NLS)**, or **Bayesian MCMC**:

```python
import numpy as np
from tippingpoint import MarketingReturnCurve

spends = np.array([1200, 5000, 15000, 25000, 40000])
returns = np.array([200, 1500, 12000, 22000, 28000])

# 1. Unified Interface (Auto-selects best fitting method)
model = MarketingReturnCurve.fit(
    spend_array=spends,
    return_array=returns,
    channel_name="YouTube Performance",
    adstock_type="bounded",
    adstock_bounds=(1.0, 14.0)
)

# Or explicitly choose an engine:

# 1a. Gradient Descent (MLE)
model_mle = MarketingReturnCurve.fit_gradient_descent(
    spend_array=spends,
    return_array=returns,
    channel_name="YouTube Performance",
    adstock_type="bounded",
    adstock_bounds=(1.0, 14.0)
)

# 1b. Frequentist NLS (includes standard errors & 95% confidence intervals)
model_freq = MarketingReturnCurve.fit_frequentist(
    spend_array=spends,
    return_array=returns,
    channel_name="YouTube Performance",
    adstock_type="bounded",
    adstock_bounds=(1.0, 14.0),
    confidence_level=0.95
)

# 1c. Bayesian MCMC (includes posterior distributions & experimental calibration)
model_bayes = MarketingReturnCurve.fit_bayesian(
    spend_array=spends,
    return_array=returns,
    channel_name="YouTube Performance",
    adstock_type="bounded",
    adstock_bounds=(1.0, 14.0)
)
```


### 2. Extracting Intelligence & Inflection Points

```python
# Evaluate current headroom and efficiency status
model.evaluate_current_budget(current_spend=12000, target_mroas=1.5)

# Programmatically retrieve key boundaries
inflection = model.get_inflection_point()
opt_window = model.get_optimal_scaling_window(target_mroas=1.0)

print(f"Peak Efficiency Spend: ${inflection:,.2f}")
print(f"Optimal Scaling Window: ${opt_window[0]:,.2f} - ${opt_window[1]:,.2f}")
```

**Example Output:**
```text
--- Budget Evaluation: YouTube Performance ---
Current Spend: $12,000.00 | Current mROAS: 2.10
Status: OPTIMAL SCALING ZONE
Recommendation: You are operating within the highly efficient growth window.
```

### 3. Statistical Fit Evaluation & Goodness-of-Fit
Assess the quality of fitted saturation and adstock curves with standard statistical diagnostic metrics ($R^2$, Adjusted $R^2$, RMSE, MAE, MAPE, AIC, BIC):

```python
# Compute and print goodness-of-fit metrics
metrics = model.evaluate_fit(verbose=True)
print(f"R²: {metrics['r_squared']:.4f} | RMSE: {metrics['rmse']:,.2f} | MAPE: {metrics['mape']:.2f}%")
```

### 4. Predictive Uncertainty & Confidence Intervals
Generate point predictions and statistical uncertainty intervals using the **Frequentist Delta Method** or **Bayesian Posterior Samples**:

```python
# Returns (point_prediction, lower_ci, upper_ci)
pred, low, high = model.predict_incremental_return(spend=15000, return_interval=True, confidence_level=0.95)
print(f"Expected Incremental Return: {pred:,.1f} (95% CI: [{low:,.1f}, {high:,.1f}])")
```

### 5. Incrementality Experiment Calibration & Parallel Association
Associate causal lift test results (e.g. geo-experiments, conversion lift studies) in parallel across individual channels, joint MMMs, or portfolio allocators:

```python
# 5a. Associate experiments directly on a single-channel model
model_youtube.add_experiment(spend=15000, lift=11500, se=800, name="YT_Lift_Q1")
model_youtube.validate_experiments(verbose=True)

# 5b. Associate experiments in parallel across multiple channels in an MMM
mmm.attach_experiments({
    "YouTube": [{"name": "YT_Q1", "spend": 15000, "lift": 11500, "se": 800}],
    "Paid Search": {"name": "Search_Q1", "spend": 25000, "lift": 38000, "se": 1200}
})
mmm_validation = mmm.validate_experiments(verbose=True)

# 5c. Audit calibration across portfolio allocator channels before optimization
allocator = PortfolioAllocator([model_search, model_youtube, model_social])
calibration_audit = allocator.get_calibration_summary()
```

### 6. Cross-Channel Portfolio Optimization (Scenario Planning)
Once you have fitted single-channel curves, the `PortfolioAllocator` calculates the budget distribution that maximizes total portfolio return:

```python
from tippingpoint import PortfolioAllocator

# Initialize the Allocator with fitted channel models
allocator = PortfolioAllocator([model_search, model_youtube, model_social])

# Run scenario analysis for a $1,000,000 budget
scenario = allocator.allocate_budget(
    total_budget=1000000,
    channel_bounds={"Paid Search": (50000, 300000)} # Optional constraints
)

print(scenario["allocation"])
print(f"Expected Portfolio Return: ${scenario['expected_total_return']:,.2f}")
```

### 7. Interactive Dashboard & Example Notebooks
*   **Web App Dashboard:** Launch the built-in Streamlit app to explore single-channel curves, adstock carryover timelines, and cross-channel allocation simulations:
    ```bash
    tipp dashboard
    ```
*   **Single-Channel YouTube Saturation Example:** See [`examples/single_channel_youtube_branded_search.ipynb`](examples/single_channel_youtube_branded_search.ipynb) for a concise, step-by-step engineering tutorial on fitting daily YouTube video spend to Attributed Branded Search volume, calculating the geometric carryover half-life, locating Peak Efficiency ($f''(x) = 0$), and identifying the Stop Scaling Point against a $16.00 target CPA.
*   **Causal Experiment Calibration Example:** See [`examples/single_channel_incrementality_calibration.ipynb`](examples/single_channel_incrementality_calibration.ipynb) for an engineering case study integrating holdout conversion lift studies via Bayesian MCMC to decouple organic baseline demand from paid media lift.
*   **Multi-Channel Stacked Walkthrough:** See [`examples/tippingpoint_walkthrough.ipynb`](examples/tippingpoint_walkthrough.ipynb) for an end-to-end tutorial on multi-channel budget allocation and visualizing how brand consideration campaigns shift response curves upward.

---

## Exploring Multi-Channel Dynamics: The Lightweight MMM Framework

While Tipping Point is built around lightweight single-channel curve fitting and portfolio allocation, it also provides a multi-channel modeling class (`MultiChannelMMM`) that allows practitioners to explore how individual channels interact under a unified framework.

> [!IMPORTANT]
> **Not a Substitute for Full MMM:**
> `MultiChannelMMM` is a lightweight, exploratory tool designed to help users examine joint adstock carryover, saturation, and preliminary historical attribution across channels. **It does not provide a full, production-grade Marketing Mix Model.**
>
> A full MMM—such as **[Google's Meridian](https://github.com/google/meridian)**—incorporates rich macroeconomic controls, pricing/promotions, non-media baseline variables, reach and frequency transformations, and comprehensive prior elicitation. For enterprise budget decisions, causal attribution, and complete cross-media measurement, **Google Meridian should always be used to produce robust results.**

### What `MultiChannelMMM` Provides
When you need to analyze multiple spend series simultaneously:
*   **Joint Parameter Estimation:** Simultaneously estimates adstock decay ($\theta_m$), Hill saturation ($\alpha_m, K_m$), baseline ($\beta_0$), and channel scale ($\beta_m$) via MCMC.
*   **Hierarchical Partial Pooling:** Stabilizes estimates for smaller or noisy channels by pooling across channel distributions.
*   **Geo/Regional Hierarchy:** Fits geo-specific multipliers when regional panel data is available.
*   **Historical Contribution Decomposition:** Breaks down historical revenue into organic baseline and channel-specific return series.

```python
import pandas as pd
from tippingpoint import MultiChannelMMM

df = pd.read_csv("weekly_marketing_data.csv")

mmm = MultiChannelMMM(channel_names=["Search", "YouTube", "Social"])
mmm.fit(
    spend_data=df[["Search", "YouTube", "Social"]],
    return_array=df["Revenue"],
    fit_baseline=True,
    n_samples=2000,
    burn_in=500
)

# Decompose historical contributions and channel ROIs
decomp = mmm.decompose_historical_contributions(
    spend_data=df[["Search", "YouTube", "Social"]],
    return_array=df["Revenue"]
)
print(decomp["summary_table"])
```

---

## Integrating with Existing MMMs (Google Meridian)

If you already run Google Meridian or PyMC-Marketing, you can extract your posterior mean parameters and initialize `MarketingReturnCurve` directly without refitting:

```python
# Initialize directly from your existing MMM posterior outputs
model = MarketingReturnCurve(
    beta=120000.0,
    alpha=1.65,
    half_saturation_k=25000.0,
    theta=0.6,
    baseline=5000.0,
    channel_name="YouTube"
)
```
