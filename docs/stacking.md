# Stacked Analysis

In response to customer questions and desired complexity, additionaly signal and analytical rigor should be added

graph TD
    classDef foundation fill:#F8F9FA,stroke:#4285F4,stroke-width:2px,color:#202124;
    classDef adstock fill:#F8F9FA,stroke:#34A853,stroke-width:2px,color:#202124;
    classDef eval fill:#F8F9FA,stroke:#5F6368,stroke-width:2px,color:#202124;
    classDef causal fill:#F8F9FA,stroke:#EA4335,stroke-width:2px,color:#202124;
    classDef multi fill:#F8F9FA,stroke:#FBBC04,stroke-width:2px,color:#202124;
    classDef portfolio fill:#4285F4,stroke:#1A73E8,stroke-width:2px,color:#FFFFFF;

    subgraph Level1["Level 1: Single-Channel Static Saturation"]
        Q1["Question: At what spend level does an isolated channel exit warm-up and hit diminishing returns?"]
        M1["Engine: Hill Function Saturation<br/>y = (beta * x^alpha) / (K^alpha + x^alpha)"]
        O1["Output: Peak Efficiency Point (f''(x)=0) & Stop Scaling Point (f'(x)=Target mROAS)"]
        Q1 --> M1 --> O1
    end

    subgraph Level2["Level 2: Saturation with Adstock Carryover"]
        Q2["Question: How does brand memory and delayed conversion impact daily spend headroom?"]
        M2["Engine: Vectorized Geometric / Weibull Adstock<br/>S_eff,t = S_t + theta * S_eff,t-1"]
        O2["Output: Carryover Half-Life & Steady-State Daily Headroom: S_daily = S_eff * (1 - theta)"]
        Q2 --> M2 --> O2
    end

    subgraph Level3["Level 3: Statistical Assessment & Uncertainty"]
        Q3["Question: How well does the model fit historical data, and what is the uncertainty band?"]
        M3["Engine: Goodness-of-Fit & Delta Method<br/>R², Adj R², RMSE, AIC/BIC, Covariance Matrix"]
        O3["Output: 95% Confidence Intervals (Frequentist) & 90% Credible Intervals (Bayesian)"]
        Q3 --> M3 --> O3
    end

    subgraph Level4["Level 4: Incrementality Experiment Validation"]
        Q4["Question: Is observational regression inflated by organic demand, and does it match causal tests?"]
        M4["Engine: Incrementality Validation & Bayesian Holdout Anchoring<br/>Z-scores, Reduced Chi-Square, Organic Baseline beta_0"]
        O4["Output: Decoupled Organic Baseline vs True Causal Lift & Calibrated Saturation Curves"]
        Q4 --> M4 --> O4
    end

    subgraph Level5["Level 5: Multi-Channel Synergy & Attribution"]
        Q5["Question: How do multiple channels interact, and how does upper-funnel brand lift shift lower-funnel ceilings?"]
        M5["Engine: MultiChannelMMM Joint Estimation<br/>Y_t = Baseline + sum(Hill_m(Adstock_m(S_m,t)))"]
        O5["Output: Historical Return Attribution & Curve-Shifting Synergy Analysis"]
        Q5 --> M5 --> O5
    end

    subgraph Level6["Level 6: Cross-Channel Portfolio Optimization"]
        Q6["Question: Given a fixed total budget, what spend distribution maximizes total portfolio return?"]
        M6["Engine: PortfolioAllocator Constrained Non-Linear Optimization<br/>Equalize Marginal ROAS across all channels"]
        O6["Output: Mathematically Optimal Budget Split & Projected Portfolio Returns"]
        Q6 --> M6 --> O6
    end

    O1 --> Level2
    O2 --> Level3
    O3 --> Level4
    O4 --> Level5
    O5 --> Level6

    class Q1,M1,O1 foundation;
    class Q2,M2,O2 adstock;
    class Q3,M3,O3 eval;
    class Q4,M4,O4 causal;
    class Q5,M5,O5 multi;
    class Q6,M6,O6 portfolio;


#### Level 1: Single-Channel Static Saturation

  • Core Question: "Given immediate spend, what is my channel capacity, where is peak efficiency, and when do I hit diminishing returns?"
  • Component: models.py fitting the Hill response function:

             βxᵅ
    f(x) = ───────
           Kᵅ + xᵅ

  • Strategic Output:
      • Peak Efficiency Point (f''(x) = 0): The inflection point where marginal return peaks. Spend below this threshold is in the inefficient warm-up zone.
      • Stop Scaling Point (f'(x) = Target mROAS): The exact spend level where marginal return drops below the profitability hurdle rate.
      • Optimal Scaling Zone: The operating window between peak efficiency and diminishing returns.

  ──────
  #### Level 2: Saturation with Adstock Carryover

  • Core Question: "How does advertising memory and delayed conversion alter daily spend capacity?"
  • Component: Vectorized geometric (θ) or Weibull (k,λ) adstock transformations in math.py.
  • Strategic Output:
      • Carryover Half-Life:


             ln 2
    t    = - ────
     1/2     ln θ

  days.

  • Steady-State Daily Scaling Limit: Converts effective adstocked headroom (

    S
     effective

  ) into actionable daily spend limits:

    S      = S         ·(1 - θ)
     daily    effective
  ──────
  #### Level 3: Statistical Assessment & Uncertainty

  • Core Question: "Can I trust this curve, and what is the margin of error on expected returns?"
  • Component: Goodness-of-fit suite in evaluation.py and Delta-method predictive variance in models.py.
  • Strategic Output:
      • Fit Diagnostics: R², Adjusted R², RMSE, MAE, MAPE, AIC, and BIC.
      • Predictive Uncertainty Intervals: 95% Confidence Intervals (Frequentist NLS) or 90% Credible Intervals (Bayesian MCMC) surrounding both total return y and marginal ROAS f'(x).

  ──────
  #### Level 4: Incrementality Experiment Validation

  • Core Question: "Is observational regression taking credit for organic brand demand, and does the fitted curve align with ground-truth lift tests?"
  • Component: Standalone validation engine in validation.py and Bayesian prior calibration in bayesian.py.
  • Strategic Output:
      • Validation Scoring: Computes Z-scores, 95% CI coverage, and reduced χ² alignment statistics against user lift studies.
      • Causal Calibration: Decouples non-paid organic baseline volume (β₀) from true paid media lift (β).

  ──────
  #### Level 5: Multi-Channel Synergy & Attribution

  • Core Question: "How do multiple channels interact, and how does upper-funnel brand awareness expand the saturation ceiling of lower-funnel performance channels?"
  • Component: mmm.py joint estimation:

               M
    Y  = β  +  ∑   Hill  ⎛Adstock  ⎛S   ⎞⎞ + ε
     t    0   m=1      m ⎝       m ⎝ m,t⎠⎠    t

  • Strategic Output:
      • Historical Contribution Decomposition: Time-series attribution across Baseline, Paid Search, Paid Social, and YouTube.
      • Curve-Shifting Synergy: Visualizes how brand consideration raises the maximum incremental capacity (β) of performance channels.

  ──────
  #### Level 6: Cross-Channel Portfolio Optimization

  • Core Question: "Given a fixed total budget, what is the globally optimal spend split across all channels to maximize portfolio return?"
  • Component: portfolio.py constrained non-linear solver.
  • Strategic Output:
      • Equalizing Marginal ROAS: Reallocates spend from saturated channel tails (mROAS < 1.0) into channels with high marginal headroom.
      • Optimal Allocation Splits: Returns exact dollar allocations for each channel subject to custom minimum/maximum business constraints.
