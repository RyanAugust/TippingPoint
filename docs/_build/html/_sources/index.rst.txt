.. Tipping Point documentation master file.

Tipping Point
=============

**Tipping Point** is a lightweight, high-performance marketing intelligence library that uses machine learning and calculus to determine the exact inflection points of media response curves.

Inspired by modern Marketing Mix Modeling (MMM) principles—specifically the methodologies popularized by Google Meridian—Tipping Point helps growth marketers make optimal, data-driven budget allocation and scaling decisions.

Primary Focus: Single-Channel Curves & Portfolio Allocation
===========================================================

Tipping Point focuses primarily on **single-channel saturation curve fitting** and **cross-channel portfolio planning**:

1. **Lightweight Single-Channel Analysis (:class:`tippingpoint.models.MarketingReturnCurve`)**:
   Targeted, standalone saturation curves for individual marketing channels (e.g., YouTube, Paid Search, Paid Social). Supports fast gradient descent optimization (MLE via Tinygrad) and Bayesian MCMC sampling, customizable adstock decay, unobserved organic baseline estimation, and incrementality experiment calibration.

2. **Cross-Channel Portfolio Allocation (:class:`tippingpoint.portfolio.PortfolioAllocator`)**:
   Ingests multiple fitted channel curves and uses Sequential Least Squares Programming (SLSQP) to find the budget allocation that equalizes marginal ROAS across channels, maximizing total portfolio revenue under global and per-channel spend constraints.

Exploratory Multi-Channel Dynamics (:class:`tippingpoint.mmm.MultiChannelMMM`)
=================================================================================

While Tipping Point is built around lightweight single-channel curve fitting and portfolio allocation, it also provides a *Meridian-lite* multi-channel MMM class (:class:`tippingpoint.mmm.MultiChannelMMM`) to help users explore how individual channels interact.

.. note::
   **Not a Substitute for Full MMM:**
   :class:`tippingpoint.mmm.MultiChannelMMM` is a lightweight, exploratory tool designed to help users examine joint adstock carryover, saturation, and preliminary historical attribution across channels. It does not provide a full, production-grade Marketing Mix Model.

   A full MMM—such as `Google Meridian <https://github.com/google/meridian>`_—incorporates rich macroeconomic controls, pricing/promotions, non-media baseline variables, reach and frequency transformations, and comprehensive prior elicitation. For enterprise budget decisions, causal attribution, and complete cross-media measurement, Google Meridian should always be used to produce robust results.

Key Mathematical Concepts
=========================

1. **Media Saturation (The Hill Function)**
   A flexible S-shaped or C-shaped response curve mapping media spend to incremental return:

   .. math::
      Return = \beta_0 + \frac{\beta \cdot S_{adstocked}^\alpha}{K^\alpha + S_{adstocked}^\alpha}

   - **Beta (:math:`\beta`):** Channel return capacity / asymptote.
   - **Alpha (:math:`\alpha`):** Shape parameter (:math:`\alpha > 1` for S-curve with initial warm-up; :math:`\alpha \le 1` for C-curve with concave returns).
   - **K (:math:`K`):** Half-saturation spend level.
   - **Baseline (:math:`\beta_0`):** Organic, non-media baseline demand.

2. **Marginal ROAS & Tipping Points**
   - **First Derivative (:math:`f'(x)`):** Marginal ROAS—the efficiency of the next dollar invested.
   - **Peak Efficiency Point (:math:`f''(x) = 0`):** The inflection point where marginal cost is lowest and return acceleration peaks.
   - **Stop Scaling Point (:math:`f'(x) = Target\_mROAS`):** The point of diminishing returns where scaling ceases to meet your hurdle rate.
   - **Optimal Scaling Window:** The high-efficiency zone between the inflection point and the stop scaling point.

3. **Adstock Transformations (Memory & Carryover)**
   - **Geometric Adstock:** Exponential decay parameterized by retention rate :math:`\theta \in [0, 1)`:

     .. math::
        S_{t\_adstocked} = S_t + \theta \cdot S_{t-1\_adstocked}

   - **Weibull Adstock:** Delayed peak effects via Weibull PDF or flexible S/C decay via Weibull CDF with shape :math:`k` and scale :math:`\lambda`.

4. **Incrementality Experiment Calibration**
   Integrate causal lift studies (geo-experiments, conversion lift) directly into single-channel or multi-channel curve fitting to anchor Bayesian posteriors to empirically validated truths.

5. **Historical Contribution Decomposition**
   Multi-channel models decompose observed revenue over time into organic baseline and per-channel adstocked contributions, computing historical ROI, spend share, and current marginal ROAS.

6. **Portfolio Optimization (Cross-Channel Scenario Planning)**
   The :class:`tippingpoint.portfolio.PortfolioAllocator` ingests multiple fitted channel curves and uses Sequential Least Squares Programming (SLSQP) to find the budget allocation that equalizes marginal ROAS across channels, maximizing total portfolio revenue under global and per-channel spend constraints.

Interactive Dashboard
=====================

Tipping Point includes an interactive Streamlit dashboard launched via ``tipp dashboard``:

- **Stage 1: Channel Configuration:** Dynamically fit, configure, and stack multiple channels. Features conversion value multipliers and interactive Adstock carryover timelines.
- **Stage 2: Portfolio Optimization:** Set global budgets and constraints. Generates optimal scale mix (stacked area) plots and cross-channel saturation overlays.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

