.. Tipping Point documentation master file, created by
   sphinx-quickstart on Thu May 28 18:42:58 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tipping Point
=============

**Tipping Point** is a lightweight, high-performance marketing intelligence module that uses machine learning and calculus to determine the exact inflection points of a media response curve.

Inspired by the Google Meridian methodology, Tipping Point helps growth marketers make optimal, data-driven budget allocation and scaling decisions.

Key Concepts
============

1. **The Hill Function**
   A S-shaped (or C-shaped) media response curve that maps media spend to incremental return.

   .. math::
      Return = \frac{\beta \cdot Spend^\alpha}{K^\alpha + Spend^\alpha}

2. **First Derivative (Marginal ROAS)**
   Determines the marginal efficiency of the next dollar spent. Peak marginal efficiency occurs at the **Inflection Point**.

3. **Tipping Points**
   - **Peak Efficiency Point (Daily):** The inflection point ($f''(x) = 0$) where acquisition cost is minimized.
   - **Stop Scaling Point (Daily/Annualized):** The point of diminishing returns where Marginal ROAS falls below your profitability threshold (default 1.0).

4. **Geometric Adstock (Prior Decay Carryover)**
   Accounts for lagged effects of prior spend on upcoming returns following a geometric decay model:

   .. math::
      S_{t\_adstocked} = S_t + \theta \cdot S_{t-1\_adstocked}

   Tipping Point supports 4 adstock modes during training:
   - **No Adstock:** Assumes immediate return.
   - **Free Adstock Fit:** Automatically learns decay parameter $\theta \in (0, 1)$ from historical data.
   - **Bounded Adstock Fit:** Constrains the learned half-life decay to a specified day interval $[x, y]$.
   - **Fixed Adstock:** Directly enforces a user-defined decay half-life in days.

5. **Portfolio Optimization (Cross-Channel Scenario Planning)**
   Tipping Point scales from single-channel analysis to a full scenario planning engine. The `PortfolioAllocator` class takes multiple fitted `MarketingReturnCurve` models and utilizes `scipy` optimization (SLSQP algorithm) to find the exact budget allocation that maximizes total incremental return for a given total budget constraint, ensuring marginal ROAS is balanced across all valid channels.

Interactive Dashboard
=====================

Tipping Point includes a fully interactive Streamlit dashboard allowing web-based exploration, separated into two powerful stages:

**Stage 1: Channel Configuration**
- **Dynamic Stacking:** Users can upload custom CSV data or input manual parameters to fit and stack multiple independent channels.
- **Value Denomination:** Optional conversion value multipliers turn raw conversions (leads, installs) into revenue-denominated curves before fitting.
- **Deep Dive Analysis:** Provides a channel-by-channel view of the Plotly saturation curve, marginal efficiency metrics, and an **Adstock Carryover** timeline displaying raw vs. accumulated spend.

**Stage 2: Portfolio Optimization**
- **Scenario Planning:** Input a total portfolio budget and optionally set hard constraints (min/max limits) on specific channels.
- **Optimal Cross-Channel Allocation:** Instantly calculates the most efficient distribution of funds across your configured channels.
- **Visual Benchmarking:** Overlays all configured saturation curves on a single Plotly axis, cleanly marking the "setpoint" for each channel (solid line for funded spend, dashed line for untapped potential).
- **Scale Mix:** A beautiful stacked area plot showing how your optimal channel mix expands, bottlenecks, and shifts weighting as your total investment ceiling increases.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

