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

Interactive Dashboard
=====================

Tipping Point includes a fully interactive Streamlit dashboard allowing web-based exploration:

- **Top-Down Web Layout:** Features a large, dual-axis **Plotly** visualization where users can zoom, drag, and view unified hover tooltips.
- **Explicit Setup Mode:** Launches in a dormant state, allowing users to upload custom CSV data, map columns, and set optional conversion value multipliers (to turn raw conversions like leads/installs into monetary value) before fitting.
- **Active Persistence:** State is managed via `st.session_state` to prevent re-runs or re-fits on slider adjustments.
- **Carryover Analysis Plot:** Shows a dedicated timeline comparing raw daily spends (bars) against accumulated adstocked spends (area) to visually display media wear-out.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

