import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import subprocess
import sys
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tippingpoint import MarketingReturnCurve

def fit_in_subprocess(spends, returns, epochs, lr, channel_name, adstock_type="none", adstock_bounds=None, adstock_fixed_days=None):
  """Fits the model in an isolated Python subprocess to prevent tinygrad from crashing Streamlit."""
  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in:
    pickle.dump((spends, returns, epochs, lr, channel_name, adstock_type, adstock_bounds, adstock_fixed_days), f_in)
    in_path = f_in.name

  with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_out:
    out_path = f_out.name

  code = f"""
import pickle
import sys
sys.path.append('src')
from tippingpoint import MarketingReturnCurve

with open({repr(in_path)}, 'rb') as f:
  spends, returns, epochs, lr, channel_name, adstock_type, adstock_bounds, adstock_fixed_days = pickle.load(f)

model = MarketingReturnCurve.from_historical_data(
  spend_array=spends,
  return_array=returns,
  channel_name=channel_name,
  epochs=epochs,
  lr=lr,
  adstock_type=adstock_type,
  adstock_bounds=adstock_bounds,
  adstock_fixed_days=adstock_fixed_days
)

with open({repr(out_path)}, 'wb') as f:
  pickle.dump(model, f)
"""

  try:
    # Run subprocess with the current python executable
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
      raise RuntimeError(f"Fitting process failed:\\n{result.stderr or result.stdout}")

    with open(out_path, "rb") as f:
      model = pickle.load(f)

    return model
  finally:
    # Clean up temp files
    if os.path.exists(in_path): os.remove(in_path)
    if os.path.exists(out_path): os.remove(out_path)

def create_plotly_plot(model, target_mroas, scatter=None):
  min_spend = model.get_minimal_marginal_cost_point()
  max_spend = model.get_diminishing_returns_point(target_mroas)

  # Prepare scatter data if adstock is active
  scatter_spend = None
  if scatter is not None:
    scatter_spend, scatter_return = scatter
    if model.theta > 0:
      from tippingpoint.math import geometric_adstock
      scatter_spend = geometric_adstock(scatter_spend, model.theta)

  # Determine plot limits
  plot_limit = (max_spend * 1.5) if max_spend else (min_spend * 4 or 100000)
  if scatter_spend is not None and len(scatter_spend) > 0:
    plot_limit = max(plot_limit, float(np.max(scatter_spend) * 1.1))

  x_vals = np.linspace(0, plot_limit, 500)
  y_return = model.predict_incremental_return(x_vals)
  y_mroas = model.predict_marginal_return(x_vals)

  # Create figure with secondary y-axis
  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Add Incremental Return trace
  fig.add_trace(
    go.Scatter(x=x_vals, y=y_return, name="Incremental Return", line=dict(color='#4285F4', width=3)),
    secondary_y=False,
  )

  # Add Marginal ROAS trace
  fig.add_trace(
    go.Scatter(x=x_vals, y=y_mroas, name="Marginal ROAS", line=dict(color='#5F6368', dash='dash', width=1.5), opacity=0.8),
    secondary_y=True,
  )

  # Add Target mROAS horizontal line
  fig.add_trace(
    go.Scatter(
      x=[0, plot_limit],
      y=[target_mroas, target_mroas],
      name=f"Target mROAS ({target_mroas})",
      line=dict(color='#EA4335', width=1, dash='dot'),
      showlegend=True
    ),
    secondary_y=True,
  )

  # Add Scatter data if enabled
  if scatter_spend is not None:
      fig.add_trace(
          go.Scatter(
              x=scatter_spend, y=scatter_return,
              mode="markers", name="Historical Data (Adstocked)" if model.theta > 0 else "Historical Data",
              marker=dict(color='#4285F4', size=8, opacity=0.6, line=dict(color='white', width=1)),
              showlegend=True
          ),
          secondary_y=False
      )
  # Highlight Optimal Scaling Zone (using shapes)
  if max_spend and max_spend > min_spend:
    fig.add_vrect(
      x0=min_spend, x1=max_spend,
      fillcolor="#34A853", opacity=0.1,
      layer="below", line_width=0,
      annotation_text="OPTIMAL ZONE", annotation_position="top left",
      annotation_font=dict(size=10, color="#34A853", weight="bold")
    )

  # Markers for key points
  if min_spend > 0:
    idx = (np.abs(x_vals - min_spend)).argmin()
    fig.add_trace(
      go.Scatter(
        x=[min_spend], y=[y_mroas[idx]],
        mode="markers", name="Peak Efficiency",
        marker=dict(color="#FBBC04", size=10, line=dict(color="#5F6368", width=1)),
        showlegend=True
      ),
      secondary_y=True
    )

  # Set x-axis title
  fig.update_xaxes(title_text="Daily Spend ($)", tickformat="$,.0f")

  # Set y-axes titles
  fig.update_yaxes(title_text="Incremental Return", secondary_y=False, tickformat=",.0f")
  fig.update_yaxes(title_text="Marginal ROAS", secondary_y=True, tickformat=",.2f", showgrid=False)

  # Layout tuning
  fig.update_layout(
    title_text=f"Media Response Analysis: {model.channel_name}",
    title_font=dict(size=20, color='#202124'),
    font=dict(size=13, color='#5F6368'),
    height=650,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
    template="plotly_white",
    margin=dict(l=20, r=20, t=60, b=20)
  )

  return fig

def create_adstock_timeline_plot(spends, model):
    """Generates an interactive Plotly timeline comparing raw vs adstocked spends."""
    adstocked = model.adstock_spend(spends)
    indices = np.arange(len(spends))

    fig = go.Figure()

    # Raw Spend (Bar Chart)
    fig.add_trace(go.Bar(
        x=indices, y=spends,
        name="Raw Spend",
        marker_color="#EA4335",
        opacity=0.4
    ))

    # Adstocked Spend (Line/Area Chart)
    fig.add_trace(go.Scatter(
        x=indices, y=adstocked,
        mode="lines",
        name="Effective Adstocked Spend",
        line=dict(color="#4285F4", width=3),
        fill='tozeroy',
        fillcolor="rgba(66, 133, 244, 0.1)"
    ))

    half_life = -np.log(2) / np.log(model.theta) if model.theta > 0 else 0.0
    fig.update_layout(
        title=f"Adstock Carryover Decay (Half-Life: {half_life:.1f} Days, Theta: {model.theta:.4f})",
        xaxis_title="Observation Timeline (Days / Periods)",
        yaxis_title="Spend ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def run_dashboard():

  st.set_page_config(page_title="Tipping Point Dashboard", layout="wide")

  st.title("Media Response Curve Dashboard")
  st.markdown("""
  Interact with your media response model to find the **Optimal Scaling Zone**.
  Adjust the target marginal ROAS (mROAS) to see how it impacts your recommended spend limits.
  """)

  # Initialize session state for the model
  if "model" not in st.session_state:
    st.session_state.model = None
  if "training_data" not in st.session_state:
    st.session_state.training_data = None

  # Check for external model from python script invocation (launch_dashboard)
  external_model_path = os.environ.get("TIPPINGPOINT_MODEL_PATH")
  if external_model_path and os.path.exists(external_model_path) and st.session_state.model is None:
    try:
      with open(external_model_path, "rb") as f:
        st.session_state.model = pickle.load(f)
      st.sidebar.success(f"Loaded model from script: {st.session_state.model.channel_name}")
    except Exception as e:
      st.sidebar.error(f"Failed to load external model: {e}")

  # Sidebar: Model Configuration / Actions
  st.sidebar.header("Model Configuration")

  if st.session_state.model is None:
    # Step 1: No model loaded yet. Show configuration options.
    data_source = st.sidebar.selectbox("Select Input Method", ["Upload CSV", "Use Sample Data", "Manual Parameters"])

    adstock_type = "none"
    adstock_bounds = None
    adstock_fixed_days = None

    if data_source in ["Upload CSV", "Use Sample Data"]:
      st.sidebar.markdown("---")
      st.sidebar.subheader("Adstock (Lagged Effects)")
      st.sidebar.markdown("Following Google Meridian, account for prior media spend on upcoming returns.")
      adstock_opt = st.sidebar.selectbox(
        "Adstock Mode",
        [
          "No Adstock",
          "Adstock completely decided by fitting",
          "Bounded fitting of the data (days)",
          "Explicitly set the adstock value (days)"
        ]
      )
      if adstock_opt == "No Adstock":
        adstock_type = "none"
      elif adstock_opt == "Adstock completely decided by fitting":
        adstock_type = "free"
      elif adstock_opt == "Bounded fitting of the data (days)":
        adstock_type = "bounded"
        min_days = st.sidebar.number_input("Minimum Decay Days (Half-life)", value=1.0, min_value=0.1, step=0.5)
        max_days = st.sidebar.number_input("Maximum Decay Days (Half-life)", value=14.0, min_value=1.0, step=1.0)
        adstock_bounds = (min_days, max_days)
      elif adstock_opt == "Explicitly set the adstock value (days)":
        adstock_type = "fixed"
        adstock_fixed_days = st.sidebar.number_input("Explicit Decay Days (Half-life)", value=3.0, min_value=0.1, step=0.5)

    if data_source == "Upload CSV":
      uploaded_file = st.sidebar.file_uploader("Upload Historical Data (CSV)", type=["csv"])
      if uploaded_file:
        try:
          df = pd.read_csv(uploaded_file)
          st.sidebar.write("Preview:")
          st.sidebar.dataframe(df.head(3))

          # Try to find columns
          spend_col = st.sidebar.selectbox("Spend Column", df.columns, index=0 if "spend" in df.columns[0].lower() else 0)
          return_col = st.sidebar.selectbox("Return/KPI Column", df.columns, index=1 if len(df.columns) > 1 and "return" in df.columns[1].lower() else 0)

          # Conversion value multiplier
          set_val = st.sidebar.checkbox("set conversion value", value=False)
          conversion_val = 1.0
          if set_val:
            conversion_val = st.sidebar.number_input("Conversion Value ($)", value=100.0, min_value=0.01, step=1.0, help="Multiply raw returns (e.g. counts) by this value to generate revenue-denominated returns.")

          epochs = st.sidebar.number_input("Fitting Epochs", value=1000, step=100)

          if st.sidebar.button("🚀 Fit Model from CSV"):
            model = None
            try:
              with st.spinner("Fitting curve using tinygrad..."):
                spends = df[spend_col].values
                returns = df[return_col].values * conversion_val
                model = fit_in_subprocess(
                  spends=spends,
                  returns=returns,
                  epochs=epochs,
                  lr=0.05,
                  channel_name="Uploaded Channel",
                  adstock_type=adstock_type,
                  adstock_bounds=adstock_bounds,
                  adstock_fixed_days=adstock_fixed_days
                )
            except Exception as e:
              st.sidebar.error(f"Error processing CSV: {e}")

            if model:
              st.session_state.model = model
              st.session_state.training_data = (spends, returns)
              st.rerun()
        except Exception as e:
          st.sidebar.error(f"Error processing CSV: {e}")
      else:
        st.info("Please upload a CSV file with spend and return columns.")

    elif data_source == "Use Sample Data":
      st.sidebar.markdown("""
      **Sample Dataset:**
      - Spends: `[1k, 5k, 15k, 25k, 40k, 60k]`
      - Returns: `[200, 1.5k, 12k, 22k, 28k, 32k]`
      """)
      epochs = st.sidebar.number_input("Fitting Epochs", value=1000, step=100)
      if st.sidebar.button("🚀 Fit Sample Model"):
        model = None
        try:
          with st.spinner("Fitting sample curve..."):
            spends = np.array([1000, 5000, 15000, 25000, 40000, 60000])
            returns = np.array([200, 1500, 12000, 22000, 28000, 32000])
            model = fit_in_subprocess(
              spends=spends,
              returns=returns,
              epochs=epochs,
              lr=0.05,
              channel_name="Sample Channel",
              adstock_type=adstock_type,
              adstock_bounds=adstock_bounds,
              adstock_fixed_days=adstock_fixed_days
            )
        except Exception as e:
          st.sidebar.error(f"Error fitting sample model: {e}")

        if model:
          st.session_state.model = model
          st.session_state.training_data = (spends, returns)
          st.rerun()

    elif data_source == "Manual Parameters":
      beta = st.sidebar.number_input("Beta (Max Capacity)", value=50000.0, step=1000.0)
      alpha = st.sidebar.number_input("Alpha (Shape Parameter)", value=1.8, min_value=0.1, step=0.1)
      k = st.sidebar.number_input("K (Half-Saturation Point)", value=20000.0, step=1000.0)
      theta = st.sidebar.number_input("Theta (Adstock Decay)", value=0.0, min_value=0.0, max_value=0.99, step=0.05)

      if st.sidebar.button("✅ Apply Parameters"):
        model = MarketingReturnCurve(beta=beta, alpha=alpha, half_saturation_k=k, theta=theta, channel_name="Custom Channel")
        st.session_state.model = model
        st.rerun()

    # Display placeholder in main area when no model is active
    st.warning("👈 Please configure and fit/apply a model in the sidebar to begin.")

    # Simple math reference for the user
    st.markdown("""
    ### Mathematical Reference
    The model fits historical performance data to a continuous **Hill Function**:

    $$Return = \\frac{\\beta \\cdot Spend^\\alpha}{K^\\alpha + Spend^\\alpha}$$

    - **$\\beta$ (Beta):** Maximum possible return capacity (asymptote).
    - **$\\alpha$ (Alpha):** Shape parameter (S-shape if $>1$, C-shape if $\\le 1$).
    - **$K$ (Half-Saturation):** Spend level where half of maximum capacity is achieved.
    """)
    return

  # Step 2: Model is active. Show full interactive dashboard.
  model = st.session_state.model

  # Option to reset/change model
  if st.sidebar.button("🔄 Reset / Load New Model"):
    st.session_state.model = None
    st.session_state.training_data = None
    if "TIPPINGPOINT_MODEL_PATH" in os.environ:
      del os.environ["TIPPINGPOINT_MODEL_PATH"]
    st.rerun()

  st.sidebar.markdown("---")
  st.sidebar.header("Optimization Settings")
  target_mroas = st.sidebar.slider(
    "Target Marginal ROAS (mROAS)",
    min_value=0.1,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="The minimum return you expect for every additional dollar spent."
  )

  # Historical Data Points toggle
  show_data_points = False
  if "training_data" in st.session_state and st.session_state.training_data is not None:
    show_data_points = st.sidebar.checkbox("Show Historical Data Points", value=True)

  # Calculations
  min_spend = model.get_minimal_marginal_cost_point()
  max_spend = model.get_diminishing_returns_point(target_mroas=target_mroas)

  # Dashboard Layout
  st.subheader("Media Response & Marginal Efficiency")
  fig = create_plotly_plot(model, target_mroas, scatter=st.session_state.training_data if show_data_points else None)
  st.plotly_chart(fig, use_container_width=True)

  st.markdown("---")
  st.subheader("Scaling Recommendations")

  if max_spend:
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Peak Efficiency Point (Daily)", f"${min_spend:,.2f}")
    with col2:
      st.metric(f"Stop Scaling Point (Daily, mROAS={target_mroas})", f"${max_spend:,.2f}")
    with col3:
      annual_spend = max_spend * 365
      annual_spend_m = annual_spend / 1_000_000
      st.metric("Stop Scaling Point (Annualized)", f"${annual_spend_m:.2f}M")

    st.info(f"💡 **Strategic Recommendation:** You should spend at least **${min_spend:,.2f}** per day to exit the inefficient warm-up phase (Peak Efficiency). However, do not scale spend beyond **${max_spend:,.2f}** per day (or **${annual_spend_m:.2f}M** annualized), as any additional dollar spent beyond this threshold will return less than your target mROAS of **{target_mroas:.2f}**.")
  else:
    st.warning(f"Target mROAS of {target_mroas} is unreachable with current model parameters.")
    st.write(f"Max possible mROAS: {model.predict_marginal_return(min_spend):.2f}")

  if model.theta > 0:
    st.markdown("---")
    st.subheader("🕰️ Adstock Carryover Analysis")

    col1, col2 = st.columns(2)
    with col1:
      st.metric("Fitted Decay Rate (Theta)", f"{model.theta:.4f}")
    with col2:
      half_life = -np.log(2) / np.log(model.theta) if model.theta > 0 else 0.0
      st.metric("Carryover Half-Life", f"{half_life:.1f} Days")

    if st.session_state.training_data is not None:
      spends, _ = st.session_state.training_data
      adstock_fig = create_adstock_timeline_plot(spends, model)
      st.plotly_chart(adstock_fig, use_container_width=True)
    else:
      st.info("No timeline data available to show carryover plot (Manual Parameters loaded).")

  st.markdown("---")
  st.subheader("Model Summary")
  st.json(model.summary())

if __name__ == "__main__":
  run_dashboard()
