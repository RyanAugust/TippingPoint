import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tippingpoint import MarketingReturnCurve

def run_dashboard():
    st.set_page_config(page_title="Tipping Point Dashboard", layout="wide")

    st.title("📈 Media Response Curve Dashboard")
    st.markdown("""
    Interact with your media response model to find the **Optimal Scaling Zone**.
    Adjust the target marginal ROAS (mROAS) to see how it impacts your recommended spend limits.
    """)

    # Initialize session state for the model
    if "model" not in st.session_state:
        st.session_state.model = None

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

                    epochs = st.sidebar.number_input("Fitting Epochs", value=1000, step=100)

                    if st.sidebar.button("🚀 Fit Model from CSV"):
                        with st.spinner("Fitting curve using tinygrad..."):
                            spends = df[spend_col].values
                            returns = df[return_col].values
                            model = MarketingReturnCurve.from_historical_data(
                                spend_array=spends,
                                return_array=returns,
                                channel_name="Uploaded Channel",
                                epochs=epochs
                            )
                            st.session_state.model = model
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
                with st.spinner("Fitting sample curve..."):
                    spends = np.array([1000, 5000, 15000, 25000, 40000, 60000])
                    returns = np.array([200, 1500, 12000, 22000, 28000, 32000])
                    model = MarketingReturnCurve.from_historical_data(
                        spend_array=spends,
                        return_array=returns,
                        channel_name="Sample Channel",
                        epochs=epochs
                    )
                    st.session_state.model = model
                    st.rerun()

        elif data_source == "Manual Parameters":
            beta = st.sidebar.number_input("Beta (Max Capacity)", value=50000.0, step=1000.0)
            alpha = st.sidebar.number_input("Alpha (Shape Parameter)", value=1.8, min_value=0.1, step=0.1)
            k = st.sidebar.number_input("K (Half-Saturation Point)", value=20000.0, step=1000.0)

            if st.sidebar.button("✅ Apply Parameters"):
                model = MarketingReturnCurve(beta=beta, alpha=alpha, half_saturation_k=k, channel_name="Custom Channel")
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

    # Calculations
    min_spend = model.get_minimal_marginal_cost_point()
    max_spend = model.get_diminishing_returns_point(target_mroas=target_mroas)

    # Dashboard Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Media Response & Marginal Efficiency")
        fig = model.plot_response_curve(target_mroas=target_mroas, show=False)
        st.pyplot(fig)

    with col2:
        st.subheader("Scaling Recommendations")

        if max_spend:
            st.metric("Peak Efficiency Point (Daily)", f"${min_spend:,.2f}")

            st.markdown("---")
            st.markdown(f"### Stop Scaling Point (mROAS = {target_mroas})")

            # Daily Metric
            st.metric("Max Recommended Daily Spend", f"${max_spend:,.2f}")

            # Annualized Metric
            annual_spend = max_spend * 365
            annual_spend_m = annual_spend / 1_000_000
            st.metric("Max Recommended Annual Spend", f"${annual_spend_m:.2f}M")

            st.info(f"Spending beyond **${max_spend:,.2f}** per day will return less than **${target_mroas:.2f}** for every additional dollar invested.")
        else:
            st.warning(f"Target mROAS of {target_mroas} is unreachable with current model parameters.")
            st.write(f"Max possible mROAS: {model.predict_marginal_return(min_spend):.2f}")

    st.markdown("---")
    st.subheader("Model Summary")
    st.json(model.summary())

if __name__ == "__main__":
    run_dashboard()
