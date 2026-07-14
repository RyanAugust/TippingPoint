import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tippingpoint import MarketingReturnCurve

@st.cache_resource
def load_external_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_sample_model():
    # Generate some dummy data
    spends = np.array([1000, 5000, 15000, 25000, 40000, 60000])
    returns = np.array([200, 1500, 12000, 22000, 28000, 32000])

    return MarketingReturnCurve.from_historical_data(
        spend_array=spends,
        return_array=returns,
        channel_name="Sample Channel",
        epochs=1000
    )

def run_dashboard():
    st.set_page_config(page_title="Tipping Point Dashboard", layout="wide")

    st.title("📈 Media Response Curve Dashboard")
    st.markdown("""
    Interact with your media response model to find the **Optimal Scaling Zone**.
    Adjust the target marginal ROAS (mROAS) to see how it impacts your recommended spend limits.
    """)

    # Sidebar for Model Parameters or Data
    st.sidebar.header("Model Configuration")

    external_model_path = os.environ.get("TIPPINGPOINT_MODEL_PATH")
    if external_model_path and os.path.exists(external_model_path):
        model = load_external_model(external_model_path)
        st.sidebar.success(f"Loaded model: {model.channel_name}")
        if st.sidebar.button("Clear External Model"):
            del os.environ["TIPPINGPOINT_MODEL_PATH"]
            st.rerun()
    else:
        data_source = st.sidebar.selectbox("Data Source", ["Sample Data", "Manual Parameters"])

        if data_source == "Sample Data":
            with st.spinner("Fitting model to sample data..."):
                model = get_sample_model()
        else:
            beta = st.sidebar.number_input("Beta (Max Capacity)", value=50000.0)
            alpha = st.sidebar.number_input("Alpha (Shape)", value=1.8, min_value=0.1)
            k = st.sidebar.number_input("K (Half-Saturation)", value=20000.0)
            model = MarketingReturnCurve(beta=beta, alpha=alpha, half_saturation_k=k, channel_name="Custom Channel")

    # Interactive Slider for Target mROAS
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
        # Use the existing plotting logic
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
