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
from tippingpoint import MarketingReturnCurve, PortfolioAllocator

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
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Fitting process failed:\\n{result.stderr or result.stdout}")

        with open(out_path, "rb") as f:
            model = pickle.load(f)

        return model
    finally:
        if os.path.exists(in_path): os.remove(in_path)
        if os.path.exists(out_path): os.remove(out_path)

def create_plotly_plot(model, target_mroas, scatter=None):
    min_spend = model.get_minimal_marginal_cost_point()
    max_spend = model.get_diminishing_returns_point(target_mroas)

    scatter_spend = None
    if scatter is not None:
        scatter_spend, scatter_return = scatter
        if model.theta > 0:
            from tippingpoint.math import geometric_adstock
            scatter_spend = geometric_adstock(scatter_spend, model.theta)

    plot_limit = (max_spend * 1.5) if max_spend else (min_spend * 4 or 100000)
    if scatter_spend is not None and len(scatter_spend) > 0:
        plot_limit = max(plot_limit, float(np.max(scatter_spend) * 1.1))

    x_vals = np.linspace(0, plot_limit, 500)
    y_return = model.predict_incremental_return(x_vals)
    y_mroas = model.predict_marginal_return(x_vals)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=x_vals, y=y_return, name="Incremental Return", line=dict(color='#4285F4', width=3)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x_vals, y=y_mroas, name="Marginal ROAS", line=dict(color='#5F6368', dash='dash', width=1.5), opacity=0.8),
        secondary_y=True,
    )

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

    if max_spend and max_spend > min_spend:
        fig.add_vrect(
            x0=min_spend, x1=max_spend,
            fillcolor="#34A853", opacity=0.1,
            layer="below", line_width=0,
            annotation_text="OPTIMAL ZONE", annotation_position="top left",
            annotation_font=dict(size=10, color="#34A853", weight="bold")
        )

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

    fig.update_xaxes(title_text="Daily Spend ($)", tickformat="$,.0f")
    fig.update_yaxes(title_text="Incremental Return", secondary_y=False, tickformat=",.0f")
    fig.update_yaxes(title_text="Marginal ROAS", secondary_y=True, tickformat=",.2f", showgrid=False)

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

def create_allocation_mix_plot(models_dict, max_budget, channel_bounds):
    """Generates a stacked area chart showing optimal channel mix as total budget increases."""
    from tippingpoint.portfolio import PortfolioAllocator
    allocator = PortfolioAllocator(list(models_dict.values()))

    # We will test 50 budget points from 0 to the target max budget
    budgets = np.linspace(max_budget * 0.1, max_budget * 1.5, 50)
    allocations_history = {cname: [] for cname in models_dict.keys()}
    valid_budgets = []

    for b in budgets:
        res = allocator.allocate_budget(total_budget=b, channel_bounds=channel_bounds)
        if res["success"]:
            valid_budgets.append(b)
            for cname in models_dict.keys():
                allocations_history[cname].append(res["allocation"][cname])

    fig = go.Figure()
    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#AB47BC', '#00ACC1', '#FF7043', '#8D6E63']

    for i, (cname, allocs) in enumerate(allocations_history.items()):
        if np.sum(allocs) > 0:  # Only plot if channel gets some budget
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=valid_budgets, y=allocs,
                mode='lines',
                name=cname,
                stackgroup='one',
                line=dict(width=0.5, color=color),
                fillcolor=color
            ))

    # Add vertical line for the current selected budget
    fig.add_vline(x=max_budget, line_width=2, line_dash="dash", line_color="#202124",
                  annotation_text="Current Budget", annotation_position="top left")

    fig.update_layout(
        title_text="Optimal Channel Mix at Scale",
        title_font=dict(size=18, color='#202124'),
        font=dict(size=13, color='#5F6368'),
        xaxis_title="Total Portfolio Budget ($)",
        yaxis_title="Allocated Spend ($)",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def create_adstock_timeline_plot(spends, model):
    adstocked = model.adstock_spend(spends)
    indices = np.arange(len(spends))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=indices, y=spends,
        name="Raw Spend",
        marker_color="#EA4335",
        opacity=0.4
    ))

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

def create_portfolio_curves_plot(models_dict, allocations):
    """Generates an overlay of saturation curves for all channels, marking optimal allocations."""
    fig = go.Figure()

    # Generate a nice color palette for channels
    colors = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#AB47BC', '#00ACC1', '#FF7043', '#8D6E63']

    max_spend_limit = 0.0
    for alloc in allocations.values():
        max_spend_limit = max(max_spend_limit, alloc * 2.5) # ensure the curve extends past the allocation
    if max_spend_limit == 0:
        max_spend_limit = 100000.0

    x_vals = np.linspace(0, max_spend_limit, 500)

    for i, (cname, model) in enumerate(models_dict.items()):
        color = colors[i % len(colors)]
        alloc_x = allocations.get(cname, 0.0)

        if alloc_x > 0:
            # Solid line before allocation
            x_solid = x_vals[x_vals <= alloc_x]
            if len(x_solid) == 0 or x_solid[-1] < alloc_x:
                x_solid = np.append(x_solid, alloc_x)
            y_solid = model.predict_incremental_return(x_solid)

            fig.add_trace(go.Scatter(
                x=x_solid, y=y_solid,
                mode='lines',
                name=cname,
                line=dict(color=color, width=3)
            ))

            # Dashed line after allocation
            x_dashed = x_vals[x_vals >= alloc_x]
            if len(x_dashed) > 0:
                y_dashed = model.predict_incremental_return(x_dashed)
                fig.add_trace(go.Scatter(
                    x=x_dashed, y=y_dashed,
                    mode='lines',
                    showlegend=False,
                    line=dict(color=color, width=3, dash='dash')
                ))

            # Marker for allocation
            alloc_y = model.predict_incremental_return(alloc_x)
            fig.add_trace(go.Scatter(
                x=[alloc_x], y=[alloc_y],
                mode='markers',
                name=f"{cname} Allocation",
                marker=dict(color=color, size=12, line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f"<b>{cname}</b><br>Spend: ${{x:,.0f}}<br>Return: ${{y:,.0f}}<extra></extra>"
            ))
        else:
            # Fully dashed if no allocation
            y_return = model.predict_incremental_return(x_vals)
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_return,
                mode='lines',
                name=cname,
                line=dict(color=color, width=3, dash='dash')
            ))

    fig.update_layout(
        title_text="Cross-Channel Saturation Curve Overlay",
        title_font=dict(size=18, color='#202124'),
        font=dict(size=13, color='#5F6368'),
        xaxis_title="Allocated Spend ($)",
        yaxis_title="Incremental Return",
        height=500,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

def run_dashboard():
    st.set_page_config(page_title="Tipping Point Dashboard", layout="wide")

    st.title("Media Response Curve Dashboard")
    st.markdown("""
    Interact with your media response models to find the **Optimal Scaling Zone** and allocate budget across multiple channels.
    """)

    if "models" not in st.session_state:
        st.session_state.models = {}
    if "training_data" not in st.session_state:
        st.session_state.training_data = {}

    external_model_path = os.environ.get("TIPPINGPOINT_MODEL_PATH")
    if external_model_path and os.path.exists(external_model_path):
        try:
            with open(external_model_path, "rb") as f:
                ext_model = pickle.load(f)
                if ext_model.channel_name not in st.session_state.models:
                    st.session_state.models[ext_model.channel_name] = ext_model
            st.sidebar.success(f"Loaded model from script: {ext_model.channel_name}")
            del os.environ["TIPPINGPOINT_MODEL_PATH"]
        except Exception as e:
            st.sidebar.error(f"Failed to load external model: {e}")

    tab1, tab2 = st.tabs(["Stage 1: Channel Configuration", "Stage 2: Portfolio Optimization"])

    with tab1:
        st.header("Configure Channels")

        if st.session_state.models:
            st.subheader("Configured Channels")
            for cname, m in list(st.session_state.models.items()):
                colA, colB = st.columns([4, 1])
                colA.write(f"**{cname}**: β={m.beta:,.0f}, α={m.alpha:.2f}, K={m.K:,.0f}, θ={m.theta:.2f}")
                if colB.button(f"Remove", key=f"rm_{cname}"):
                    del st.session_state.models[cname]
                    if cname in st.session_state.training_data:
                        del st.session_state.training_data[cname]
                    st.rerun()

        st.markdown("---")

        with st.expander("Add New Channel", expanded=len(st.session_state.models)==0):
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                new_channel_name = st.text_input("Channel Name", value=f"Channel {len(st.session_state.models)+1}")
                data_source = st.selectbox("Select Input Method", ["Upload CSV", "Use Sample Data", "Manual Parameters"])
                epochs = 1000
                if data_source in ["Upload CSV", "Use Sample Data"]:
                    epochs = st.number_input("Fitting Epochs", value=1000, step=100)

            with c_col2:
                adstock_type = "none"
                adstock_bounds = None
                adstock_fixed_days = None

                if data_source in ["Upload CSV", "Use Sample Data"]:
                    st.markdown("**Adstock (Lagged Effects)**")
                    adstock_opt = st.selectbox(
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
                        min_days = st.number_input("Minimum Decay Days", value=1.0, min_value=0.1, step=0.5)
                        max_days = st.number_input("Maximum Decay Days", value=14.0, min_value=1.0, step=1.0)
                        adstock_bounds = (min_days, max_days)
                    elif adstock_opt == "Explicitly set the adstock value (days)":
                        adstock_type = "fixed"
                        adstock_fixed_days = st.number_input("Explicit Decay Days", value=3.0, min_value=0.1, step=0.5)

            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Upload Historical Data (CSV)", type=["csv"])
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.write("Preview:")
                        st.dataframe(df.head(3))

                        spend_col = st.selectbox("Spend Column", df.columns, index=0 if "spend" in df.columns[0].lower() else 0)
                        return_col = st.selectbox("Return/KPI Column", df.columns, index=1 if len(df.columns) > 1 and "return" in df.columns[1].lower() else 0)

                        set_val = st.checkbox("set conversion value", value=False)
                        conversion_val = 1.0
                        if set_val:
                            conversion_val = st.number_input("Conversion Value ($)", value=100.0, min_value=0.01, step=1.0)

                        if st.button("🚀 Fit Model from CSV"):
                            try:
                                with st.spinner("Fitting curve using tinygrad..."):
                                    spends = df[spend_col].values
                                    returns = df[return_col].values * conversion_val
                                    model = fit_in_subprocess(
                                        spends=spends, returns=returns, epochs=epochs, lr=0.05,
                                        channel_name=new_channel_name, adstock_type=adstock_type,
                                        adstock_bounds=adstock_bounds, adstock_fixed_days=adstock_fixed_days
                                    )
                                    st.session_state.models[new_channel_name] = model
                                    st.session_state.training_data[new_channel_name] = (spends, returns)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error processing CSV: {e}")
                    except Exception as e:
                        st.error(f"Error processing CSV: {e}")

            elif data_source == "Use Sample Data":
                st.markdown("**Sample Dataset:** Spends: `[1k, 5k, 15k, 25k, 40k, 60k]`, Returns: `[200, 1.5k, 12k, 22k, 28k, 32k]`")
                if st.button("🚀 Fit Sample Model"):
                    try:
                        with st.spinner("Fitting sample curve..."):
                            spends = np.array([1000, 5000, 15000, 25000, 40000, 60000])
                            returns = np.array([200, 1500, 12000, 22000, 28000, 32000])
                            model = fit_in_subprocess(
                                spends=spends, returns=returns, epochs=epochs, lr=0.05,
                                channel_name=new_channel_name, adstock_type=adstock_type,
                                adstock_bounds=adstock_bounds, adstock_fixed_days=adstock_fixed_days
                            )
                            st.session_state.models[new_channel_name] = model
                            st.session_state.training_data[new_channel_name] = (spends, returns)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error fitting sample model: {e}")

            elif data_source == "Manual Parameters":
                m_col1, m_col2 = st.columns(2)
                beta = m_col1.number_input("Beta (Max Capacity)", value=50000.0, step=1000.0)
                alpha = m_col1.number_input("Alpha (Shape)", value=1.8, min_value=0.1, step=0.1)
                k = m_col2.number_input("K (Half-Saturation)", value=20000.0, step=1000.0)
                theta = m_col2.number_input("Theta (Adstock Decay)", value=0.0, min_value=0.0, max_value=0.99, step=0.05)

                if st.button("✅ Apply Parameters"):
                    model = MarketingReturnCurve(beta=beta, alpha=alpha, half_saturation_k=k, theta=theta, channel_name=new_channel_name)
                    st.session_state.models[new_channel_name] = model
                    st.rerun()

        if st.session_state.models:
            st.markdown("---")
            st.subheader("Channel Deep Dive")
            selected_channel = st.selectbox("Select Channel to Analyze", list(st.session_state.models.keys()))
            model = st.session_state.models[selected_channel]

            st.sidebar.markdown("---")
            st.sidebar.header("Optimization Settings")
            target_mroas = st.sidebar.slider(
                "Target Marginal ROAS (mROAS)",
                min_value=0.1, max_value=5.0, value=1.5, step=0.1,
                help="The minimum return you expect for every additional dollar spent."
            )

            show_data_points = False
            if selected_channel in st.session_state.training_data and st.session_state.training_data[selected_channel] is not None:
                show_data_points = st.sidebar.checkbox("Show Historical Data Points", value=True)

            min_spend = model.get_minimal_marginal_cost_point()
            max_spend = model.get_diminishing_returns_point(target_mroas=target_mroas)

            st.subheader("Media Response & Marginal Efficiency")
            fig = create_plotly_plot(model, target_mroas, scatter=st.session_state.training_data.get(selected_channel) if show_data_points else None)
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

                if selected_channel in st.session_state.training_data and st.session_state.training_data[selected_channel] is not None:
                    spends, _ = st.session_state.training_data[selected_channel]
                    adstock_fig = create_adstock_timeline_plot(spends, model)
                    st.plotly_chart(adstock_fig, use_container_width=True)
                else:
                    st.info("No timeline data available to show carryover plot (Manual Parameters loaded).")

            st.markdown("---")
            st.subheader("Model Summary")
            st.json(model.summary())

    with tab2:
        st.header("Portfolio Optimization")
        st.markdown("Determine the optimal budget allocation across multiple marketing channels to maximize total incremental returns.")

        if not st.session_state.models:
            st.warning("Please configure at least one channel in Stage 1.")
        else:
            opt_col1, opt_col2 = st.columns([1, 2])
            with opt_col1:
                total_budget = st.number_input("Total Portfolio Budget ($)", value=100000.0, step=10000.0)
                st.markdown("### Channel Constraints")
                st.markdown("Optionally set minimum and maximum spend limits for each channel.")

                channel_bounds = {}
                for cname in st.session_state.models.keys():
                    with st.expander(f"{cname} Bounds"):
                        use_bounds = st.checkbox(f"Constrain {cname}", value=False)
                        if use_bounds:
                            min_b = st.number_input(f"Min Spend ({cname})", value=0.0, step=1000.0, key=f"min_{cname}")
                            max_b = st.number_input(f"Max Spend ({cname})", value=total_budget, step=1000.0, key=f"max_{cname}")
                            channel_bounds[cname] = (min_b, max_b)

                run_opt = st.button("🚀 Run Portfolio Optimization", type="primary", use_container_width=True)

            with opt_col2:
                if run_opt:
                    with st.spinner("Optimizing portfolio allocation..."):
                        allocator = PortfolioAllocator(list(st.session_state.models.values()))
                        res = allocator.allocate_budget(total_budget=total_budget, channel_bounds=channel_bounds)

                        if res["success"]:
                            st.success(f"Optimization Successful! Expected Total Return: ${res['expected_total_return']:,.2f}")

                            labels = list(res["allocation"].keys())
                            values = list(res["allocation"].values())

                            fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color='#4285F4')])
                            fig.update_layout(
                                title="Optimal Budget Allocation",
                                xaxis_title="Channel",
                                yaxis_title="Allocated Spend ($)",
                                template="plotly_white",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.markdown("---")
                            st.subheader("Cross-Channel Saturation")
                            curves_fig = create_portfolio_curves_plot(st.session_state.models, res["allocation"])
                            st.plotly_chart(curves_fig, use_container_width=True)

                            st.markdown("---")
                            st.subheader("Optimal Channel Mix at Scale")
                            mix_fig = create_allocation_mix_plot(st.session_state.models, total_budget, channel_bounds)
                            st.plotly_chart(mix_fig, use_container_width=True)

                            st.subheader("Allocation Details")
                            details_data = []
                            for cname in labels:
                                spend = res["allocation"][cname]
                                mroas = res["marginal_roas_at_allocation"][cname]
                                details_data.append({"Channel": cname, "Allocated Spend ($)": spend, "Marginal ROAS": mroas})
                            st.dataframe(pd.DataFrame(details_data).style.format({"Allocated Spend ($)": "{:,.2f}", "Marginal ROAS": "{:.3f}"}))

                        else:
                            st.error(f"Optimization Failed: {res['message']}")
                else:
                    st.info("Configure budget and click Run Portfolio Optimization.")

if __name__ == "__main__":
    run_dashboard()