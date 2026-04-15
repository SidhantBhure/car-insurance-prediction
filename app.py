"""
Car Insurance Claim Prediction System - Streamlit Dashboard v2
Premium dark-themed dashboard with live prediction graphs, history tracking, and INR support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import io
import json
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_curve, auc
import base64

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Insurance Claim Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Force Dark Theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ── Global dark overrides ── */
    html, body, [class*="css"], .stApp, .main, .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp, [data-testid="stAppViewContainer"], .main,
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: #0b0f19 !important;
    }
    
    /* Sidebar dark */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* All text white */
    .stApp p, .stApp span, .stApp label, .stApp div,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #e2e8f0 !important;
    }
    
    /* Input fields dark */
    .stNumberInput input, .stTextInput input,
    [data-baseweb="input"] input,
    [data-baseweb="select"] div,
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    
    /* Select dropdowns */
    [data-baseweb="popover"] {
        background-color: #1e293b !important;
    }
    [data-baseweb="menu"] {
        background-color: #1e293b !important;
    }
    [data-baseweb="menu"] li {
        color: #e2e8f0 !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: #334155 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        background-color: #111827 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }
    
    /* Dataframe dark */
    [data-testid="stDataFrame"],
    .stDataFrame, .stDataFrame table,
    [data-testid="stTable"] {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stDataFrame"] th,
    [data-testid="stDataFrame"] td {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
        border-color: #1e293b !important;
    }
    
    /* Buttons */
    .stButton > button, .stDownloadButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.35) !important;
    }
    
    /* Form container */
    [data-testid="stForm"] {
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        padding: 24px !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #6366f1 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a1f3a 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #94a3b8 !important;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 8px;
    }
    
    /* Header */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-subtitle {
        color: #64748b !important;
        font-size: 1.05rem;
        margin-top: 4px;
    }
    
    /* Prediction result cards */
    .prediction-result {
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 16px;
    }
    .prediction-fraud {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .prediction-safe {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111827 !important;
        color: #e2e8f0 !important;
    }
    .streamlit-expanderContent {
        background-color: #0d1117 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
    
    /* History card */
    .history-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* ── Hide ONLY Deploy button, keep Theme & Print ── */
    [data-testid="stToolbar"] button[kind="header"],
    [data-testid="stStatusWidget"],
    .stDeployButton,
    button[title="Deploy"],
    #stDecoration {
        display: none !important;
    }
    footer { visibility: hidden !important; }
    
    /* Hide Streamlit version inside the portaled main menu */
    ul[data-testid="main-menu-list"] + div {
        display: none !important;
    }
    [data-baseweb="popover"] ul[role="menu"] + div {
        display: none !important;
    }
    [data-baseweb="popover"] [data-testid="stMarkdownContainer"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Force Hide "Made with Streamlit" via cross-origin-safe JS ──
import streamlit.components.v1 as components
components.html(
    """
    <script>
    const hideMadeWithStreamlit = () => {
        let els = window.parent.document.querySelectorAll('*');
        els.forEach(el => {
            // Find nodes containing the text and explicitly hide them
            if (el.innerText && el.innerText.includes("Made with Streamlit") && el.children.length === 0) {
                el.style.display = 'none';
                /* Hide its parent wrapper for cleaner spacing */
                if (el.parentElement) {
                    el.parentElement.style.display = 'none';
                }
            }
        });
    };
    hideMadeWithStreamlit();
    // Re-run the hider whenever the DOM changes (e.g., popover opens)
    const observer = new MutationObserver(() => {
        hideMadeWithStreamlit();
    });
    observer.observe(window.parent.document.body, { childList: true, subtree: true });
    </script>
    """,
    height=0,
    width=0,
)


# ── Dark plotly template ─────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template='plotly_dark',
    paper_bgcolor='#0b0f19',
    plot_bgcolor='#111827',
    font=dict(family='Inter', color='#e2e8f0'),
)

def dark_fig(fig, height=420):
    """Apply consistent dark styling to a plotly figure."""
    fig.update_layout(
        **PLOTLY_DARK,
        height=height,
        title_font_size=16,
        margin=dict(t=50, b=40, l=50, r=30),
    )
    fig.update_xaxes(gridcolor='#1e293b', zerolinecolor='#1e293b')
    fig.update_yaxes(gridcolor='#1e293b', zerolinecolor='#1e293b')
    return fig


# ── Load Models & Data ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    results = joblib.load('model_results.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return rf_model, xgb_model, results, label_encoders, feature_names

try:
    rf_model, xgb_model, results, label_encoders, feature_names = load_models()
except FileNotFoundError:
    st.error("⚠️ Models not found! Run `python train_models.py` first.")
    st.stop()


# ── Initialize session state for prediction history ──────────────────────────
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 Navigation")
    st.markdown("---")
    selected_model = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost"],
        help="Choose which model to display metrics for"
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown(f"**Samples:** 1,000")
    st.markdown(f"**Features:** {len(feature_names)}")
    st.markdown(f"**Test Size:** {len(results['y_test'])}")
    st.markdown(f"**Currency:** ₹ (INR)")
    st.markdown("---")
    st.markdown("### 🧠 Models")
    st.markdown("- 🌲 Random Forest (200 trees)")
    st.markdown("- ⚡ XGBoost (200 rounds)")
    st.markdown("---")
    st.markdown(f"### 📜 Prediction History")
    st.markdown(f"**Total predictions:** {len(st.session_state.prediction_history)}")
    if st.session_state.prediction_history:
        fraud_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 'FRAUD')
        st.markdown(f"**Fraud detected:** {fraud_count}")
        st.markdown(f"**Legitimate:** {len(st.session_state.prediction_history) - fraud_count}")
    st.markdown("---")


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="header-title">🚗 Car Insurance Claim Prediction</p>'
    '<p class="header-subtitle">AI-powered fraud detection system using ensemble machine learning</p>',
    unsafe_allow_html=True
)
st.markdown("")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Model Performance", "🔍 Detailed Analysis",
    "⚖️ Model Comparison", "🔮 Predict Fraud", "📜 History"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Model Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    metrics = results['rf_metrics'] if selected_model == "Random Forest" else results['xgb_metrics']

    st.markdown(f"### {selected_model} Performance")
    st.markdown("")

    cols = st.columns(5)
    metric_icons = ['🎯', '✅', '🔄', '📊', '📈']
    for i, (name, val) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{metric_icons[i]}</div>
                <div class="metric-value">{val:.4f}</div>
                <div class="metric-label">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pred = results['rf_pred'] if selected_model == "Random Forest" else results['xgb_pred']
        cm = confusion_matrix(results['y_test'], pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['No Fraud', 'Fraud'], y=['No Fraud', 'Fraud'],
            color_continuous_scale='Purples',
            title=f"Confusion Matrix — {selected_model}"
        )
        dark_fig(fig_cm)
        fig_cm.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': f'confusion_matrix_{selected_model}'}})

    with col2:
        proba = results['rf_proba'] if selected_model == "Random Forest" else results['xgb_proba']
        fpr, tpr, _ = roc_curve(results['y_test'], proba)
        roc_auc_val = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'{selected_model} (AUC = {roc_auc_val:.4f})',
            line=dict(color='#8b5cf6', width=3),
            fill='tozeroy', fillcolor='rgba(139, 92, 246, 0.1)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline',
            line=dict(color='#475569', width=1, dash='dash')
        ))
        fig_roc.update_layout(
            title=f"ROC Curve — {selected_model}",
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.5, y=0.05)
        )
        dark_fig(fig_roc)
        st.plotly_chart(fig_roc, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': f'roc_curve_{selected_model}'}})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Detailed Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown(f"### Feature Importance — {selected_model}")
    st.markdown("")

    fi_key = 'feature_importances_rf' if selected_model == "Random Forest" else 'feature_importances_xgb'
    fi = results[fi_key]
    fi_df = pd.DataFrame({'Feature': list(fi.keys()), 'Importance': list(fi.values())})
    fi_df = fi_df.sort_values('Importance', ascending=True).tail(15)

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi_df['Importance'], y=fi_df['Feature'],
        orientation='h',
        marker=dict(
            color=fi_df['Importance'],
            colorscale=[[0, '#312e81'], [0.5, '#6366f1'], [1, '#c084fc']],
            line=dict(color='#818cf8', width=1)
        ),
        text=[f'{v:.4f}' for v in fi_df['Importance']],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=11),
    ))
    fig_fi.update_layout(
        title=f"Top 15 Features — {selected_model}",
        xaxis_title='Importance Score',
        yaxis_title='',
        showlegend=False,
    )
    dark_fig(fig_fi, height=550)
    st.plotly_chart(fig_fi, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': f'feature_importance_{selected_model}'}})

    # If there's a last prediction, highlight which features contributed
    if st.session_state.last_prediction is not None:
        lp = st.session_state.last_prediction
        st.markdown("---")
        st.markdown(f"### 🔮 Last Prediction Input — Feature Contribution")
        st.markdown(f"*Model: **{lp['model']}** | Result: **{lp['prediction']}** | Probability: **{lp['fraud_prob']:.1f}%***")

        # Show top features and their input values
        top_features = fi_df['Feature'].tolist()[::-1]  # descending
        input_vals = lp['input_data']
        contrib_data = []
        for feat in top_features:
            val = input_vals.get(feat, 'N/A')
            importance = fi.get(feat, 0)
            contrib_data.append({'Feature': feat, 'Your Input Value': val, 'Importance': f'{importance:.4f}'})
        contrib_df = pd.DataFrame(contrib_data)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Prediction distribution
    st.markdown("### Prediction Probability Distribution")
    proba_dist = results['rf_proba'] if selected_model == "Random Forest" else results['xgb_proba']

    fig_dist = go.Figure()
    mask_0 = results['y_test'] == 0
    mask_1 = results['y_test'] == 1

    fig_dist.add_trace(go.Histogram(
        x=proba_dist[mask_0], name='No Fraud',
        marker_color='#6366f1', opacity=0.7, nbinsx=30
    ))
    fig_dist.add_trace(go.Histogram(
        x=proba_dist[mask_1], name='Fraud',
        marker_color='#ef4444', opacity=0.7, nbinsx=30
    ))

    # Mark last prediction on distribution
    if st.session_state.last_prediction is not None:
        lp_prob = st.session_state.last_prediction['fraud_prob'] / 100
        fig_dist.add_vline(
            x=lp_prob, line_width=3, line_dash="dash", line_color="#22c55e",
            annotation_text=f"Your Prediction: {lp_prob*100:.1f}%",
            annotation_position="top",
            annotation_font_color="#22c55e"
        )

    fig_dist.update_layout(
        barmode='overlay',
        xaxis_title='Predicted Fraud Probability',
        yaxis_title='Count',
    )
    dark_fig(fig_dist, 400)
    st.plotly_chart(fig_dist, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'probability_distribution'}})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Model Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### Head-to-Head Model Comparison")
    st.markdown("")

    rf_m = results['rf_metrics']
    xgb_m = results['xgb_metrics']

    comp_df = pd.DataFrame({
        'Metric': list(rf_m.keys()),
        'Random Forest': [f"{v:.4f}" for v in rf_m.values()],
        'XGBoost': [f"{v:.4f}" for v in xgb_m.values()],
        'Winner': ['🌲 RF' if rf_m[k] >= xgb_m[k] else '⚡ XGB' for k in rf_m]
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.markdown("")

    # Grouped bar chart
    metrics_list = list(rf_m.keys())
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=metrics_list, y=list(rf_m.values()),
        name='🌲 Random Forest',
        marker_color='#6366f1',
        marker_line_color='#818cf8', marker_line_width=1,
        text=[f'{v:.3f}' for v in rf_m.values()],
        textposition='outside', textfont=dict(color='#e2e8f0')
    ))
    fig_comp.add_trace(go.Bar(
        x=metrics_list, y=list(xgb_m.values()),
        name='⚡ XGBoost',
        marker_color='#a78bfa',
        marker_line_color='#c4b5fd', marker_line_width=1,
        text=[f'{v:.3f}' for v in xgb_m.values()],
        textposition='outside', textfont=dict(color='#e2e8f0')
    ))
    fig_comp.update_layout(
        barmode='group',
        title="Metric Comparison",
        yaxis_title='Score',
        yaxis=dict(range=[0, 1.1])
    )
    dark_fig(fig_comp, 450)
    st.plotly_chart(fig_comp, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'model_comparison'}})

    # ROC Comparison
    st.markdown("### ROC Curve Comparison")
    fpr_rf, tpr_rf, _ = roc_curve(results['y_test'], results['rf_proba'])
    fpr_xgb, tpr_xgb, _ = roc_curve(results['y_test'], results['xgb_proba'])

    fig_roc_comp = go.Figure()
    fig_roc_comp.add_trace(go.Scatter(
        x=fpr_rf, y=tpr_rf, mode='lines',
        name=f"Random Forest (AUC = {rf_m['ROC-AUC']:.4f})",
        line=dict(color='#6366f1', width=3)
    ))
    fig_roc_comp.add_trace(go.Scatter(
        x=fpr_xgb, y=tpr_xgb, mode='lines',
        name=f"XGBoost (AUC = {xgb_m['ROC-AUC']:.4f})",
        line=dict(color='#a78bfa', width=3)
    ))
    fig_roc_comp.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random Baseline',
        line=dict(color='#475569', width=1, dash='dash')
    ))
    fig_roc_comp.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
    )
    dark_fig(fig_roc_comp, 450)
    st.plotly_chart(fig_roc_comp, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'roc_comparison'}})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: Predict Fraud (with LIVE graphs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### 🔮 Predict Insurance Fraud")
    st.markdown("Enter claim details below. **All monetary values are in ₹ (INR)**. Charts update live after prediction.")
    st.markdown("")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Policy Details**")
            months_as_customer = st.number_input("Months as Customer", 0, 500, 150)
            age = st.number_input("Age", 18, 80, 35)
            policy_deductable = st.selectbox("Policy Deductible (₹)", [500, 1000, 2000])
            policy_annual_premium = st.number_input("Annual Premium (₹)", 500.0, 2500.0, 1200.0)
            umbrella_limit = st.selectbox("Umbrella Limit (₹)", [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000])
            capital_gains = st.number_input("Capital Gains (₹)", 0, 10000000, 0)
            capital_loss = st.number_input("Capital Loss (₹)", -10000000, 0, 0)

        with col2:
            st.markdown("**🚨 Incident Details**")
            incident_type = st.selectbox("Incident Type",
                ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Vehicle Theft', 'Parked Car'])
            incident_severity = st.selectbox("Incident Severity",
                ['Minor Damage', 'Major Damage', 'Total Loss', 'Trivial Damage'])
            incident_hour = st.slider("Incident Hour", 0, 23, 12)
            num_vehicles = st.selectbox("Vehicles Involved", [1, 2, 3, 4])
            bodily_injuries = st.selectbox("Bodily Injuries", [0, 1, 2])
            witnesses = st.selectbox("Witnesses", [0, 1, 2, 3])

        with col3:
            st.markdown("**💰 Claim Details (₹)**")
            total_claim_amount = st.number_input("Total Claim (₹)", 0, 12000000, 5000000, step=100000)
            injury_claim = st.number_input("Injury Claim (₹)", 0, 2500000, 500000, step=50000)
            property_claim = st.number_input("Property Claim (₹)", 0, 2500000, 500000, step=50000)
            vehicle_claim = st.number_input("Vehicle Claim (₹)", 0, 8000000, 4000000, step=100000)
            auto_year = st.number_input("Auto Year", 1995, 2026, 2010)
            property_damage = st.selectbox("Property Damage", ['YES', 'NO'])
            police_report = st.selectbox("Police Report Available", ['YES', 'NO'])

        predict_model = st.selectbox("Predict using", ["Random Forest", "XGBoost"])
        submitted = st.form_submit_button("🔍 Predict Fraud", use_container_width=True)

    if submitted:
        # Build feature vector
        input_data = {f: 0 for f in feature_names}

        direct_map = {
            'months_as_customer': months_as_customer,
            'age': age,
            'policy_deductable': policy_deductable,
            'policy_annual_premium': policy_annual_premium,
            'umbrella_limit': umbrella_limit,
            'capital-gains': capital_gains,
            'capital-loss': capital_loss,
            'incident_hour_of_the_day': incident_hour,
            'number_of_vehicles_involved': num_vehicles,
            'bodily_injuries': bodily_injuries,
            'witnesses': witnesses,
            'total_claim_amount': total_claim_amount,
            'injury_claim': injury_claim,
            'property_claim': property_claim,
            'vehicle_claim': vehicle_claim,
            'auto_year': auto_year,
        }

        for k, v in direct_map.items():
            if k in input_data:
                input_data[k] = v

        cat_map = {
            'incident_type': incident_type,
            'incident_severity': incident_severity,
            'property_damage': property_damage,
            'police_report_available': police_report,
        }
        for k, v in cat_map.items():
            if k in input_data and k in label_encoders:
                try:
                    input_data[k] = label_encoders[k].transform([v])[0]
                except ValueError:
                    input_data[k] = 0

        input_df = pd.DataFrame([input_data])

        model = rf_model if predict_model == "Random Forest" else xgb_model
        pred = model.predict(input_df)[0]
        proba_vals = model.predict_proba(input_df)[0]
        fraud_prob = proba_vals[1] * 100

        # Store prediction in session state
        prediction_record = {
            'id': len(st.session_state.prediction_history) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': predict_model,
            'prediction': 'FRAUD' if pred == 1 else 'LEGITIMATE',
            'fraud_prob': fraud_prob,
            'input_data': input_data,
            'input_display': {
                'Age': age,
                'Months as Customer': months_as_customer,
                'Incident Type': incident_type,
                'Incident Severity': incident_severity,
                'Total Claim (₹)': f"₹{total_claim_amount:,}",
                'Injury Claim (₹)': f"₹{injury_claim:,}",
                'Property Claim (₹)': f"₹{property_claim:,}",
                'Vehicle Claim (₹)': f"₹{vehicle_claim:,}",
                'Annual Premium (₹)': f"₹{policy_annual_premium:,.0f}",
                'Auto Year': auto_year,
                'Police Report': police_report,
                'Property Damage': property_damage,
            }
        }
        st.session_state.prediction_history.append(prediction_record)
        st.session_state.last_prediction = prediction_record

        # ── RESULT CARD ──
        st.markdown("")
        if pred == 1:
            st.markdown(f"""
            <div class="prediction-result prediction-fraud">
                <div style="font-size:3rem;">🚨</div>
                <div style="font-size:1.8rem;font-weight:700;color:#ef4444;">FRAUD DETECTED</div>
                <div style="color:#94a3b8;margin-top:8px;">
                    Fraud Probability: <strong style="color:#ef4444;">{fraud_prob:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result prediction-safe">
                <div style="font-size:3rem;">✅</div>
                <div style="font-size:1.8rem;font-weight:700;color:#22c55e;">LEGITIMATE CLAIM</div>
                <div style="color:#94a3b8;margin-top:8px;">
                    Fraud Probability: <strong style="color:#22c55e;">{fraud_prob:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── LIVE CHARTS (only after prediction) ──
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Fraud Risk Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_prob,
                title={'text': "Fraud Risk Score", 'font': {'size': 18, 'color': '#e2e8f0'}},
                number={'suffix': '%', 'font': {'color': '#e2e8f0', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#475569', 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': '#8b5cf6'},
                    'bgcolor': '#1e293b',
                    'bordercolor': '#334155',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(234, 179, 8, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ],
                    'threshold': {'line': {'color': '#ef4444', 'width': 3}, 'value': 50}
                }
            ))
            dark_fig(fig_gauge, 320)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'fraud_risk_gauge'}})

        with chart_col2:
            # Probability pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraud'],
                values=[proba_vals[0] * 100, proba_vals[1] * 100],
                hole=0.55,
                marker_colors=['#22c55e', '#ef4444'],
                textinfo='label+percent',
                textfont=dict(color='#e2e8f0', size=14),
                pull=[0, 0.05]
            )])
            fig_pie.update_layout(
                title="Prediction Breakdown",
                annotations=[dict(text=f"{fraud_prob:.0f}%", x=0.5, y=0.5,
                                  font_size=28, font_color='#e2e8f0', showarrow=False)],
                showlegend=True,
                legend=dict(font=dict(color='#e2e8f0'))
            )
            dark_fig(fig_pie, 320)
            st.plotly_chart(fig_pie, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'prediction_breakdown'}})

        # ── Live Feature Importance with your input highlighted ──
        st.markdown("### 📊 Feature Importance vs Your Input")
        fi_key_pred = 'feature_importances_rf' if predict_model == "Random Forest" else 'feature_importances_xgb'
        fi_pred = results[fi_key_pred]
        fi_pred_df = pd.DataFrame({'Feature': list(fi_pred.keys()), 'Importance': list(fi_pred.values())})
        fi_pred_df = fi_pred_df.sort_values('Importance', ascending=True).tail(15)

        # Normalize input values for comparison
        input_vals_norm = []
        for feat in fi_pred_df['Feature']:
            raw_val = input_data.get(feat, 0)
            input_vals_norm.append(float(raw_val) if isinstance(raw_val, (int, float, np.integer, np.floating)) else 0)

        fig_fi_live = make_subplots(
            rows=1, cols=2, shared_yaxes=True,
            subplot_titles=[f'Feature Importance ({predict_model})', 'Your Input Values'],
            horizontal_spacing=0.05
        )

        fig_fi_live.add_trace(go.Bar(
            x=fi_pred_df['Importance'], y=fi_pred_df['Feature'],
            orientation='h', name='Importance',
            marker=dict(color='#6366f1', line=dict(color='#818cf8', width=1)),
            text=[f'{v:.4f}' for v in fi_pred_df['Importance']],
            textposition='outside', textfont=dict(color='#e2e8f0', size=10),
        ), row=1, col=1)

        fig_fi_live.add_trace(go.Bar(
            x=input_vals_norm, y=fi_pred_df['Feature'].tolist(),
            orientation='h', name='Your Input',
            marker=dict(color='#a78bfa', line=dict(color='#c4b5fd', width=1)),
            text=[f'{v:,.0f}' for v in input_vals_norm],
            textposition='outside', textfont=dict(color='#e2e8f0', size=10),
        ), row=1, col=2)

        dark_fig(fig_fi_live, 550)
        fig_fi_live.update_annotations(font=dict(color='#e2e8f0', size=14))
        st.plotly_chart(fig_fi_live, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'feature_importance_vs_input'}})

        # ── Where your prediction falls in the distribution ──
        st.markdown("### 📍 Where Your Prediction Falls")
        proba_all = results['rf_proba'] if predict_model == "Random Forest" else results['xgb_proba']
        m0 = results['y_test'] == 0
        m1 = results['y_test'] == 1

        fig_pos = go.Figure()
        fig_pos.add_trace(go.Histogram(x=proba_all[m0], name='No Fraud (Test Set)', marker_color='#6366f1', opacity=0.6, nbinsx=30))
        fig_pos.add_trace(go.Histogram(x=proba_all[m1], name='Fraud (Test Set)', marker_color='#ef4444', opacity=0.6, nbinsx=30))
        fig_pos.add_vline(x=fraud_prob / 100, line_width=4, line_dash="solid",
                          line_color="#22c55e",
                          annotation_text=f"⬆ YOUR CLAIM: {fraud_prob:.1f}%",
                          annotation_position="top",
                          annotation_font=dict(color="#22c55e", size=14))
        fig_pos.update_layout(barmode='overlay', xaxis_title='Fraud Probability', yaxis_title='Count')
        dark_fig(fig_pos, 400)
        st.plotly_chart(fig_pos, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'prediction_position'}})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5: History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("### 📜 Prediction History")
    st.markdown("")

    if not st.session_state.prediction_history:
        st.info("No predictions yet. Go to the **Predict Fraud** tab to make your first prediction!")
    else:
        # Summary metrics
        history = st.session_state.prediction_history
        total = len(history)
        fraud_count = sum(1 for p in history if p['prediction'] == 'FRAUD')
        legit_count = total - fraud_count
        avg_prob = np.mean([p['fraud_prob'] for p in history])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Predictions</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="-webkit-text-fill-color:#ef4444;">{fraud_count}</div>
                <div class="metric-label">Fraud Detected</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="-webkit-text-fill-color:#22c55e;">{legit_count}</div>
                <div class="metric-label">Legitimate</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_prob:.1f}%</div>
                <div class="metric-label">Avg Fraud Prob</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # History trend chart
        if len(history) > 1:
            st.markdown("### 📈 Prediction Trend")
            trend_df = pd.DataFrame([
                {'Prediction #': p['id'], 'Fraud Probability (%)': p['fraud_prob'],
                 'Result': p['prediction'], 'Model': p['model']}
                for p in history
            ])
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=trend_df['Prediction #'], y=trend_df['Fraud Probability (%)'],
                mode='lines+markers',
                marker=dict(
                    size=12,
                    color=['#ef4444' if r == 'FRAUD' else '#22c55e' for r in trend_df['Result']],
                    line=dict(color='#e2e8f0', width=1)
                ),
                line=dict(color='#6366f1', width=2),
                text=[f"{r} ({m})" for r, m in zip(trend_df['Result'], trend_df['Model'])],
                hovertemplate='Prediction #%{x}<br>Fraud Prob: %{y:.1f}%<br>%{text}<extra></extra>'
            ))
            fig_trend.add_hline(y=50, line_dash="dash", line_color="#eab308",
                                annotation_text="Fraud Threshold (50%)",
                                annotation_font_color="#eab308")
            fig_trend.update_layout(
                xaxis_title='Prediction #', yaxis_title='Fraud Probability (%)',
                yaxis=dict(range=[0, 105])
            )
            dark_fig(fig_trend, 350)
            st.plotly_chart(fig_trend, use_container_width=True, config={'toImageButtonOptions': {'format': 'svg', 'filename': 'prediction_trend'}})

        # History table
        st.markdown("### 📋 All Predictions")
        table_data = []
        for p in reversed(history):
            row = {
                '#': p['id'],
                'Time': p['timestamp'],
                'Model': p['model'],
                'Result': f"🚨 {p['prediction']}" if p['prediction'] == 'FRAUD' else f"✅ {p['prediction']}",
                'Fraud %': f"{p['fraud_prob']:.1f}%",
            }
            row.update(p['input_display'])
            table_data.append(row)
        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        # ── Download Section ──
        st.markdown("---")
        st.markdown("### 💾 Download History")

        # Build CSV with clean numeric values (no ₹ symbols)
        csv_rows = []
        for p in history:
            csv_rows.append({
                'ID': p['id'],
                'Timestamp': p['timestamp'],
                'Model': p['model'],
                'Prediction': p['prediction'],
                'Fraud_Probability_Percent': round(p['fraud_prob'], 2),
                'Age': p['input_display'].get('Age', ''),
                'Months_as_Customer': p['input_display'].get('Months as Customer', ''),
                'Incident_Type': p['input_display'].get('Incident Type', ''),
                'Incident_Severity': p['input_display'].get('Incident Severity', ''),
                'Total_Claim_INR': str(p['input_display'].get('Total Claim (₹)', '')).replace('₹', '').replace(',', ''),
                'Injury_Claim_INR': str(p['input_display'].get('Injury Claim (₹)', '')).replace('₹', '').replace(',', ''),
                'Property_Claim_INR': str(p['input_display'].get('Property Claim (₹)', '')).replace('₹', '').replace(',', ''),
                'Vehicle_Claim_INR': str(p['input_display'].get('Vehicle Claim (₹)', '')).replace('₹', '').replace(',', ''),
                'Annual_Premium_INR': str(p['input_display'].get('Annual Premium (₹)', '')).replace('₹', '').replace(',', ''),
                'Auto_Year': p['input_display'].get('Auto Year', ''),
                'Police_Report': p['input_display'].get('Police Report', ''),
                'Property_Damage': p['input_display'].get('Property Damage', ''),
            })

        csv_df = pd.DataFrame(csv_rows)
        csv_string = csv_df.to_csv(index=False)
        
        # Generate proper downloadable CSV via HTML link (fixes UUID filename issue)
        b64_csv = base64.b64encode(csv_string.encode()).decode()
        csv_filename = f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.markdown(f"""
        <a href="data:text/csv;base64,{b64_csv}" download="{csv_filename}" 
           style="display:inline-block;width:100%;text-align:center;padding:14px 24px;
                  background:linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                  color:white;border-radius:10px;text-decoration:none;
                  font-weight:600;font-size:1rem;font-family:Inter,sans-serif;
                  transition:transform 0.2s ease,box-shadow 0.2s ease;"
           onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 24px rgba(99,102,241,0.35)';"
           onmouseout="this.style.transform='none';this.style.boxShadow='none';">
            📥 Download Prediction History as CSV
        </a>
        """, unsafe_allow_html=True)

        # Clear history button
        st.markdown("")
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.prediction_history = []
            st.session_state.last_prediction = None
            st.rerun()
