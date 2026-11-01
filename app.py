"""
HC-SmartPulse: AI-Powered Employee Flight Risk & Talent Analytics
Enhanced Streamlit Dashboard with Modern UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
from datetime import datetime
import xgboost as xgb

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from recommendation_engine import RecommendationEngine

# Page configuration
st.set_page_config(
    page_title="HC-SmartPulse | Talent Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/DianPandus/hc-smartpulse-talent-analytics',
        'Report a bug': 'https://github.com/DianPandus/hc-smartpulse-talent-analytics/issues',
        'About': "### HC-SmartPulse\nEmployee attrition risk prediction with XGBoost and SHAP"
    }
)

# Inject Custom CSS
st.markdown("""
<style>
    /* Midnight Luxury Color Palette */
    :root {
        --deep-black: #0B0B0C;
        --dark-purple: #2E1A47;
        --royal-violet: #4B3061;
        --soft-lavender: #D1C4E9;
        --accent-gold: #FFD700;
        --gradient-primary: linear-gradient(135deg, #2E1A47, #4B3061);
        --gradient-dark: linear-gradient(135deg, #0B0B0C, #2E1A47);
        --gradient-luxury: linear-gradient(135deg, #4B3061, #2E1A47);
    }
    
    /* Main Container Styling */
    .stApp {
        background: linear-gradient(135deg, #0B0B0C 0%, #2E1A47 50%, #4B3061 100%);
        background-attachment: fixed;
        color: white;
    }
    
    /* Card Styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(209, 196, 233, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric Card */
    .metric-card {
        background: var(--gradient-luxury);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(75, 48, 97, 0.5);
        border: 1px solid rgba(209, 196, 233, 0.1);
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #D1C4E9, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        text-align: center;
        min-width: 120px;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #8B0000, #DC143C);
        color: white;
        box-shadow: 0 4px 15px rgba(220, 20, 60, 0.6);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #FF8C00, #FFA500);
        color: white;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.5);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4B3061, #D1C4E9);
        color: white;
        box-shadow: 0 4px 15px rgba(75, 48, 97, 0.5);
    }
    
    /* Button Styling */
    .stButton > button {
        background: var(--gradient-luxury);
        color: white;
        border: 1px solid rgba(209, 196, 233, 0.2);
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(75, 48, 97, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(75, 48, 97, 0.6);
        background: var(--gradient-primary);
    }
    
    /* Form Styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #4B3061;
        box-shadow: 0 0 0 3px rgba(75, 48, 97, 0.3);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2E1A47 0%, #4B3061 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(209, 196, 233, 0.1);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white !important;
        border: 1px solid rgba(209, 196, 233, 0.2);
    }
    
    /* Custom Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #4361ee, #3a0ca3);
        border: none;
        margin: 2rem 0;
        border-radius: 3px;
    }
    
    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Employee Card */
    .employee-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4B3061;
        box-shadow: 0 5px 15px rgba(75, 48, 97, 0.3);
        border: 1px solid rgba(209, 196, 233, 0.1);
    }
    
    /* Text Color Override */
    .stMarkdown, .stMarkdown p, h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Input Labels */
    label {
        color: white !important;
    }
    
    /* Main content area background */
    .main .block-container {
        background-color: transparent;
    }
    
    /* Remove white backgrounds from containers */
    .element-container {
        background-color: transparent;
    }
    
    /* Remove Streamlit header white background */
    header {
        background-color: transparent !important;
    }
    
    /* Remove toolbar background */
    .stApp > header {
        background-color: transparent !important;
    }
    
    /* Remove all default white backgrounds */
    .stApp section {
        background-color: transparent !important;
    }
    
    /* Caption text color */
    .stCaption, .css-1629p8f, [data-testid="stCaptionContainer"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    # /* System info text */
    # .stText, p, div {
    #     color: white;
    # }
    
    /* Sidebar toggle button - make icon white */
    [data-testid="collapsedControl"] {
        color: white !important;
    }
    
    [data-testid="collapsedControl"] svg {
        fill: white !important;
        stroke: white !important;
    }
    
    button[kind="header"] {
        color: white !important;
    }
    
    /* Dropdown/selectbox text should be black for visibility */
    .stSelectbox div[data-baseweb="select"] div {
        color: black !important;
    }
    
    /* Dropdown options background white with black text - MORE SPECIFIC */
    [role="option"] {
        color: black !important;
        background-color: white !important;
    }
    
    [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="menu"] {
        background-color: white !important;
    }
    
    [data-baseweb="menu"] li {
        color: black !important;
        background-color: white !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background-color: #f0f0f0 !important;
    }
    
    /* Popover */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    /* ALL list items in dropdowns */
    ul[role="listbox"] li {
        color: black !important;
        background-color: white !important;
    }
    
    ul[role="listbox"] {
        background-color: white !important;
    }
    
    /* Multiselect selected items (tags) */
    [data-baseweb="tag"] {
        background-color: #4B3061 !important;
        color: white !important;
    }
    
    [data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Input fields text should be black */
    input {
        color: black !important;
        background-color: white !important;
    }
    
    /* Number input fields */
    .stNumberInput input {
        color: black !important;
        background-color: white !important;
    }
    
    /* Text input fields */
    .stTextInput input {
        color: black !important;
        background-color: white !important;
    }
    
    /* Multiselect dropdown */
    .stMultiSelect div[data-baseweb="select"] {
        background-color: white !important;
    }
    
    .stMultiSelect div[data-baseweb="select"] span {
        color: black !important;
    }
    
    .stMultiSelect [role="button"] {
        background-color: white !important;
        color: black !important;
    }
    
    /* Date picker */
    .stDateInput input {
        color: black !important;
        background-color: white !important;
    }
    
    /* Select slider */
    .stSelectSlider [data-baseweb="select"] div {
        background-color: white !important;
        color: black !important;
    }
    
    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    
    /* Checkbox in sidebar */
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }
    
    /* Link buttons in sidebar (Support section) */
    [data-testid="stSidebar"] .stButton button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Dropdown selected value display - make background white */
    [data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
    }
    
    /* All inner divs in select */
    [data-baseweb="select"] div {
        color: black !important;
    }
    
    /* Ensure dropdown arrow is visible */
    [data-baseweb="select"] svg {
        fill: black !important;
    }
    
    /* Select value container */
    [data-baseweb="select"] [data-baseweb="input"] {
        color: black !important;
    }
    
    /* ULTRA AGGRESSIVE - Force all dropdown content to be black */
    [data-baseweb="popover"] * {
        color: black !important;
    }
    
    [data-baseweb="menu"] * {
        color: black !important;
    }
    
    ul[role="listbox"] * {
        color: black !important;
    }
    
    /* Target the actual text content in options */
    [role="option"] > div {
        color: black !important;
    }
    
    [role="option"] span {
        color: black !important;
    }
    
    /* Multiselect specific */
    .stMultiSelect li {
        color: black !important;
    }
    
    .stMultiSelect [role="option"] {
        color: black !important;
    }
    
    /* Date picker calendar */
    [data-baseweb="calendar"] {
        background-color: white !important;
    }
    
    [data-baseweb="calendar"] * {
        color: black !important;
    }
    
    /* Month/Year dropdowns in date picker */
    [data-baseweb="calendar"] select {
        color: black !important;
        background-color: white !important;
    }


</style>
""", unsafe_allow_html=True)

RISK_COLORS = {'high': '#ff416c', 'medium': '#ffb347', 'low': '#4facfe'}


@st.cache_resource
def load_models():
    """Load trained model and preprocessing artifacts"""
    model = joblib.load('models/xgboost_model.pkl')
    encoders = joblib.load('models/feature_encoder.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    metrics = joblib.load('models/model_metrics.pkl')
    feature_importance = pd.read_csv('models/feature_importance.csv')
    return model, encoders, scaler, feature_columns, metrics, feature_importance


@st.cache_data
def load_dataset():
    return pd.read_csv('data/dataset.csv', encoding='utf-8-sig')


@st.cache_data
def score_dataset(df, _model, _encoders, _scaler, feature_columns):
    """Add the model's predicted attrition probability to every employee record"""
    X = encode_features(df.drop(columns=['Attrition']), _encoders, _scaler, feature_columns)
    scored = df.copy()
    scored['RiskScore'] = _model.predict_proba(X)[:, 1]
    return scored


def create_animated_gauge(probability, rec_engine, employee_name="Employee"):
    """Create a gauge whose bands follow the active risk thresholds"""
    high = rec_engine.risk_threshold * 100
    medium = rec_engine.medium_threshold * 100
    _, level = rec_engine.get_risk_level(probability)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{employee_name}'s Flight Risk", 'font': {'size': 24, 'color': 'white'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': 'white', 'family': "Arial, sans-serif"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'size': 14, 'color': 'white'}},
            'bar': {'color': RISK_COLORS[level], 'thickness': 0.5},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, medium], 'color': '#e3f2fd'},
                {'range': [medium, high], 'color': '#fff3e0'},
                {'range': [high, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "#ff416c", 'width': 4},
                'thickness': 0.8,
                'value': high
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial, sans-serif"}
    )

    return fig


def profile_scores(input_data):
    """Rescale selected form inputs to 0-100 for the profile radar chart"""
    return {
        'Job Satisfaction': input_data['JobSatisfaction'] / 4 * 100,
        'Compensation': min(input_data['MonthlyIncome'] / 20000 * 100, 100),  # form max is $20,000
        'Promotion Recency': max(0, 1 - input_data['YearsSinceLastPromotion'] / 15) * 100,
        'Work-Life Balance': input_data['WorkLifeBalance'] / 4 * 100,
        'Engagement': input_data['JobInvolvement'] / 4 * 100
    }

def create_radar_chart(profile):
    """Create a radar chart for employee profile"""

    categories = list(profile.keys())
    values = list(profile.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(67, 97, 238, 0.3)',
        line_color='#4361ee',
        line_width=2
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont_size=10,
                tickfont_color='white'
            ),
            bgcolor='rgba(0,0,0,0)',
            angularaxis=dict(
                tickfont_color='white'
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    
    return fig

def create_comparison_chart(employee_prob, avg_prob):
    """Create comparison chart against company average"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['This Employee', 'Active Employee Avg'],
        x=[employee_prob * 100, avg_prob * 100],
        orientation='h',
        marker_color=['#4361ee', '#4cc9f0'],
        text=[f'{employee_prob*100:.1f}%', f'{avg_prob*100:.1f}%'],
        textposition='auto',
        width=0.5
    ))
    
    fig.update_layout(
        title='Risk Comparison',
        title_font_color='white',
        xaxis_title='Attrition Probability (%)',
        xaxis_title_font_color='white',
        yaxis_title='',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickfont_color='white',
            gridcolor='rgba(255,255,255,0.1)'
        ),
        yaxis=dict(
            tickfont_color='white'
        ),
        font={'color': 'white'}
    )
    
    return fig

def encode_features(df, encoders, scaler, feature_columns):
    """Apply the training-time encoding and scaling to raw employee records"""
    df = df.copy()

    for col, encoder in encoders.items():
        if col != 'Attrition' and col in df.columns:
            unknown = set(df[col]) - set(encoder.classes_)
            if unknown:
                raise ValueError(f"Unknown {col} value(s): {sorted(unknown)}")
            df[col] = encoder.transform(df[col])

    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing feature(s): {missing}")

    df = df[feature_columns]
    return pd.DataFrame(scaler.transform(df), columns=feature_columns)


def preprocess_input(input_data, encoders, scaler, feature_columns):
    """Preprocess a single employee from the form for prediction"""
    return encode_features(pd.DataFrame([input_data]), encoders, scaler, feature_columns)


def main():
    """Main application with enhanced UI"""

    # Load models
    with st.spinner('🚀 Loading AI Models...'):
        try:
            model, encoders, scaler, feature_columns, metrics, feature_importance = load_models()
        except Exception as e:
            st.error(f"Could not load model artifacts: {e}. "
                     "Run `python src/model_training.py` first.")
            st.stop()

    df_full = score_dataset(load_dataset(), model, encoders, scaler, feature_columns)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🚀</div>
            <h2 style="margin: 0;">HC-SmartPulse</h2>
            <p style="color: #666; margin: 5px 0;">Talent Retention Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Risk threshold, defaulting to the one tuned during training
        st.markdown("### 🎚️ Risk Threshold")
        risk_threshold = st.slider(
            "High-risk threshold",
            0.05, 0.95, float(round(metrics['threshold'], 2)), 0.01,
            help="Employees at or above this predicted probability are flagged as high risk. "
                 f"The default ({metrics['threshold']:.2f}) is the F1-optimal threshold "
                 "found with cross-validation on the training set."
        )
        rec_engine = RecommendationEngine(risk_threshold=risk_threshold)

        st.markdown("---")

        # Filters
        st.markdown("### 🔍 Filters")

        # Get unique departments from dataset
        available_departments = ["All"] + sorted(df_full['Department'].unique().tolist())

        department_filter = st.multiselect(
            "Department",
            available_departments,
            default=["All"],
            help="Filter dashboard by department"
        )

        # Attrition status filter
        attrition_filter = st.selectbox(
            "Attrition Status",
            ["All", "Active Employees", "Left Company"],
            help="Filter by employment status"
        )

    # ===== APPLY FILTERS =====
    filtered_df = df_full.copy()

    # Department filter
    if "All" not in department_filter:
        filtered_df = filtered_df[filtered_df['Department'].isin(department_filter)]

    # Attrition filter
    if attrition_filter == "Active Employees":
        filtered_df = filtered_df[filtered_df['Attrition'] == 'No']
    elif attrition_filter == "Left Company":
        filtered_df = filtered_df[filtered_df['Attrition'] == 'Yes']

    # Calculate metrics from filtered data
    total_employees = len(filtered_df)
    attrition_rate = (filtered_df['Attrition'] == 'Yes').mean() * 100 if total_employees > 0 else 0
    active_df = filtered_df[filtered_df['Attrition'] == 'No']
    high_risk_active = int((active_df['RiskScore'] >= risk_threshold).sum())
    avg_active_risk = df_full.loc[df_full['Attrition'] == 'No', 'RiskScore'].mean()

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Employees", f"{total_employees:,}")
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
        st.metric("High-Risk Active Employees", f"{high_risk_active:,}")

    # Header with animated elements
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<h1 class="main-header">🚀 HC-SmartPulse</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Talent Retention Platform</p>', unsafe_allow_html=True)
        st.caption(f"📊 Employees: {len(df_full):,} | 🎯 Model recall: {metrics['recall']*100:.1f}% | "
                   f"F1: {metrics['f1_score']*100:.1f}% (held-out test set)")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "👤 Employee Risk", "📈 Analytics", "📁 Export & Model Info"])
    
    with tab1:
        # Dashboard View
        st.markdown("## 📊 Executive Dashboard")
        st.caption("Risk scores are model predictions for the historical IBM HR dataset. These records "
                   "include the training data, so the dashboard demonstrates the workflow rather than "
                   "an out-of-sample forecast.")

        if total_employees == 0:
            st.warning("No employees match the selected filters.")
        else:
            # KPI Cards
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 2rem;">🔴</div>
                    <div style="font-size: 2rem; font-weight: bold;">{high_risk_active:,}</div>
                    <div>High-Risk Active Employees</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4cc9f0, #4895ef);">
                    <div style="font-size: 2rem;">🔄</div>
                    <div style="font-size: 2rem; font-weight: bold;">{attrition_rate:.1f}%</div>
                    <div>Attrition Rate</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #134E5E, #71B280);">
                    <div style="font-size: 2rem;">📊</div>
                    <div style="font-size: 2rem; font-weight: bold;">{filtered_df['RiskScore'].mean()*100:.1f}%</div>
                    <div>Avg Predicted Risk</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #7209b7, #560bad);">
                    <div style="font-size: 2rem;">🎯</div>
                    <div style="font-size: 2rem; font-weight: bold;">{metrics['recall']*100:.1f}%</div>
                    <div>Model Recall (test set)</div>
                </div>
                """, unsafe_allow_html=True)

            # Charts Row
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📈 Risk Distribution")
                risk_levels = filtered_df['RiskScore'].apply(lambda p: rec_engine.get_risk_level(p)[0])
                risk_data = (risk_levels.value_counts()
                             .reindex(['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0)
                             .rename_axis('Risk Level').reset_index(name='Count'))

                fig = px.pie(risk_data, values='Count', names='Risk Level',
                            color='Risk Level', color_discrete_map={
                                'Low Risk': RISK_COLORS['low'],
                                'Medium Risk': RISK_COLORS['medium'],
                                'High Risk': RISK_COLORS['high']
                            })
                fig.update_traces(textposition='inside', textinfo='percent+label', textfont_color='white')
                fig.update_layout(
                    showlegend=False,
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("### 🏢 Attrition Rate by Department")
                dept_data = (filtered_df.assign(Left=filtered_df['Attrition'].eq('Yes'))
                             .groupby('Department')['Left'].mean().mul(100).round(1)
                             .rename('Attrition %').reset_index())

                fig = px.bar(dept_data, x='Department', y='Attrition %',
                            color='Attrition %', color_continuous_scale='Blues')
                fig.update_layout(
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        tickfont_color='white',
                        title_font_color='white'
                    ),
                    yaxis=dict(
                        tickfont_color='white',
                        title_font_color='white',
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    font={'color': 'white'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Active employees the model ranks highest
            st.markdown("### 🔥 Active Employees with the Highest Predicted Risk")

            watchlist = active_df.nlargest(5, 'RiskScore')
            if watchlist.empty:
                st.info("No active employees in the current filter.")

            for _, emp in watchlist.iterrows():
                _, level = rec_engine.get_risk_level(emp['RiskScore'])
                st.markdown(f"""
                <div class="employee-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0;">Employee #{emp['EmployeeNumber']}</h4>
                            <p style="margin: 5px 0; color: #666;">{emp['JobRole']} • {emp['Department']}</p>
                        </div>
                        <span class="risk-badge risk-{level}">{emp['RiskScore']*100:.0f}% Risk</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <small>🕐 Overtime: {emp['OverTime']} • 💼 {emp['YearsAtCompany']} yrs at company • 😊 Job satisfaction {emp['JobSatisfaction']}/4</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        # Employee Risk Assessment
        st.markdown("## 👤 Employee Risk Assessment")
        
        # Create two-column layout for input form
        form_col1, form_col2 = st.columns(2)
        
        with form_col1:
            with st.container():
                st.markdown("### 📋 Basic Information")
                employee_name = st.text_input("Employee Name", placeholder="Enter employee name")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    age = st.slider("Age", 18, 65, 30, help="Employee's age")
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                with col_b:
                    department = st.selectbox("Department", 
                        ["Sales", "Research & Development", "Human Resources"])
                    job_role = st.selectbox("Job Role", 
                        ["Sales Executive", "Research Scientist", "Laboratory Technician",
                         "Manufacturing Director", "Healthcare Representative", "Manager",
                         "Sales Representative", "Research Director", "Human Resources"])
                    distance_from_home = st.slider("Distance From Home (km)", 1, 30, 10)
                
                st.markdown("### 💼 Job Details")
                col_c, col_d = st.columns(2)
                with col_c:
                    job_level = st.select_slider("Job Level", options=[1, 2, 3, 4, 5], value=3)
                    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 6500, step=100)
                    job_involvement = st.slider("Job Involvement", 1, 4, 3)
                with col_d:
                    total_working_years = st.slider("Total Working Years", 0, 40, 10)
                    years_at_company = st.slider("Years at Company", 0, 40, 5)
                    years_in_current_role = st.slider("Years in Current Role", 0, 20, 3)
                
                st.markdown("### 📚 Education & Experience")
                col_e, col_f = st.columns(2)
                with col_e:
                    education = st.select_slider("Education Level", 
                        options=[1, 2, 3, 4, 5], value=3,
                        format_func=lambda x: ["Below College", "College", "Bachelor", "Master", "Doctor"][x-1])
                    education_field = st.selectbox("Education Field",
                        ["Life Sciences", "Medical", "Marketing", "Technical Degree", 
                         "Human Resources", "Other"])
                with col_f:
                    num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 2)
                    years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
                    years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 3)
        
        with form_col2:
            with st.container():
                st.markdown("### 😊 Satisfaction Scores")
                
                satisfaction_cols = st.columns(2)
                with satisfaction_cols[0]:
                    job_satisfaction = st.select_slider("Job Satisfaction", 
                        options=[1, 2, 3, 4], value=3,
                        format_func=lambda x: ["😞 Low", "😐 Medium", "😊 High", "🤩 Very High"][x-1])
                    environment_satisfaction = st.select_slider("Environment Satisfaction", 
                        options=[1, 2, 3, 4], value=3,
                        format_func=lambda x: ["😞 Low", "😐 Medium", "😊 High", "🤩 Very High"][x-1])
                
                with satisfaction_cols[1]:
                    relationship_satisfaction = st.select_slider("Relationship Satisfaction", 
                        options=[1, 2, 3, 4], value=3,
                        format_func=lambda x: ["😞 Low", "😐 Medium", "😊 High", "🤩 Very High"][x-1])
                    work_life_balance = st.select_slider("Work-Life Balance", 
                        options=[1, 2, 3, 4], value=3,
                        format_func=lambda x: ["😞 Poor", "😐 Fair", "😊 Good", "🤩 Excellent"][x-1])
                
                st.markdown("### ⚙️ Work Conditions")
                col_e, col_f = st.columns(2)
                with col_e:
                    overtime = st.radio("Over Time", ["No", "Yes"], horizontal=True)
                    business_travel = st.selectbox("Business Travel", 
                        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
                    stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
                with col_f:
                    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
                    performance_rating = st.select_slider("Performance Rating", 
                        options=[1, 2, 3, 4], value=3,
                        format_func=lambda x: ["📉 Low", "📈 Good", "🚀 Excellent", "🏆 Outstanding"][x-1])
                    
                st.markdown("### 📊 Additional Metrics")
                col_g, col_h = st.columns(2)
                with col_g:
                    hourly_rate = st.number_input("Hourly Rate ($)", 30, 100, 65)
                    daily_rate = st.number_input("Daily Rate ($)", 100, 1500, 800)
                with col_h:
                    monthly_rate = st.number_input("Monthly Rate ($)", 2000, 30000, 14000)
                    percent_salary_hike = st.slider("Last Salary Hike %", 10, 25, 15)
        
        # Predict Button
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            if st.button("🎯 Analyze Flight Risk", type="primary", use_container_width=True):
                with st.spinner('🤖 Analyzing employee data...'):
                    
                    # Prepare input data dictionary
                    input_data = {
                        'Age': age,
                        'Gender': gender,
                        'MaritalStatus': marital_status,
                        'DistanceFromHome': distance_from_home,
                        'Department': department,
                        'JobRole': job_role,
                        'JobLevel': job_level,
                        'JobInvolvement': job_involvement,
                        'MonthlyIncome': monthly_income,
                        'TotalWorkingYears': total_working_years,
                        'YearsAtCompany': years_at_company,
                        'YearsInCurrentRole': years_in_current_role,
                        'YearsSinceLastPromotion': years_since_promotion,
                        'YearsWithCurrManager': years_with_curr_manager,
                        'NumCompaniesWorked': num_companies_worked,
                        'EnvironmentSatisfaction': environment_satisfaction,
                        'JobSatisfaction': job_satisfaction,
                        'RelationshipSatisfaction': relationship_satisfaction,
                        'WorkLifeBalance': work_life_balance,
                        'Education': education,
                        'EducationField': education_field,
                        'TrainingTimesLastYear': training_times_last_year,
                        'OverTime': overtime,
                        'BusinessTravel': business_travel,
                        'StockOptionLevel': stock_option_level,
                        'PerformanceRating': performance_rating,
                        'HourlyRate': hourly_rate,
                        'DailyRate': daily_rate,
                        'MonthlyRate': monthly_rate,
                        'PercentSalaryHike': percent_salary_hike
                    }
                    
                    # Predict with the trained model; never show a made-up score
                    try:
                        processed_data = preprocess_input(input_data, encoders, scaler, feature_columns)
                        probability = float(model.predict_proba(processed_data)[0][1])
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        st.stop()

                    # Display results with animation
                    st.markdown('<div class="fade-in">', unsafe_allow_html=True)

                    # Results Header
                    st.markdown("---")
                    st.markdown(f"## 📊 Analysis Results for {employee_name or 'Employee'}")

                    # Risk Visualization in columns
                    viz_col1, viz_col2 = st.columns([2, 1])

                    with viz_col1:
                        # Gauge Chart
                        st.plotly_chart(create_animated_gauge(probability, rec_engine, employee_name or "Employee"),
                                      use_container_width=True)

                    with viz_col2:
                        # Risk Badge and Stats
                        risk_level, level = rec_engine.get_risk_level(probability)

                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div class="risk-badge risk-{level}" style="font-size: 1.2rem; min-width: 150px; margin: 0 auto;">
                                {risk_level}
                            </div>
                            <div style="font-size: 3rem; font-weight: bold; margin: 20px 0; color: {RISK_COLORS[level]}">
                                {probability*100:.1f}%
                            </div>
                            <div style="color: #666;">
                                Predicted attrition probability
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Comparison chart
                        st.plotly_chart(create_comparison_chart(probability, avg_active_risk),
                                      use_container_width=True)

                    # Radar Chart for Profile
                    st.markdown("### 🎯 Employee Profile Snapshot")
                    radar_col1, radar_col2 = st.columns([1, 2])
                    profile = profile_scores(input_data)

                    with radar_col1:
                        strongest = max(profile, key=profile.get)
                        weakest = min(profile, key=profile.get)
                        st.markdown(f"""
                        #### Key Insights:
                        - **Strongest area**: {strongest} ({profile[strongest]:.0f}/100)
                        - **Weakest area**: {weakest} ({profile[weakest]:.0f}/100)
                        - **Overtime**: {input_data['OverTime']}
                        """)
                        st.caption("Profile scores are the form inputs rescaled to 0-100, not model outputs.")

                    with radar_col2:
                        st.plotly_chart(create_radar_chart(profile),
                                      use_container_width=True)

                    # Recommendations
                    st.markdown("### 💡 Action Plan & Recommendations")

                    recommendations = rec_engine.get_recommendations(probability, input_data)

                    for i, rec in enumerate(recommendations, 1):
                        icon = "🔴" if rec["priority"] in ("Critical", "High") else "🟡" if rec["priority"] == "Medium" else "🟢"

                        with st.expander(f"{icon} {rec['category']} ({rec['priority']} Priority)", expanded=True if i == 1 else False):
                            st.success(rec["action"])

                    st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        # Analytics Page
        st.markdown("## 📈 Advanced Analytics")

        # Feature Importance
        st.markdown("### 🔍 Top Factors Influencing Attrition")

        fig = px.bar(feature_importance.head(10).sort_values('importance'),
                    x='importance',
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Viridis',
                    title='Mean |SHAP value| on the test set')

        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font_color='white',
            xaxis=dict(
                tickfont_color='white',
                title_font_color='white',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                tickfont_color='white',
                title_font_color='white'
            ),
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model Performance Metrics
        st.markdown("### 🎯 Model Performance (held-out test set)")
        st.caption(f"{metrics['n_test']} test employees, decision threshold {metrics['threshold']:.2f} "
                   "(tuned with cross-validation on the training set only).")

        col1, col2, col3, col4 = st.columns(4)
        metrics_display = [
            ("Recall", metrics['recall'], "#f72585"),
            ("Precision", metrics['precision'], "#4cc9f0"),
            ("F1-Score", metrics['f1_score'], "#7209b7"),
            ("PR-AUC", metrics['pr_auc'], "#4361ee")
        ]

        for (name, value, color), col in zip(metrics_display, [col1, col2, col3, col4]):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.05); border-radius: 15px; box-shadow: 0 5px 15px rgba(75, 48, 97, 0.3); border: 1px solid rgba(209, 196, 233, 0.1);">
                    <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                        {value*100:.1f}%
                    </div>
                    <div style="color: white; margin-top: 5px;">
                        {name}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Baselines
        st.markdown("### ⚖️ Comparison with Baselines")
        rows = [{'Model': f"XGBoost (threshold {metrics['threshold']:.2f})", **metrics}]
        rows += [{'Model': f"{name} (threshold 0.50)", **scores} for name, scores in metrics['baselines'].items()]
        columns = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall',
                   'f1_score': 'F1', 'pr_auc': 'PR-AUC', 'cv_pr_auc': 'CV PR-AUC'}
        comparison = pd.DataFrame(rows).set_index('Model')[list(columns)].rename(columns=columns)
        st.dataframe(comparison.style.format('{:.1%}'), use_container_width=True)
        st.caption("Accuracy is misleading on this dataset: predicting \"No\" for everyone already scores "
                   "~84%. Compare models on recall, F1 and PR-AUC. CV PR-AUC is the 5-fold "
                   "cross-validated score on the training set.")

        # Confusion matrix
        st.markdown("### 🧮 Confusion Matrix (test set)")
        fig = px.imshow(metrics['confusion_matrix'], text_auto=True, color_continuous_scale='Purples',
                        x=['Predicted: Stay', 'Predicted: Leave'], y=['Actual: Stay', 'Actual: Leave'])
        fig.update_layout(
            height=350,
            coloraxis_showscale=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("## 📁 Export & Model Info")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📥 Export Employee Risk Scores")

            export_departments = st.multiselect(
                "Department",
                ["All"] + sorted(df_full['Department'].unique().tolist()),
                default=["All"],
                help="Select departments to include in export"
            )

            export_attrition = st.selectbox(
                "Attrition Status",
                ["All", "Yes - Left Company", "No - Still Active"],
                help="Filter by employee attrition status"
            )

            export_job_role = st.multiselect(
                "Job Role (optional)",
                ["All"] + sorted(df_full['JobRole'].unique().tolist()),
                default=["All"],
                help="Optionally filter by specific job roles"
            )

            export_format = st.radio("Export Format", ["CSV", "Excel"], horizontal=True)

            df_export = df_full.copy()
            if "All" not in export_departments:
                df_export = df_export[df_export['Department'].isin(export_departments)]
            if export_attrition != "All":
                attrition_value = "Yes" if "Yes" in export_attrition else "No"
                df_export = df_export[df_export['Attrition'] == attrition_value]
            if "All" not in export_job_role:
                df_export = df_export[df_export['JobRole'].isin(export_job_role)]

            if df_export.empty:
                st.error("❌ No data matches the selected filters. Please adjust your filters.")
            else:
                st.info(f"📊 {len(df_export):,} of {len(df_full):,} employees selected, "
                        "including the predicted RiskScore column.")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                if export_format == "CSV":
                    st.download_button(
                        label="💾 Download CSV",
                        data=df_export.to_csv(index=False).encode('utf-8'),
                        file_name=f"employee_risk_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_export.to_excel(writer, sheet_name='Employee Risk', index=False)
                    st.download_button(
                        label="💾 Download Excel",
                        data=output.getvalue(),
                        file_name=f"employee_risk_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        with col2:
            st.markdown("### 🛠️ Model Information")

            model_info = {
                "Model": "XGBoost classifier",
                "XGBoost version": xgb.__version__,
                "Training / test employees": f"{metrics['n_train']:,} / {metrics['n_test']:,}",
                "Tuned decision threshold": f"{metrics['threshold']:.3f}",
                "Cross-validated PR-AUC": f"{metrics['cv_pr_auc']:.3f}",
                "Test ROC-AUC": f"{metrics['roc_auc']:.3f}"
            }

            for key, value in model_info.items():
                st.markdown(f"<p style='color: white; margin: 5px 0;'>• {key}: {value}</p>", unsafe_allow_html=True)

# Sidebar

if __name__ == "__main__":
    main()
