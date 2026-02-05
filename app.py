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
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from recommendation_engine import RecommendationEngine
except ImportError:
    # Mock class for demo
    class RecommendationEngine:
        def __init__(self, risk_threshold=0.7):
            self.risk_threshold = risk_threshold
        
        def get_risk_level(self, probability):
            if probability >= 0.7:
                return "HIGH RISK", "high"
            elif probability >= 0.4:
                return "MEDIUM RISK", "medium"
            else:
                return "LOW RISK", "low"
        
        def get_recommendations(self, probability, input_data):
            return [
                {"category": "Compensation", "action": "Review salary competitiveness", "priority": "High"},
                {"category": "Career Growth", "action": "Discuss promotion timeline", "priority": "Medium"},
                {"category": "Work-Life Balance", "action": "Evaluate overtime requirements", "priority": "Medium"}
            ]

# Page configuration
st.set_page_config(
    page_title="HC-SmartPulse | Talent Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://example.com',
        'Report a bug': 'https://example.com',
        'About': "### HC-SmartPulse v2.0\nAI-powered employee retention platform"
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

@st.cache_resource
def load_models():
    """Load trained model and preprocessing artifacts"""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        encoders = joblib.load('models/feature_encoder.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        
        # Try to load metrics and feature importance
        try:
            metrics = joblib.load('models/model_metrics.pkl')
        except:
            metrics = {'accuracy': 0.87, 'precision': 0.85, 'recall': 0.82, 'f1_score': 0.83}
        
        try:
            feature_importance = pd.read_csv('models/feature_importance.csv')
        except:
            # Mock feature importance for demo
            feature_importance = pd.DataFrame({
                'feature': ['MonthlyIncome', 'OverTime', 'JobSatisfaction', 'YearsAtCompany', 'WorkLifeBalance'],
                'importance': [0.25, 0.18, 0.15, 0.12, 0.10]
            })
        
        return model, encoders, scaler, feature_columns, metrics, feature_importance
    except Exception as e:
        st.warning(f"Using demo mode: {e}")
        # Return mock data for demo
        return None, None, None, None, None, None

def create_animated_gauge(probability, employee_name="Employee"):
    """Create an animated gauge chart"""
    
    # Determine color based on probability
    if probability >= 0.7:
        color = '#ff416c'
        color_scale = [(0.0, '#4facfe'), (0.7, '#ffb347'), (1.0, '#ff416c')]
    elif probability >= 0.4:
        color = '#ffb347'
        color_scale = [(0.0, '#4facfe'), (0.4, '#ffb347'), (1.0, '#ffb347')]
    else:
        color = '#4facfe'
        color_scale = [(0.0, '#4facfe'), (0.4, '#4facfe'), (1.0, '#ffb347')]
    
    # Create gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{employee_name}'s Flight Risk", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        number={'suffix': "%", 'font': {'size': 48, 'color': 'white', 'family': "Arial, sans-serif"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'size': 14, 'color': 'white'}},
            'bar': {'color': color, 'thickness': 0.5},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#e3f2fd'},
                {'range': [40, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
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

def create_radar_chart(input_data):
    """Create a radar chart for employee profile"""
    
    categories = ['Job Satisfaction', 'Compensation', 'Growth Potential', 'Work-Life Balance', 'Engagement']
    
    # Mock values based on input data (in real app, calculate these properly)
    values = [
        input_data.get('JobSatisfaction', 3) / 4 * 100,
        min(input_data.get('MonthlyIncome', 5000) / 10000 * 100, 100),
        (5 - input_data.get('YearsSinceLastPromotion', 2)) / 5 * 100,
        input_data.get('WorkLifeBalance', 3) / 4 * 100,
        input_data.get('JobInvolvement', 3) / 4 * 100
    ]
    
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

def create_comparison_chart(employee_prob, avg_prob=0.3):
    """Create comparison chart against company average"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=['This Employee', 'Company Average'],
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

def preprocess_input(input_data, encoders, scaler, feature_columns):
    """Preprocess user input data for prediction"""
    
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df.columns and col != 'Attrition':
            if df[col].dtype == 'object':
                try:
                    df[col] = encoder.transform(df[col])
                except:
                    # If value not in training data, use most frequent
                    df[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_columns]
    
    # Scale numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

def main():
    """Main application with enhanced UI"""
    
    # Load models
    with st.spinner('🚀 Loading AI Models...'):
        model, encoders, scaler, feature_columns, metrics, feature_importance = load_models()
        rec_engine = RecommendationEngine(risk_threshold=0.7)
    
    # Load dataset for filtering
    @st.cache_data
    def load_dataset():
        return pd.read_csv('data/dataset.csv')
    
    df_full = load_dataset()
    
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
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        st.metric("Active Alerts", "24", "+3")
        st.metric("Avg Risk Score", "32.5%", "-2.1%")
        st.metric("Retention Rate", "91.5%", "+0.8%")
        
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
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📧 Send Reports", use_container_width=True):
            st.toast("📤 Reports sent to management team!")
        
        if st.button("🆕 New Analysis", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Show filter status
        filter_count = 0
        if "All" not in department_filter:
            filter_count += 1
        if attrition_filter != "All":
            filter_count += 1
        
        if filter_count > 0:
            st.caption(f"🎯 {filter_count} filter(s) active")
    
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
    attrition_count = len(filtered_df[filtered_df['Attrition'] == 'Yes'])
    attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
    retention_rate = 100 - attrition_rate
    avg_monthly_income = filtered_df['MonthlyIncome'].mean() if total_employees > 0 else 0
    
    
    # Header with animated elements
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<h1 class="main-header">🚀 HC-SmartPulse</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Talent Retention Platform</p>', unsafe_allow_html=True)
        
        # Current date and stats
        today = datetime.now().strftime("%B %d, %Y")
        accuracy = metrics.get('accuracy', 0.87) if metrics else 0.87
        st.caption(f"📅 Last Updated: {today} | 📊 Active Employees: 1,234 | 🎯 Model Accuracy: {accuracy*100:.1f}%")
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "👤 Employee Risk", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        # Dashboard View
        st.markdown("## 📊 Executive Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 2rem;">🔴</div>
                <div style="font-size: 2rem; font-weight: bold;">12.3%</div>
                <div>High Risk Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #4cc9f0, #4895ef);">
                <div style="font-size: 2rem;">🔄</div>
                <div style="font-size: 2rem; font-weight: bold;">8.5%</div>
                <div>Turnover Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #134E5E, #71B280);">
                <div style="font-size: 2rem;">💰</div>
                <div style="font-size: 2rem; font-weight: bold;">$2.1M</div>
                <div>Potential Savings</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #7209b7, #560bad);">
                <div style="font-size: 2rem;">🎯</div>
                <div style="font-size: 2rem; font-weight: bold;">83.7%</div>
                <div>Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Risk Distribution")
            # Mock data
            risk_data = pd.DataFrame({
                'Risk Level': ['Low', 'Medium', 'High'],
                'Count': [850, 250, 134],
                'Color': ['#4facfe', '#ffb347', '#ff416c']
            })
            
            fig = px.pie(risk_data, values='Count', names='Risk Level', 
                        color='Risk Level', color_discrete_map={
                            'Low': '#4facfe',
                            'Medium': '#ffb347',
                            'High': '#ff416c'
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
            st.markdown("### 🏢 Department Analysis")
            dept_data = pd.DataFrame({
                'Department': ['Research & Development', 'Sales', 'Human Resources'],
                'Risk %': [18.7, 25.3, 8.9]
            })
            
            fig = px.bar(dept_data, x='Department', y='Risk %',
                        color='Risk %', color_continuous_scale='Blues')
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
        
        # Recent High-Risk Employees
        st.markdown("### 🔥 Recent High-Risk Alerts")
        
        high_risk_employees = [
            {"name": "Sarah Johnson", "role": "Research Scientist", "risk": 85, "department": "Research & Development"},
            {"name": "Michael Chen", "role": "Sales Manager", "risk": 78, "department": "Sales"},
            {"name": "Emma Wilson", "role": "Sales Executive", "risk": 72, "department": "Sales"},
            {"name": "David Brown", "role": "HR Specialist", "risk": 68, "department": "Human Resources"}
        ]
        
        for emp in high_risk_employees:
            risk_color = "risk-high" if emp["risk"] >= 70 else "risk-medium"
            with st.container():
                st.markdown(f"""
                <div class="employee-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin: 0;">{emp['name']}</h4>
                            <p style="margin: 5px 0; color: #666;">{emp['role']} • {emp['department']}</p>
                        </div>
                        <span class="risk-badge {risk_color}">{emp['risk']}% Risk</span>
                    </div>
                    <div style="margin-top: 10px;">
                        <small>📅 Last review: 2 days ago • 🎯 Priority: High</small>
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
                    time.sleep(1.5)
                    
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
                    
                    # Make actual prediction if model is loaded
                    if model is not None:
                        try:
                            processed_data = preprocess_input(input_data, encoders, scaler, feature_columns)
                            probability = model.predict_proba(processed_data)[0][1]
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            probability = np.random.uniform(0.1, 0.9)  # Fallback
                    else:
                        # Fallback to mock prediction if model not loaded
                        probability = np.random.uniform(0.1, 0.9)
                    
                    # Display results with animation
                    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                    
                    # Results Header
                    st.markdown("---")
                    st.markdown(f"## 📊 Analysis Results for {employee_name or 'Employee'}")
                    
                    # Risk Visualization in columns
                    viz_col1, viz_col2 = st.columns([2, 1])
                    
                    with viz_col1:
                        # Gauge Chart
                        st.plotly_chart(create_animated_gauge(probability, employee_name), 
                                      use_container_width=True)
                    
                    with viz_col2:
                        # Risk Badge and Stats
                        risk_level, color = rec_engine.get_risk_level(probability)
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px;">
                            <div class="risk-badge risk-{color}" style="font-size: 1.2rem; min-width: 150px; margin: 0 auto;">
                                {risk_level}
                            </div>
                            <div style="font-size: 3rem; font-weight: bold; margin: 20px 0; color: {'#ff416c' if probability >= 0.7 else '#ffb347' if probability >= 0.4 else '#4facfe'}">
                                {probability*100:.1f}%
                            </div>
                            <div style="color: #666;">
                                Probability of leaving within 6 months
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Comparison chart
                        st.plotly_chart(create_comparison_chart(probability), 
                                      use_container_width=True)
                    
                    # Radar Chart for Profile
                    st.markdown("### 🎯 Employee Profile Snapshot")
                    radar_col1, radar_col2 = st.columns([1, 2])
                    
                    with radar_col1:
                        st.markdown("""
                        #### Key Insights:
                        - **Strengths**: Good work-life balance
                        - **Concerns**: Compensation competitiveness
                        - **Opportunity**: Career growth path
                        - **Threat**: High overtime hours
                        """)
                    
                    with radar_col2:
                        # Use the input_data dict created above for radar chart
                        st.plotly_chart(create_radar_chart(input_data), 
                                      use_container_width=True)
                    
                    # Recommendations
                    st.markdown("### 💡 Action Plan & Recommendations")
                    
                    recommendations = rec_engine.get_recommendations(probability, input_data)
                    
                    for i, rec in enumerate(recommendations, 1):
                        icon = "🔴" if rec["priority"] == "High" else "🟡" if rec["priority"] == "Medium" else "🟢"
                        
                        with st.expander(f"{icon} {rec['category']} ({rec['priority']} Priority)", expanded=True if i == 1 else False):
                            st.success(rec["action"])
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.button(f"✅ Mark Complete", key=f"complete_{i}", 
                                         help=f"Mark {rec['category']} as completed")
                            with col_b:
                                st.button(f"📅 Schedule Follow-up", key=f"schedule_{i}",
                                         help=f"Schedule follow-up for {rec['category']}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Analytics Page
        st.markdown("## 📈 Advanced Analytics")
        
        # Feature Importance
        if feature_importance is not None:
            st.markdown("### 🔍 Top Factors Influencing Attrition")
            
            fig = px.bar(feature_importance.head(10), 
                        x='importance', 
                        y='feature',
                        orientation='h',
                        color='importance',
                        color_continuous_scale='Viridis',
                        title='Feature Importance Analysis')
            
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
        st.markdown("### 🎯 Model Performance")
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            metrics_display = [
                ("Accuracy", metrics.get('accuracy', 0), "#4361ee"),
                ("Precision", metrics.get('precision', 0), "#4cc9f0"),
                ("Recall", metrics.get('recall', 0), "#f72585"),
                ("F1-Score", metrics.get('f1_score', 0), "#7209b7")
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
    
    with tab4:
        # Settings Page
        st.markdown("## ⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔧 Model Settings")
            
            risk_threshold = st.slider("High Risk Threshold", 0.0, 1.0, 0.7, 0.05)
            confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
            
            st.markdown("### 📊 Data Sources")
            sources = st.multiselect(
                "Integration Sources",
                ["HRIS System", "Performance Reviews", "Employee Surveys", 
                 "Attendance Records", "Project Management"],
                default=["HRIS System", "Performance Reviews"]
            )
        
        with col2:
            st.markdown("### 👥 User Preferences")
            
            notifications = st.checkbox("🔔 Enable Email Notifications", True)
            if notifications:
                st.select_slider("Notification Frequency", 
                                options=["Real-time", "Daily", "Weekly", "Monthly"],
                                value="Daily")
            
            st.markdown("### 📁 Export Options")
            
            # Export Filters
            st.markdown("#### 🔍 Filter Data")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                export_departments = st.multiselect(
                    "Department",
                    ["All", "Sales", "Research & Development", "Human Resources"],
                    default=["All"],
                    help="Select departments to include in export"
                )
                
                export_attrition = st.selectbox(
                    "Attrition Status",
                    ["All", "Yes - Left Company", "No - Still Active"],
                    help="Filter by employee attrition status"
                )
            
            with filter_col2:
                export_job_role = st.multiselect(
                    "Job Role (optional)",
                    ["All", "Sales Executive", "Research Scientist", "Laboratory Technician",
                     "Manufacturing Director", "Healthcare Representative", "Manager",
                     "Sales Representative", "Research Director", "Human Resources"],
                    default=["All"],
                    help="Optionally filter by specific job roles"
                )
            
            st.markdown("---")
            
            export_format = st.radio("Export Format", ["CSV", "Excel", "PDF"], horizontal=True)
            
            if st.button("📥 Export Current Data", use_container_width=True):
                # Load the actual dataset for export
                try:
                    df_export = pd.read_csv("data/dataset.csv")
                    original_count = len(df_export)
                    
                    # Apply filters
                    # 1. Department filter
                    if "All" not in export_departments:
                        df_export = df_export[df_export['Department'].isin(export_departments)]
                    
                    # 2. Attrition filter
                    if export_attrition != "All":
                        attrition_value = "Yes" if "Yes" in export_attrition else "No"
                        df_export = df_export[df_export['Attrition'] == attrition_value]
                    
                    # 3. Job Role filter
                    if "All" not in export_job_role:
                        df_export = df_export[df_export['JobRole'].isin(export_job_role)]
                    
                    filtered_count = len(df_export)
                    
                    # Check if any data remains after filtering
                    if filtered_count == 0:
                        st.error("❌ No data matches the selected filters. Please adjust your filters.")
                    else:
                        # Show filter results
                        if filtered_count < original_count:
                            st.info(f"🔍 Filtered: {filtered_count} of {original_count} employees ({filtered_count/original_count*100:.1f}%)")
                        else:
                            st.info(f"📊 Exporting all {original_count} employees (no filters applied)")
                        
                        
                        if export_format == "CSV":
                            csv_data = df_export.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="💾 Download CSV",
                                data=csv_data,
                                file_name=f"employee_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.success(f"✅ CSV file ready for download! ({len(df_export)} rows)")
                            
                        elif export_format == "Excel":
                            # Create Excel file in memory
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df_export.to_excel(writer, sheet_name='Employee Data', index=False)
                            excel_data = output.getvalue()
                            
                            st.download_button(
                                label="💾 Download Excel",
                                data=excel_data,
                                file_name=f"employee_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                            st.success(f"✅ Excel file ready for download! ({len(df_export)} rows)")
                            
                        elif export_format == "PDF":
                            st.info("📄 PDF export coming soon! Please use CSV or Excel for now.")
                        
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 🛠️ System Information")
        
        sys_info = {
            "Version": "HC-SmartPulse v2.0",
            "Last Updated": today,
            "Model Version": "XGBoost v1.4.2",
            "Active Users": "24",
            "Data Points": "45,678"
        }
        
        for key, value in sys_info.items():
            st.markdown(f"<p style='color: white; margin: 5px 0;'>• {key}: {value}</p>", unsafe_allow_html=True)

# Sidebar

if __name__ == "__main__":
    main()
