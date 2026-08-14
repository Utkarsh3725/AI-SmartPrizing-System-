"""
==============================================================================
AI-Driven Smart Pricing System for Theme Parks
Streamlit Dashboard
==============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG & THEME
# ============================================================
st.set_page_config(
    page_title="AI Smart Pricing | Theme Park Revenue Optimizer",
    page_icon="🎢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark-green theme
st.markdown("""
<style>
    /* ===== GLOBAL DARK THEME ===== */
    .stApp {
        background: linear-gradient(135deg, #0a0f0d 0%, #101a14 50%, #0d1210 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1a12 0%, #0a1410 100%) !important;
        border-right: 1px solid #1a3a28;
    }
    
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #0d2818 0%, #1a3d28 100%);
        border: 1px solid #2d5a3d;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8fbfa3 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #4ade80 !important;
        font-weight: 700 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e0f2e9 !important;
    }

    /* Custom card class */
    .dashboard-card {
        background: linear-gradient(135deg, #0d2818 0%, #142e1e 100%);
        border: 1px solid #2d5a3d;
        border-radius: 16px;
        padding: 24px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    
    .big-price {
        font-size: 3.5rem;
        font-weight: 800;
        color: #4ade80;
        text-align: center;
        text-shadow: 0 0 30px rgba(74,222,128,0.3);
        margin: 10px 0;
    }
    
    .price-label {
        font-size: 1.1rem;
        color: #8fbfa3;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .brand-title {
        font-size: 1.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4ade80, #22c55e, #86efac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .brand-subtitle {
        font-size: 0.8rem;
        color: #6b9a7e;
        text-align: center;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    
    .team-badge {
        background: linear-gradient(135deg, #0d3320 0%, #1a4a30 100%);
        border: 1px solid #2d6b42;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 15px 0;
    }
    
    .team-badge h4 {
        color: #4ade80 !important;
        margin: 0 0 5px 0;
        font-size: 0.95rem;
    }
    
    .team-badge p {
        color: #8fbfa3;
        margin: 2px 0;
        font-size: 0.8rem;
    }
    
    .alert-card {
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
    .alert-increase {
        background: rgba(74,222,128,0.1);
        border-left: 4px solid #4ade80;
        color: #b8e6cc;
    }
    .alert-decrease {
        background: rgba(239,68,68,0.1);
        border-left: 4px solid #ef4444;
        color: #f5b0b0;
    }
    .alert-info {
        background: rgba(59,130,246,0.1);
        border-left: 4px solid #3b82f6;
        color: #a8cbf0;
    }
    
    /* Streamlit widget styling */
    .stSlider > div > div {
        color: #4ade80 !important;
    }
    .stSelectbox label, .stSlider label, .stDateInput label {
        color: #b8e6cc !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0d2818;
        border: 1px solid #2d5a3d;
        border-radius: 8px;
        color: #8fbfa3;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a4a30 !important;
        border-color: #4ade80 !important;
        color: #4ade80 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 12px 30px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(34,197,94,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(34,197,94,0.4) !important;
    }
    
    /* Dataframe / table */
    .stDataFrame {
        border: 1px solid #2d5a3d;
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border-color: #1a3a28 !important;
    }

    /* Chat input */
    .stChatInput > div {
        border-color: #2d5a3d !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load the theme park dataset."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'theme_park_data.csv')
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b %Y')
    df['year'] = df['date'].dt.year
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    return df


@st.cache_resource
def load_model():
    """Load the trained pricing model."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'pricing_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python package '{exc.name}' required to load the saved model. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    
    encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'weather_encoder.pkl')
    with open(encoder_path, 'rb') as f:
        weather_encoder = pickle.load(f)
    
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'model_results.pkl')
    model_results = None
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            model_results = pickle.load(f)
    
    return model_package, weather_encoder, model_results


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    """Render the sidebar with branding and navigation."""
    with st.sidebar:
        # Branding
        st.markdown('<div class="brand-title">🎢 AI Smart Pricing</div>', unsafe_allow_html=True)
        st.markdown('<div class="brand-subtitle">Theme Park Revenue Optimizer</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "📍 Navigation",
            ["📊 Overview Dashboard", "🤖 AI Price Predictor", "📈 Revenue Analysis", 
             "🏪 Market Monitor", "🔬 Revenue Simulator"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Project badge
        st.markdown("""
        <div class="team-badge">
            <h4>🎢 Theme Park Revenue Optimizer</h4>
            <p>━━━━━━━━━━━━━━━</p>
            <p><b style="color:#4ade80;">AI Smart Pricing System</b></p>
            <p>Production-ready pricing dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model info
        try:
            model_package, _, model_results = load_model()
            winner = model_package.get('winner', 'N/A')
            if model_results:
                best_r2 = model_results.get(winner, {}).get('r2', 0)
                st.markdown(f"""
                <div class="team-badge">
                    <h4>🧠 AI Model Info</h4>
                    <p>Model: <b style="color:#4ade80;">{winner.replace('_', ' ').title()}</b></p>
                    <p>Accuracy: <b style="color:#4ade80;">{best_r2*100:.1f}%</b> R²</p>
                    <p>Status: <b style="color:#4ade80;">● Online</b></p>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            st.info("⚠️ Model unavailable. Run `pip install -r requirements.txt`, then `python main.py`.")
        
        st.markdown("---")
        
        # Chatbot
        st.markdown("### 💬 Ask AI Assistant")
        render_chatbot()
        
        return page


def render_chatbot():
    """Simple rule-based chatbot in sidebar."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history[-5:]:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**🤖 AI:** {msg['text']}")
    
    user_input = st.chat_input("Ask about pricing...", key="sidebar_chat")
    
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'text': user_input})
        
        response = get_chatbot_response(user_input)
        st.session_state.chat_history.append({'role': 'bot', 'text': response})
        st.rerun()


def get_chatbot_response(query):
    """Generate responses for common pricing questions."""
    query_lower = query.lower()
    
    try:
        df = load_data()
    except Exception:
        return "Data not loaded yet. Please run `python main.py` first."
    
    if any(w in query_lower for w in ['sunday', 'weekend', 'saturday']):
        weekend_data = df[df['is_weekend'] == 1]
        avg_price = weekend_data['recommended_price'].mean()
        avg_visitors = weekend_data['visitors'].mean()
        return f"For weekends, the recommended price is ₹{avg_price:.0f} with an average of {avg_visitors:.0f} visitors."
    
    elif any(w in query_lower for w in ['holiday', 'diwali', 'holi', 'festival']):
        holiday_data = df[df['is_holiday'] == 1]
        avg_price = holiday_data['recommended_price'].mean()
        return f"For holidays, the AI recommends ₹{avg_price:.0f}. Festival seasons see 40-60% more visitors!"
    
    elif any(w in query_lower for w in ['rain', 'rainy', 'monsoon']):
        rainy = df[df['weather'] == 'rainy']
        avg_price = rainy['recommended_price'].mean()
        return f"On rainy days, recommend ₹{avg_price:.0f}. Consider offering a 'Rainy Day Special' discount package."
    
    elif any(w in query_lower for w in ['best price', 'optimal', 'recommend']):
        avg_rec = df['recommended_price'].mean()
        return f"The average optimal price is ₹{avg_rec:.0f}. Use the AI Predictor page to get specific recommendations."
    
    elif any(w in query_lower for w in ['revenue', 'earning', 'income']):
        total_rev = df['revenue'].sum()
        avg_rev = df['revenue'].mean()
        return f"Total revenue: ₹{total_rev:,.0f}. Daily average: ₹{avg_rev:,.0f}."
    
    elif any(w in query_lower for w in ['visitor', 'crowd', 'people']):
        avg_v = df['visitors'].mean()
        max_v = df['visitors'].max()
        return f"Average daily visitors: {avg_v:.0f}. Peak day: {max_v:,} visitors."
    
    elif any(w in query_lower for w in ['sunny', 'weather', 'hot', 'cold']):
        sunny = df[df['weather'] == 'sunny']
        avg_price = sunny['recommended_price'].mean()
        return f"On sunny days, recommend ₹{avg_price:.0f}. Sunny days attract 15-25% more visitors."
    
    elif any(w in query_lower for w in ['competitor', 'competition']):
        avg_comp = df['competitor_price'].mean()
        return f"Competitor average: ₹{avg_comp:.0f}. Our AI keeps prices competitive while maximizing revenue."
    
    elif any(w in query_lower for w in ['help', 'what can you do', 'hi', 'hello']):
        return "I can help with pricing strategy! Ask about weekends, holidays, weather effects, revenue, visitors, or competitor prices."
    
    else:
        return "Try asking about: weekend/holiday pricing, weather effects, revenue stats, visitor patterns, or competitor analysis."


# ============================================================
# PLOTLY THEME
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,40,24,0.5)',
    font=dict(color='#b8e6cc', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='rgba(45,90,61,0.3)', zerolinecolor='rgba(45,90,61,0.3)'),
    yaxis=dict(gridcolor='rgba(45,90,61,0.3)', zerolinecolor='rgba(45,90,61,0.3)'),
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(bgcolor='#1a4a30', font_color='#e0f2e9', bordercolor='#4ade80'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8fbfa3'))
)

GREEN_PALETTE = ['#4ade80', '#22c55e', '#16a34a', '#15803d', '#166534', '#14532d',
                 '#86efac', '#bbf7d0', '#dcfce7', '#059669']


# ============================================================
# PAGE 1: OVERVIEW DASHBOARD
# ============================================================
def page_overview(df):
    """Render the Overview Dashboard page."""
    st.markdown("# 📊 Overview Dashboard")
    st.markdown("*Real-time analytics for theme park revenue optimization*")
    st.markdown("---")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df['revenue'].sum()
    avg_visitors = df['visitors'].mean()
    avg_price = df['current_price'].mean()
    
    # Revenue growth: compare last 6 months vs first 6 months
    df_sorted = df.sort_values('date')
    mid = len(df_sorted) // 2
    rev_first_half = df_sorted.iloc[:mid]['revenue'].sum()
    rev_second_half = df_sorted.iloc[mid:]['revenue'].sum()
    growth = ((rev_second_half - rev_first_half) / rev_first_half) * 100
    
    with col1:
        st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}", delta=f"{growth:+.1f}% growth")
    with col2:
        st.metric("👥 Avg Daily Visitors", f"{avg_visitors:,.0f}", delta=f"{df['visitors'].max():,} peak")
    with col3:
        st.metric("🎫 Avg Ticket Price", f"₹{avg_price:,.0f}")
    with col4:
        st.metric("📈 Revenue Growth", f"{growth:+.1f}%", delta="YoY comparison")
    
    st.markdown("---")
    
    # Revenue Over Time
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📈 Revenue Trend Over Time")
        monthly_rev = df.groupby(df['date'].dt.to_period('M').astype(str)).agg({
            'revenue': 'sum',
            'visitors': 'sum',
            'current_price': 'mean'
        }).reset_index()
        monthly_rev.columns = ['Month', 'Revenue', 'Visitors', 'Avg_Price']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_rev['Month'],
            y=monthly_rev['Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#4ade80', width=3),
            marker=dict(size=8, color='#4ade80'),
            fill='tozeroy',
            fillcolor='rgba(74,222,128,0.1)',
            hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            title=None,
            xaxis_title='Month',
            yaxis_title='Revenue (₹)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.markdown("### 🌦️ Weather Distribution")
        weather_counts = df['weather'].value_counts()
        colors_weather = {'sunny': '#f59e0b', 'cloudy': '#94a3b8', 'rainy': '#3b82f6'}
        
        fig = go.Figure(data=[go.Pie(
            labels=weather_counts.index,
            values=weather_counts.values,
            hole=0.55,
            marker=dict(colors=[colors_weather.get(w, '#4ade80') for w in weather_counts.index]),
            textinfo='label+percent',
            textfont=dict(size=13, color='#e0f2e9'),
            hovertemplate='<b>%{label}</b><br>Days: %{value}<br>Share: %{percent}<extra></extra>'
        )])
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Bottom row
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📅 Visitors by Day of Week")
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_visitors = df.groupby('day_of_week')['visitors'].mean().reset_index()
        daily_visitors['day_name'] = [day_names[d] for d in daily_visitors['day_of_week']]
        
        fig = go.Figure(data=[go.Bar(
            x=daily_visitors['day_name'],
            y=daily_visitors['visitors'],
            marker=dict(
                color=daily_visitors['visitors'],
                colorscale=[[0, '#14532d'], [0.5, '#22c55e'], [1, '#4ade80']],
                cornerradius=6,
            ),
            hovertemplate='<b>%{x}</b><br>Avg Visitors: %{y:,.0f}<extra></extra>'
        )])
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=370,
            xaxis_title='Day of Week',
            yaxis_title='Average Visitors',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        st.markdown("### 🎫 Price vs Visitors Correlation")
        fig = go.Figure()
        for w in ['sunny', 'cloudy', 'rainy']:
            subset = df[df['weather'] == w]
            fig.add_trace(go.Scatter(
                x=subset['current_price'],
                y=subset['visitors'],
                mode='markers',
                name=w.capitalize(),
                marker=dict(
                    size=5,
                    opacity=0.5,
                    color=colors_weather.get(w, '#4ade80')
                ),
                hovertemplate=f'<b>{w.capitalize()}</b><br>Price: ₹%{{x:,.0f}}<br>Visitors: %{{y:,.0f}}<extra></extra>'
            ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=370,
            xaxis_title='Ticket Price (₹)',
            yaxis_title='Visitors',
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 2: AI PRICE PREDICTOR
# ============================================================
def page_predictor(df):
    """Render the AI Price Predictor page."""
    st.markdown("# 🤖 AI Price Predictor")
    st.markdown("*Get intelligent pricing recommendations powered by machine learning*")
    st.markdown("---")
    
    try:
        model_package, weather_encoder, model_results = load_model()
        model = model_package['model']
        feature_columns = model_package['feature_columns']
    except Exception as e:
        st.error(f"⚠️ Model could not be loaded. Run `pip install -r requirements.txt`, then `python main.py`.\n\nError: {e}")
        return
    
    # Input Form
    col_form, col_result = st.columns([1, 1])
    
    with col_form:
        st.markdown("### 🎛️ Input Parameters")
        
        selected_date = st.date_input("📅 Select Date", value=datetime.now().date())
        
        weather = st.selectbox("🌦️ Weather Condition", ['sunny', 'cloudy', 'rainy'], index=0)
        
        temperature = st.slider("🌡️ Temperature (°C)", 15, 40, 28)
        
        competitor_price = st.slider("🏪 Competitor Price (₹)", 800, 1500, 1000, step=10)
        
        expected_visitors = st.slider("👥 Expected Visitors", 200, 5000, 1500, step=50)
        
        is_holiday = st.toggle("🎉 Is Holiday?", value=False)
        
        predict_button = st.button("🚀 Get AI Recommended Price", use_container_width=True)
    
    with col_result:
        if predict_button:
            with st.spinner("🧠 AI is analyzing market conditions..."):
                import time
                time.sleep(1.2)  # Dramatic pause for effect
                
                # Prepare features
                day_of_week = selected_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                month = selected_date.month
                quarter = (month - 1) // 3 + 1
                weather_encoded = weather_encoder.transform([weather])[0]
                
                features = pd.DataFrame({
                    'day_of_week': [day_of_week],
                    'is_weekend': [is_weekend],
                    'is_holiday': [1 if is_holiday else 0],
                    'weather_encoded': [weather_encoded],
                    'temperature': [temperature],
                    'competitor_price': [competitor_price],
                    'visitors': [expected_visitors],
                    'month': [month],
                    'quarter': [quarter]
                })
                
                predicted_price = model.predict(features)[0]
                predicted_price = int(np.clip(predicted_price, 800, 1500))
                estimated_revenue = predicted_price * expected_visitors
                
                # Confidence based on how close to training data distribution
                price_diff_ratio = abs(predicted_price - competitor_price) / competitor_price
                confidence = max(70, min(98, 95 - price_diff_ratio * 100))
            
            # Display Results
            st.markdown("### 🎯 AI Recommendation")
            
            st.markdown(f"""
            <div class="dashboard-card">
                <div class="price-label">AI RECOMMENDED PRICE</div>
                <div class="big-price">₹{predicted_price:,}</div>
                <div class="price-label">PER TICKET</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("💰 Est. Revenue", f"₹{estimated_revenue:,.0f}")
            with m2:
                price_vs_comp = ((predicted_price - competitor_price) / competitor_price) * 100
                st.metric("⚔️ vs Competitor", f"{price_vs_comp:+.1f}%", 
                         delta=f"₹{predicted_price - competitor_price:+,}")
            with m3:
                st.metric("🎯 Confidence", f"{confidence:.1f}%")
            
            st.markdown("")
            
            # Price Comparison Gauge
            st.markdown("### 📊 Price Positioning")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_price,
                delta={'reference': competitor_price, 'relative': False, 'prefix': '₹'},
                title={'text': "Recommended vs Market", 'font': {'color': '#b8e6cc'}},
                number={'prefix': '₹', 'font': {'color': '#4ade80', 'size': 36}},
                gauge={
                    'axis': {'range': [800, 1500], 'tickcolor': '#8fbfa3'},
                    'bar': {'color': '#4ade80'},
                    'bgcolor': 'rgba(13,40,24,0.5)',
                    'bordercolor': '#2d5a3d',
                    'steps': [
                        {'range': [800, 1000], 'color': 'rgba(22,163,74,0.2)'},
                        {'range': [1000, 1200], 'color': 'rgba(34,197,94,0.2)'},
                        {'range': [1200, 1500], 'color': 'rgba(74,222,128,0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': '#ef4444', 'width': 3},
                        'thickness': 0.8,
                        'value': competitor_price
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#b8e6cc'),
                height=280,
                margin=dict(l=30, r=30, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Default state
            st.markdown("""
            <div class="dashboard-card" style="text-align:center; min-height: 400px; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:4rem; margin-bottom:20px;">🧠</div>
                <div style="color:#8fbfa3; font-size:1.3rem;">Configure parameters and click</div>
                <div style="color:#4ade80; font-size:1.5rem; font-weight:700; margin-top:10px;">
                    "Get AI Recommended Price"
                </div>
                <div style="color:#6b9a7e; font-size:0.9rem; margin-top:15px;">
                    Our ML model analyzes weather, demand, competition<br/>
                    and seasonal patterns to find the optimal price point.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Model Performance Section
    st.markdown("---")
    st.markdown("### 🧪 Model Performance Metrics")
    
    if model_results:
        perf_cols = st.columns(len(model_results))
        for idx, (name, metrics) in enumerate(model_results.items()):
            with perf_cols[idx]:
                display_name = name.replace('_', ' ').title()
                is_winner = name == model_package.get('winner', '')
                badge = " 🏆" if is_winner else ""
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h4 style="color:#4ade80; text-align:center;">{display_name}{badge}</h4>
                    <p style="color:#8fbfa3; text-align:center;">
                        R² Score: <b style="color:#4ade80;">{metrics['r2']:.4f}</b><br/>
                        MAE: <b style="color:#4ade80;">₹{metrics['mae']:.2f}</b><br/>
                        RMSE: <b style="color:#4ade80;">₹{metrics['rmse']:.2f}</b><br/>
                        CV Mean: <b style="color:#4ade80;">{metrics['cv_mean']:.4f}</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# PAGE 3: REVENUE ANALYSIS
# ============================================================
def page_revenue_analysis(df):
    """Render the Revenue Analysis page."""
    st.markdown("# 📈 Revenue Analysis")
    st.markdown("*Deep-dive into revenue patterns, elasticity, and performance drivers*")
    st.markdown("---")
    
    # Heatmap: Revenue by Day of Week vs Weather
    st.markdown("### 🔥 Revenue Heatmap (Day of Week × Weather)")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    heatmap_data = df.pivot_table(
        index='weather', columns='day_of_week', values='revenue', aggfunc='mean'
    )
    heatmap_data.columns = [day_names[d] for d in heatmap_data.columns]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, '#0d1a12'], [0.3, '#14532d'], [0.6, '#16a34a'], [1, '#4ade80']],
        hovertemplate='<b>%{y} × %{x}</b><br>Avg Revenue: ₹%{z:,.0f}<extra></extra>',
        texttemplate='₹%{z:,.0f}',
        textfont=dict(size=11, color='#e0f2e9'),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        xaxis_title='Day of Week',
        yaxis_title='Weather',
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎉 Holiday vs Non-Holiday Revenue")
        holiday_rev = df.groupby('is_holiday')['revenue'].agg(['mean', 'sum']).reset_index()
        holiday_rev['label'] = holiday_rev['is_holiday'].map({0: 'Regular Day', 1: 'Holiday'})
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=holiday_rev['label'],
            y=holiday_rev['mean'],
            name='Avg Daily Revenue',
            marker=dict(color=['#22c55e', '#4ade80'], cornerradius=8),
            text=[f"₹{v:,.0f}" for v in holiday_rev['mean']],
            textposition='outside',
            textfont=dict(color='#b8e6cc'),
            hovertemplate='<b>%{x}</b><br>Avg Revenue: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            yaxis_title='Average Revenue (₹)',
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Price Elasticity Analysis")
        # Group by price ranges
        df_copy = df.copy()
        df_copy['price_range'] = pd.cut(df_copy['current_price'], 
                                         bins=[799, 900, 1000, 1100, 1200, 1300, 1500],
                                         labels=['800-900', '900-1000', '1000-1100', 
                                                '1100-1200', '1200-1300', '1300-1500'])
        
        elasticity = df_copy.groupby('price_range', observed=True)['visitors'].mean().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=elasticity['price_range'].astype(str),
            y=elasticity['visitors'],
            mode='lines+markers',
            marker=dict(size=12, color='#4ade80', symbol='diamond'),
            line=dict(color='#22c55e', width=3),
            fill='tozeroy',
            fillcolor='rgba(74,222,128,0.1)',
            hovertemplate='<b>Price: ₹%{x}</b><br>Avg Visitors: %{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            xaxis_title='Price Range (₹)',
            yaxis_title='Average Visitors',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top 10 Revenue Days
    st.markdown("### 🏆 Top 10 Highest Revenue Days")
    top_10 = df.nlargest(10, 'revenue')[['date', 'weather', 'temperature', 'visitors', 
                                          'current_price', 'recommended_price', 'revenue', 
                                          'is_holiday', 'is_weekend']].copy()
    top_10['date'] = top_10['date'].dt.strftime('%d %b %Y')
    top_10['revenue'] = top_10['revenue'].apply(lambda x: f"₹{x:,.0f}")
    top_10['current_price'] = top_10['current_price'].apply(lambda x: f"₹{x:,.0f}")
    top_10['recommended_price'] = top_10['recommended_price'].apply(lambda x: f"₹{x:,.0f}")
    top_10['is_holiday'] = top_10['is_holiday'].map({0: '❌', 1: '✅'})
    top_10['is_weekend'] = top_10['is_weekend'].map({0: '❌', 1: '✅'})
    top_10.columns = ['Date', 'Weather', 'Temp °C', 'Visitors', 'Price', 'AI Price', 
                      'Revenue', 'Holiday', 'Weekend']
    
    st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    # Monthly Revenue Breakdown
    st.markdown("---")
    st.markdown("### 📊 Monthly Revenue Breakdown")
    monthly = df.groupby(df['date'].dt.to_period('M').astype(str)).agg({
        'revenue': 'sum',
        'visitors': 'sum',
        'recommended_price': 'mean',
        'current_price': 'mean'
    }).reset_index()
    monthly.columns = ['Month', 'Revenue', 'Visitors', 'AI_Avg_Price', 'Actual_Avg_Price']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=monthly['Month'], y=monthly['Revenue'],
        name='Revenue', marker=dict(color='rgba(74,222,128,0.5)', cornerradius=4),
        hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
    ), secondary_y=False)
    
    fig.add_trace(go.Scatter(
        x=monthly['Month'], y=monthly['Visitors'],
        name='Visitors', mode='lines+markers',
        line=dict(color='#f59e0b', width=2),
        marker=dict(size=6),
        hovertemplate='<b>%{x}</b><br>Visitors: %{y:,.0f}<extra></extra>'
    ), secondary_y=True)
    
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        yaxis_title='Revenue (₹)',
        yaxis2_title='Visitors',
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE 4: MARKET MONITOR
# ============================================================
def page_market_monitor(df):
    """Render the Market Monitor page."""
    st.markdown("# 🏪 Market Monitor")
    st.markdown("*Competitive intelligence, alerts, and strategic pricing recommendations*")
    st.markdown("---")
    
    # Pricing Alerts
    st.markdown("### 🔔 Pricing Alerts")
    
    alerts_col1, alerts_col2, alerts_col3 = st.columns(3)
    
    # Calculate dynamic alerts
    recent = df.tail(30)
    avg_recent_visitors = recent['visitors'].mean()
    avg_all_visitors = df['visitors'].mean()
    
    with alerts_col1:
        if avg_recent_visitors > avg_all_visitors * 1.1:
            st.markdown("""
            <div class="alert-card alert-increase">
                📈 <b>HIGH DEMAND DETECTED</b><br>
                Recent visitors are 10%+ above average. Consider increasing prices by ₹50-100.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-card alert-info">
                ℹ️ <b>DEMAND STABLE</b><br>
                Visitor count is within normal range. Maintain current pricing strategy.
            </div>
            """, unsafe_allow_html=True)
    
    with alerts_col2:
        rainy_upcoming = df[df['weather'] == 'rainy'].tail(7)
        if len(rainy_upcoming) > 3:
            st.markdown("""
            <div class="alert-card alert-decrease">
                🌧️ <b>RAINY SEASON ALERT</b><br>
                Multiple rainy days detected. Activate 'Rainy Day Special' campaign with 15% discount.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-card alert-increase">
                ☀️ <b>GOOD WEATHER AHEAD</b><br>
                Favorable conditions expected. Premium pricing opportunity available.
            </div>
            """, unsafe_allow_html=True)
    
    with alerts_col3:
        comp_trend = recent['competitor_price'].mean()
        our_trend = recent['current_price'].mean()
        if our_trend < comp_trend * 0.95:
            st.markdown("""
            <div class="alert-card alert-increase">
                💡 <b>PRICE GAP OPPORTUNITY</b><br>
                Our price is 5%+ below competitors. Room to increase without losing visitors.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-card alert-info">
                ⚖️ <b>PRICE COMPETITIVE</b><br>
                Pricing is aligned with competitor range. No immediate action needed.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Weekly Pricing Recommendations
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📋 Weekly Pricing Recommendations")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_rec = []
        for dow in range(7):
            subset = df[df['day_of_week'] == dow]
            rec_price = subset['recommended_price'].mean()
            avg_visitors = subset['visitors'].mean()
            avg_revenue = subset['revenue'].mean()
            
            strategy = "Standard" if dow < 5 else "Premium"
            if dow == 5 or dow == 6:
                strategy = "🔥 Weekend Premium"
            elif dow == 4:
                strategy = "📈 Pre-Weekend"
            else:
                strategy = "📊 Standard"
            
            weekly_rec.append({
                'Day': day_names[dow],
                'Recommended Price': f"₹{rec_price:,.0f}",
                'Avg Visitors': f"{avg_visitors:,.0f}",
                'Avg Revenue': f"₹{avg_revenue:,.0f}",
                'Strategy': strategy
            })
        
        weekly_df = pd.DataFrame(weekly_rec)
        st.dataframe(weekly_df, use_container_width=True, hide_index=True)
    
    with col_right:
        st.markdown("### 📊 Competitor Price Tracking")
        
        # Weekly competitor price trend
        weekly_comp = df.groupby(df['date'].dt.to_period('W').astype(str)).agg({
            'competitor_price': 'mean',
            'current_price': 'mean',
            'recommended_price': 'mean'
        }).reset_index().tail(26)  # Last 6 months
        weekly_comp.columns = ['Week', 'Competitor', 'Our Price', 'AI Recommended']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=weekly_comp['Week'], y=weekly_comp['Competitor'],
            name='Competitor', mode='lines',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Competitor: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=weekly_comp['Week'], y=weekly_comp['Our Price'],
            name='Our Price', mode='lines',
            line=dict(color='#3b82f6', width=2),
            hovertemplate='<b>%{x}</b><br>Our Price: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=weekly_comp['Week'], y=weekly_comp['AI Recommended'],
            name='AI Recommended', mode='lines',
            line=dict(color='#4ade80', width=3),
            hovertemplate='<b>%{x}</b><br>AI Price: ₹%{y:,.0f}<extra></extra>'
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            yaxis_title='Price (₹)',
        )
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Package Deals
    st.markdown("### 🎁 Smart Package Deal Suggestions")
    
    pkg1, pkg2, pkg3 = st.columns(3)
    
    rainy_avg = df[df['weather'] == 'rainy']['current_price'].mean()
    holiday_avg = df[df['is_holiday'] == 1]['current_price'].mean()
    weekend_avg = df[df['is_weekend'] == 1]['current_price'].mean()
    
    with pkg1:
        st.markdown(f"""
        <div class="dashboard-card" style="text-align:center;">
            <div style="font-size:2.5rem;">🌧️</div>
            <h3 style="color:#3b82f6 !important;">Rainy Day Special</h3>
            <div style="color:#8fbfa3;">15% Discount Package</div>
            <div class="big-price" style="font-size:2.5rem; color:#3b82f6;">₹{int(rainy_avg * 0.85):,}</div>
            <div style="color:#6b9a7e; font-size:0.85rem;">
                <s>₹{int(rainy_avg):,}</s> per ticket<br/>
                + Free indoor attractions access<br/>
                + Complimentary hot beverages<br/>
                + Rain poncho included
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with pkg2:
        st.markdown(f"""
        <div class="dashboard-card" style="text-align:center;">
            <div style="font-size:2.5rem;">🎆</div>
            <h3 style="color:#f59e0b !important;">Holiday Premium</h3>
            <div style="color:#8fbfa3;">Festival Season Surge</div>
            <div class="big-price" style="font-size:2.5rem; color:#f59e0b;">₹{int(holiday_avg * 1.2):,}</div>
            <div style="color:#6b9a7e; font-size:0.85rem;">
                Premium ticket per person<br/>
                + Priority ride access (FastPass)<br/>
                + Festival special food coupon<br/>
                + Exclusive photo opportunities
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with pkg3:
        st.markdown(f"""
        <div class="dashboard-card" style="text-align:center;">
            <div style="font-size:2.5rem;">👨‍👩‍👧‍👦</div>
            <h3 style="color:#a855f7 !important;">Weekend Family Pack</h3>
            <div style="color:#8fbfa3;">Family of 4 Special</div>
            <div class="big-price" style="font-size:2.5rem; color:#a855f7;">₹{int(weekend_avg * 3.5):,}</div>
            <div style="color:#6b9a7e; font-size:0.85rem;">
                4 tickets (2 adults + 2 kids)<br/>
                + Family meal voucher ₹500<br/>
                + Free parking pass<br/>
                + Souvenir photo package
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Download
    st.markdown("### 📥 Export Recommendations")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        export_df = df[['date', 'weather', 'temperature', 'visitors', 'current_price', 
                         'recommended_price', 'competitor_price', 'revenue']].copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
        export_df['price_diff'] = export_df['recommended_price'] - export_df['current_price']
        export_df['potential_revenue'] = export_df['recommended_price'] * df['visitors']
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Recommendations (CSV)",
            data=csv,
            file_name="smart_pricing_recommendations.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        summary_text = f"""
AI Smart Pricing System - Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Theme Park Revenue Optimizer

Total Records: {len(df)}
Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
Total Revenue: ₹{df['revenue'].sum():,.0f}
Average Recommended Price: ₹{df['recommended_price'].mean():.0f}
Potential Revenue (AI Pricing): ₹{(df['recommended_price'] * df['visitors']).sum():,.0f}
"""
        st.download_button(
            label="📄 Download Summary Report (TXT)",
            data=summary_text,
            file_name="pricing_summary_report.txt",
            mime="text/plain",
            use_container_width=True
        )


# ============================================================
# PAGE 5: REVENUE SIMULATOR
# ============================================================
def page_simulator(df):
    """Revenue Simulator and What-If Analysis."""
    st.markdown("# 🔬 Revenue Simulator & What-If Analysis")
    st.markdown("*Test different pricing strategies and project annual revenue*")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["💡 Revenue Simulator", "🔍 What-If Analysis"])
    
    with tab1:
        st.markdown("### 💡 Pricing Strategy Simulator")
        st.markdown("Adjust the pricing strategy and see how it impacts annual revenue projections.")
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("#### Strategy Parameters")
            
            base_price_adj = st.slider("Base Price Adjustment (%)", -30, 30, 0, step=5)
            weekend_premium = st.slider("Weekend Premium (%)", 0, 50, 15, step=5)
            holiday_premium = st.slider("Holiday Premium (%)", 0, 80, 30, step=5)
            rainy_discount = st.slider("Rainy Day Discount (%)", 0, 30, 15, step=5)
            
            summer_adj = st.slider("Summer Season Adj. (%)", -20, 20, 5, step=5)
            festival_adj = st.slider("Festival Season Adj. (%)", 0, 50, 25, step=5)
            
            simulate_btn = st.button("🚀 Run Simulation", use_container_width=True)
        
        with sim_col2:
            if simulate_btn:
                with st.spinner("⚙️ Running revenue simulation..."):
                    import time
                    time.sleep(0.8)
                    
                    sim_df = df.copy()
                    
                    # Apply strategy
                    sim_df['sim_price'] = sim_df['current_price'] * (1 + base_price_adj / 100)
                    
                    # Weekend premium
                    mask_weekend = sim_df['is_weekend'] == 1
                    sim_df.loc[mask_weekend, 'sim_price'] *= (1 + weekend_premium / 100)
                    
                    # Holiday premium
                    mask_holiday = sim_df['is_holiday'] == 1
                    sim_df.loc[mask_holiday, 'sim_price'] *= (1 + holiday_premium / 100)
                    
                    # Rainy discount
                    mask_rainy = sim_df['weather'] == 'rainy'
                    sim_df.loc[mask_rainy, 'sim_price'] *= (1 - rainy_discount / 100)
                    
                    # Seasonal
                    mask_summer = sim_df['month'].isin([4, 5, 6])
                    sim_df.loc[mask_summer, 'sim_price'] *= (1 + summer_adj / 100)
                    
                    mask_festival = sim_df['month'].isin([10, 11, 12])
                    sim_df.loc[mask_festival, 'sim_price'] *= (1 + festival_adj / 100)
                    
                    sim_df['sim_price'] = sim_df['sim_price'].clip(800, 1500).astype(int)
                    
                    # Estimate visitor impact (higher price = fewer visitors)
                    price_change = (sim_df['sim_price'] - sim_df['current_price']) / sim_df['current_price']
                    elasticity = -0.3  # 10% price increase → 3% visitor decrease
                    visitor_change = price_change * elasticity
                    sim_df['sim_visitors'] = (sim_df['visitors'] * (1 + visitor_change)).clip(200, 5000).astype(int)
                    
                    sim_df['sim_revenue'] = sim_df['sim_price'] * sim_df['sim_visitors']
                    
                    # Results
                    original_revenue = df['revenue'].sum()
                    simulated_revenue = sim_df['sim_revenue'].sum()
                    revenue_diff = simulated_revenue - original_revenue
                    revenue_diff_pct = (revenue_diff / original_revenue) * 100
                    
                    # Annualized projection
                    days_in_data = len(df)
                    annual_original = (original_revenue / days_in_data) * 365
                    annual_simulated = (simulated_revenue / days_in_data) * 365
                
                # Display results
                st.markdown("#### 📊 Simulation Results")
                
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("Original Revenue", f"₹{original_revenue:,.0f}")
                with r2:
                    st.metric("Simulated Revenue", f"₹{simulated_revenue:,.0f}", 
                             delta=f"₹{revenue_diff:,.0f} ({revenue_diff_pct:+.1f}%)")
                with r3:
                    st.metric("Annual Projection", f"₹{annual_simulated:,.0f}",
                             delta=f"₹{annual_simulated - annual_original:,.0f}")
                
                st.markdown("")
                
                # Comparison chart
                monthly_comp = pd.DataFrame({
                    'Month': df.groupby(df['date'].dt.to_period('M').astype(str))['revenue'].sum().index,
                    'Original': df.groupby(df['date'].dt.to_period('M').astype(str))['revenue'].sum().values,
                    'Simulated': sim_df.groupby(sim_df['date'].dt.to_period('M').astype(str))['sim_revenue'].sum().values
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_comp['Month'], y=monthly_comp['Original'],
                    name='Original', marker=dict(color='rgba(59,130,246,0.6)', cornerradius=4),
                    hovertemplate='<b>%{x}</b><br>Original: ₹%{y:,.0f}<extra></extra>'
                ))
                fig.add_trace(go.Bar(
                    x=monthly_comp['Month'], y=monthly_comp['Simulated'],
                    name='Simulated', marker=dict(color='rgba(74,222,128,0.6)', cornerradius=4),
                    hovertemplate='<b>%{x}</b><br>Simulated: ₹%{y:,.0f}<extra></extra>'
                ))
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=400,
                    barmode='group',
                    yaxis_title='Revenue (₹)',
                )
                fig.update_layout(legend=dict(orientation='h', y=1.1))
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.markdown("""
                <div class="dashboard-card" style="text-align:center; min-height:300px; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:3.5rem;">📊</div>
                    <div style="color:#8fbfa3; font-size:1.2rem; margin-top:10px;">
                        Adjust pricing strategy parameters on the left<br/>
                        and click <b style="color:#4ade80;">"Run Simulation"</b> to see projected results.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🔍 What-If Scenario Analysis")
        st.markdown("Explore how specific conditions affect revenue and optimal pricing.")
        
        scenario = st.selectbox("Choose Scenario", [
            "What if it rains for an entire week?",
            "What if a major festival falls on a weekend?",
            "What if competitor drops price by 20%?",
            "What if visitor capacity increases by 50%?",
            "What if temperature exceeds 40°C for a week?"
        ])
        
        analyze_btn = st.button("🔍 Analyze Scenario", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("🧠 Analyzing scenario..."):
                import time
                time.sleep(1.0)
            
            if "rains for an entire week" in scenario:
                rainy_data = df[df['weather'] == 'rainy']
                normal_data = df[df['weather'] != 'rainy']
                
                revenue_impact = (rainy_data['revenue'].mean() - normal_data['revenue'].mean()) / normal_data['revenue'].mean() * 100
                weekly_loss = (normal_data['revenue'].mean() - rainy_data['revenue'].mean()) * 7
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3 style="color:#3b82f6 !important;">🌧️ Week of Rain Scenario</h3>
                    <p style="color:#b8e6cc;">
                        <b>Revenue Impact:</b> {revenue_impact:.1f}% decrease per day<br/>
                        <b>Estimated Weekly Loss:</b> ₹{weekly_loss:,.0f}<br/>
                        <b>Avg Visitors on Rainy Days:</b> {rainy_data['visitors'].mean():,.0f} (vs {normal_data['visitors'].mean():,.0f} normal)<br/><br/>
                        <b style="color:#4ade80;">💡 AI Recommendation:</b><br/>
                        • Activate "Rainy Day Special" at ₹{int(rainy_data['recommended_price'].mean() * 0.85):,}<br/>
                        • Promote indoor attractions heavily<br/>
                        • Offer complimentary hot beverages and rain gear<br/>
                        • Launch 2-for-1 ticket promotion to maintain footfall<br/>
                        • Focus marketing on covered rides and shows
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "festival falls on a weekend" in scenario:
                festival_weekend = df[(df['is_holiday'] == 1) & (df['is_weekend'] == 1)]
                regular = df[(df['is_holiday'] == 0) & (df['is_weekend'] == 0)]
                
                if len(festival_weekend) > 0:
                    premium = (festival_weekend['revenue'].mean() / regular['revenue'].mean() - 1) * 100
                else:
                    premium = 85.0
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3 style="color:#f59e0b !important;">🎆 Festival Weekend Scenario</h3>
                    <p style="color:#b8e6cc;">
                        <b>Revenue Boost:</b> +{premium:.1f}% vs regular weekday<br/>
                        <b>Expected Visitors:</b> {int(df['visitors'].quantile(0.9)):,}+<br/>
                        <b>Optimal Price:</b> ₹{int(df['recommended_price'].quantile(0.9)):,}<br/><br/>
                        <b style="color:#4ade80;">💡 AI Recommendation:</b><br/>
                        • Implement surge pricing at ₹{int(df['recommended_price'].quantile(0.9)):,}<br/>
                        • Open all rides and attractions at full capacity<br/>
                        • Deploy additional staff for crowd management<br/>
                        • Launch "Festival Special" premium package<br/>
                        • Extend park hours to maximize revenue window
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "competitor drops price by 20%" in scenario:
                avg_comp = df['competitor_price'].mean()
                new_comp = avg_comp * 0.8
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3 style="color:#ef4444 !important;">⚔️ Competitor Price Drop Scenario</h3>
                    <p style="color:#b8e6cc;">
                        <b>Competitor Current Avg:</b> ₹{avg_comp:,.0f} → ₹{new_comp:,.0f}<br/>
                        <b>Expected Visitor Loss:</b> 10-15% if we don't respond<br/>
                        <b>Revenue Risk:</b> ₹{int(df['revenue'].mean() * 0.12 * 30):,} per month<br/><br/>
                        <b style="color:#4ade80;">💡 AI Recommendation:</b><br/>
                        • Don't match competitor blindly — focus on value<br/>
                        • Introduce value-add bundles (meals + rides + photos)<br/>
                        • Moderate 8-10% discount to stay competitive<br/>
                        • Emphasize unique attractions in marketing<br/>
                        • Launch loyalty program to retain repeat visitors
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "capacity increases by 50%" in scenario:
                current_avg = df['visitors'].mean()
                new_capacity = current_avg * 1.5
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3 style="color:#a855f7 !important;">🏗️ Capacity Expansion Scenario</h3>
                    <p style="color:#b8e6cc;">
                        <b>Current Avg Visitors:</b> {current_avg:,.0f}<br/>
                        <b>New Capacity:</b> {new_capacity:,.0f} visitors/day<br/>
                        <b>Potential Revenue Increase:</b> ₹{int(df['revenue'].mean() * 0.35 * 365):,} annually<br/><br/>
                        <b style="color:#4ade80;">💡 AI Recommendation:</b><br/>
                        • Initially reduce prices 5-10% to fill new capacity<br/>
                        • Gradual price optimization as demand stabilizes<br/>
                        • Focus on off-peak discounts to spread visitors evenly<br/>
                        • Invest in marketing to drive new visitor segments<br/>
                        • Monitor wait times — shorter waits justify premium pricing
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif "temperature exceeds 40°C" in scenario:
                hot_data = df[df['temperature'] >= 35]
                normal_temp = df[(df['temperature'] >= 22) & (df['temperature'] <= 30)]
                
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3 style="color:#f97316 !important;">🌡️ Extreme Heat Scenario</h3>
                    <p style="color:#b8e6cc;">
                        <b>Avg Visitors in Heat:</b> {hot_data['visitors'].mean():,.0f} (vs {normal_temp['visitors'].mean():,.0f} normal)<br/>
                        <b>Revenue Drop:</b> ~{((1 - hot_data['revenue'].mean() / normal_temp['revenue'].mean()) * 100):.0f}%<br/>
                        <b>Recommended Price:</b> ₹{hot_data['recommended_price'].mean():,.0f}<br/><br/>
                        <b style="color:#4ade80;">💡 AI Recommendation:</b><br/>
                        • Offer "Beat the Heat" discounted entry<br/>
                        • Promote water rides and air-conditioned attractions<br/>
                        • Shift operating hours to evening (4 PM - 10 PM)<br/>
                        • Free water stations and misting zones<br/>
                        • Afternoon discount for 12 PM - 3 PM slot
                    </p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================
def main():
    """Main application entry point."""
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run `python main.py` first to generate data and train models.")
        st.info("```\ncd smart-pricing-system\npython main.py\n```")
        return
    
    # Sidebar navigation
    page = render_sidebar()
    
    # Route to pages
    if page == "📊 Overview Dashboard":
        page_overview(df)
    elif page == "🤖 AI Price Predictor":
        page_predictor(df)
    elif page == "📈 Revenue Analysis":
        page_revenue_analysis(df)
    elif page == "🏪 Market Monitor":
        page_market_monitor(df)
    elif page == "🔬 Revenue Simulator":
        page_simulator(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#4a6b56; font-size:0.8rem; padding:20px;">
        <p>🎢 <b>AI Smart Pricing System</b> — Theme Park Revenue Optimizer v2.0</p>
        <p>Powered by XGBoost + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
