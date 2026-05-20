"""
Streamlit Dashboard - Interactive web interface for GNN electricity price forecasting

Run with: streamlit run gnn_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json

from gnn_predict import GNNPredictor, evaluate_predictions
from gnn_utils import (
    calculate_savings, 
    find_optimal_charging_window,
    generate_recommendations,
    analyze_hourly_patterns
)
from gnn_config import ARTIFACTS_DIR


# ────────────────────────────────────────────────────────────────
# Page Configuration
# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# Initialize Session State
# ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    """Load the GNN predictor (cached)."""
    try:
        return GNNPredictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# ────────────────────────────────────────────────────────────────
# Main App
# ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown('<div class="main-header">⚡ GNN Electricity Price Forecasting</div>', 
                unsafe_allow_html=True)
    
    st.markdown("### 🇩🇰 Danish Electricity Spot Prices - DK1 Zone")
    st.markdown("---")
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("❌ Model not loaded. Please run the training pipeline first.")
        st.code("python gnn_pipeline.py")
        return
    
    # ────────────────────────────────────────────────────────────
    # Sidebar Configuration
    # ────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Settings")
    
    # Forecast horizon
    forecast_hours = st.sidebar.slider(
        "Forecast Horizon (hours)",
        min_value=6,
        max_value=72,
        value=24,
        step=6
    )
    
    # Consumption profile
    st.sidebar.markdown("### 🔌 Consumption Profile")
    
    ev_kwh = st.sidebar.number_input(
        "EV Charging (kWh)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )
    
    ev_duration = st.sidebar.slider(
        "EV Charging Duration (hours)",
        min_value=1,
        max_value=12,
        value=4,
        step=1
    )
    
    # ────────────────────────────────────────────────────────────
    # Get Predictions
    # ────────────────────────────────────────────────────────────
    with st.spinner("🔮 Generating predictions..."):
        future_df = predictor.predict_future(num_hours=forecast_hours)
        test_df = predictor.predict_test_set()
    
    # ────────────────────────────────────────────────────────────
    # Tab Layout
    # ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast", 
        "💰 Savings Analysis", 
        "📊 Model Performance",
        "💡 Recommendations"
    ])
    
    # ═══════════════════════════════════════════════════════════
    # TAB 1: FORECAST
    # ═══════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"### 🔮 Next {forecast_hours} Hours Forecast")
        
        # Price statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Price",
                f"{future_df['predicted_price'].mean():.4f} DKK/kWh"
            )
        
        with col2:
            st.metric(
                "Minimum Price",
                f"{future_df['predicted_price'].min():.4f} DKK/kWh"
            )
        
        with col3:
            st.metric(
                "Maximum Price",
                f"{future_df['predicted_price'].max():.4f} DKK/kWh"
            )
        
        with col4:
            price_range = future_df['predicted_price'].max() - future_df['predicted_price'].min()
            st.metric(
                "Price Range",
                f"{price_range:.4f} DKK/kWh"
            )
        
        # Line chart
        st.markdown("#### Price Forecast Over Time")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=future_df['timestamp'],
            y=future_df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        # Highlight cheapest hour
        cheapest_idx = future_df['predicted_price'].idxmin()
        fig.add_trace(go.Scatter(
            x=[future_df.loc[cheapest_idx, 'timestamp']],
            y=[future_df.loc[cheapest_idx, 'predicted_price']],
            mode='markers',
            name='Cheapest Hour',
            marker=dict(size=15, color='green', symbol='star')
        ))
        
        # Highlight most expensive hour
        expensive_idx = future_df['predicted_price'].idxmax()
        fig.add_trace(go.Scatter(
            x=[future_df.loc[expensive_idx, 'timestamp']],
            y=[future_df.loc[expensive_idx, 'predicted_price']],
            mode='markers',
            name='Most Expensive Hour',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Price (DKK/kWh)",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cheapest and most expensive hours
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💚 Top 5 Cheapest Hours")
            cheapest = future_df.nsmallest(5, 'predicted_price')[['timestamp', 'predicted_price']]
            cheapest['timestamp'] = cheapest['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            cheapest['predicted_price'] = cheapest['predicted_price'].round(6)
            st.dataframe(cheapest, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 💸 Top 5 Most Expensive Hours")
            expensive = future_df.nlargest(5, 'predicted_price')[['timestamp', 'predicted_price']]
            expensive['timestamp'] = expensive['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            expensive['predicted_price'] = expensive['predicted_price'].round(6)
            st.dataframe(expensive, hide_index=True, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════
    # TAB 2: SAVINGS ANALYSIS
    # ═══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 💰 Potential Savings Analysis")
        
        # General savings
        savings = calculate_savings(future_df, consumption_kwh=ev_kwh)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Cost at Cheapest Hour",
                f"{savings['cost_at_min']:.2f} DKK",
                help=f"For {ev_kwh} kWh at {savings['min_price']:.4f} DKK/kWh"
            )
        
        with col2:
            st.metric(
                "Cost at Average Time",
                f"{savings['cost_at_avg']:.2f} DKK",
                help=f"For {ev_kwh} kWh at {savings['avg_price']:.4f} DKK/kWh"
            )
        
        with col3:
            st.metric(
                "Potential Savings",
                f"{savings['savings_vs_avg']:.2f} DKK",
                f"{savings['savings_percentage_vs_avg']:.1f}%",
                help="Savings by charging at cheapest vs average time"
            )
        
        # EV Charging window optimization
        st.markdown(f"#### 🔌 Optimal EV Charging Window ({ev_duration} hours)")
        
        if len(future_df) >= ev_duration:
            window = find_optimal_charging_window(
                future_df, 
                duration_hours=ev_duration,
                consumption_kwh=ev_kwh
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### ✅ Best Charging Window")
                st.markdown(f"**Start:** {window['optimal_window']['start_time']}")
                st.markdown(f"**End:** {window['optimal_window']['end_time']}")
                st.markdown(f"**Avg Price:** {window['optimal_window']['avg_price']:.4f} DKK/kWh")
                st.markdown(f"**Total Cost:** {window['optimal_window']['total_cost']:.2f} DKK")
            
            with col2:
                st.markdown("##### ❌ Worst Charging Window")
                st.markdown(f"**Start:** {window['worst_window']['start_time']}")
                st.markdown(f"**End:** {window['worst_window']['end_time']}")
                st.markdown(f"**Avg Price:** {window['worst_window']['avg_price']:.4f} DKK/kWh")
                st.markdown(f"**Total Cost:** {window['worst_window']['total_cost']:.2f} DKK")
            
            st.success(f"💰 Save **{window['potential_savings']:.2f} DKK** ({window['savings_percentage']:.1f}%) by charging at optimal time!")
        else:
            st.warning(f"Need at least {ev_duration} hours of predictions")
        
        # Hourly pattern analysis
        st.markdown("#### ⏰ Price by Hour of Day")
        
        hourly_stats = analyze_hourly_patterns(future_df)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_stats['hour'],
            y=hourly_stats['mean'],
            name='Average Price',
            marker_color='lightblue',
            error_y=dict(type='data', array=hourly_stats['std'])
        ))
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Price (DKK/kWh)",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════
    # TAB 3: MODEL PERFORMANCE
    # ═══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 📊 Model Performance Metrics")
        
        # Calculate metrics
        metrics = evaluate_predictions(test_df)
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.6f}", help="Mean Absolute Error (DKK/kWh)")
            
            with col2:
                st.metric("RMSE", f"{metrics['RMSE']:.6f}", help="Root Mean Squared Error (DKK/kWh)")
            
            with col3:
                st.metric("R²", f"{metrics['R2']:.4f}", help="Coefficient of Determination")
            
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%", help="Mean Absolute Percentage Error")
            
            # Predictions vs Actual
            st.markdown("#### Predictions vs Actual (Test Set)")
            
            # Sample test data for visualization (last 100 points)
            sample_df = test_df.tail(100)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sample_df['timestamp'],
                y=sample_df['actual'],
                mode='lines',
                name='Actual',
                line=dict(color='green', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=sample_df['timestamp'],
                y=sample_df['predicted'],
                mode='lines',
                name='Predicted',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Price (DKK/kWh)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            st.markdown("#### Prediction Accuracy Scatter")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=test_df['actual'],
                y=test_df['predicted'],
                mode='markers',
                name='Predictions',
                marker=dict(size=5, opacity=0.6)
            ))
            
            # Add diagonal line
            min_val = min(test_df['actual'].min(), test_df['predicted'].min())
            max_val = max(test_df['actual'].max(), test_df['predicted'].max())
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title="Actual Price (DKK/kWh)",
                yaxis_title="Predicted Price (DKK/kWh)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════
    # TAB 4: RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### 💡 Smart Usage Recommendations")
        
        consumption_profile = {
            'ev_charging': ev_kwh,
            'dishwasher': 1.5,
            'washing_machine': 2.0,
            'dryer': 3.0,
            'water_heater': 4.0
        }
        
        recommendations = generate_recommendations(future_df, consumption_profile)
        
        for rec in recommendations:
            activity_icons = {
                'ev_charging': '🚗',
                'dishwasher': '🍽️',
                'washing_machine': '🧺',
                'dryer': '👕',
                'water_heater': '🚿'
            }
            
            icon = activity_icons.get(rec['activity'], '⚡')
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>{icon} {rec['activity'].replace('_', ' ').title()}</h4>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <p><strong>Estimated Cost:</strong> {rec['estimated_cost']:.2f} DKK 
                   ({rec['consumption_kwh']} kWh)</p>
                <p><strong>Potential Savings:</strong> {rec['potential_savings']:.2f} DKK 
                   vs average timing</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download data
        st.markdown("### 📥 Download Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = future_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = future_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # ────────────────────────────────────────────────────────────
    # Footer
    # ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>
        ⚡ Powered by Graph Neural Networks | 
        Data from Energy-Charts API & Open-Meteo | 
        🇩🇰 Denmark DK1 Zone
        </small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()