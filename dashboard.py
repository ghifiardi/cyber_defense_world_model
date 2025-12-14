#!/usr/bin/env python3
"""
Cybersecurity World Model - Real-Time Monitoring Dashboard

Interactive dashboard for monitoring:
- Real-time attack predictions
- Attack scenario timelines
- Threat levels and confidence scores
- Early warnings and alerts
- Defense recommendations
- Network state visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from pathlib import Path

from cybersecurity_world_model import CyberWorldModel, PredictiveDefenseOrchestrator
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.threat_models import ThreatModelLoader, ThreatScenarioGenerator
from cybersecurity_world_model.utils.logging import setup_logging

# Page configuration
st.set_page_config(
    page_title="Cybersecurity World Model Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .warning-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .warning-low {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'world_model' not in st.session_state:
    st.session_state.world_model = None

@st.cache_resource
def initialize_models():
    """Initialize models (cached to avoid re-initialization)."""
    config = Config()
    world_model = CyberWorldModel(
        feature_dim=config.get('model.feature_dim', 256),
        latent_dim=config.get('model.latent_dim', 256),
        action_dim=config.get('model.action_dim', 50)
    )
    orchestrator = PredictiveDefenseOrchestrator(config=config)
    return world_model, orchestrator

def generate_sample_telemetry(days=7):
    """Generate sample telemetry data."""
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=days * 24,
        freq='h'
    )
    data = np.random.randn(len(timestamps), 256)
    # Add some attack patterns
    data[-48:, 10:15] += np.linspace(0, 2, 48).reshape(-1, 1)
    data[-24:, 50:55] += np.random.choice([0, 3], size=(24, 5), p=[0.7, 0.3])
    return pd.DataFrame(data, index=timestamps)

def generate_current_network_state():
    """Generate current network state."""
    import torch
    return {
        'network': torch.randn(1, 256),
        'flows': torch.randn(1, 10, 20),
        'events': torch.randn(1, 10, 64),
        'timestamp': datetime.now()
    }

def run_prediction():
    """Run prediction and update session state."""
    with st.spinner("Running prediction analysis..."):
        # Generate telemetry
        telemetry_data = generate_sample_telemetry(days=7)
        
        # Run prediction
        predictions = st.session_state.orchestrator.predict_attacks(
            telemetry_data=telemetry_data,
            forecast_hours=48
        )
        
        # Generate scenarios
        threat_generator = ThreatScenarioGenerator()
        threat_scenarios = threat_generator.generate_multiple_scenarios(max_scenarios=5)
        
        # Simulate scenarios (simplified for dashboard)
        scenarios = []
        for i, threat_scenario in enumerate(threat_scenarios):
            scenario_info = {
                'scenario_id': i + 1,
                'use_case_id': threat_scenario.get('scenario_id', ''),
                'scenario_name': threat_scenario.get('scenario_name', ''),
                'category': threat_scenario.get('category', ''),
                'expert_type': threat_scenario.get('expert_type', ''),
                'attack_sequence': threat_scenario.get('attack_sequence', []),
                'threat_level': np.random.uniform(0.45, 0.55),  # Simulated for dashboard
                'num_steps': len(threat_scenario.get('attack_sequence', []))
            }
            scenarios.append(scenario_info)
        
        st.session_state.predictions = predictions
        st.session_state.scenarios = scenarios
        st.session_state.last_update = datetime.now()

def main():
    """Main dashboard function."""
    
    # Header
    st.markdown('<div class="main-header">🛡️ Cybersecurity World Model Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Real-Time Proactive Attack Prediction & Monitoring")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Initialize models button
        if st.button("🔄 Initialize Models", use_container_width=True):
            with st.spinner("Initializing..."):
                world_model, orchestrator = initialize_models()
                st.session_state.world_model = world_model
                st.session_state.orchestrator = orchestrator
            st.success("Models initialized!")
        
        # Run prediction button
        if st.session_state.orchestrator is None:
            st.warning("Please initialize models first")
        else:
            if st.button("🔮 Run Prediction", use_container_width=True, type="primary"):
                run_prediction()
                st.success("Prediction completed!")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        # Settings
        st.header("📊 Settings")
        forecast_hours = st.slider("Forecast Horizon (hours)", 12, 72, 48)
        threat_threshold = st.slider("High Threat Threshold (%)", 40, 80, 50)
        
        # Status
        st.header("📈 Status")
        if st.session_state.last_update:
            st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
        else:
            st.info("No predictions yet")
        
        if st.session_state.predictions:
            st.metric("Confidence", f"{st.session_state.predictions['confidence_level']:.1%}")
            st.metric("Scenarios", len(st.session_state.scenarios))
    
    # Main content area
    if st.session_state.orchestrator is None:
        st.info("👈 Please initialize models from the sidebar to begin")
        return
    
    if st.session_state.predictions is None:
        st.warning("⚠️ No prediction data available. Click 'Run Prediction' to generate predictions.")
        if st.button("Generate Sample Prediction"):
            run_prediction()
        return
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Key Metrics Row
    st.header("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        confidence = st.session_state.predictions['confidence_level']
        confidence_color = "🟢" if confidence > 0.7 else "🟠" if confidence > 0.5 else "🔴"
        st.metric("Prediction Confidence", f"{confidence:.1%}", delta=None)
        st.caption(f"{confidence_color} {'High' if confidence > 0.7 else 'Medium' if confidence > 0.5 else 'Low'}")
    
    with col2:
        anomalies = st.session_state.predictions['anomalies_detected']
        st.metric("Anomalies Detected", anomalies, delta=None)
        st.caption("🟡 Behavioral anomalies")
    
    with col3:
        warnings = len(st.session_state.predictions.get('early_warnings', []))
        st.metric("Early Warnings", warnings, delta=None)
        st.caption("⚠️ Active alerts")
    
    with col4:
        scenarios_count = len(st.session_state.scenarios)
        high_threat = sum(1 for s in st.session_state.scenarios if s.get('threat_level', 0) > threat_threshold/100)
        st.metric("Attack Scenarios", scenarios_count, delta=f"{high_threat} high-risk")
        st.caption("🎯 Simulated scenarios")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Attack Scenarios", 
        "📈 Predictions", 
        "⚠️ Early Warnings", 
        "🛡️ Recommendations",
        "📊 Network State"
    ])
    
    with tab1:
        st.header("🎯 Predicted Attack Scenarios")
        
        # Scenario cards
        for i, scenario in enumerate(st.session_state.scenarios):
            threat_level = scenario.get('threat_level', 0.5)
            threat_pct = threat_level * 100
            
            # Determine warning class
            if threat_pct > 70:
                warning_class = "warning-high"
            elif threat_pct > 40:
                warning_class = "warning-medium"
            else:
                warning_class = "warning-low"
            
            with st.expander(
                f"Scenario {scenario['scenario_id']}: {scenario.get('scenario_name', 'Unknown')} "
                f"({threat_pct:.1f}% Threat)",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Use Case:** {scenario.get('use_case_id', 'N/A')}")
                    st.markdown(f"**Category:** {scenario.get('category', 'N/A')}")
                    st.markdown(f"**Expert Type:** {scenario.get('expert_type', 'N/A')}")
                    
                    # Attack chain visualization
                    attack_chain = " → ".join([step['name'] for step in scenario.get('attack_sequence', [])])
                    st.markdown(f"**Attack Chain:** {attack_chain}")
                    
                    # Timeline
                    st.markdown("**Timeline:**")
                    current_time = datetime.now()
                    for j, step in enumerate(scenario.get('attack_sequence', [])):
                        step_time = current_time + timedelta(hours=j*2)
                        st.markdown(f"  • {step_time.strftime('%H:%M')} - {step['name']}: {step.get('description', '')}")
                
                with col2:
                    # Threat level gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = threat_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Threat Level"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📈 Attack Predictions")
        
        # Predicted attacks chart
        if st.session_state.predictions.get('predicted_attacks'):
            pred_data = st.session_state.predictions['predicted_attacks']
            df_pred = pd.DataFrame(pred_data)
            
            # Bar chart
            fig = px.bar(
                df_pred,
                x='attack_type',
                y='probability',
                title='Predicted Attack Probabilities',
                labels={'attack_type': 'Attack Type', 'probability': 'Probability'},
                color='probability',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Table
            st.dataframe(df_pred, use_container_width=True)
        else:
            st.info("No predicted attacks above threshold")
        
        # Confidence over time (simulated)
        st.subheader("Confidence Trend")
        hours = list(range(48))
        confidence_trend = [st.session_state.predictions['confidence_level'] + np.random.uniform(-0.1, 0.1) for _ in hours]
        
        fig = px.line(
            x=hours,
            y=confidence_trend,
            title='Prediction Confidence Over Forecast Horizon',
            labels={'x': 'Hours Ahead', 'y': 'Confidence'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("⚠️ Early Warnings & Alerts")
        
        warnings = st.session_state.predictions.get('early_warnings', [])
        
        if warnings:
            for warning in warnings:
                level = warning.get('level', 'MEDIUM')
                level_color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }.get(level, '⚪')
                
                st.markdown(f"### {level_color} {level}: {warning.get('type', 'Unknown')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Attack Type:** {warning.get('attack_type', 'N/A')}")
                    st.markdown(f"**Time Window:** {warning.get('predicted_time_window', 'N/A')}")
                with col2:
                    st.markdown(f"**Confidence:** {warning.get('confidence', 0):.1%}")
                    st.markdown(f"**Rationale:** {warning.get('rationale', 'N/A')}")
                
                st.markdown("**Recommended Actions:**")
                for action in warning.get('recommended_actions', [])[:3]:
                    st.markdown(f"  • {action}")
                
                st.divider()
        else:
            st.success("✅ No active warnings at this time")
    
    with tab4:
        st.header("🛡️ Defense Recommendations")
        
        recommendations = st.session_state.predictions.get('defense_recommendations', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🚨 Immediate Actions")
            st.markdown("*(Next 24 hours)*")
            for i, rec in enumerate(recommendations.get('immediate', []), 1):
                st.markdown(f"{i}. {rec}")
            if not recommendations.get('immediate'):
                st.info("No immediate actions required")
        
        with col2:
            st.subheader("📅 Short-Term Actions")
            st.markdown("*(Next week)*")
            for i, rec in enumerate(recommendations.get('short_term', []), 1):
                st.markdown(f"{i}. {rec}")
        
        with col3:
            st.subheader("🔮 Long-Term Actions")
            st.markdown("*(Next month)*")
            for i, rec in enumerate(recommendations.get('long_term', []), 1):
                st.markdown(f"{i}. {rec}")
        
        # Recommendation priority matrix
        st.subheader("Priority Matrix")
        
        # Create priority data
        priority_data = []
        for category, recs in recommendations.items():
            for rec in recs:
                priority = {
                    'immediate': 3,
                    'short_term': 2,
                    'long_term': 1
                }.get(category, 1)
                priority_data.append({
                    'Recommendation': rec[:50] + '...' if len(rec) > 50 else rec,
                    'Priority': priority,
                    'Category': category.replace('_', ' ').title()
                })
        
        if priority_data:
            df_priority = pd.DataFrame(priority_data)
            fig = px.scatter(
                df_priority,
                x='Category',
                y='Priority',
                size=[10]*len(df_priority),
                hover_data=['Recommendation'],
                title='Recommendation Priority Distribution',
                labels={'Priority': 'Priority Level', 'Category': 'Timeframe'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📊 Network State & Telemetry")
        
        # Generate sample network metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Connections", "1,234", delta="+12%")
        with col2:
            st.metric("Traffic Volume", "2.4 TB", delta="+5%")
        with col3:
            st.metric("Security Events", "156", delta="-8%")
        with col4:
            st.metric("Threat Indicators", "23", delta="+3")
        
        # Network activity timeline
        st.subheader("Network Activity Timeline")
        hours = pd.date_range(end=datetime.now(), periods=24, freq='h')
        activity_data = {
            'Time': hours,
            'Traffic': np.random.randn(24) * 100 + 500,
            'Events': np.random.randn(24) * 10 + 50,
            'Threats': np.random.randn(24) * 2 + 5
        }
        df_activity = pd.DataFrame(activity_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_activity['Time'],
            y=df_activity['Traffic'],
            name='Traffic',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=df_activity['Time'],
            y=df_activity['Events'] * 10,
            name='Security Events (×10)',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=df_activity['Time'],
            y=df_activity['Threats'] * 50,
            name='Threat Indicators (×50)',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Network Activity Over Last 24 Hours',
            xaxis_title='Time',
            yaxis_title='Activity Level',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario distribution
        st.subheader("Scenario Distribution by Category")
        if st.session_state.scenarios:
            category_counts = {}
            for scenario in st.session_state.scenarios:
                category = scenario.get('category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            fig = px.pie(
                values=list(category_counts.values()),
                names=list(category_counts.keys()),
                title='Attack Scenarios by Category'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_update else 'Never'} | "
        f"Cybersecurity World Model v1.0"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()

