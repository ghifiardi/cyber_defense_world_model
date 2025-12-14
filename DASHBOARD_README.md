# Cybersecurity World Model Dashboard

## Overview

Interactive real-time dashboard for monitoring the Cybersecurity World Model predictions, attack scenarios, and proactive defense recommendations.

## Features

### 📊 Key Metrics
- **Prediction Confidence**: Overall confidence level of predictions
- **Anomalies Detected**: Number of behavioral anomalies
- **Early Warnings**: Active security alerts
- **Attack Scenarios**: Number of simulated attack scenarios

### 🎯 Attack Scenarios Tab
- Detailed view of each predicted attack scenario
- Use case information (ID, category, expert type)
- Attack chain visualization
- Timeline of attack phases
- Threat level gauges

### 📈 Predictions Tab
- Predicted attack probabilities (bar chart)
- Confidence trends over forecast horizon
- Detailed prediction table

### ⚠️ Early Warnings Tab
- Real-time security alerts
- Alert severity levels (CRITICAL, HIGH, MEDIUM, LOW)
- Recommended actions for each warning
- Time windows for predicted attacks

### 🛡️ Recommendations Tab
- Immediate actions (next 24 hours)
- Short-term actions (next week)
- Long-term actions (next month)
- Priority matrix visualization

### 📊 Network State Tab
- Network activity metrics
- Traffic and event timelines
- Scenario distribution by category
- Real-time telemetry visualization

## Installation

### Install Dashboard Dependencies

```bash
pip install streamlit plotly
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

### Start the Dashboard

**Option 1: Using the run script (Recommended)**
```bash
./run_dashboard.sh
```

Or specify a custom port:
```bash
./run_dashboard.sh 8503
```

**Option 2: Direct Streamlit command**
```bash
streamlit run dashboard.py --server.port 8502
```

**Option 3: Custom port**
```bash
streamlit run dashboard.py --server.port 8503
```

The dashboard will open in your default web browser at `http://localhost:8502` (or your specified port)

### Port Configuration

The dashboard uses port **8502** by default (instead of Streamlit's default 8501) to avoid conflicts.

To use a different port:
- **Method 1**: Pass port as argument: `./run_dashboard.sh 8503`
- **Method 2**: Use Streamlit flag: `streamlit run dashboard.py --server.port 8503`
- **Method 3**: Edit `.streamlit/config.toml` and change the port number

### Dashboard Workflow

1. **Initialize Models**: Click "🔄 Initialize Models" in the sidebar
2. **Run Prediction**: Click "🔮 Run Prediction" to generate predictions
3. **View Results**: Navigate through tabs to see different views
4. **Auto-refresh**: Enable auto-refresh for continuous monitoring

## Dashboard Components

### Sidebar Controls
- **Initialize Models**: Load the world model and orchestrator
- **Run Prediction**: Generate new predictions
- **Auto-refresh**: Automatically refresh every 30 seconds
- **Settings**: Adjust forecast horizon and threat thresholds
- **Status**: View last update time and key metrics

### Main Views

#### 1. Attack Scenarios
- Expandable cards for each scenario
- Threat level gauges
- Attack chain visualization
- Timeline with phase descriptions

#### 2. Predictions
- Bar chart of predicted attacks
- Confidence trend over time
- Detailed prediction data table

#### 3. Early Warnings
- Color-coded alert levels
- Attack type and time windows
- Recommended actions
- Alert rationale

#### 4. Recommendations
- Categorized by timeframe
- Priority matrix visualization
- Actionable defense steps

#### 5. Network State
- Real-time network metrics
- Activity timeline charts
- Scenario distribution
- Telemetry visualization

## Customization

### Adjusting Forecast Horizon

In the sidebar, use the slider to adjust the forecast horizon (12-72 hours).

### Threat Thresholds

Adjust the "High Threat Threshold" slider to change what's considered high-risk (40-80%).

### Auto-Refresh

Enable auto-refresh to continuously monitor predictions. The dashboard will refresh every 30 seconds.

## Integration with Real Data

To connect to real data sources, modify the data generation functions:

```python
def generate_sample_telemetry(days=7):
    # Replace with actual SIEM/EDR data
    # Example: Query Splunk, QRadar, or data lake
    pass
```

## Screenshots

The dashboard includes:
- Interactive charts using Plotly
- Real-time metrics
- Color-coded threat levels
- Expandable scenario cards
- Priority visualizations

## Troubleshooting

### Dashboard won't start
- Ensure Streamlit is installed: `pip install streamlit`
- Check Python version (3.8+)
- **Port already in use**: Try a different port: `streamlit run dashboard.py --server.port 8503`
- Check if another Streamlit instance is running: `lsof -i :8502` (or your port)

### Models won't initialize
- Ensure all dependencies are installed
- Check that threat models directory exists

### No predictions shown
- Click "Run Prediction" after initializing models
- Check console for error messages

## Next Steps

1. **Connect Real Data**: Integrate with SIEM/EDR APIs
2. **Add Alerts**: Configure email/Slack notifications
3. **Export Reports**: Add PDF/CSV export functionality
4. **Historical Analysis**: Add historical prediction tracking
5. **User Authentication**: Add login/role-based access

## Related Documentation

- [README.md](README.md) - Main project documentation
- [USE_CASE_README.md](USE_CASE_README.md) - Use case documentation
- [THREAT_MODELS_INTEGRATION.md](THREAT_MODELS_INTEGRATION.md) - Threat models integration

