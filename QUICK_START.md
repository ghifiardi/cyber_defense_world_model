# Quick Start - Dashboard

Since you're already in a virtual environment, you can start the dashboard directly:

## Option 1: Direct Command (Easiest)

```bash
# Make sure streamlit is installed
pip install streamlit plotly

# Start dashboard on port 8503
streamlit run dashboard.py --server.port 8503
```

## Option 2: Use the Script

The script has been updated to work with your active venv:

```bash
./start_dashboard.sh 8503
```

## Option 3: Any Port

```bash
streamlit run dashboard.py --server.port 8504
```

Then open: `http://localhost:8504`

## Check if Streamlit is Installed

```bash
python -c "import streamlit; print('Streamlit installed!')"
```

If not installed:
```bash
pip install streamlit plotly
```

