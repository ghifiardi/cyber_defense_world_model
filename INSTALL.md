# Installation Guide

## Quick Install

If you're in a virtual environment (venv), install dependencies:

```bash
pip install -r requirements.txt
```

Or if `pip` is not available, try:

```bash
python -m pip install -r requirements.txt
```

Or:

```bash
pip3 install -r requirements.txt
```

## Manual Installation

If you prefer to install dependencies one by one:

```bash
# Core ML Framework
pip install torch torchvision

# Data Processing
pip install numpy pandas scikit-learn

# Network Analysis
pip install networkx

# Visualization
pip install matplotlib

# Configuration (Required for YAML support)
pip install pyyaml

# Utilities
pip install python-dateutil
```

## Verify Installation

After installation, verify everything works:

```bash
python -c "import cybersecurity_world_model; print('✓ Installation successful!')"
```

## Common Issues

### ModuleNotFoundError: No module named 'yaml'

This means PyYAML is not installed. Install it with:

```bash
pip install pyyaml
```

### ModuleNotFoundError: No module named 'torch'

Install PyTorch:

```bash
pip install torch torchvision
```

For CUDA support, visit: https://pytorch.org/get-started/locally/

### Virtual Environment Issues

If you're using a virtual environment, make sure it's activated:

```bash
# Activate venv (Linux/Mac)
source venv/bin/activate

# Activate venv (Windows)
venv\Scripts\activate
```

## Development Setup

For development, you may also want:

```bash
pip install pytest  # For testing
pip install black   # For code formatting
pip install flake8  # For linting
```


