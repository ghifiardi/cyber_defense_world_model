# Cybersecurity World Model

A comprehensive AI-powered proactive defense system for predicting and preventing cyber attacks before they occur using world model techniques.

## Overview

The Cybersecurity World Model is a machine learning system that:

- **Predicts attacks** before they happen using temporal fusion transformers
- **Detects anomalies** in network behavior using self-supervised learning
- **Generates attack graphs** showing potential attack paths through your network
- **Orchestrates proactive defenses** based on predicted threats

## Architecture

The system is organized into the following components:

```
cybersecurity_world_model/
├── core/              # Core world model (encoder, dynamics, world model)
├── defense/           # Proactive defense components
│   ├── predictors.py      # Temporal attack prediction
│   ├── detectors.py        # Anomaly detection
│   ├── graph_generator.py   # Attack graph generation
│   └── orchestrator.py     # Main defense orchestrator
├── training/          # Training utilities
├── simulation/        # Simulation environments
├── config/            # Configuration management
├── utils/             # Utility functions (logging, etc.)
└── integrations/      # External system integrations
```

## Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Verify installation:**

```python
import cybersecurity_world_model
print(cybersecurity_world_model.__version__)
```

## Quick Start

### Training a Model

Train the cybersecurity world model on simulated network data:

```bash
python train.py --epochs 50 --batch-size 8 --checkpoint-dir checkpoints/
```

Options:
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 4)
- `--sequence-length`: Length of training sequences (default: 100)
- `--checkpoint-dir`: Directory to save model checkpoints (default: checkpoints/)
- `--config`: Path to configuration file (optional)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Running Predictions

Predict attacks from telemetry data:

```bash
python predict.py --telemetry data/telemetry.csv --forecast-hours 24
```

Options:
- `--telemetry`: Path to telemetry data file (CSV format)
- `--forecast-hours`: Hours to forecast ahead (default: 24)
- `--config`: Path to configuration file (optional)
- `--log-level`: Logging level

If no telemetry file is provided, the system will generate sample data for demonstration.

## Usage Examples

### Basic Usage

```python
from cybersecurity_world_model import CyberWorldModel, PredictiveDefenseOrchestrator
from cybersecurity_world_model.config import Config
import pandas as pd
import numpy as np

# Initialize with configuration
config = Config()
orchestrator = PredictiveDefenseOrchestrator(config=config)

# Load or generate telemetry data
telemetry_data = pd.DataFrame(np.random.randn(100, 256))

# Predict attacks
results = orchestrator.predict_attacks(telemetry_data, forecast_hours=24)

# Access results
print(f"Confidence: {results['confidence_level']:.2%}")
print(f"Warnings: {len(results['early_warnings'])}")
for warning in results['early_warnings']:
    print(f"  [{warning['level']}] {warning['type']}")
```

### Training a Custom Model

```python
from cybersecurity_world_model import CyberWorldModel
from cybersecurity_world_model.simulation import NetworkSimulator
from cybersecurity_world_model.training import train_cybersecurity_world_model

# Initialize model
model = CyberWorldModel(
    feature_dim=256,
    latent_dim=256,
    action_dim=50
)

# Initialize simulator
simulator = NetworkSimulator(grid_size=24, feature_dim=256)

# Train
history = train_cybersecurity_world_model(
    model=model,
    simulator=simulator,
    epochs=50,
    batch_size=8,
    sequence_length=100,
    checkpoint_dir='checkpoints/'
)
```

### Using the Core World Model

```python
from cybersecurity_world_model import CyberWorldModel
import torch

# Initialize model
model = CyberWorldModel()

# Simulate an attack scenario
initial_state = {
    'network': torch.randn(1, 256),
    'flows': torch.randn(1, 10, 20),
    'events': torch.randn(1, 10, 64)
}

attack_sequence = [
    {'action': 0},  # RECONNAISSANCE
    {'action': 1},  # INITIAL_ACCESS
    {'action': 2},  # EXECUTION
]

states, predictions = model.simulate_attack_scenario(
    initial_state, 
    attack_sequence
)

print(f"Simulated {len(states)} states")
print(f"Generated {len(predictions)} predictions")
```

## Configuration

Configuration can be provided via:

1. **YAML file:**
```yaml
model:
  latent_dim: 256
  feature_dim: 256
  device: cuda

training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 50

defense:
  forecast_horizon_hours: 24
  warning_thresholds:
    high_confidence: 0.85
    anomaly_threshold: 0.8
```

2. **Environment variables:**
```bash
export CWM_MODEL_DEVICE=cuda
export CWM_TRAINING_BATCH_SIZE=8
export CWM_LOG_LEVEL=INFO
```

3. **Python code:**
```python
from cybersecurity_world_model.config import Config

config = Config()
config.set('model.latent_dim', 512)
config.set('training.batch_size', 16)
```

## Components

### Core World Model

- **CyberStateEncoder**: Encodes network state into latent representation
- **AttackDynamicsModel**: Predicts next network state given current state and attack action
- **CyberWorldModel**: Complete world model orchestrator

### Defense Components

- **TemporalAttackPredictor**: Multi-horizon attack prediction using temporal fusion
- **BehavioralAnomalyDetector**: Self-supervised anomaly detection
- **AttackGraphGenerator**: Generates probabilistic attack graphs
- **PredictiveDefenseOrchestrator**: Main orchestrator for proactive defense

## Project Structure

The project follows the planned architecture:

```
proactive-defense-system/
├── cybersecurity_world_model/    # Main package
├── train.py                       # Training entry point
├── predict.py                     # Prediction entry point
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Development

### Running Tests

```bash
# Run basic import test
python -c "import cybersecurity_world_model; print('OK')"
```

### Code Structure

- All core components are in `cybersecurity_world_model/`
- Entry points are in the root directory
- Configuration and utilities are modular and reusable
- Logging is structured and configurable

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- NetworkX >= 3.0
- PyYAML >= 6.0

See `requirements.txt` for complete list.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here]


