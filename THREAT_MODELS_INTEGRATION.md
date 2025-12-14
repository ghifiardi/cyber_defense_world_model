# Threat Models Integration

## Overview

The Cybersecurity World Model now integrates with real-world threat modeling use cases from the IOH Documentation SCC Transformation Project. This integration enables the system to simulate attacks based on actual threat intelligence and documented attack patterns.

## Integrated Use Cases

The system includes the following threat model use cases:

### UC-011: Ransomware Early Warning & Group Monitoring
- **Category:** Advanced Threats
- **Expert Type:** Threat Intelligence Expert
- **MITRE Tactics:** TA0001, TA0009, TA0010, TA0040
- **Description:** Proactive detection of ransomware campaigns targeting telecom sector, with 7-30 day early warning window

### UC-010: APT Malware Signature & Behaviour Detection
- **Category:** Advanced Threats
- **Expert Type:** Malware Analysis Expert
- **MITRE Tactics:** TA0002, TA0003, TA0005, TA0011
- **Description:** Detection of APT malware families (SEASPY, SALTWATER, AngryRebel.Linux, BIRDROBAT, etc.)

### UC-013: Vulnerability Exploit Forecast & Patching Prioritization
- **Category:** Vulnerability Management
- **Expert Type:** Vulnerability Intelligence Expert
- **MITRE Tactics:** TA0043, TA0001
- **Description:** Predict which vulnerabilities are likely to be exploited within 7-30 days

## Usage

### Basic Usage

The use case automatically loads threat models when available:

```python
python use_case_proactive_prediction.py
```

The system will:
1. Load threat models from `threat_models/` directory
2. Generate scenarios based on real threat intelligence
3. Fall back to default scenarios if threat models are unavailable

### Programmatic Usage

```python
from cybersecurity_world_model.threat_models import ThreatModelLoader, ThreatScenarioGenerator

# Load threat models
loader = ThreatModelLoader()

# Get specific use case
uc011 = loader.get_use_case('UC011')
print(f"Use Case: {uc011['title']}")

# Generate scenario from use case
generator = ThreatScenarioGenerator(loader)
ransomware_scenario = generator.generate_ransomware_scenario()

# Get attack sequence
attack_sequence = loader.get_attack_sequence_from_use_case('UC011')
```

### Available Methods

#### ThreatModelLoader

- `get_use_case(use_case_id)`: Get use case by ID
- `get_use_cases_by_category(category)`: Get all use cases in a category
- `get_use_cases_by_expert(expert_type)`: Get use cases for an expert type
- `get_mitre_tactics(use_case_id)`: Get MITRE tactics for a use case
- `get_attack_sequence_from_use_case(use_case_id)`: Extract attack sequence
- `list_all_use_cases()`: Get all available use cases

#### ThreatScenarioGenerator

- `generate_scenario_from_use_case(use_case_id)`: Generate scenario from use case
- `generate_ransomware_scenario()`: Generate ransomware attack scenario
- `generate_apt_scenario()`: Generate APT malware scenario
- `generate_vulnerability_exploit_scenario()`: Generate vulnerability exploit scenario
- `generate_multiple_scenarios(max_scenarios=5)`: Generate multiple scenarios

## Threat Model Structure

Threat models are stored in the `threat_models/` directory:

```
threat_models/
├── usecases_manifest.json          # Use cases index
├── UC-011_Ransomware_Early_Warning.md
├── UC-010_APT_Malware_Detection.md
└── UC-013_Vulnerability_Exploit_Forecast_Complete.md
```

## Integration with World Model

The threat models are automatically integrated into the proactive prediction use case:

1. **Scenario Generation**: Real threat models are used to generate attack scenarios
2. **Attack Sequences**: MITRE ATT&CK tactics from threat models map to world model actions
3. **Realistic Timelines**: Attack phases based on documented threat intelligence
4. **Context-Aware Predictions**: Predictions consider actual threat actor behaviors

## Benefits

1. **Real-World Accuracy**: Scenarios based on documented threat intelligence
2. **Industry-Specific**: Use cases tailored for telecom/infrastructure sectors
3. **Comprehensive Coverage**: Multiple threat categories (ransomware, APT, vulnerabilities)
4. **MITRE ATT&CK Alignment**: All scenarios map to MITRE ATT&CK framework
5. **Expert-Driven**: Scenarios designed by security experts for specific threat types

## Adding New Threat Models

To add new threat models:

1. Add the use case markdown file to `threat_models/` directory
2. Update `usecases_manifest.json` with the new use case entry
3. Add attack descriptions to `ThreatScenarioGenerator.ATTACK_DESCRIPTIONS` if needed
4. The system will automatically load and use the new threat model

## Example Output

When using threat models, the output will show:

```
--- Scenario 1 ---
Source: UC011 - Ransomware Early Warning & Group Monitoring
Attack Chain: RECONNAISSANCE -> INITIAL_ACCESS -> COLLECTION -> EXFILTRATION -> IMPACT
Predicted Steps: 5
Average Threat Level: 52.3%
```

This indicates the scenario is based on the real UC-011 threat model rather than a generic scenario.

## Related Documentation

- [USE_CASE_README.md](USE_CASE_README.md) - Main use case documentation
- [README.md](README.md) - Project overview
- Threat model files in `threat_models/` directory

