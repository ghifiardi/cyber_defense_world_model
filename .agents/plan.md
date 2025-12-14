# Goal & Non-Goals

## Goal
Organize the Cybersecurity World Model project into a structured, production-ready architecture following the planned directory structure. Integrate all components into a cohesive system with proper dependencies, error handling, and observability.

## Non-Goals
- Rewriting core ML models (preserve existing implementations)
- Creating full UI dashboard (focus on backend structure)
- Implementing all integrations (create framework/stubs)
- Full test suite (add basic structure and examples)
- Complete deployment automation (structure for future deployment)

# Perubahan per File

## File Organization & Structure

### 1. Create requirements.txt
- Lokasi: Root directory
- Perubahan: Create new file with all Python dependencies
- Mengapa: Missing dependency management identified in research
- Dampak: Enables reproducible environment setup

### 2. Create main package structure
- Lokasi: Create `cybersecurity_world_model/` package directory
- Perubahan: 
  - `__init__.py` with version and exports
  - `core/` subpackage for world model components
  - `defense/` subpackage for defense components
  - `training/` subpackage for training utilities
  - `simulation/` subpackage for simulators
  - `utils/` subpackage for shared utilities
- Mengapa: Organize code according to Project Structure.txt [L1-35]
- Dampak: Enables proper imports and modularity

### 3. Refactor Core Architecture Cyber_Threat_World_Model.py
- Lokasi: Move to `cybersecurity_world_model/core/world_model.py`
- Perubahan:
  - Split into separate modules: encoder.py, dynamics.py, world_model.py
  - Add error handling and logging
  - Add type hints
  - Create unified interface
- Mengapa: Core component needs proper structure [research: Core Models section]
- Dampak: Other components depend on this

### 4. Refactor AI-Powered Proactive Defense System.py
- Lokasi: Move to `cybersecurity_world_model/defense/` subpackage
- Perubahan:
  - Split PredictiveDefenseOrchestrator into orchestrator.py
  - Move TemporalAttackPredictor to predictors.py
  - Move BehavioralAnomalyDetector to detectors.py
  - Move AttackGraphGenerator to graph_generator.py
  - Add configuration management
  - Add structured logging
- Mengapa: Main orchestrator needs modular structure [research: Prediction Flow]
- Dampak: Core integration point for defense system

### 5. Integrate Training Components
- Lokasi: `cybersecurity_world_model/training/`
- Perubahan:
  - Move Train Cybersecurity World Model.py to `trainer.py`
  - Move Network Simulator.py to `simulation/network_simulator.py`
  - Create training configuration class
  - Add checkpoint saving/loading
  - Add training metrics logging
- Mengapa: Training flow needs integration [research: Training Flow]
- Dampak: Enables reproducible model training

### 6. Integrate Defense Components
- Lokasi: `cybersecurity_world_model/defense/`
- Perubahan:
  - Move Threat Hunting and Prediction.py to `threat_hunting.py`
  - Move Automated Incident Response.py to `incident_response.py`
  - Move Blue Team Defense Optimization.py to `defense_optimizer.py`
  - Create unified defense API
- Mengapa: Defense components need integration [research: Defense Optimization]
- Dampak: Enables coordinated defense strategies

### 7. Create Configuration System
- Lokasi: `cybersecurity_world_model/config/`
- Perubahan:
  - Create `config.py` with Config class
  - Support YAML/JSON configuration files
  - Environment variable overrides
  - Default configurations
- Mengapa: Hardcoded values need centralization [research: Risks]
- Dampak: All components can use shared config

### 8. Add Logging Infrastructure
- Lokasi: `cybersecurity_world_model/utils/logging.py`
- Perubahan:
  - Structured logging setup
  - Log levels configuration
  - File and console handlers
  - Metrics collection hooks
- Mengapa: Missing observability [research: Missing Observability]
- Dampak: All components can log properly

### 9. Create Main Entry Points
- Lokasi: Root directory
- Perubahan:
  - `train.py` - Training entry point
  - `predict.py` - Prediction entry point
  - `simulate.py` - Simulation entry point
  - `main.py` - Full system orchestration
- Mengapa: Need clear entry points [research: How to Run]
- Dampak: Users can run system easily

### 10. Create Integration Framework
- Lokasi: `cybersecurity_world_model/integrations/`
- Perubahan:
  - Base connector interface
  - SIEM connector stubs
  - EDR connector stubs
  - Cloud log connector stubs
  - Factory pattern for connectors
- Mengapa: Integration structure needed [research: Deployment Uncertainty]
- Dampak: Enables real-world integration

### 11. Add Error Handling
- Lokasi: Throughout codebase
- Perubahan:
  - Add try-catch blocks in critical paths
  - Custom exception classes in `cybersecurity_world_model/exceptions.py`
  - Graceful degradation patterns
  - Input validation
- Mengapa: No error handling identified [research: Risks]
- Dampak: System robustness

### 12. Create README.md
- Lokasi: Root directory
- Perubahan:
  - Project overview
  - Installation instructions
  - Usage examples
  - Architecture diagram reference
  - API documentation links
- Mengapa: Missing documentation [research: How to Run]
- Dampak: User onboarding

### 13. Create .gitignore
- Lokasi: Root directory
- Perubahan:
  - Python patterns
  - Model checkpoints
  - Data files
  - IDE files
  - Environment files
- Mengapa: Version control hygiene
- Dampak: Clean repository

# Urutan Eksekusi (Step 1..n + "uji cepat" per step)

## Step 1: Create Foundation
- Create requirements.txt with dependencies
- Create .gitignore
- Create main package structure with __init__.py files
- Uji cepat: `python -c "import cybersecurity_world_model; print('OK')"`

## Step 2: Setup Utilities
- Create config system
- Create logging infrastructure
- Create exception classes
- Uji cepat: Import and instantiate config, verify logging works

## Step 3: Refactor Core World Model
- Split Core Architecture into modules
- Add error handling and logging
- Update imports
- Uji cepat: Import core modules, instantiate CyberWorldModel

## Step 4: Refactor Defense System
- Split AI-Powered Proactive Defense into modules
- Integrate with core world model
- Add configuration support
- Uji cepat: Import PredictiveDefenseOrchestrator, instantiate

## Step 5: Integrate Training
- Move training components
- Create trainer with checkpoint support
- Integrate with network simulator
- Uji cepat: Import trainer, verify it can be instantiated

## Step 6: Integrate Defense Components
- Move defense components to defense package
- Create unified API
- Integrate with orchestrator
- Uji cepat: Import all defense components

## Step 7: Create Entry Points
- Create train.py, predict.py, simulate.py, main.py
- Wire up components
- Add CLI argument parsing
- Uji cepat: Run `python train.py --help`, `python predict.py --help`

## Step 8: Create Integration Framework
- Create base connector interface
- Create stub connectors
- Integrate with orchestrator
- Uji cepat: Import connectors, verify interface

## Step 9: Documentation
- Create README.md
- Add docstrings to key classes
- Create example usage scripts
- Uji cepat: Verify README instructions work

## Step 10: Final Integration Test
- Run full system test
- Verify all imports work
- Test basic training flow
- Test basic prediction flow
- Uji cepat: End-to-end smoke test

# Acceptance Criteria (incl. edge-cases)

1. **Package Structure**: All code organized in `cybersecurity_world_model/` package with proper subpackages
2. **Imports Work**: All components can be imported without errors
3. **Configuration**: System can be configured via config files or environment variables
4. **Logging**: All components log to structured logger
5. **Error Handling**: Critical paths have try-catch with meaningful errors
6. **Entry Points**: train.py, predict.py, simulate.py, main.py all work
7. **Dependencies**: requirements.txt includes all necessary packages
8. **Documentation**: README.md provides clear setup and usage instructions
9. **Edge Cases**:
   - Missing config file → uses defaults
   - Invalid input data → raises clear error
   - Missing dependencies → clear error message
   - Empty telemetry data → handles gracefully
   - Model file not found → clear error message

# Rollback & Guardrails (feature flag/circuit breaker)

1. **Feature Flags**: Use config to enable/disable components (e.g., `enable_automated_response: false`)
2. **Circuit Breakers**: Add circuit breakers for external integrations (SIEM, EDR)
3. **Checkpointing**: Training saves checkpoints, can resume from last checkpoint
4. **Dry Run Mode**: Prediction mode can run without taking actions
5. **Logging Levels**: Can reduce logging verbosity for production
6. **Rollback Strategy**: Keep original files in backup/ directory during refactoring

# Risiko Sisa & Mitigasi

1. **Risk**: Breaking existing functionality during refactoring
   - **Mitigation**: Keep original files, test after each step, incremental changes

2. **Risk**: Missing dependencies in requirements.txt
   - **Mitigation**: Test in clean environment, document all imports

3. **Risk**: Import path issues after reorganization
   - **Mitigation**: Use relative imports where possible, test imports at each step

4. **Risk**: Configuration complexity
   - **Mitigation**: Provide sensible defaults, clear documentation

5. **Risk**: Performance regression
   - **Mitigation**: Preserve original implementations, only reorganize structure

6. **Risk**: Integration connectors incomplete
   - **Mitigation**: Create clear interface, document expected behavior, provide stubs

