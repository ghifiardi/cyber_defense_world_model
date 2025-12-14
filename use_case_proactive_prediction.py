#!/usr/bin/env python3
"""
Use Case: Proactive Attack Prediction Using World Model

This use case demonstrates how the Cybersecurity World Model can:
1. Analyze current network state and telemetry
2. Simulate potential attack scenarios that haven't occurred yet
3. Predict upcoming attacks before they happen
4. Generate proactive defense recommendations
5. Visualize attack paths and timelines

Scenario: A security team wants to anticipate potential attacks
based on current network patterns and threat intelligence.
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from cybersecurity_world_model import CyberWorldModel, PredictiveDefenseOrchestrator
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = setup_logging(log_level="INFO")


def generate_current_network_state() -> Dict[str, torch.Tensor]:
    """
    Generate a simulated current network state.
    In production, this would come from real network monitoring tools.
    """
    # Simulate network topology data
    network_tensor = torch.randn(1, 256)  # Network state features
    
    # Simulate traffic flow data (last 10 timesteps)
    flow_tensor = torch.randn(1, 10, 20)  # (batch, seq_len, features)
    
    # Simulate security events (last 10 timesteps)
    event_tensor = torch.randn(1, 10, 64)  # (batch, seq_len, features)
    
    # Add some suspicious patterns (simulating early indicators)
    # Increased scanning activity
    flow_tensor[:, -3:, 5:8] += 2.0
    # Unusual authentication attempts
    event_tensor[:, -2:, 10:15] += 1.5
    # New outbound connections
    flow_tensor[:, -1:, 12:15] += 1.8
    
    return {
        'network': network_tensor,
        'flows': flow_tensor,
        'events': event_tensor,
        'timestamp': datetime.now()
    }


def generate_telemetry_data(days: int = 7) -> pd.DataFrame:
    """
    Generate historical telemetry data for analysis.
    In production, this would come from SIEM, EDR, or network monitoring tools.
    """
    # Generate time series data
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=days * 24,  # Hourly data
        freq='h'  # Use lowercase 'h' instead of 'H' to avoid deprecation warning
    )
    
    # Generate 256 features (network metrics, security events, etc.)
    data = np.random.randn(len(timestamps), 256)
    
    # Add some patterns that might indicate upcoming attacks:
    # 1. Gradual increase in reconnaissance activity (last 2 days)
    data[-48:, 10:15] += np.linspace(0, 2, 48).reshape(-1, 1)
    
    # 2. Spikes in failed authentication attempts (last day)
    data[-24:, 50:55] += np.random.choice([0, 3], size=(24, 5), p=[0.7, 0.3])
    
    # 3. Unusual network traffic patterns (last 12 hours)
    data[-12:, 100:110] += 1.5
    
    # 4. New service discovery attempts (last 6 hours)
    data[-6:, 150:155] += 2.0
    
    telemetry_df = pd.DataFrame(data, index=timestamps)
    return telemetry_df


def simulate_attack_scenarios(
    world_model: CyberWorldModel,
    current_state: Dict[str, torch.Tensor],
    num_scenarios: int = 5
) -> List[Dict[str, Any]]:
    """
    Use the world model to simulate potential attack scenarios.
    These are attacks that haven't happened yet but could occur.
    """
    print("\n" + "="*70)
    print("SIMULATING POTENTIAL ATTACK SCENARIOS")
    print("="*70)
    
    scenarios = []
    
    # Generate multiple attack scenarios
    for i in range(num_scenarios):
        print(f"\n--- Scenario {i+1} ---")
        
        # Define potential attack sequence based on MITRE ATT&CK
        # This represents a potential kill chain that could unfold
        attack_sequence = []
        
        # Scenario variations
        if i == 0:
            # Scenario 1: Phishing -> Initial Access -> Execution -> Persistence
            attack_sequence = [
                {'action': 0, 'name': 'RECONNAISSANCE', 'description': 'Email reconnaissance and target research'},
                {'action': 1, 'name': 'INITIAL_ACCESS', 'description': 'Phishing email with malicious attachment'},
                {'action': 2, 'name': 'EXECUTION', 'description': 'Malware execution from attachment'},
                {'action': 3, 'name': 'PERSISTENCE', 'description': 'Registry modification for persistence'}
            ]
        elif i == 1:
            # Scenario 2: Web Exploit -> Privilege Escalation -> Lateral Movement
            attack_sequence = [
                {'action': 0, 'name': 'RECONNAISSANCE', 'description': 'Web application scanning'},
                {'action': 1, 'name': 'INITIAL_ACCESS', 'description': 'Exploit public-facing web application'},
                {'action': 4, 'name': 'PRIVILEGE_ESCALATION', 'description': 'Exploit local privilege escalation'},
                {'action': 8, 'name': 'LATERAL_MOVEMENT', 'description': 'Move to internal network segments'}
            ]
        elif i == 2:
            # Scenario 3: Credential Theft -> Discovery -> Collection -> Exfiltration
            attack_sequence = [
                {'action': 6, 'name': 'CREDENTIAL_ACCESS', 'description': 'Credential dumping from memory'},
                {'action': 7, 'name': 'DISCOVERY', 'description': 'Network and system discovery'},
                {'action': 9, 'name': 'COLLECTION', 'description': 'Data collection from multiple sources'},
                {'action': 10, 'name': 'EXFILTRATION', 'description': 'Data exfiltration to external server'}
            ]
        elif i == 3:
            # Scenario 4: Ransomware attack chain
            attack_sequence = [
                {'action': 0, 'name': 'RECONNAISSANCE', 'description': 'Network mapping and vulnerability scanning'},
                {'action': 1, 'name': 'INITIAL_ACCESS', 'description': 'RDP brute force or exploit'},
                {'action': 8, 'name': 'LATERAL_MOVEMENT', 'description': 'Spread across network'},
                {'action': 12, 'name': 'IMPACT', 'description': 'Encrypt files and demand ransom'}
            ]
        else:
            # Scenario 5: Advanced Persistent Threat (APT) style
            attack_sequence = [
                {'action': 0, 'name': 'RECONNAISSANCE', 'description': 'Long-term reconnaissance'},
                {'action': 1, 'name': 'INITIAL_ACCESS', 'description': 'Spear-phishing campaign'},
                {'action': 3, 'name': 'PERSISTENCE', 'description': 'Multiple persistence mechanisms'},
                {'action': 5, 'name': 'DEFENSE_EVASION', 'description': 'Obfuscation and evasion techniques'},
                {'action': 8, 'name': 'LATERAL_MOVEMENT', 'description': 'Stealthy lateral movement'},
                {'action': 9, 'name': 'COLLECTION', 'description': 'Long-term data collection'},
                {'action': 10, 'name': 'EXFILTRATION', 'description': 'Staged exfiltration'}
            ]
        
        # Simulate the attack scenario
        try:
            # Ensure current_state tensors have correct shapes
            # Make a copy to avoid modifying the original
            state_copy = {
                'network': current_state['network'].clone(),
                'flows': current_state['flows'].clone(),
                'events': current_state['events'].clone()
            }
            
            states, predictions = world_model.simulate_attack_scenario(
                state_copy,
                attack_sequence
            )
            
            # Extract key information from predictions
            # Normalize threat levels to 0-1 range using sigmoid
            def normalize_threat(threat_val):
                """Normalize threat level to 0-1 range using sigmoid"""
                import torch.nn.functional as F
                return float(F.sigmoid(torch.tensor(threat_val)).item())
            
            threat_levels = []
            for pred in predictions:
                raw_threat = pred['threat_level'].mean()
                normalized = normalize_threat(raw_threat)
                threat_levels.append(normalized)
            
            scenario_info = {
                'scenario_id': i + 1,
                'attack_sequence': attack_sequence,
                'num_steps': len(attack_sequence),
                'predicted_states': len(states),
                'threat_levels': threat_levels,
                'vulnerabilities': [float(pred['new_vulnerabilities'].mean()) for pred in predictions],
                'anomalous_flows': [float(pred['anomalous_flows'].mean()) for pred in predictions],
                'timeline': []
            }
            
            # Build timeline
            current_time = datetime.now()
            for j, step in enumerate(attack_sequence):
                if j < len(predictions):
                    raw_threat = float(predictions[j]['threat_level'].mean())
                    threat_level = normalize_threat(raw_threat)
                    scenario_info['timeline'].append({
                        'time': current_time + timedelta(hours=j*2),  # Assume 2 hours per step
                        'phase': step['name'],
                        'description': step['description'],
                        'threat_level': threat_level,
                        'vulnerability_score': float(predictions[j]['new_vulnerabilities'].mean()),
                        'anomaly_score': float(predictions[j]['anomalous_flows'].mean())
                    })
            
            scenarios.append(scenario_info)
            
            # Print scenario summary
            print(f"Attack Chain: {' -> '.join([s['name'] for s in attack_sequence])}")
            print(f"Predicted Steps: {len(attack_sequence)}")
            avg_threat = np.mean(scenario_info['threat_levels'])
            threat_percentage = avg_threat * 100
            print(f"Average Threat Level: {threat_percentage:.1f}%")
            if scenario_info['timeline']:
                print(f"Estimated Timeline: {scenario_info['timeline'][0]['time']} to {scenario_info['timeline'][-1]['time']}")
            
        except Exception as e:
            print(f"Error simulating scenario {i+1}: {e}")
            continue
    
    return scenarios


def predict_upcoming_attacks(
    orchestrator: PredictiveDefenseOrchestrator,
    telemetry_data: pd.DataFrame,
    forecast_hours: int = 48
) -> Dict[str, Any]:
    """
    Use the proactive defense orchestrator to predict upcoming attacks.
    """
    print("\n" + "="*70)
    print(f"PREDICTING ATTACKS FOR NEXT {forecast_hours} HOURS")
    print("="*70)
    
    # Run prediction
    prediction_results = orchestrator.predict_attacks(
        telemetry_data=telemetry_data,
        forecast_hours=forecast_hours
    )
    
    return prediction_results


def generate_proactive_recommendations(
    scenarios: List[Dict[str, Any]],
    predictions: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Generate proactive defense recommendations based on predicted scenarios.
    """
    recommendations = {
        'immediate': [],
        'short_term': [],
        'long_term': []
    }
    
    # Analyze scenarios - use normalized threat levels (0-1 range)
    high_threat_scenarios = [s for s in scenarios if s.get('threat_levels') and np.mean(s['threat_levels']) > 0.5]
    
    if high_threat_scenarios:
        recommendations['immediate'].append(
            f"Monitor for {len(high_threat_scenarios)} high-probability attack scenarios"
        )
        
        # Get unique attack types
        attack_types = set()
        for scenario in high_threat_scenarios:
            for step in scenario['attack_sequence']:
                attack_types.add(step['name'])
        
        recommendations['immediate'].append(
            f"Focus on: {', '.join(list(attack_types)[:3])}"
        )
    
    # Add predictions-based recommendations
    if predictions.get('early_warnings'):
        for warning in predictions['early_warnings']:
            if warning['level'] in ['CRITICAL', 'HIGH']:
                recommendations['immediate'].extend(
                    warning.get('recommended_actions', [])[:2]
                )
    
    # Short-term recommendations
    recommendations['short_term'].extend([
        "Implement network segmentation for critical assets",
        "Deploy deception technology (honeypots) in key network segments",
        "Increase monitoring frequency for predicted attack vectors",
        "Review and update firewall rules based on predicted attack paths"
    ])
    
    # Long-term recommendations
    recommendations['long_term'].extend([
        "Conduct red team exercises based on predicted scenarios",
        "Update incident response playbooks with predicted attack patterns",
        "Implement zero-trust architecture",
        "Enhance threat intelligence feeds with predicted attack indicators"
    ])
    
    return recommendations


def visualize_prediction_timeline(scenarios: List[Dict[str, Any]]):
    """
    Visualize the predicted attack timeline.
    """
    print("\n" + "="*70)
    print("PREDICTED ATTACK TIMELINE")
    print("="*70)
    
    for scenario in scenarios:
        print(f"\n--- Scenario {scenario['scenario_id']} ---")
        print(f"Attack Chain: {' -> '.join([s['name'] for s in scenario['attack_sequence']])}")
        avg_threat = np.mean(scenario['threat_levels']) if scenario['threat_levels'] else 0
        print(f"Overall Threat Level: {avg_threat*100:.1f}%")
        print("\nTimeline:")
        for event in scenario['timeline']:
            threat_pct = event['threat_level'] * 100
            threat_emoji = "🔴" if threat_pct > 70 else "🟠" if threat_pct > 40 else "🟡" if threat_pct > 20 else "🟢"
            print(f"  {event['time'].strftime('%Y-%m-%d %H:%M')} - {event['phase']} {threat_emoji}")
            print(f"    Threat Level: {threat_pct:.1f}%")
            print(f"    Description: {event['description']}")


def main():
    """
    Main use case demonstration: Proactive Attack Prediction
    """
    print("="*70)
    print("USE CASE: PROACTIVE ATTACK PREDICTION")
    print("Predicting Attacks Before They Happen")
    print("="*70)
    
    # Step 1: Initialize components
    print("\n[1] Initializing Cybersecurity World Model...")
    config = Config()
    world_model = CyberWorldModel(
        feature_dim=config.get('model.feature_dim', 256),
        latent_dim=config.get('model.latent_dim', 256),
        action_dim=config.get('model.action_dim', 50)
    )
    
    orchestrator = PredictiveDefenseOrchestrator(config=config)
    print("✓ Components initialized")
    
    # Step 2: Generate current network state
    print("\n[2] Analyzing current network state...")
    current_state = generate_current_network_state()
    print("✓ Network state captured")
    print(f"  - Network features: {current_state['network'].shape}")
    print(f"  - Flow sequences: {current_state['flows'].shape}")
    print(f"  - Security events: {current_state['events'].shape}")
    
    # Step 3: Load historical telemetry
    print("\n[3] Loading historical telemetry data...")
    telemetry_data = generate_telemetry_data(days=7)
    print(f"✓ Loaded {len(telemetry_data)} hours of telemetry data")
    print(f"  - Features: {telemetry_data.shape[1]}")
    print(f"  - Time range: {telemetry_data.index[0]} to {telemetry_data.index[-1]}")
    
    # Step 4: Simulate potential attack scenarios
    print("\n[4] Simulating potential attack scenarios...")
    scenarios = simulate_attack_scenarios(
        world_model=world_model,
        current_state=current_state,
        num_scenarios=5
    )
    print(f"✓ Generated {len(scenarios)} attack scenarios")
    
    # Step 5: Predict upcoming attacks
    print("\n[5] Running proactive attack prediction...")
    predictions = predict_upcoming_attacks(
        orchestrator=orchestrator,
        telemetry_data=telemetry_data,
        forecast_hours=48
    )
    print("✓ Predictions completed")
    
    # Step 6: Generate recommendations
    print("\n[6] Generating proactive defense recommendations...")
    recommendations = generate_proactive_recommendations(scenarios, predictions)
    print("✓ Recommendations generated")
    
    # Step 7: Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nConfidence Level: {predictions['confidence_level']:.2%}")
    print(f"Forecast Horizon: {predictions['forecast_horizon']}")
    print(f"Anomalies Detected: {predictions['anomalies_detected']}")
    
    print("\nTop Predicted Attacks:")
    for i, attack in enumerate(predictions['predicted_attacks'][:5], 1):
        print(f"  {i}. {attack['attack_type']}: {attack['probability']:.1%}")
    
    print("\nEarly Warnings:")
    for warning in predictions['early_warnings'][:5]:
        print(f"  [{warning['level']}] {warning['type']}")
        if 'attack_type' in warning:
            print(f"     Attack Type: {warning['attack_type']}")
        print(f"     Time Window: {warning.get('predicted_time_window', 'N/A')}")
        print(f"     Actions: {', '.join(warning.get('recommended_actions', [])[:2])}")
    
    # Visualize timelines
    visualize_prediction_timeline(scenarios)
    
    # Display recommendations
    print("\n" + "="*70)
    print("PROACTIVE DEFENSE RECOMMENDATIONS")
    print("="*70)
    
    print("\nIMMEDIATE ACTIONS (Next 24 hours):")
    for i, rec in enumerate(recommendations['immediate'], 1):
        print(f"  {i}. {rec}")
    
    print("\nSHORT-TERM ACTIONS (Next week):")
    for i, rec in enumerate(recommendations['short_term'], 1):
        print(f"  {i}. {rec}")
    
    print("\nLONG-TERM ACTIONS (Next month):")
    for i, rec in enumerate(recommendations['long_term'], 1):
        print(f"  {i}. {rec}")
    
    # Save results
    print("\n[7] Saving prediction results...")
    results = {
        'timestamp': datetime.now().isoformat(),
        'scenarios': [
            {
                'scenario_id': s['scenario_id'],
                'attack_chain': [step['name'] for step in s['attack_sequence']],
                'avg_threat_level': float(np.mean(s['threat_levels'])),
                'timeline': [
                    {
                        'time': e['time'].isoformat(),
                        'phase': e['phase'],
                        'threat_level': float(e['threat_level'])
                    }
                    for e in s['timeline']
                ]
            }
            for s in scenarios
        ],
        'predictions': {
            'confidence': float(predictions['confidence_level']),
            'predicted_attacks': predictions['predicted_attacks'],
            'warnings': len(predictions['early_warnings'])
        },
        'recommendations': recommendations
    }
    
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved to prediction_results.json")
    
    print("\n" + "="*70)
    print("USE CASE COMPLETE")
    print("="*70)
    print("\nThe system has successfully:")
    print("  ✓ Analyzed current network state")
    print("  ✓ Simulated 5 potential attack scenarios")
    print("  ✓ Predicted upcoming attacks for the next 48 hours")
    print("  ✓ Generated proactive defense recommendations")
    print("  ✓ Created actionable timeline for security team")
    print("\nThese predictions allow security teams to:")
    print("  • Prepare defenses before attacks occur")
    print("  • Allocate resources to high-probability threats")
    print("  • Update monitoring rules based on predicted patterns")
    print("  • Test incident response procedures for predicted scenarios")


if __name__ == '__main__':
    main()

