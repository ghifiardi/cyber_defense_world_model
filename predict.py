#!/usr/bin/env python3
"""Prediction entry point for Cybersecurity World Model."""

import argparse
import numpy as np
import pandas as pd
from cybersecurity_world_model.defense.orchestrator import PredictiveDefenseOrchestrator
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Predict cyber attacks using Cybersecurity World Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--telemetry', type=str, help='Path to telemetry data file (CSV)')
    parser.add_argument('--forecast-hours', type=int, default=24, help='Hours to forecast ahead')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = Config(config_path=args.config) if args.config else Config()
    
    # Initialize orchestrator
    orchestrator = PredictiveDefenseOrchestrator(config=config)
    
    # Load telemetry data
    if args.telemetry:
        telemetry_data = pd.read_csv(args.telemetry)
    else:
        # Generate sample telemetry data
        print("No telemetry file provided, generating sample data...")
        telemetry_data = pd.DataFrame(np.random.randn(100, 256))
    
    # Run prediction
    results = orchestrator.predict_attacks(
        telemetry_data=telemetry_data,
        forecast_hours=args.forecast_hours
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nConfidence Level: {results['confidence_level']:.2%}")
    print(f"Forecast Horizon: {results['forecast_horizon']}")
    print(f"Anomalies Detected: {results['anomalies_detected']}")
    
    print("\nPredicted Attacks (Top 3):")
    for i, attack in enumerate(results['predicted_attacks'][:3], 1):
        print(f"  {i}. {attack['attack_type']}: {attack['probability']:.1%}")
    
    print("\nEarly Warnings:")
    for warning in results['early_warnings'][:3]:
        print(f"  [{warning['level']}] {warning['type']}: {warning.get('attack_type', 'N/A')}")
        print(f"     Recommended: {', '.join(warning.get('recommended_actions', [])[:2])}")
    
    print("\nImmediate Defense Recommendations:")
    for i, rec in enumerate(results['defense_recommendations']['immediate'][:5], 1):
        print(f"  {i}. {rec}")

if __name__ == '__main__':
    main()

