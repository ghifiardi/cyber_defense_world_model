#!/usr/bin/env python3
"""Training entry point for Cybersecurity World Model."""

import argparse
from cybersecurity_world_model.core.world_model import CyberWorldModel
from cybersecurity_world_model.simulation.network_simulator import NetworkSimulator
from cybersecurity_world_model.training.trainer import train_cybersecurity_world_model
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description='Train Cybersecurity World Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = Config(config_path=args.config) if args.config else Config()
    
    # Initialize model
    model = CyberWorldModel(
        feature_dim=config.get('model.feature_dim', 256),
        latent_dim=config.get('model.latent_dim', 256),
        action_dim=config.get('model.action_dim', 50),
        learning_rate=config.get('training.learning_rate', 1e-4),
        weight_decay=config.get('training.weight_decay', 1e-5)
    )
    
    # Initialize simulator
    simulator = NetworkSimulator(grid_size=24, feature_dim=config.get('model.feature_dim', 256))
    
    # Train
    training_history = train_cybersecurity_world_model(
        model=model,
        simulator=simulator,
        epochs=args.epochs or config.get('training.epochs', 10),
        sequence_length=args.sequence_length or config.get('training.sequence_length', 100),
        batch_size=args.batch_size or config.get('training.batch_size', 4),
        checkpoint_dir=args.checkpoint_dir or config.get('training.checkpoint_dir', 'checkpoints'),
        save_interval=config.get('training.save_interval', 10)
    )
    
    print(f"\nTraining completed!")
    print(f"Final loss: {training_history['losses'][-1]:.6f}")

if __name__ == '__main__':
    main()


