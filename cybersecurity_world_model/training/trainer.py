"""Training utilities for Cybersecurity World Model."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from pathlib import Path
from cybersecurity_world_model.core.world_model import CyberWorldModel
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import TrainingError

logger = get_logger(__name__)


def train_cybersecurity_world_model(
    model: CyberWorldModel,
    simulator: Any,
    epochs: int = 10,
    sequence_length: int = 100,
    batch_size: int = 4,
    checkpoint_dir: Optional[str] = None,
    save_interval: int = 10
) -> Dict[str, Any]:
    """
    Train the cybersecurity world model.
    
    Args:
        model: CyberWorldModel instance to train
        simulator: Simulator that provides training sequences
        epochs: Number of training epochs
        sequence_length: Length of each training sequence
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints
        save_interval: Save checkpoint every N epochs
        
    Returns:
        Dict with training metrics and history
        
    Raises:
        TrainingError: If training fails
    """
    try:
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, seq_len={sequence_length}")
        
        training_history = {
            'losses': [],
            'epochs': []
        }
        
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx in range(100):  # 100 batches per epoch
                try:
                    # Generate a batch of sequences
                    batch_states = []
                    batch_actions = []
                    
                    for _ in range(batch_size):
                        states, actions = simulator.generate_sequence(sequence_length)
                        batch_states.append(states)
                        batch_actions.append(actions)
                    
                    # Convert to tensors
                    batch_states = torch.tensor(batch_states, dtype=torch.float32)
                    batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
                    
                    # Move to device if model is on GPU
                    device = next(model.encoder.parameters()).device
                    batch_states = batch_states.to(device)
                    batch_actions = batch_actions.to(device)
                    
                    # Train on predicting the next state given current state and action
                    # So for each sequence, we have T-1 transitions
                    for t in range(sequence_length - 1):
                        state_t = batch_states[:, t]
                        action_t = batch_actions[:, t].long()  # Convert to long for embedding
                        state_t1 = batch_states[:, t + 1]
                        
                        # Forward pass
                        # Note: This is a simplified training loop
                        # In practice, you'd use the full world model interface
                        latent_t = model.encoder(
                            state_t,  # network
                            state_t[:, :20] if state_t.shape[1] >= 20 else state_t,  # flows (simplified)
                            state_t[:, 20:84] if state_t.shape[1] >= 84 else state_t  # events (simplified)
                        )
                        
                        next_latent, pred_flows, pred_threats, pred_vulns = model.dynamics(
                            latent_t, action_t
                        )
                        
                        # Encode actual next state
                        latent_t1_true = model.encoder(
                            state_t1,
                            state_t1[:, :20] if state_t1.shape[1] >= 20 else state_t1,
                            state_t1[:, 20:84] if state_t1.shape[1] >= 84 else state_t1
                        )
                        
                        # Compute loss
                        loss = F.mse_loss(next_latent, latent_t1_true)
                        
                        model.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(model.dynamics.parameters(), 1.0)
                        model.optimizer.step()
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                except Exception as e:
                    logger.warning(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            training_history['losses'].append(avg_loss)
            training_history['epochs'].append(epoch)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            if checkpoint_dir and (epoch % save_interval == 0 or epoch == epochs - 1):
                checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch}.pth"
                model.save_checkpoint(str(checkpoint_file))
                logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        logger.info("Training completed successfully")
        return training_history
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise TrainingError(f"Failed to train model: {e}") from e


