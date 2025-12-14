"""BehavioralAnomalyDetector - Self-supervised anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError

logger = get_logger(__name__)


class BehavioralAnomalyDetector(nn.Module):
    """
    Self-supervised anomaly detection using contrastive learning.
    """
    
    def __init__(self, feature_dim: int = 128, memory_bank_size: int = 1000):
        """
        Initialize the anomaly detector.
        
        Args:
            feature_dim: Input feature dimension
            memory_bank_size: Size of memory bank for normal patterns
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.memory_bank_size = memory_bank_size
        
        # Autoencoder for normal pattern reconstruction
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, feature_dim)
        )
        
        # Contrastive learning projection head
        self.projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Anomaly scoring network
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Memory bank for normal patterns
        self.register_buffer('memory_bank', torch.randn(memory_bank_size, 64))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for anomaly detection.
        
        Args:
            x: Input tensor (batch, feature_dim)
            
        Returns:
            Dict with 'latent', 'reconstruction', 'projection', 'anomaly_score', 'reconstruction_loss'
            
        Raises:
            ModelError: If detection fails
        """
        try:
            # Encode to latent space
            z = self.encoder(x)
            
            # Reconstruct
            x_recon = self.decoder(z)
            
            # Project for contrastive learning
            z_proj = self.projection(z)
            
            # Calculate anomaly score
            anomaly_score = self.anomaly_scorer(z)
            
            return {
                'latent': z,
                'reconstruction': x_recon,
                'projection': z_proj,
                'anomaly_score': anomaly_score,
                'reconstruction_loss': F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            raise ModelError(f"Failed to detect anomalies: {e}") from e
    
    def update_memory_bank(self, normal_patterns: torch.Tensor):
        """
        Update memory bank with new normal patterns.
        
        Args:
            normal_patterns: Tensor of normal patterns (batch, feature_dim)
        """
        try:
            with torch.no_grad():
                z_normal = self.encoder(normal_patterns)
                # Update via FIFO
                self.memory_bank = torch.cat([self.memory_bank, z_normal])[-self.memory_bank_size:]
            logger.debug(f"Memory bank updated with {len(normal_patterns)} patterns")
        except Exception as e:
            logger.warning(f"Error updating memory bank: {e}")

