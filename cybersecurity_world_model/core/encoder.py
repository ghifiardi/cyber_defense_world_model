"""CyberStateEncoder - Encodes network state into latent representation."""

import torch
import torch.nn as nn
from typing import Tuple
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError

logger = get_logger(__name__)


class CyberStateEncoder(nn.Module):
    """
    Encodes network state into latent representation.
    Network state includes: topology, traffic flows, security events.
    """
    
    def __init__(self, feature_dim: int = 256):
        """
        Initialize the encoder.
        
        Args:
            feature_dim: Dimension of the output latent representation
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Multi-modal encoder for different data types
        self.network_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(128*64, 512)
        )
        
        self.flow_encoder = nn.LSTM(
            input_size=20,  # flow features
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        self.event_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        )
    
    def forward(
        self, 
        network_tensor: torch.Tensor, 
        flow_tensor: torch.Tensor, 
        event_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode network state into latent representation.
        
        Args:
            network_tensor: Network topology tensor
            flow_tensor: Traffic flow tensor (batch, seq_len, 20)
            event_tensor: Security events tensor (batch, seq_len, 64)
            
        Returns:
            Latent representation tensor (batch, feature_dim)
            
        Raises:
            ModelError: If encoding fails
        """
        try:
            # Encode network topology
            net_feat = self.network_encoder(network_tensor.unsqueeze(1))
            
            # Encode traffic flows
            _, (flow_feat, _) = self.flow_encoder(flow_tensor)
            flow_feat = flow_feat[-1]  # Last layer hidden state
            
            # Encode security events
            event_feat = self.event_encoder(event_tensor)
            event_feat = event_feat.mean(dim=1)
            
            # Fusion
            combined = torch.cat([net_feat, flow_feat, event_feat], dim=1)
            latent = self.fusion(combined)
            
            return latent
        except Exception as e:
            logger.error(f"Error encoding network state: {e}")
            raise ModelError(f"Failed to encode network state: {e}") from e

