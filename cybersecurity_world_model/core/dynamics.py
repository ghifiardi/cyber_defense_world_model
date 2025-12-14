"""AttackDynamicsModel - Predicts next network state given current state and attack action."""

import torch
import torch.nn as nn
from typing import Tuple
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError

logger = get_logger(__name__)


class AttackDynamicsModel(nn.Module):
    """
    Predicts next network state given current state and attack action.
    """
    
    def __init__(self, latent_dim: int = 256, action_dim: int = 50):
        """
        Initialize the dynamics model.
        
        Args:
            latent_dim: Dimension of latent state representation
            action_dim: Number of possible attack actions
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Attack action embedding (tactics, techniques, procedures)
        self.action_embedder = nn.Embedding(action_dim, 64)
        
        # Dynamics predictor
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 64, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, latent_dim)
        )
        
        # Multi-task decoders for different prediction tasks
        self.flow_predictor = nn.Linear(latent_dim, 20)  # Traffic features
        self.threat_predictor = nn.Linear(latent_dim, 10)  # Threat levels
        self.vulnerability_predictor = nn.Linear(latent_dim, 5)  # New vulns
    
    def forward(
        self, 
        latent_state: torch.Tensor, 
        attack_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state given current latent state and attack action.
        
        Args:
            latent_state: Current latent state (batch, latent_dim)
            attack_action: Attack action indices (batch,)
            
        Returns:
            Tuple of (next_latent, predicted_flows, predicted_threats, predicted_vulns)
            
        Raises:
            ModelError: If prediction fails
        """
        try:
            # Embed attack action
            action_emb = self.action_embedder(attack_action)
            
            # Predict next latent state
            next_latent = self.dynamics(torch.cat([latent_state, action_emb], dim=1))
            
            # Predict various security metrics
            predicted_flows = self.flow_predictor(next_latent)
            predicted_threats = self.threat_predictor(next_latent)
            predicted_vulns = self.vulnerability_predictor(next_latent)
            
            return next_latent, predicted_flows, predicted_threats, predicted_vulns
        except Exception as e:
            logger.error(f"Error predicting next state: {e}")
            raise ModelError(f"Failed to predict next state: {e}") from e


