"""TemporalAttackPredictor - Multi-horizon attack prediction model."""

import torch
import torch.nn as nn
from typing import Tuple
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError

logger = get_logger(__name__)


class TemporalAttackPredictor(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon attack prediction.
    """
    
    def __init__(self, input_dim: int = 256, forecast_horizon: int = 24):
        """
        Initialize the temporal attack predictor.
        
        Args:
            input_dim: Input feature dimension
            forecast_horizon: Number of future timesteps to predict
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        
        # Temporal attention layers
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Temporal convolutional network for pattern extraction
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, input_dim, kernel_size=3, dilation=3, padding=3),
            nn.ReLU()
        )
        
        # Gated recurrent units for sequential patterns
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Multi-horizon forecasting heads
        self.forecast_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),  # GRU is bidirectional
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 5)  # 5 threat metrics per timestep
            ) for _ in range(forecast_horizon)
        ])
        
        # Attack type classifier
        self.attack_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 13)  # MITRE ATT&CK tactics
        )
        
        # Confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for attack prediction.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (forecasts, attack_probs, confidence)
            - forecasts: (batch, horizon, 5) threat metrics
            - attack_probs: (batch, 13) attack type probabilities
            - confidence: (batch, 1) prediction confidence
            
        Raises:
            ModelError: If prediction fails
        """
        try:
            # Temporal attention
            attended, _ = self.temporal_attention(x, x, x)
            
            # Temporal convolution
            x_tcn = x.transpose(1, 2)
            x_tcn = self.tcn(x_tcn)
            x_tcn = x_tcn.transpose(1, 2)
            
            # Combine features
            combined = attended + x_tcn
            
            # GRU for sequential patterns
            gru_out, _ = self.gru(combined)
            
            # Multi-horizon forecasts
            forecasts = []
            last_hidden = gru_out[:, -1, :]  # Last timestep
            
            for head in self.forecast_heads:
                forecast = head(last_hidden)
                forecasts.append(forecast)
            
            forecasts = torch.stack(forecasts, dim=1)  # (batch, horizon, 5)
            
            # Attack classification
            attack_probs = self.attack_classifier(last_hidden)
            
            # Prediction confidence
            confidence = self.confidence_net(last_hidden)
            
            return forecasts, attack_probs, confidence
        except Exception as e:
            logger.error(f"Error in attack prediction: {e}")
            raise ModelError(f"Failed to predict attacks: {e}") from e


