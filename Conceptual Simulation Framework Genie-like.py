import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentWorldModel(nn.Module):
    """
    Simplified world model inspired by Genie's architecture
    """
    def __init__(self, latent_dim=128, action_dim=8):
        super().__init__()
        
        # Encoder (image to latent)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*6*6, latent_dim)  # Adjusted based on input size
        )
        
        # Dynamics model (predicts next latent)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder (latent to image)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*6*6),
            nn.Unflatten(1, (128, 6, 6)),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid()
        )
        
        # Action predictor (learns plausible actions)
        self.action_predictor = nn.Linear(latent_dim*2, action_dim)
    
    def forward(self, x, action=None):
        # Encode to latent
        z = self.encoder(x)
        
        if action is None:
            # Sample random action
            action = torch.randn(z.shape[0], 8)
        
        # Predict next latent
        z_next = self.dynamics(torch.cat([z, action], dim=1))
        
        # Decode to image
        x_recon = self.decoder(z_next)
        
        return x_recon, z, z_next
    
    def generate_trajectory(self, initial_image, steps=10):
        """Generate a sequence of frames from initial image"""
        frames = [initial_image]
        z = self.encoder(initial_image)
        
        for _ in range(steps):
            # Sample action (in practice, this would be learned)
            action = torch.randn(1, 8)
            z = self.dynamics(torch.cat([z, action], dim=1))
            frame = self.decoder(z)
            frames.append(frame)
        
        return torch.cat(frames, dim=0)
