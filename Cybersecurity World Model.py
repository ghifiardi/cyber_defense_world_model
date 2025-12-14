class CybersecurityWorldModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=3, input_channels=3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*6*6, latent_dim)
        )
        
        # Action embedding: map 3 continuous values (x, y, type) to 16
        self.action_embedding = nn.Linear(action_dim, 16)
        
        # Dynamics model
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*6*6),
            nn.Unflatten(1, (128, 6, 6)),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x, action):
        # x: (batch, C, H, W)
        # action: (batch, 3) where the last dimension is [x_norm, y_norm, type]
        
        z = self.encoder(x)
        
        # Embed the action
        action_emb = self.action_embedding(action)  # (batch, 16)
        
        z_next = self.dynamics(torch.cat([z, action_emb], dim=1))
        x_recon = self.decoder(z_next)
        
        return x_recon, z, z_next
    
    def generate_trajectory(self, initial_state, action_sequence):
        # initial_state: (1, C, H, W)
        # action_sequence: list of action vectors (each is [x_norm, y_norm, type]) of length T
        
        states = [initial_state]
        z = self.encoder(initial_state)
        
        for action_vec in action_sequence:
            action = torch.tensor(action_vec).unsqueeze(0).float().to(initial_state.device)
            action_emb = self.action_embedding(action)
            z = self.dynamics(torch.cat([z, action_emb], dim=1))
            next_state = self.decoder(z)
            states.append(next_state)
        
        return torch.cat(states, dim=0)
