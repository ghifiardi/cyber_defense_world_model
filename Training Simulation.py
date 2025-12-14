class GenieSimulation:
    def __init__(self):
        self.model = LatentWorldModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train_step(self, video_batch):
        """
        Simplified training on video sequences
        video_batch: (batch, frames, C, H, W)
        """
        batch_size, num_frames, C, H, W = video_batch.shape
        
        # Reformat to pairs of consecutive frames
        frames_t = video_batch[:, :-1].reshape(-1, C, H, W)
        frames_t1 = video_batch[:, 1:].reshape(-1, C, H, W)
        
        # Encode both
        z_t = self.model.encoder(frames_t)
        z_t1 = self.model.encoder(frames_t1)
        
        # Learn to predict actions between frames
        with torch.no_grad():
            # In real Genie, this is where the action tokenizer learns
            # Simplified: random actions for demonstration
            actions = torch.randn(z_t.shape[0], 8)
        
        # Predict next latent
        z_t1_pred = self.model.dynamics(torch.cat([z_t, actions], dim=1))
        
        # Reconstruction loss
        recon_loss = F.mse_loss(z_t1_pred, z_t1)
        
        # Decoder reconstruction
        frames_recon = self.model.decoder(z_t1_pred)
        pixel_loss = F.mse_loss(frames_recon, frames_t1)
        
        total_loss = recon_loss + pixel_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def generate_world(self, prompt_image, num_steps=30):
        """Generate an interactive world from single image"""
        self.model.eval()
        with torch.no_grad():
            trajectory = self.model.generate_trajectory(prompt_image, num_steps)
        return trajectory
    
    def interactive_play(self, initial_state, action_sequence):
        """Simulate playing in the generated world"""
        states = [initial_state]
        current_latent = self.model.encoder(initial_state)
        
        for action in action_sequence:
            action_tensor = torch.tensor(action).unsqueeze(0)
            current_latent = self.model.dynamics(
                torch.cat([current_latent, action_tensor], dim=1)
            )
            next_state = self.model.decoder(current_latent)
            states.append(next_state)
        
        return torch.cat(states, dim=0)
