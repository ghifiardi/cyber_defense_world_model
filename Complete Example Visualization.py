import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def run_genie_simulation():
    # Initialize simulation
    simulator = GenieSimulation()
    
    # Simulated training data (in practice, this would be real videos)
    # For demo purposes, we'll use synthetic data
    def create_synthetic_videos(batch_size=4, frames=16, size=64):
        # Create simple moving shapes
        videos = []
        for b in range(batch_size):
            video = []
            for t in range(frames):
                img = np.zeros((3, size, size))
                # Moving circle
                x = size//4 + int(size//2 * np.sin(t/frames * 2*np.pi))
                y = size//4 + int(size//2 * np.cos(t/frames * 2*np.pi))
                for i in range(size):
                    for j in range(size):
                        if (i-x)**2 + (j-y)**2 < 25:
                            img[:, i, j] = [1.0, 0.5, 0.2]  # Orange circle
                video.append(img)
            videos.append(video)
        return torch.tensor(videos).float()
    
    # Training loop (simplified)
    print("Training simplified world model...")
    for epoch in range(10):  # In reality, Genie trains on billions of frames
        videos = create_synthetic_videos()
        loss = simulator.train_step(videos)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Generate a world from a prompt
    print("\nGenerating world from prompt...")
    prompt = create_synthetic_videos(1, 1, 64)[0, 0].unsqueeze(0)
    
    with torch.no_grad():
        generated_world = simulator.generate_world(prompt, num_steps=8)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show prompt
    axes[0].imshow(prompt[0].permute(1, 2, 0).numpy())
    axes[0].set_title("Prompt Image")
    axes[0].axis('off')
    
    # Show generated frames
    grid = make_grid(generated_world, nrow=3)
    axes[1].imshow(grid.permute(1, 2, 0).numpy())
    axes[1].set_title("Generated World Frames")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return simulator

# Run the simulation
if __name__ == "__main__":
    simulator = run_genie_simulation()
