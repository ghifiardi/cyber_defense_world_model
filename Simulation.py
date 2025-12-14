if __name__ == "__main__":
    simulator = NetworkSimulator(grid_size=64)
    model = CybersecurityWorldModel(latent_dim=128, action_dim=3, input_channels=3).cuda()
    
    train_cybersecurity_world_model(model, simulator, epochs=10, sequence_length=20, batch_size=8)
