def train_cybersecurity_world_model(model, simulator, epochs=10, sequence_length=100, batch_size=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        for _ in range(100):  # 100 batches per epoch
            # Generate a batch of sequences
            batch_states = []
            batch_actions = []
            for _ in range(batch_size):
                states, actions = simulator.generate_sequence(sequence_length)
                batch_states.append(states)
                batch_actions.append(actions)
            
            # Convert to tensors
            batch_states = torch.tensor(np.array(batch_states)).float()  # (batch, T, C, H, W)
            batch_actions = torch.tensor(np.array(batch_actions)).float()  # (batch, T, 3)
            
            # We train on predicting the next state given current state and action
            # So for each sequence, we have T-1 transitions
            for t in range(sequence_length-1):
                state_t = batch_states[:, t].cuda()
                action_t = batch_actions[:, t].cuda()
                state_t1 = batch_states[:, t+1].cuda()
                
                # Forward pass
                state_t1_pred, _, _ = model(state_t, action_t)
                
                # Compute loss
                loss = F.mse_loss(state_t1_pred, state_t1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss/(100*(sequence_length-1)):.6f}")
