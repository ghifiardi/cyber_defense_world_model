    def generate_sequence(self, length=100):
        states = []
        actions = []
        
        for _ in range(length):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            action_type = np.random.randint(0, 3)
            action = (x, y, action_type)
            
            state = self.step(action)
            states.append(state)
            # Normalize x and y
            actions.append([x/self.grid_size, y/self.grid_size, action_type])
        
        return np.array(states), np.array(actions)
