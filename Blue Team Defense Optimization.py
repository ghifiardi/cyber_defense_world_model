class DefenseOptimizer:
    """
    Uses world model to optimize defense strategies
    """
    def __init__(self, world_model):
        self.world_model = world_model
        self.defense_strategies = []
        
    def optimize_defense_posture(self, network_state, budget_constraint):
        """
        Find optimal defense configuration using reinforcement learning
        """
        # Define defense action space
        defense_actions = list(self.world_model.defense_actions.values())
        
        # Q-learning for defense optimization
        q_table = np.zeros((len(defense_actions), len(defense_actions)))
        
        # Simulate attacks and learn optimal responses
        for episode in range(1000):
            current_state = network_state.clone()
            total_reward = 0
            
            for step in range(10):
                # Choose defense action
                defense_action = self._epsilon_greedy(q_table, step, 0.1)
                
                # Simulate random attack
                attack_action = np.random.randint(0, 13)
                
                # Get outcome from world model
                outcome = self._simulate_interaction(
                    current_state, 
                    attack_action, 
                    defense_action
                )
                
                # Update Q-table
                reward = self._calculate_defense_reward(outcome)
                q_table = self._update_q_table(
                    q_table, defense_action, reward, step
                )
                
                total_reward += reward
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
        
        # Extract optimal defense strategy
        optimal_strategy = self._extract_strategy(q_table)
        
        return optimal_strategy
    
    def _simulate_interaction(self, state, attack_action, defense_action):
        """Simulate attack-defense interaction"""
        return {
            'attack_success': np.random.random() < 0.5,
            'detection_time': np.random.exponential(5),
            'damage_prevented': np.random.randint(0, 100),
            'false_positives': np.random.randint(0, 5),
            'resource_cost': np.random.randint(1, 10)
        }
    
    def _calculate_defense_reward(self, outcome):
        """Calculate reward for defense action"""
        reward = 0
        reward += outcome['damage_prevented'] * 10
        reward -= outcome['detection_time'] * 2
        reward -= outcome['false_positives'] * 5
        reward -= outcome['resource_cost']
        
        if not outcome['attack_success']:
            reward += 50
        
        return reward
    
    def _epsilon_greedy(self, q_table, state, epsilon):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(len(q_table))
        else:
            return np.argmax(q_table[state])
    
    def _update_q_table(self, q_table, action, reward, state):
        """Update Q-table using Q-learning"""
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        
        # Q-learning update rule
        old_value = q_table[state, action]
        next_max = np.max(q_table[state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        
        return q_table
    
    def _extract_strategy(self, q_table):
        """Extract optimal defense strategy from Q-table"""
        optimal_actions = np.argmax(q_table, axis=1)
        
        strategy = {
            'reconnaissance_phase': self.world_model.defense_actions[optimal_actions[0]],
            'initial_access_phase': self.world_model.defense_actions[optimal_actions[1]],
            'execution_phase': self.world_model.defense_actions[optimal_actions[2]],
            'lateral_movement_phase': self.world_model.defense_actions[optimal_actions[8]],
            'exfiltration_phase': self.world_model.defense_actions[optimal_actions[10]]
        }
        
        return strategy
