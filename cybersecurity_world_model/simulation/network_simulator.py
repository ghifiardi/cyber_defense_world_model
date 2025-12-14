"""Network simulator for generating training sequences."""

import numpy as np
from typing import Tuple
from cybersecurity_world_model.utils.logging import get_logger

logger = get_logger(__name__)


class NetworkSimulator:
    """Simulator for generating network state sequences."""
    
    def __init__(self, grid_size: int = 24, feature_dim: int = 256):
        """
        Initialize the network simulator.
        
        Args:
            grid_size: Size of the network grid
            feature_dim: Dimension of feature vectors
        """
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        logger.info(f"NetworkSimulator initialized: grid_size={grid_size}, feature_dim={feature_dim}")
    
    def generate_sequence(self, length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sequence of network states and actions.
        
        Args:
            length: Length of the sequence
            
        Returns:
            Tuple of (states, actions) where:
            - states: (length, feature_dim) array of network states
            - actions: (length, 3) array of actions [x_norm, y_norm, type]
        """
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
            actions.append([x / self.grid_size, y / self.grid_size, action_type])
        
        return np.array(states), np.array(actions)
    
    def step(self, action: Tuple[int, int, int]) -> np.ndarray:
        """
        Execute an action and return the next state.
        
        Args:
            action: Tuple of (x, y, action_type)
            
        Returns:
            Next state vector (feature_dim,)
        """
        # Generate a random state vector
        # In a real implementation, this would simulate actual network dynamics
        state = np.random.randn(self.feature_dim)
        return state


