"""CyberWorldModel - Complete cybersecurity world model for threat simulation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
from cybersecurity_world_model.core.encoder import CyberStateEncoder
from cybersecurity_world_model.core.dynamics import AttackDynamicsModel
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError, TrainingError

logger = get_logger(__name__)


class CyberWorldModel:
    """
    Complete cybersecurity world model for threat simulation.
    """
    
    # Attack dictionary (MITRE ATT&CK inspired)
    ATTACK_TYPES = {
        0: "RECONNAISSANCE",
        1: "INITIAL_ACCESS",
        2: "EXECUTION",
        3: "PERSISTENCE",
        4: "PRIVILEGE_ESCALATION",
        5: "DEFENSE_EVASION",
        6: "CREDENTIAL_ACCESS",
        7: "DISCOVERY",
        8: "LATERAL_MOVEMENT",
        9: "COLLECTION",
        10: "EXFILTRATION",
        11: "COMMAND_CONTROL",
        12: "IMPACT"
    }
    
    # Defense actions
    DEFENSE_ACTIONS = {
        20: "BLOCK_IP",
        21: "QUARANTINE_HOST",
        22: "PATCH_SYSTEM",
        23: "ISOLATE_NETWORK",
        24: "DECEPTION_HONEYPOT",
        25: "RATE_LIMITING",
        26: "INCREASE_MONITORING",
        27: "CHANGE_CREDENTIALS",
        28: "ENABLE_FIREWALL_RULE",
        29: "ALERT_SOC"
    }
    
    def __init__(
        self, 
        feature_dim: int = 256,
        latent_dim: int = 256,
        action_dim: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the cybersecurity world model.
        
        Args:
            feature_dim: Dimension of encoder output
            latent_dim: Dimension of latent state
            action_dim: Number of possible attack actions
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.encoder = CyberStateEncoder(feature_dim=feature_dim)
        self.dynamics = AttackDynamicsModel(latent_dim=latent_dim, action_dim=action_dim)
        
        self.attack_types = self.ATTACK_TYPES.copy()
        self.defense_actions = self.DEFENSE_ACTIONS.copy()
        
        # Training optimizers
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        logger.info("CyberWorldModel initialized")
    
    def train_on_attack_sequences(self, attack_dataset: List, epochs: int = 100):
        """
        Train on historical attack sequences.
        
        Args:
            attack_dataset: List of attack sequences, each sequence is a list of dicts
                with 'state' and 'action' keys
            epochs: Number of training epochs
            
        Raises:
            TrainingError: If training fails
        """
        try:
            self.encoder.train()
            self.dynamics.train()
            
            logger.info(f"Starting training on {len(attack_dataset)} sequences for {epochs} epochs")
            
            for epoch in range(epochs):
                total_loss = 0
                num_batches = 0
                
                for sequence in attack_dataset:
                    # Each sequence: [state0, action0, state1, action1, ...]
                    for i in range(len(sequence) - 1):
                        try:
                            current_state = sequence[i]['state']
                            action = sequence[i]['action']
                            next_state = sequence[i+1]['state']
                            
                            # Encode current state
                            latent = self.encoder(
                                current_state['network'],
                                current_state['flows'],
                                current_state['events']
                            )
                            
                            # Convert action to tensor if needed
                            if not isinstance(action, torch.Tensor):
                                action = torch.tensor([action], dtype=torch.long)
                            
                            # Predict next state
                            next_latent, pred_flows, pred_threats, pred_vulns = \
                                self.dynamics(latent, action)
                            
                            # Encode actual next state
                            next_latent_true = self.encoder(
                                next_state['network'],
                                next_state['flows'],
                                next_state['events']
                            )
                            
                            # Compute losses
                            latent_loss = F.mse_loss(next_latent, next_latent_true)
                            
                            # Additional prediction losses
                            flow_loss = F.mse_loss(
                                pred_flows, 
                                next_state['flows'].mean(dim=1)
                            )
                            
                            # Total loss
                            loss = latent_loss + 0.1 * flow_loss
                            
                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                self.encoder.parameters(), 1.0
                            )
                            torch.nn.utils.clip_grad_norm_(
                                self.dynamics.parameters(), 1.0
                            )
                            self.optimizer.step()
                            
                            total_loss += loss.item()
                            num_batches += 1
                        except Exception as e:
                            logger.warning(f"Error processing sequence {i}: {e}")
                            continue
                
                if epoch % 10 == 0:
                    avg_loss = total_loss / max(num_batches, 1)
                    logger.info(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            logger.info("Training completed")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise TrainingError(f"Failed to train model: {e}") from e
    
    def simulate_attack_scenario(
        self, 
        initial_state: Dict[str, torch.Tensor], 
        attack_sequence: List[Dict[str, Any]]
    ) -> tuple:
        """
        Simulate a complete attack kill chain.
        
        Args:
            initial_state: Initial network state dict with 'network', 'flows', 'events'
            attack_sequence: List of attack steps, each with 'action' key
            
        Returns:
            Tuple of (states, predictions) where states is list of state dicts
            and predictions is list of prediction dicts
            
        Raises:
            ModelError: If simulation fails
        """
        try:
            self.encoder.eval()
            self.dynamics.eval()
            
            states = [initial_state]
            predictions = []
            
            with torch.no_grad():
                current_state = initial_state
                
                for attack_step in attack_sequence:
                    # Encode current state
                    latent = self.encoder(
                        current_state['network'],
                        current_state['flows'],
                        current_state['events']
                    )
                    
                    # Get attack action
                    action = attack_step.get('action', 0)
                    if not isinstance(action, torch.Tensor):
                        attack_action = torch.tensor([action], dtype=torch.long)
                    else:
                        attack_action = action
                    
                    # Predict next state
                    next_latent, pred_flows, pred_threats, pred_vulns = \
                        self.dynamics(latent, attack_action)
                    
                    # Decode predictions (simplified)
                    predicted_state = {
                        'threat_level': pred_threats.cpu().numpy(),
                        'new_vulnerabilities': pred_vulns.cpu().numpy(),
                        'anomalous_flows': pred_flows.cpu().numpy()
                    }
                    
                    predictions.append(predicted_state)
                    
                    # Update current state (in reality, we'd decode properly)
                    # Ensure flows tensor has correct shape (batch, seq_len, features)
                    if len(pred_flows.shape) == 1:
                        # If 1D, reshape to (1, 1, features) then repeat for seq_len
                        pred_flows_expanded = pred_flows.unsqueeze(0).unsqueeze(0)  # (1, 1, 20)
                        pred_flows_expanded = pred_flows_expanded.repeat(1, current_state['flows'].shape[1], 1)  # (1, seq_len, 20)
                    elif len(pred_flows.shape) == 2:
                        # If 2D (batch, features), add sequence dimension
                        pred_flows_expanded = pred_flows.unsqueeze(1)  # (1, 1, 20)
                        pred_flows_expanded = pred_flows_expanded.repeat(1, current_state['flows'].shape[1], 1)  # (1, seq_len, 20)
                    else:
                        pred_flows_expanded = pred_flows
                    
                    current_state = {
                        'network': current_state['network'],  # Keep original network state
                        'flows': pred_flows_expanded,  # Updated flows with correct shape
                        'events': current_state['events']  # Keep original events
                    }
                    
                    states.append(current_state)
            
            logger.info(f"Simulated {len(attack_sequence)} attack steps")
            return states, predictions
        except Exception as e:
            logger.error(f"Error simulating attack scenario: {e}")
            raise ModelError(f"Failed to simulate attack scenario: {e}") from e
    
    def generate_adversarial_scenarios(
        self, 
        network_config: Dict[str, Any], 
        num_scenarios: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate novel attack scenarios using the world model.
        
        Args:
            network_config: Network configuration dict
            num_scenarios: Number of scenarios to generate
            
        Returns:
            List of scenario dicts with phases, timeline, IOCs, and impact
        """
        scenarios = []
        
        for _ in range(num_scenarios):
            scenario = {
                'phases': [],
                'timeline': [],
                'indicators_of_compromise': [],
                'predicted_impact': None
            }
            
            # Start with reconnaissance
            current_phase = 0
            success_probability = 1.0
            
            while current_phase < len(self.attack_types) and success_probability > 0.3:
                # Select next attack technique
                if current_phase == 0:
                    # Initial phase: reconnaissance
                    next_action = 0
                else:
                    # Progress through kill chain
                    next_action = current_phase
                
                # Simulate this phase
                phase_result = self._simulate_attack_phase(
                    network_config, 
                    next_action,
                    success_probability
                )
                
                scenario['phases'].append(phase_result)
                scenario['timeline'].append({
                    'phase': self.attack_types[next_action],
                    'time_estimate': phase_result['estimated_time'],
                    'detection_probability': phase_result['detection_risk']
                })
                
                # Update for next phase
                success_probability *= phase_result['success_rate']
                current_phase += 1
            
            scenarios.append(scenario)
        
        logger.info(f"Generated {num_scenarios} adversarial scenarios")
        return scenarios
    
    def _simulate_attack_phase(
        self, 
        network: Dict[str, Any], 
        attack_action: int, 
        prev_success_rate: float
    ) -> Dict[str, Any]:
        """
        Simulate a single attack phase.
        
        Args:
            network: Network configuration
            attack_action: Attack action index
            prev_success_rate: Previous phase success rate
            
        Returns:
            Dict with attack phase results
        """
        # This would use the dynamics model to simulate the phase
        # For demonstration, return simulated results
        return {
            'attack_type': self.attack_types[attack_action],
            'success_rate': np.random.uniform(0.6, 0.95),
            'detection_risk': np.random.uniform(0.1, 0.7),
            'estimated_time': np.random.randint(1, 24),  # hours
            'required_resources': np.random.randint(1, 5),
            'footprint': np.random.choice(['Low', 'Medium', 'High'])
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'dynamics_state_dict': self.dynamics.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.dynamics.load_state_dict(checkpoint['dynamics_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {filepath}")

