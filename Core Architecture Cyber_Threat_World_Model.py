import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from collections import deque
import json

class CyberStateEncoder(nn.Module):
    """
    Encodes network state into latent representation
    Network state includes: topology, traffic flows, security events
    """
    def __init__(self, feature_dim=256):
        super().__init__()
        
        # Multi-modal encoder for different data types
        self.network_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(128*64, 512)
        )
        
        self.flow_encoder = nn.LSTM(
            input_size=20,  # flow features
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        self.event_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=256,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + 64, 1024),
            nn.ReLU(),
            nn.Linear(1024, feature_dim)
        )
    
    def forward(self, network_tensor, flow_tensor, event_tensor):
        # Encode network topology
        net_feat = self.network_encoder(network_tensor.unsqueeze(1))
        
        # Encode traffic flows
        _, (flow_feat, _) = self.flow_encoder(flow_tensor)
        flow_feat = flow_feat[-1]  # Last layer hidden state
        
        # Encode security events
        event_feat = self.event_encoder(event_tensor)
        event_feat = event_feat.mean(dim=1)
        
        # Fusion
        combined = torch.cat([net_feat, flow_feat, event_feat], dim=1)
        latent = self.fusion(combined)
        
        return latent

class AttackDynamicsModel(nn.Module):
    """
    Predicts next network state given current state and attack action
    """
    def __init__(self, latent_dim=256, action_dim=50):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Attack action embedding (tactics, techniques, procedures)
        self.action_embedder = nn.Embedding(action_dim, 64)
        
        # Dynamics predictor
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + 64, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, latent_dim)
        )
        
        # Multi-task decoders for different prediction tasks
        self.flow_predictor = nn.Linear(latent_dim, 20)  # Traffic features
        self.threat_predictor = nn.Linear(latent_dim, 10)  # Threat levels
        self.vulnerability_predictor = nn.Linear(latent_dim, 5)  # New vulns
        
    def forward(self, latent_state, attack_action):
        # Embed attack action
        action_emb = self.action_embedder(attack_action)
        
        # Predict next latent state
        next_latent = self.dynamics(torch.cat([latent_state, action_emb], dim=1))
        
        # Predict various security metrics
        predicted_flows = self.flow_predictor(next_latent)
        predicted_threats = self.threat_predictor(next_latent)
        predicted_vulns = self.vulnerability_predictor(next_latent)
        
        return next_latent, predicted_flows, predicted_threats, predicted_vulns

class CyberWorldModel:
    """
    Complete cybersecurity world model for threat simulation
    """
    def __init__(self):
        self.encoder = CyberStateEncoder()
        self.dynamics = AttackDynamicsModel()
        
        # Attack dictionary (MITRE ATT&CK inspired)
        self.attack_types = {
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
        self.defense_actions = {
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
        
        # Training optimizers
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        
    def train_on_attack_sequences(self, attack_dataset):
        """
        Train on historical attack sequences
        """
        self.encoder.train()
        self.dynamics.train()
        
        for epoch in range(100):
            total_loss = 0
            
            for sequence in attack_dataset:
                # Each sequence: [state0, action0, state1, action1, ...]
                for i in range(len(sequence) - 1):
                    current_state = sequence[i]['state']
                    action = sequence[i]['action']
                    next_state = sequence[i+1]['state']
                    
                    # Encode current state
                    latent = self.encoder(
                        current_state['network'],
                        current_state['flows'],
                        current_state['events']
                    )
                    
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
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def simulate_attack_scenario(self, initial_state, attack_sequence):
        """
        Simulate a complete attack kill chain
        """
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
                attack_action = torch.tensor([attack_step['action']])
                
                # Predict next state
                next_latent, pred_flows, pred_threats, pred_vulns = \
                    self.dynamics(latent, attack_action)
                
                # Decode predictions (simplified)
                predicted_state = {
                    'threat_level': pred_threats.numpy(),
                    'new_vulnerabilities': pred_vulns.numpy(),
                    'anomalous_flows': pred_flows.numpy()
                }
                
                predictions.append(predicted_state)
                
                # Update current state (in reality, we'd decode properly)
                current_state = {
                    'network': current_state['network'],  # Simplified
                    'flows': torch.tensor(pred_flows).unsqueeze(0),
                    'events': current_state['events']  # Simplified
                }
                
                states.append(current_state)
        
        return states, predictions
    
    def generate_adversarial_scenarios(self, network_config, num_scenarios=5):
        """
        Generate novel attack scenarios using the world model
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
        
        return scenarios
    
    def _simulate_attack_phase(self, network, attack_action, prev_success_rate):
        """
        Simulate a single attack phase
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
