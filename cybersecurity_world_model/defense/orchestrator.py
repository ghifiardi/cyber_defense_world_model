"""PredictiveDefenseOrchestrator - Main proactive defense system orchestrator."""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Optional
from cybersecurity_world_model.defense.predictors import TemporalAttackPredictor
from cybersecurity_world_model.defense.detectors import BehavioralAnomalyDetector
from cybersecurity_world_model.defense.graph_generator import AttackGraphGenerator
from cybersecurity_world_model.config import Config
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import PredictionError, DataError

logger = get_logger(__name__)


class PredictiveDefenseOrchestrator:
    """
    Main proactive defense system that orchestrates all components.
    """
    
    ATTACK_TYPES = [
        'RECONNAISSANCE', 'INITIAL_ACCESS', 'EXECUTION', 'PERSISTENCE',
        'PRIVILEGE_ESCALATION', 'DEFENSE_EVASION', 'CREDENTIAL_ACCESS',
        'DISCOVERY', 'LATERAL_MOVEMENT', 'COLLECTION', 'EXFILTRATION',
        'COMMAND_CONTROL', 'IMPACT'
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the proactive defense orchestrator.
        
        Args:
            config: Optional configuration object
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING PROACTIVE DEFENSE SYSTEM")
        logger.info("=" * 70)
        
        self.config = config or Config()
        
        # Initialize components
        self.predictor = TemporalAttackPredictor(
            input_dim=256,
            forecast_horizon=self.config.get('defense.forecast_horizon_hours', 24)
        )
        self.anomaly_detector = BehavioralAnomalyDetector()
        self.attack_graph_generator = AttackGraphGenerator()
        
        # Threat intelligence feeds
        self.threat_intel = self._init_threat_intel()
        
        # Attack prediction history
        self.prediction_history = deque(maxlen=1000)
        
        # Network model
        self.network_model = None
        
        # Early warning thresholds
        self.warning_thresholds = self.config.get('defense.warning_thresholds', {
            'high_confidence': 0.85,
            'medium_confidence': 0.65,
            'imminent_threat': 0.9,
            'anomaly_threshold': 0.8
        })
        
        # Defense actions database
        self.defense_actions = self._init_defense_actions()
        
        logger.info("[✓] Proactive Defense System Initialized")
    
    def predict_attacks(
        self, 
        telemetry_data: Any, 
        forecast_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Main prediction method - predicts attacks before they happen.
        
        Args:
            telemetry_data: Telemetry data (DataFrame, numpy array, or tensor)
            forecast_hours: Hours to forecast ahead
            
        Returns:
            Dict with prediction results, warnings, and recommendations
            
        Raises:
            PredictionError: If prediction fails
        """
        try:
            logger.info(f"[PREDICTION] Analyzing telemetry for next {forecast_hours} hours...")
            
            # Preprocess telemetry
            processed_data = self._preprocess_telemetry(telemetry_data)
            
            # Generate attack predictions
            with torch.no_grad():
                forecasts, attack_probs, confidence = self.predictor(processed_data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(processed_data)
            
            # Generate attack graphs if network model exists
            attack_graphs = []
            if self.network_model:
                attack_graphs = self._generate_attack_graphs()
            
            # Correlate predictions with threat intelligence
            correlated_threats = self._correlate_with_intel(
                forecasts, attack_probs, anomalies
            )
            
            # Generate early warnings
            warnings = self._generate_early_warnings(
                correlated_threats, confidence, anomalies
            )
            
            # Generate proactive defense recommendations
            recommendations = self._generate_defense_recommendations(
                warnings, attack_graphs
            )
            
            # Log prediction
            prediction_record = {
                'timestamp': datetime.now(),
                'forecasts': forecasts.cpu().numpy() if isinstance(forecasts, torch.Tensor) else forecasts,
                'attack_probs': attack_probs.cpu().numpy() if isinstance(attack_probs, torch.Tensor) else attack_probs,
                'confidence': confidence.cpu().numpy() if isinstance(confidence, torch.Tensor) else confidence,
                'anomalies': anomalies,
                'warnings': warnings,
                'recommendations': recommendations
            }
            
            self.prediction_history.append(prediction_record)
            
            return {
                'status': 'PREDICTION_COMPLETE',
                'timestamp': datetime.now().isoformat(),
                'forecast_horizon': f'{forecast_hours} hours',
                'confidence_level': float(confidence.mean()) if isinstance(confidence, torch.Tensor) else float(np.mean(confidence)),
                'predicted_attacks': self._format_predictions(attack_probs),
                'anomalies_detected': len(anomalies),
                'early_warnings': warnings,
                'defense_recommendations': recommendations,
                'attack_graphs': attack_graphs[:3] if attack_graphs else []
            }
        except Exception as e:
            logger.error(f"Error in attack prediction: {e}")
            raise PredictionError(f"Failed to predict attacks: {e}") from e
    
    def _preprocess_telemetry(self, telemetry_data: Any) -> torch.Tensor:
        """Preprocess telemetry data for prediction."""
        try:
            # Convert to tensor
            if isinstance(telemetry_data, pd.DataFrame):
                tensor_data = torch.tensor(telemetry_data.values, dtype=torch.float32)
            elif isinstance(telemetry_data, np.ndarray):
                tensor_data = torch.tensor(telemetry_data, dtype=torch.float32)
            elif isinstance(telemetry_data, torch.Tensor):
                tensor_data = telemetry_data
            else:
                raise DataError(f"Unsupported telemetry data type: {type(telemetry_data)}")
            
            # Normalize
            tensor_data = (tensor_data - tensor_data.mean()) / (tensor_data.std() + 1e-8)
            
            # Ensure correct shape: (batch, seq_len, features)
            if len(tensor_data.shape) == 2:
                tensor_data = tensor_data.unsqueeze(0)
            
            return tensor_data
        except Exception as e:
            logger.error(f"Error preprocessing telemetry: {e}")
            raise DataError(f"Failed to preprocess telemetry: {e}") from e
    
    def _detect_anomalies(self, telemetry_data: torch.Tensor) -> List[Dict[str, Any]]:
        """Detect anomalous behavior patterns."""
        anomalies = []
        
        try:
            with torch.no_grad():
                for i in range(telemetry_data.shape[0]):
                    # Get the last sequence (most recent)
                    sequence = telemetry_data[i:i+1, -10:, :]  # Last 10 timesteps
                    
                    # Reshape for anomaly detector
                    batch_size, seq_len, features = sequence.shape
                    reshaped = sequence.reshape(-1, features)
                    
                    # Get anomaly scores
                    outputs = self.anomaly_detector(reshaped)
                    
                    # Find anomalies
                    anomaly_mask = outputs['anomaly_score'] > self.warning_thresholds['anomaly_threshold']
                    
                    if anomaly_mask.any():
                        anomaly_indices = torch.where(anomaly_mask)[0]
                        for idx in anomaly_indices:
                            anomalies.append({
                                'timestamp_offset': idx.item(),
                                'anomaly_score': float(outputs['anomaly_score'][idx]),
                                'reconstruction_error': float(outputs['reconstruction_loss'][idx])
                            })
        except Exception as e:
            logger.warning(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _correlate_with_intel(
        self, 
        forecasts: torch.Tensor, 
        attack_probs: torch.Tensor, 
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correlate predictions with threat intelligence."""
        correlated_threats = []
        
        try:
            # Get predicted attack types
            if isinstance(attack_probs, torch.Tensor):
                predicted_attack_indices = torch.argsort(attack_probs[0], descending=True)[:3]
            else:
                predicted_attack_indices = np.argsort(attack_probs[0])[::-1][:3]
            
            for idx in predicted_attack_indices:
                idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                attack_type = self.ATTACK_TYPES[idx_val] if idx_val < len(self.ATTACK_TYPES) else 'UNKNOWN'
                
                if isinstance(attack_probs, torch.Tensor):
                    attack_prob = float(attack_probs[0, idx_val])
                else:
                    attack_prob = float(attack_probs[0, idx_val])
                
                # Check threat intel for this attack type
                intel_matches = self._check_threat_intel(attack_type)
                
                if intel_matches or len(anomalies) > 0:
                    correlated_threats.append({
                        'attack_type': attack_type,
                        'probability': attack_prob,
                        'intel_matches': intel_matches,
                        'correlation_score': attack_prob * (1 + len(intel_matches) * 0.2)
                    })
            
            # Sort by correlation score
            correlated_threats.sort(key=lambda x: x['correlation_score'], reverse=True)
        except Exception as e:
            logger.warning(f"Error correlating with intel: {e}")
        
        return correlated_threats
    
    def _generate_early_warnings(
        self, 
        correlated_threats: List[Dict[str, Any]], 
        confidence: torch.Tensor, 
        anomalies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate early warning alerts."""
        warnings = []
        
        try:
            # Check confidence threshold
            if isinstance(confidence, torch.Tensor):
                avg_confidence = float(confidence.mean())
            else:
                avg_confidence = float(np.mean(confidence))
            
            if avg_confidence > self.warning_thresholds['high_confidence']:
                warning_level = 'CRITICAL'
            elif avg_confidence > self.warning_thresholds['medium_confidence']:
                warning_level = 'HIGH'
            else:
                warning_level = 'MEDIUM'
            
            # Generate warnings for correlated threats
            for threat in correlated_threats:
                if threat['correlation_score'] > 0.7:
                    warning = {
                        'level': warning_level,
                        'type': 'PREDICTED_ATTACK',
                        'attack_type': threat['attack_type'],
                        'confidence': threat['probability'],
                        'predicted_time_window': self._estimate_time_window(threat),
                        'recommended_actions': self._get_preemptive_actions(threat['attack_type']),
                        'rationale': f"Correlated with {len(threat['intel_matches'])} intel sources"
                    }
                    warnings.append(warning)
            
            # Generate warnings for anomalies
            for anomaly in anomalies:
                if anomaly['anomaly_score'] > self.warning_thresholds['anomaly_threshold']:
                    warning = {
                        'level': 'HIGH',
                        'type': 'BEHAVIORAL_ANOMALY',
                        'anomaly_score': anomaly['anomaly_score'],
                        'timestamp_offset': anomaly['timestamp_offset'],
                        'recommended_actions': ['Investigate anomalous behavior', 'Increase monitoring'],
                        'rationale': 'Unusual behavior pattern detected'
                    }
                    warnings.append(warning)
        except Exception as e:
            logger.warning(f"Error generating warnings: {e}")
        
        return warnings
    
    def _generate_defense_recommendations(
        self, 
        warnings: List[Dict[str, Any]], 
        attack_graphs: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate proactive defense recommendations."""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        try:
            # Immediate recommendations based on warnings
            for warning in warnings:
                if warning['level'] in ['CRITICAL', 'HIGH']:
                    recommendations['immediate'].extend(warning.get('recommended_actions', [])[:3])
            
            # Recommendations based on attack graphs
            if attack_graphs:
                for graph in attack_graphs[:2]:  # Top 2 graphs
                    for path in graph.get('critical_paths', []):
                        if path.get('probability', 0) > 0.7:
                            recommendations['short_term'].append(
                                f"Strengthen defenses on path: {' -> '.join(path['path'][:3])}"
                            )
            
            # Add general proactive measures
            recommendations['long_term'].extend([
                "Implement zero-trust architecture",
                "Deploy deception technology",
                "Enhance endpoint detection and response",
                "Conduct regular penetration testing",
                "Update incident response plan"
            ])
            
            # Remove duplicates
            for key in recommendations:
                recommendations[key] = list(set(recommendations[key]))
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _generate_attack_graphs(self) -> List[Dict[str, Any]]:
        """Generate attack graphs for current network."""
        if not self.network_model:
            return []
        
        try:
            # Simulate network assets (in real implementation, this would come from CMDB)
            network_assets = [
                {'id': 'EXT-1', 'type': 'external', 'criticality': 0.3, 'vulnerabilities': ['CVE-2023-1234']},
                {'id': 'FW-1', 'type': 'firewall', 'criticality': 0.8, 'controls': ['ACL', 'IDS']},
                {'id': 'WEB-1', 'type': 'web_server', 'criticality': 0.7, 'vulnerabilities': ['CVE-2023-5678']},
                {'id': 'DB-1', 'type': 'database', 'criticality': 0.9, 'controls': ['Encryption', 'Access Control']},
                {'id': 'AD-1', 'type': 'domain_controller', 'criticality': 0.95}
            ]
            
            vulnerabilities = ['CVE-2023-1234', 'CVE-2023-5678', 'CVE-2023-9101']
            
            attack_graph = self.attack_graph_generator.generate_attack_graph(
                network_assets, vulnerabilities
            )
            
            return [attack_graph]
        except Exception as e:
            logger.warning(f"Error generating attack graphs: {e}")
            return []
    
    def _init_threat_intel(self) -> Dict[str, Any]:
        """Initialize threat intelligence sources."""
        return {
            'ioc_feed': self._fetch_live_iocs(),
            'vulnerability_feed': self._fetch_vulnerabilities(),
            'threat_actor_profiles': self._load_threat_actors(),
            'geo_threat_data': self._load_geo_threats()
        }
    
    def _init_defense_actions(self) -> Dict[str, List[str]]:
        """Initialize proactive defense actions."""
        return {
            'preemptive': [
                'Patch critical vulnerabilities',
                'Update firewall rules',
                'Rotate credentials',
                'Isolate suspicious segments',
                'Deploy deception technology',
                'Increase monitoring',
                'Update IDS/IPS signatures',
                'Test backup systems'
            ],
            'reactive': [
                'Block malicious IPs',
                'Quarantine compromised hosts',
                'Kill malicious processes',
                'Revoke compromised credentials',
                'Initiate incident response',
                'Notify stakeholders',
                'Engage threat hunting',
                'Collect forensic evidence'
            ]
        }
    
    def _check_threat_intel(self, attack_type: str) -> List[str]:
        """Check threat intelligence for specific attack type."""
        # Simulated threat intelligence check
        intel_sources = [
            'CISA Alerts',
            'MITRE ATT&CK',
            'Vendor Advisories',
            'Open Source Intelligence'
        ]
        
        # Simulate matches (in reality, this would query threat intel databases)
        matches = np.random.choice(intel_sources, 
                                 size=np.random.randint(0, 3), 
                                 replace=False)
        
        return list(matches)
    
    def _estimate_time_window(self, threat: Dict[str, Any]) -> str:
        """Estimate when attack is likely to occur."""
        prob = threat.get('probability', 0.5)
        
        if prob > 0.8:
            return "0-6 hours"
        elif prob > 0.6:
            return "6-24 hours"
        elif prob > 0.4:
            return "1-3 days"
        else:
            return "3-7 days"
    
    def _get_preemptive_actions(self, attack_type: str) -> List[str]:
        """Get preemptive defense actions for specific attack type."""
        actions_map = {
            'RECONNAISSANCE': [
                'Monitor for scanning activity',
                'Deploy honeypots',
                'Obscure system information'
            ],
            'INITIAL_ACCESS': [
                'Patch public-facing applications',
                'Review firewall rules',
                'Implement WAF rules'
            ],
            'EXECUTION': [
                'Restrict script execution',
                'Implement application whitelisting',
                'Monitor process creation'
            ],
            'PERSISTENCE': [
                'Review autostart locations',
                'Monitor registry changes',
                'Check scheduled tasks'
            ],
            'PRIVILEGE_ESCALATION': [
                'Apply least privilege principle',
                'Monitor privilege escalation attempts',
                'Review user permissions'
            ]
        }
        
        return actions_map.get(attack_type, [
            'Increase monitoring',
            'Review logs',
            'Check for indicators of compromise'
        ])
    
    def _format_predictions(self, attack_probs: torch.Tensor) -> List[Dict[str, Any]]:
        """Format predictions for output."""
        predictions = []
        
        try:
            if isinstance(attack_probs, torch.Tensor):
                probs_array = attack_probs[0].cpu().numpy()
            else:
                probs_array = attack_probs[0]
            
            for i in range(len(probs_array)):
                prob = float(probs_array[i])
                if prob > 0.3:  # Only show predictions with >30% probability
                    predictions.append({
                        'attack_type': self.ATTACK_TYPES[i] if i < len(self.ATTACK_TYPES) else 'UNKNOWN',
                        'probability': prob,
                        'confidence_band': [prob * 0.8, min(prob * 1.2, 1.0)]
                    })
            
            # Sort by probability
            predictions.sort(key=lambda x: x['probability'], reverse=True)
        except Exception as e:
            logger.warning(f"Error formatting predictions: {e}")
        
        return predictions
    
    def _fetch_live_iocs(self) -> List[Dict[str, str]]:
        """Simulate fetching live IOCs."""
        return [
            {'type': 'IP', 'value': '192.168.1.100', 'threat': 'C2 Server'},
            {'type': 'Domain', 'value': 'evil.com', 'threat': 'Phishing'},
            {'type': 'Hash', 'value': 'abc123', 'threat': 'Ransomware'}
        ]
    
    def _fetch_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Simulate vulnerability feed."""
        return [
            {'cve': 'CVE-2024-12345', 'severity': 'CRITICAL', 'patch_available': True},
            {'cve': 'CVE-2024-12346', 'severity': 'HIGH', 'patch_available': False}
        ]
    
    def _load_threat_actors(self) -> Dict[str, Dict[str, Any]]:
        """Load threat actor profiles."""
        return {
            'APT29': {
                'targets': ['Government', 'Healthcare'],
                'techniques': ['Spearphishing', 'Credential Theft', 'Lateral Movement'],
                'tools': ['Cobalt Strike', 'Mimikatz']
            },
            'FIN7': {
                'targets': ['Financial', 'Retail'],
                'techniques': ['Card Skimming', 'POS Malware', 'Backdoors'],
                'tools': ['Carbanak', 'GrimPlant']
            }
        }
    
    def _load_geo_threats(self) -> Dict[str, Any]:
        """Load geographical threat data."""
        return {
            'high_risk_regions': ['Eastern Europe', 'Southeast Asia', 'Middle East'],
            'recent_attacks_by_region': {
                'North America': 45,
                'Europe': 32,
                'Asia': 28,
                'Other': 15
            }
        }

