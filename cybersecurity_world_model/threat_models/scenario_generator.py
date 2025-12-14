"""Generate attack scenarios from threat models."""

from typing import Dict, List, Any, Optional
from cybersecurity_world_model.threat_models.loader import ThreatModelLoader
from cybersecurity_world_model.utils.logging import get_logger

logger = get_logger(__name__)


class ThreatScenarioGenerator:
    """Generate realistic attack scenarios from threat models."""
    
    # Attack type descriptions from threat models
    ATTACK_DESCRIPTIONS = {
        'UC011': {  # Ransomware Early Warning
            'RECONNAISSANCE': 'Ransomware group reconnaissance targeting telecom sector',
            'INITIAL_ACCESS': 'Exploit public-facing application (VPN, firewall, web apps)',
            'EXECUTION': 'Ransomware payload execution',
            'PERSISTENCE': 'Establish persistence mechanisms',
            'COLLECTION': 'Data collection before encryption',
            'EXFILTRATION': 'Data exfiltration to leak site',
            'IMPACT': 'Data encrypted for impact (ransomware encryption)'
        },
        'UC010': {  # APT Malware Detection
            'INITIAL_ACCESS': 'APT malware delivered via phishing, exploit, or supply chain',
            'EXECUTION': 'Malware binary execution (SEASPY, SALTWATER, AngryRebel.Linux)',
            'PERSISTENCE': 'Establish persistence (registry, scheduled task, service)',
            'DEFENSE_EVASION': 'Process injection into legitimate process for stealth',
            'CREDENTIAL_ACCESS': 'Credential harvesting (Mimikatz-like functionality)',
            'LATERAL_MOVEMENT': 'Move to additional systems within network',
            'EXFILTRATION': 'Exfiltrate sensitive data via C2 channel'
        },
        'UC013': {  # Vulnerability Exploit Forecast
            'RECONNAISSANCE': 'Vulnerability scanning and research',
            'INITIAL_ACCESS': 'Exploit predicted high-probability vulnerability',
            'EXECUTION': 'Exploit code execution',
            'PRIVILEGE_ESCALATION': 'Privilege escalation via exploited vulnerability',
            'LATERAL_MOVEMENT': 'Lateral movement using compromised credentials'
        }
    }
    
    def __init__(self, threat_model_loader: Optional[ThreatModelLoader] = None):
        """
        Initialize scenario generator.
        
        Args:
            threat_model_loader: ThreatModelLoader instance
        """
        self.loader = threat_model_loader or ThreatModelLoader()
        logger.info("ThreatScenarioGenerator initialized")
    
    def generate_scenario_from_use_case(
        self,
        use_case_id: str,
        scenario_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate attack scenario from a threat model use case.
        
        Args:
            use_case_id: Use case ID (e.g., "UC011", "UC010")
            scenario_name: Optional custom scenario name
            
        Returns:
            Scenario dictionary with attack sequence and metadata
        """
        uc = self.loader.get_use_case(use_case_id)
        if not uc:
            logger.warning(f"Use case {use_case_id} not found")
            return {}
        
        attack_sequence = self.loader.get_attack_sequence_from_use_case(use_case_id)
        descriptions = self.ATTACK_DESCRIPTIONS.get(use_case_id, {})
        
        # Enhance attack sequence with descriptions
        enhanced_sequence = []
        for step in attack_sequence:
            action_name = self._get_action_name(step['action'])
            enhanced_sequence.append({
                'action': step['action'],
                'name': action_name,
                'description': descriptions.get(
                    action_name,
                    f"{action_name} phase from {uc.get('title', '')}"
                ),
                'tactic': step.get('tactic', ''),
                'use_case_id': use_case_id,
                'use_case_title': uc.get('title', '')
            })
        
        scenario = {
            'scenario_id': use_case_id,  # This is the use case ID (UC011, etc.)
            'use_case_id': use_case_id,   # Alias for compatibility
            'scenario_name': scenario_name or uc.get('title', ''),
            'category': uc.get('category', ''),
            'expert_type': uc.get('expert_type', ''),
            'mitre_tactics': uc.get('mitre_tactics', []),
            'attack_sequence': enhanced_sequence,
            'severity': 'HIGH',  # Most threat models are high severity
            'source': 'threat_model'
        }
        
        logger.info(f"Generated scenario from {use_case_id}: {uc.get('title', '')}")
        return scenario
    
    def generate_ransomware_scenario(self) -> Dict[str, Any]:
        """Generate ransomware attack scenario from UC011."""
        return self.generate_scenario_from_use_case('UC011', 'Ransomware Early Warning Attack')
    
    def generate_apt_scenario(self) -> Dict[str, Any]:
        """Generate APT malware scenario from UC010."""
        return self.generate_scenario_from_use_case('UC010', 'APT Malware Attack')
    
    def generate_vulnerability_exploit_scenario(self) -> Dict[str, Any]:
        """Generate vulnerability exploit scenario from UC013."""
        return self.generate_scenario_from_use_case('UC013', 'Vulnerability Exploit Forecast Attack')
    
    def generate_multiple_scenarios(
        self,
        use_case_ids: Optional[List[str]] = None,
        max_scenarios: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple scenarios from threat models.
        
        Args:
            use_case_ids: Optional list of specific use case IDs
            max_scenarios: Maximum number of scenarios to generate
            
        Returns:
            List of scenario dictionaries
        """
        scenarios = []
        
        if use_case_ids:
            use_cases_to_process = use_case_ids[:max_scenarios]
        else:
            # Get high-priority use cases
            priority_use_cases = ['UC011', 'UC010', 'UC013', 'UC012', 'UC014']
            use_cases_to_process = priority_use_cases[:max_scenarios]
        
        for uc_id in use_cases_to_process:
            scenario = self.generate_scenario_from_use_case(uc_id)
            if scenario:
                scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} scenarios from threat models")
        return scenarios
    
    def _get_action_name(self, action_id: int) -> str:
        """Map action ID to attack type name."""
        action_map = {
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
        return action_map.get(action_id, "UNKNOWN")

