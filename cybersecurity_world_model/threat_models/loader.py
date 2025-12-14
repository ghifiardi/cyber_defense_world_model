"""Threat model loader for use case integration."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import DataError

logger = get_logger(__name__)


class ThreatModelLoader:
    """Load and parse threat modeling use cases."""
    
    def __init__(self, threat_models_dir: Optional[str] = None):
        """
        Initialize threat model loader.
        
        Args:
            threat_models_dir: Directory containing threat model files
        """
        if threat_models_dir is None:
            # Default to threat_models directory in project root
            project_root = Path(__file__).parent.parent.parent
            threat_models_dir = project_root / "threat_models"
        
        self.threat_models_dir = Path(threat_models_dir)
        self.manifest_path = self.threat_models_dir / "usecases_manifest.json"
        self.manifest = None
        self.use_cases = {}
        
        self._load_manifest()
        logger.info(f"ThreatModelLoader initialized with {len(self.use_cases)} use cases")
    
    def _load_manifest(self):
        """Load the use cases manifest."""
        try:
            if not self.manifest_path.exists():
                logger.warning(f"Manifest not found at {self.manifest_path}")
                return
            
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            # Index use cases by ID
            for uc in self.manifest.get('use_cases', []):
                self.use_cases[uc['id']] = uc
            
            logger.info(f"Loaded manifest with {len(self.use_cases)} use cases")
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            raise DataError(f"Failed to load threat model manifest: {e}") from e
    
    def get_use_case(self, use_case_id: str) -> Optional[Dict[str, Any]]:
        """
        Get use case by ID.
        
        Args:
            use_case_id: Use case ID (e.g., "UC011", "UC010")
            
        Returns:
            Use case dictionary or None if not found
        """
        return self.use_cases.get(use_case_id.upper())
    
    def get_use_cases_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all use cases in a category.
        
        Args:
            category: Category name (e.g., "Advanced Threats", "Data Protection")
            
        Returns:
            List of use case dictionaries
        """
        return [
            uc for uc in self.use_cases.values()
            if uc.get('category') == category
        ]
    
    def get_use_cases_by_expert(self, expert_type: str) -> List[Dict[str, Any]]:
        """
        Get all use cases for an expert type.
        
        Args:
            expert_type: Expert type (e.g., "Threat Intelligence Expert")
            
        Returns:
            List of use case dictionaries
        """
        return [
            uc for uc in self.use_cases.values()
            if uc.get('expert_type') == expert_type
        ]
    
    def get_mitre_tactics(self, use_case_id: str) -> List[str]:
        """
        Get MITRE ATT&CK tactics for a use case.
        
        Args:
            use_case_id: Use case ID
            
        Returns:
            List of MITRE tactic IDs
        """
        uc = self.get_use_case(use_case_id)
        if uc:
            return uc.get('mitre_tactics', [])
        return []
    
    def list_all_use_cases(self) -> List[Dict[str, Any]]:
        """Get all use cases."""
        return list(self.use_cases.values())
    
    def get_attack_sequence_from_use_case(
        self, 
        use_case_id: str
    ) -> List[Dict[str, int]]:
        """
        Extract attack sequence from use case based on MITRE tactics.
        
        Args:
            use_case_id: Use case ID
            
        Returns:
            List of attack action dictionaries
        """
        uc = self.get_use_case(use_case_id)
        if not uc:
            return []
        
        # Map MITRE tactics to world model attack actions
        mitre_to_action = {
            'TA0001': 1,  # INITIAL_ACCESS
            'TA0002': 2,  # EXECUTION
            'TA0003': 3,  # PERSISTENCE
            'TA0004': 4,  # PRIVILEGE_ESCALATION
            'TA0005': 5,  # DEFENSE_EVASION
            'TA0006': 6,  # CREDENTIAL_ACCESS
            'TA0007': 7,  # DISCOVERY
            'TA0008': 8,  # LATERAL_MOVEMENT
            'TA0009': 9,  # COLLECTION
            'TA0010': 10, # EXFILTRATION
            'TA0011': 11, # COMMAND_CONTROL
            'TA0040': 12, # IMPACT
        }
        
        tactics = uc.get('mitre_tactics', [])
        attack_sequence = []
        
        for tactic in tactics:
            action = mitre_to_action.get(tactic)
            if action is not None:
                attack_sequence.append({
                    'action': action,
                    'tactic': tactic,
                    'use_case_id': use_case_id,
                    'use_case_title': uc.get('title', '')
                })
        
        return attack_sequence
    
    def get_ransomware_scenarios(self) -> List[Dict[str, Any]]:
        """Get ransomware-related use cases."""
        return self.get_use_cases_by_category('Advanced Threats') + \
               [uc for uc in self.use_cases.values() 
                if 'ransomware' in uc.get('title', '').lower()]
    
    def get_apt_scenarios(self) -> List[Dict[str, Any]]:
        """Get APT-related use cases."""
        return [uc for uc in self.use_cases.values() 
                if 'APT' in uc.get('title', '') or 'apt' in uc.get('title', '').lower()]

