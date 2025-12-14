"""AttackGraphGenerator - Generates probabilistic attack graphs."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from cybersecurity_world_model.utils.logging import get_logger
from cybersecurity_world_model.exceptions import ModelError

logger = get_logger(__name__)


class AttackGraphGenerator:
    """
    Generates probabilistic attack graphs showing potential attack paths.
    """
    
    def __init__(self):
        """Initialize the attack graph generator."""
        self.mitre_techniques, self.technique_relationships = self._load_mitre_matrix()
        self.attack_graphs = {}
        logger.info("AttackGraphGenerator initialized")
    
    def _load_mitre_matrix(self) -> Tuple[Dict, Dict]:
        """Load MITRE ATT&CK techniques and relationships."""
        techniques = {
            'T1595': {'name': 'Active Scanning', 'tactic': 'Reconnaissance', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1190': {'name': 'Exploit Public-Facing Application', 'tactic': 'Initial Access', 'platforms': ['Web']},
            'T1059': {'name': 'Command and Scripting Interpreter', 'tactic': 'Execution', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1547': {'name': 'Boot or Logon Autostart Execution', 'tactic': 'Persistence', 'platforms': ['Windows', 'Linux']},
            'T1134': {'name': 'Access Token Manipulation', 'tactic': 'Privilege Escalation', 'platforms': ['Windows']},
            'T1027': {'name': 'Obfuscated Files or Information', 'tactic': 'Defense Evasion', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1003': {'name': 'OS Credential Dumping', 'tactic': 'Credential Access', 'platforms': ['Windows', 'Linux']},
            'T1018': {'name': 'Remote System Discovery', 'tactic': 'Discovery', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1021': {'name': 'Remote Services', 'tactic': 'Lateral Movement', 'platforms': ['Windows', 'Linux']},
            'T1119': {'name': 'Automated Collection', 'tactic': 'Collection', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1048': {'name': 'Exfiltration Over Alternative Protocol', 'tactic': 'Exfiltration', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1105': {'name': 'Ingress Tool Transfer', 'tactic': 'Command and Control', 'platforms': ['Windows', 'Linux', 'macOS']},
            'T1486': {'name': 'Data Encrypted for Impact', 'tactic': 'Impact', 'platforms': ['Windows', 'Linux', 'macOS']}
        }
        
        # Technique relationships (prerequisites)
        relationships = {
            'T1190': ['T1595'],  # Exploit requires scanning
            'T1059': ['T1190'],  # Execution requires initial access
            'T1547': ['T1059'],  # Persistence requires execution
            'T1134': ['T1547'],  # Privilege escalation requires persistence
            'T1027': ['T1134'],  # Defense evasion requires privilege
            'T1003': ['T1027'],  # Credential dumping requires evasion
            'T1018': ['T1003'],  # Discovery requires credentials
            'T1021': ['T1018'],  # Lateral movement requires discovery
            'T1119': ['T1021'],  # Collection requires lateral movement
            'T1048': ['T1119'],  # Exfiltration requires collection
            'T1105': ['T1048'],  # C2 requires exfiltration
            'T1486': ['T1105']   # Impact requires C2
        }
        
        return techniques, relationships
    
    def generate_attack_graph(
        self, 
        network_assets: List[Dict[str, Any]], 
        vulnerabilities: List[str]
    ) -> Dict[str, Any]:
        """
        Generate probabilistic attack graph for given network.
        
        Args:
            network_assets: List of network asset dicts with id, type, criticality, etc.
            vulnerabilities: List of vulnerability identifiers (e.g., CVE IDs)
            
        Returns:
            Dict with 'nodes', 'edges', 'attack_paths', 'critical_paths'
            
        Raises:
            ModelError: If graph generation fails
        """
        try:
            graph = {
                'nodes': [],
                'edges': [],
                'attack_paths': [],
                'critical_paths': []
            }
            
            # Add network assets as nodes
            for asset in network_assets:
                graph['nodes'].append({
                    'id': asset['id'],
                    'type': asset['type'],
                    'value': asset.get('criticality', 0.5),
                    'vulnerabilities': asset.get('vulnerabilities', []),
                    'security_controls': asset.get('controls', [])
                })
            
            # Generate possible attack edges based on vulnerabilities
            for i, src_asset in enumerate(network_assets):
                for j, dst_asset in enumerate(network_assets):
                    if i == j:
                        continue
                    
                    # Calculate attack probability based on vulnerabilities
                    attack_prob = self._calculate_attack_probability(
                        src_asset, dst_asset, vulnerabilities
                    )
                    
                    if attack_prob > 0.1:  # Only include significant probabilities
                        # Find suitable attack techniques
                        techniques = self._find_applicable_techniques(
                            src_asset, dst_asset
                        )
                        
                        for technique_id in techniques:
                            graph['edges'].append({
                                'source': src_asset['id'],
                                'target': dst_asset['id'],
                                'technique': technique_id,
                                'probability': attack_prob,
                                'difficulty': np.random.uniform(0.3, 0.9),
                                'detection_risk': np.random.uniform(0.2, 0.8)
                            })
            
            # Find all possible attack paths
            graph['attack_paths'] = self._find_all_paths(graph)
            
            # Calculate critical paths (highest probability or impact)
            graph['critical_paths'] = self._find_critical_paths(graph)
            
            logger.info(f"Generated attack graph with {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")
            return graph
        except Exception as e:
            logger.error(f"Error generating attack graph: {e}")
            raise ModelError(f"Failed to generate attack graph: {e}") from e
    
    def _calculate_attack_probability(
        self, 
        src: Dict[str, Any], 
        dst: Dict[str, Any], 
        vulnerabilities: List[str]
    ) -> float:
        """Calculate probability of successful attack."""
        base_prob = 0.3
        
        # Adjust based on vulnerabilities
        vuln_bonus = len(dst.get('vulnerabilities', [])) * 0.1
        
        # Adjust based on security controls
        control_penalty = len(dst.get('security_controls', [])) * 0.05
        
        # Network connectivity factor
        if src.get('network_segment') == dst.get('network_segment'):
            connectivity_bonus = 0.2
        else:
            connectivity_bonus = 0.1
        
        probability = base_prob + vuln_bonus - control_penalty + connectivity_bonus
        return min(max(probability, 0), 0.95)  # Clamp between 0 and 0.95
    
    def _find_applicable_techniques(
        self, 
        src: Dict[str, Any], 
        dst: Dict[str, Any]
    ) -> List[str]:
        """Find MITRE techniques applicable to this edge."""
        techniques = []
        
        # Based on source and destination types
        if src['type'] == 'external' and dst['type'] == 'dmz':
            techniques.extend(['T1595', 'T1190'])
        elif src['type'] == 'dmz' and dst['type'] == 'internal':
            techniques.extend(['T1059', 'T1134'])
        elif src['type'] == 'internal' and dst['type'] == 'database':
            techniques.extend(['T1003', 'T1119'])
        
        return techniques[:2]  # Return top 2 techniques
    
    def _find_all_paths(
        self, 
        graph: Dict[str, Any], 
        start: Optional[str] = None, 
        end: Optional[str] = None, 
        path: Optional[List[str]] = None
    ) -> List[List[str]]:
        """Find all possible attack paths in the graph."""
        if path is None:
            path = []
        
        if start is None:
            start_nodes = [n for n in graph['nodes'] if n['type'] == 'external']
            all_paths = []
            for start_node in start_nodes:
                all_paths.extend(self._find_all_paths(
                    graph, start_node['id'], None, [start_node['id']]
                ))
            return all_paths
        
        # If we reached a critical asset, save the path
        if end is None:
            current_node = next((n for n in graph['nodes'] if n['id'] == start), None)
            if current_node and current_node.get('criticality', 0) > 0.8:
                return [path.copy()]
        
        # Find all outgoing edges
        outgoing_edges = [e for e in graph['edges'] if e['source'] == start]
        
        if not outgoing_edges:
            return [path.copy()]
        
        # Explore each outgoing edge
        all_paths = []
        for edge in outgoing_edges:
            if edge['target'] not in path:  # Avoid cycles
                new_path = path + [edge['target']]
                paths_from_here = self._find_all_paths(
                    graph, edge['target'], end, new_path
                )
                all_paths.extend(paths_from_here)
        
        return all_paths
    
    def _find_critical_paths(self, graph: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Find the most critical attack paths."""
        paths_with_scores = []
        
        for path in graph['attack_paths']:
            if len(path) < 2:
                continue
            
            # Calculate path score
            path_probability = 1.0
            
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                
                # Find edge between these nodes
                edge = next(
                    (e for e in graph['edges'] 
                     if e['source'] == src and e['target'] == dst),
                    None
                )
                
                if edge:
                    path_probability *= edge['probability']
            
            # Get target criticality
            target_id = path[-1]
            target_node = next((n for n in graph['nodes'] if n['id'] == target_id), None)
            if target_node:
                target_criticality = target_node.get('criticality', 0.5)
                path_score = path_probability * target_criticality * len(path)
                
                paths_with_scores.append({
                    'path': path,
                    'score': path_score,
                    'probability': path_probability,
                    'target_criticality': target_criticality
                })
        
        # Sort by score and return top N
        paths_with_scores.sort(key=lambda x: x['score'], reverse=True)
        return paths_with_scores[:top_n]


