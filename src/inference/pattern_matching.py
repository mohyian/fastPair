"""
Pattern Matching Module

This module provides functionality for matching vulnerability patterns in code graphs.
It uses the learned patterns from the pattern learning module to identify potential
vulnerabilities in new code.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from src.utils.logger import get_logger
from src.models.pattern_learning import PatternLearningModule

logger = get_logger(__name__)

class PatternMatcher:
    """
    Class for matching vulnerability patterns in code graphs.
    """
    
    def __init__(
        self, 
        pattern_module: PatternLearningModule,
        config: Dict[str, Any]
    ):
        """
        Initialize the pattern matcher.
        
        Args:
            pattern_module: Trained pattern learning module
            config: Configuration dictionary
        """
        self.pattern_module = pattern_module
        self.config = config
        
        # Set device
        self.device = torch.device(config["training"]["device"])
        
        # Load or initialize pattern repository
        self.vuln_patterns = None
        self.security_patterns = None
        self._load_pattern_repository()
        
        # Make sure pattern module is in eval mode
        self.pattern_module.eval()
    
    def _load_pattern_repository(self) -> None:
        """
        Load the pattern repository from the pattern learning module.
        """
        if hasattr(self.pattern_module, 'vuln_pattern_repository') and self.pattern_module.vuln_pattern_repository is not None:
            self.vuln_patterns = self.pattern_module.vuln_pattern_repository.to(self.device)
            self.security_patterns = self.pattern_module.security_pattern_repository.to(self.device)
            logger.info(f"Loaded pattern repository with {self.vuln_patterns.size(0)} patterns")
        else:
            logger.warning("Pattern repository not available")
    
    def match_patterns(
        self, 
        graph: HeteroData,
        node_embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Match vulnerability patterns in a code graph.
        
        Args:
            graph: Heterogeneous graph representation of code
            node_embeddings: Pre-computed node embeddings
            
        Returns:
            Dictionary with pattern matching results
        """
        # Extract patterns from the graph
        if "code" in node_embeddings:
            # For a single-type graph
            node_type = "code"
        elif "vuln" in node_embeddings:
            # For a vulnerable-patch pair
            node_type = "vuln"
        else:
            # Use the first available node type
            node_type = list(node_embeddings.keys())[0]
        
        # Extract patterns using the pattern module
        with torch.no_grad():
            patterns, attention = self.pattern_module.extract_patterns_from_graph(
                node_embeddings[node_type],
                is_vulnerable=True  # Assume we're looking for vulnerabilities
            )
        
        # Get pattern matching results
        results = self._match_with_repository(patterns, attention)
        
        # Add node subgraph information for each match
        results = self._add_subgraph_info(graph, node_type, results, attention)
        
        return results
    
    def _match_with_repository(
        self, 
        patterns: torch.Tensor,
        attention: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Match extracted patterns with the pattern repository.
        
        Args:
            patterns: Extracted pattern embeddings
            attention: Pattern attention weights
            
        Returns:
            Dictionary with pattern matching results
        """
        # Skip if repository is not available
        if self.vuln_patterns is None:
            return {"matches": []}
        
        # Normalize patterns
        patterns_norm = torch.nn.functional.normalize(patterns, p=2, dim=1)
        vuln_patterns_norm = torch.nn.functional.normalize(self.vuln_patterns, p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mm(patterns_norm, vuln_patterns_norm.t())
        
        # Get top matches for each pattern
        top_k = min(5, similarities.size(1))
        top_scores, top_indices = torch.topk(similarities, k=top_k, dim=1)
        
        # Prepare results
        matches = []
        
        for pattern_idx in range(patterns.size(0)):
            # Get top nodes for this pattern
            _, top_node_indices = torch.topk(attention[:, pattern_idx], k=min(5, attention.size(0)))
            
            # Get repository matches
            repo_matches = []
            for i in range(top_k):
                repo_idx = top_indices[pattern_idx, i].item()
                score = top_scores[pattern_idx, i].item()
                
                repo_matches.append({
                    "repository_pattern_id": repo_idx,
                    "similarity_score": score
                })
            
            matches.append({
                "pattern_id": pattern_idx,
                "top_nodes": top_node_indices.cpu().tolist(),
                "repository_matches": repo_matches
            })
        
        return {"matches": matches}
    
    def _add_subgraph_info(
        self, 
        graph: HeteroData,
        node_type: str,
        results: Dict[str, Any],
        attention: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Add subgraph information for each pattern match.
        
        Args:
            graph: Heterogeneous graph representation of code
            node_type: Type of nodes to consider
            results: Pattern matching results
            attention: Pattern attention weights
            
        Returns:
            Updated results with subgraph information
        """
        # Convert the graph to NetworkX for easier subgraph extraction
        # We only consider the specific node type
        nx_graph = to_networkx(graph, node_attrs=["x"], edge_attrs=[], to_undirected=True, node_type=node_type)
        
        # Process each match
        for match in results["matches"]:
            pattern_id = match["pattern_id"]
            top_nodes = match["top_nodes"]
            
            # Extract the subgraph containing the top nodes and their neighbors
            subgraph_nodes = set(top_nodes)
            
            # Add first-order neighbors
            for node in top_nodes:
                if node in nx_graph:
                    subgraph_nodes.update(nx_graph.neighbors(node))
            
            # Create the subgraph
            subgraph = nx_graph.subgraph(list(subgraph_nodes))
            
            # Add subgraph information to the match
            match["subgraph"] = {
                "nodes": list(subgraph.nodes()),
                "edges": list(subgraph.edges()),
                "num_nodes": subgraph.number_of_nodes(),
                "num_edges": subgraph.number_of_edges()
            }
        
        return results
    
    def find_similar_patterns(
        self, 
        pattern_embedding: torch.Tensor,
        top_k: int = 5,
        use_vuln_patterns: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find similar patterns in the repository.
        
        Args:
            pattern_embedding: Pattern embedding to find similar patterns to
            top_k: Number of top similar patterns to return
            use_vuln_patterns: Whether to use vulnerability patterns or security patterns
            
        Returns:
            List of dictionaries with similar pattern information
        """
        # Skip if repository is not available
        if self.vuln_patterns is None:
            return []
        
        # Select the appropriate pattern repository
        if use_vuln_patterns:
            repo_patterns = self.vuln_patterns
        else:
            repo_patterns = self.security_patterns
        
        # Normalize patterns
        pattern_norm = torch.nn.functional.normalize(pattern_embedding, p=2, dim=0)
        repo_patterns_norm = torch.nn.functional.normalize(repo_patterns, p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mv(repo_patterns_norm, pattern_norm)
        
        # Get top matches
        top_k = min(top_k, similarities.size(0))
        top_scores, top_indices = torch.topk(similarities, k=top_k)
        
        # Prepare results
        similar_patterns = []
        
        for i in range(top_k):
            pattern_id = top_indices[i].item()
            score = top_scores[i].item()
            
            similar_patterns.append({
                "pattern_id": pattern_id,
                "similarity_score": score
            })
        
        return similar_patterns
    
    def detect_vulnerability_type(
        self, 
        pattern_embedding: torch.Tensor,
        vulnerability_embeddings: Dict[str, torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Detect the type of vulnerability based on pattern embedding.
        
        Args:
            pattern_embedding: Pattern embedding
            vulnerability_embeddings: Dictionary mapping vulnerability types to embeddings
            
        Returns:
            Dictionary mapping vulnerability types to confidence scores
        """
        # Use default vulnerability type embeddings if not provided
        if vulnerability_embeddings is None:
            # This is a simplified approach. In a real implementation,
            # you would have proper embeddings for different vulnerability types.
            return {"generic_vulnerability": 0.8}
        
        # Calculate similarity with each vulnerability type
        scores = {}
        
        pattern_norm = torch.nn.functional.normalize(pattern_embedding, p=2, dim=0)
        
        for vuln_type, embedding in vulnerability_embeddings.items():
            embedding_norm = torch.nn.functional.normalize(embedding, p=2, dim=0)
            similarity = torch.dot(pattern_norm, embedding_norm).item()
            scores[vuln_type] = similarity
        
        # Normalize scores using softmax
        total = sum(np.exp(score) for score in scores.values())
        normalized_scores = {vuln_type: np.exp(score) / total for vuln_type, score in scores.items()}
        
        return normalized_scores
    
    def suggest_fix_pattern(
        self, 
        vulnerability_pattern: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Suggest a fix pattern for a vulnerability pattern.
        
        Args:
            vulnerability_pattern: Vulnerability pattern embedding
            
        Returns:
            Dictionary with fix suggestion information
        """
        # Skip if transformation rules are not available
        if not hasattr(self.pattern_module, 'transformation_rules') or self.pattern_module.transformation_rules is None:
            return {"has_suggestion": False}
        
        # Get the transformation for this pattern
        with torch.no_grad():
            transform = self.pattern_module.get_transformation_for_pattern(vulnerability_pattern)
            
            # Apply transformation to get security pattern
            security_pattern = vulnerability_pattern + transform.to(self.device)
            
            # Find similar security patterns
            similar_patterns = self.find_similar_patterns(
                security_pattern, 
                top_k=3, 
                use_vuln_patterns=False
            )
        
        return {
            "has_suggestion": True,
            "security_pattern": security_pattern.cpu().tolist(),
            "similar_patterns": similar_patterns
        }