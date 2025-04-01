"""
Vulnerability Scoring Module

This module provides functionality to score code graphs for potential vulnerabilities
based on learned patterns. It integrates the pattern matching and model predictions
to provide a comprehensive vulnerability assessment.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from torch_geometric.data import HeteroData

from src.utils.logger import get_logger
from src.models.graphsage import HeteroGraphSAGE
from src.models.pattern_learning import PatternLearningModule

logger = get_logger(__name__)

class VulnerabilityScorer:
    """
    Class for scoring code graphs for vulnerability detections.
    """
    
    def __init__(
        self, 
        model: HeteroGraphSAGE,
        pattern_module: PatternLearningModule,
        config: Dict[str, Any]
    ):
        """
        Initialize the vulnerability scorer.
        
        Args:
            model: Trained GraphSAGE model
            pattern_module: Trained pattern learning module
            config: Configuration dictionary
        """
        self.model = model
        self.pattern_module = pattern_module
        self.config = config
        
        # Set device
        self.device = torch.device(config["training"]["device"])
        
        # Set confidence threshold
        self.confidence_threshold = config["inference"]["confidence_threshold"]
        
        # Set top-k patterns
        self.top_k_patterns = config["inference"]["top_k_patterns"]
        
        # Make sure model and pattern module are in eval mode
        self.model.eval()
        self.pattern_module.eval()
    
    def score_graph(
        self, 
        graph: HeteroData
    ) -> Dict[str, Any]:
        """
        Score a code graph for vulnerabilities.
        
        Args:
            graph: Heterogeneous graph representation of code
            
        Returns:
            Dictionary containing vulnerability scores and related information
        """
        # Move graph to device
        graph = graph.to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(graph)
            
            # Get node embeddings
            node_embeddings = self.model.get_node_embeddings(graph)
            
            # Extract patterns from graph
            patterns, attention = self._extract_patterns(node_embeddings)
            
            # Calculate pattern match scores
            pattern_scores = self._calculate_pattern_match_scores(patterns)
            
            # Get top nodes with high pattern match scores
            top_nodes = self._get_top_pattern_nodes(attention)
        
        # Convert prediction to score
        vuln_score = prediction.item()
        
        # Prepare the result dictionary
        result = {
            "vulnerability_score": vuln_score,
            "is_vulnerable": vuln_score >= self.confidence_threshold,
            "pattern_match_scores": pattern_scores,
            "top_vulnerable_nodes": top_nodes,
            "confidence": vuln_score
        }
        
        return result
    
    def _extract_patterns(
        self, 
        node_embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patterns from node embeddings.
        
        Args:
            node_embeddings: Dictionary of node embeddings for each node type
            
        Returns:
            Tuple of (pattern embeddings, attention weights)
        """
        # Check if this is a vulnerable-patch pair or a single graph
        if "vuln" in node_embeddings and "patch" in node_embeddings:
            # This is a pair - use the vulnerable part
            patterns, attention = self.pattern_module.extract_patterns_from_graph(
                node_embeddings["vuln"], 
                is_vulnerable=True
            )
        elif "code" in node_embeddings:
            # This is a single graph - we don't know if it's vulnerable
            patterns, attention = self.pattern_module.extract_patterns_from_graph(
                node_embeddings["code"],
                is_vulnerable=True  # Assume vulnerable to check against vulnerability patterns
            )
        else:
            # Use the first available node type
            node_type = list(node_embeddings.keys())[0]
            patterns, attention = self.pattern_module.extract_patterns_from_graph(
                node_embeddings[node_type],
                is_vulnerable=True
            )
        
        return patterns, attention
    
    def _calculate_pattern_match_scores(self, patterns: torch.Tensor) -> List[float]:
        """
        Calculate similarity scores between extracted patterns and known vulnerability patterns.
        
        Args:
            patterns: Extracted pattern embeddings
            
        Returns:
            List of pattern match scores
        """
        # Get vulnerability patterns from repository
        if hasattr(self.pattern_module, 'vuln_pattern_repository') and self.pattern_module.vuln_pattern_repository is not None:
            vuln_patterns = self.pattern_module.vuln_pattern_repository.to(self.device)
            
            # Normalize patterns
            patterns_norm = torch.nn.functional.normalize(patterns, p=2, dim=1)
            vuln_patterns_norm = torch.nn.functional.normalize(vuln_patterns, p=2, dim=1)
            
            # Calculate similarities
            similarities = torch.mm(patterns_norm, vuln_patterns_norm.t())
            
            # Get top-k scores for each pattern
            top_k = min(self.top_k_patterns, similarities.size(1))
            top_scores, _ = torch.topk(similarities, k=top_k, dim=1)
            
            # Average the top scores
            avg_scores = torch.mean(top_scores, dim=1)
            
            return avg_scores.cpu().tolist()
        else:
            # If no repository is available, use raw pattern attention scores
            return [0.5] * patterns.size(0)  # Default score
    
    def _get_top_pattern_nodes(self, attention: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Get the top nodes with high pattern match scores.
        
        Args:
            attention: Pattern attention weights [num_nodes, num_patterns]
            
        Returns:
            List of dictionaries with node indices and pattern scores
        """
        # Get top node indices for each pattern
        top_nodes = []
        
        for pattern_idx in range(attention.size(1)):
            # Get attention weights for this pattern
            pattern_attention = attention[:, pattern_idx]
            
            # Get top-k node indices
            top_k = min(5, pattern_attention.size(0))
            scores, indices = torch.topk(pattern_attention, k=top_k)
            
            # Create result for this pattern
            pattern_result = {
                "pattern_id": pattern_idx,
                "nodes": [
                    {"node_idx": idx.item(), "score": score.item()}
                    for idx, score in zip(indices, scores)
                ]
            }
            
            top_nodes.append(pattern_result)
        
        return top_nodes
    
    def suggest_fix(
        self, 
        graph: HeteroData,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest potential fixes for identified vulnerabilities.
        
        Args:
            graph: Heterogeneous graph representation of code
            result: Result from score_graph method
            
        Returns:
            Dictionary with fix suggestions
        """
        # Skip if the graph is not vulnerable
        if not result["is_vulnerable"]:
            return {"suggestions": []}
        
        # Get node embeddings
        with torch.no_grad():
            node_embeddings = self.model.get_node_embeddings(graph)
            
            # Extract patterns from graph
            patterns, attention = self._extract_patterns(node_embeddings)
            
            # Get top patterns and nodes
            top_pattern_idx = torch.argmax(torch.max(attention, dim=0)[0]).item()
            
            # Get the most relevant vulnerability pattern
            top_pattern = patterns[torch.argmax(attention[:, top_pattern_idx])]
            
            # Apply transformation to get suggested fix pattern
            if hasattr(self.pattern_module, 'transformation_rules') and self.pattern_module.transformation_rules is not None:
                transform = self.pattern_module.get_transformation_for_pattern(top_pattern)
                
                # Apply transformation to get security pattern
                security_pattern = top_pattern + transform.to(self.device)
                
                # Find most similar nodes to the security pattern
                node_type = "code" if "code" in node_embeddings else "vuln"
                similarities = torch.nn.functional.cosine_similarity(
                    security_pattern.unsqueeze(0),
                    node_embeddings[node_type],
                    dim=1
                )
                
                # Get top similar nodes
                top_k = min(5, similarities.size(0))
                scores, indices = torch.topk(similarities, k=top_k)
                
                # Create suggestion
                suggestion = {
                    "pattern_id": top_pattern_idx,
                    "suggestion_type": "node_substitution",
                    "vulnerable_nodes": result["top_vulnerable_nodes"][top_pattern_idx]["nodes"],
                    "suggested_nodes": [
                        {"node_idx": idx.item(), "similarity_score": score.item()}
                        for idx, score in zip(indices, scores)
                    ]
                }
                
                return {"suggestions": [suggestion]}
            else:
                return {"suggestions": []}