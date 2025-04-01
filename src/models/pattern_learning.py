"""
Pattern Learning Module

This module provides the implementation for vulnerability pattern extraction,
security pattern identification, and transformation rule learning. It uses
contrastive learning to learn patterns from vulnerable-patched code pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from src.utils.logger import get_logger
from src.models.attention import PatternAttention

logger = get_logger(__name__)

class PatternLearningModule(nn.Module):
    """
    Module for learning vulnerability and security patterns from code graphs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern learning module.
        
        Args:
            config: Configuration dictionary
        """
        super(PatternLearningModule, self).__init__()
        
        self.config = config
        self.hidden_dim = config["model"]["hidden_channels"]
        self.pattern_dim = config["model"]["pattern_learning"]["pattern_dim"]
        self.num_patterns = config["model"]["pattern_learning"]["num_patterns"]
        self.contrastive_margin = config["model"]["pattern_learning"]["contrastive_margin"]
        self.pattern_dropout = config["model"]["pattern_learning"]["pattern_dropout"]
        self.transformation_hidden_dim = config["model"]["pattern_learning"]["transformation_hidden_dim"]
        
        # Pattern attention for vulnerability patterns
        self.vuln_pattern_attention = PatternAttention(
            hidden_dim=self.hidden_dim,
            num_patterns=self.num_patterns
        )
        
        # Pattern attention for security (patch) patterns
        self.security_pattern_attention = PatternAttention(
            hidden_dim=self.hidden_dim,
            num_patterns=self.num_patterns
        )
        
        # Pattern embeddings projection
        self.pattern_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.pattern_dim),
            nn.ReLU(),
            nn.Dropout(self.pattern_dropout),
            nn.Linear(self.pattern_dim, self.pattern_dim)
        )
        
        # Transformation rule learning
        self.transformation_net = nn.Sequential(
            nn.Linear(2 * self.pattern_dim, self.transformation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.pattern_dropout),
            nn.Linear(self.transformation_hidden_dim, self.transformation_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.pattern_dropout),
            nn.Linear(self.transformation_hidden_dim, self.pattern_dim)
        )
        
        # Pattern repository
        self.vuln_pattern_repository = None
        self.security_pattern_repository = None
        self.transformation_rules = None
    
    def forward(
        self, 
        vuln_embeddings: Dict[str, torch.Tensor],
        patch_embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for pattern learning.
        
        Args:
            vuln_embeddings: Node embeddings for vulnerable code nodes
            patch_embeddings: Node embeddings for patched code nodes
            
        Returns:
            Tuple of (contrastive loss, reconstruction loss, pattern similarity matrix, pattern outputs)
        """
        # Apply pattern attention to get pattern-focused embeddings
        if "vuln" in vuln_embeddings and "patch" in patch_embeddings:
            vuln_pattern_embeddings, vuln_attention = self.vuln_pattern_attention(vuln_embeddings["vuln"])
            security_pattern_embeddings, security_attention = self.security_pattern_attention(patch_embeddings["patch"])
        else:
            # Handle case where we're using "code" node type instead
            node_type = "code" if "code" in vuln_embeddings else list(vuln_embeddings.keys())[0]
            vuln_pattern_embeddings, vuln_attention = self.vuln_pattern_attention(vuln_embeddings[node_type])
            security_pattern_embeddings, security_attention = self.security_pattern_attention(patch_embeddings[node_type])
        
        # Project embeddings to pattern space
        vuln_patterns = self.pattern_projection(vuln_pattern_embeddings)
        security_patterns = self.pattern_projection(security_pattern_embeddings)
        
        # Learn transformation rules
        predicted_security_patterns = self._apply_transformation(vuln_patterns)
        
        # Calculate reconstruction loss - HANDLE DIFFERENT SIZES
        # Check if tensors have the same size and handle accordingly
        if vuln_patterns.size(0) == security_patterns.size(0):
            # Original implementation when sizes match
            reconstruction_loss = F.mse_loss(predicted_security_patterns, security_patterns)
        else:
            # Alternative approach for different sizes - use global representations
            # Option 1: Use mean pooling to get fixed-size representations
            global_vuln = torch.mean(vuln_patterns, dim=0, keepdim=True)
            global_security = torch.mean(security_patterns, dim=0, keepdim=True)
            global_predicted = torch.mean(predicted_security_patterns, dim=0, keepdim=True)
            
            # Calculate loss on global representations
            reconstruction_loss = F.mse_loss(global_predicted, global_security)
        
        # Calculate contrastive loss
        if vuln_patterns.size(0) == security_patterns.size(0):
            # Original contrastive loss when sizes match
            contrastive_loss = self._contrastive_loss(vuln_patterns, security_patterns)
        else:
            # Use sample-based contrastive loss for different sizes
            contrastive_loss = self._sample_based_contrastive_loss(vuln_patterns, security_patterns)
        
        # Calculate pattern similarity matrix - use the pattern embeddings directly
        pattern_similarity = self._calculate_pattern_similarity()
        
        # Return losses, similarity matrix, and pattern outputs
        pattern_outputs = {
            "vuln_patterns": vuln_patterns,
            "security_patterns": security_patterns,
            "predicted_security_patterns": predicted_security_patterns,
            "vuln_attention": vuln_attention,
            "security_attention": security_attention
        }
        
        return contrastive_loss, reconstruction_loss, pattern_similarity, pattern_outputs

    def _sample_based_contrastive_loss(
        self, 
        vuln_patterns: torch.Tensor,
        security_patterns: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss between vulnerability and security patterns
        when they have different sizes.
        
        Args:
            vuln_patterns: Vulnerability pattern embeddings
            security_patterns: Security pattern embeddings
            
        Returns:
            Contrastive loss
        """
        # Sample min(100, size) patterns from each to keep computation manageable
        vuln_size = min(100, vuln_patterns.size(0))
        sec_size = min(100, security_patterns.size(0))
        
        # Random sampling to get equal numbers from both tensors
        size = min(vuln_size, sec_size)
        
        if vuln_patterns.size(0) > size:
            indices = torch.randperm(vuln_patterns.size(0), device=vuln_patterns.device)[:size]
            vuln_samples = vuln_patterns[indices]
        else:
            vuln_samples = vuln_patterns
        
        if security_patterns.size(0) > size:
            indices = torch.randperm(security_patterns.size(0), device=security_patterns.device)[:size]
            security_samples = security_patterns[indices]
        else:
            security_samples = security_patterns
        
        # Normalize samples
        vuln_norm = F.normalize(vuln_samples, p=2, dim=1)
        security_norm = F.normalize(security_samples, p=2, dim=1)
        
        # Calculate similarity matrix between samples
        similarity = torch.mm(vuln_norm, security_norm.transpose(0, 1))
        
        # Use a simplified contrastive loss for samples
        # Maximizing diagonal (similar positions) and minimizing off-diagonal elements
        target = torch.eye(size, device=similarity.device)
        
        # Mean squared error between similarity and target
        contrastive_loss = F.mse_loss(similarity, target)
        
        return contrastive_loss
    
    def _apply_transformation(self, vuln_patterns: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation rules to vulnerability patterns to predict security patterns.
        
        Args:
            vuln_patterns: Vulnerability pattern embeddings
            
        Returns:
            Predicted security pattern embeddings
        """
        # Get global vulnerability pattern by mean pooling
        global_vuln_pattern = torch.mean(vuln_patterns, dim=0, keepdim=True).expand(vuln_patterns.size(0), -1)
        
        # Concatenate each vulnerability pattern with the global pattern
        concat_patterns = torch.cat([vuln_patterns, global_vuln_pattern], dim=1)
        
        # Apply transformation network
        predicted_security_patterns = self.transformation_net(concat_patterns)
        
        return predicted_security_patterns
    
    def _contrastive_loss(
        self, 
        vuln_patterns: torch.Tensor,
        security_patterns: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss between vulnerability and security patterns.
        
        Args:
            vuln_patterns: Vulnerability pattern embeddings
            security_patterns: Security pattern embeddings
            
        Returns:
            Contrastive loss
        """
        # Calculate cosine similarity matrix
        vuln_norm = F.normalize(vuln_patterns, p=2, dim=1)
        security_norm = F.normalize(security_patterns, p=2, dim=1)
        
        similarity = torch.mm(vuln_norm, security_norm.transpose(0, 1))
        
        # Set target: diagonal elements should be 1 (positive pairs), off-diagonal should be 0 (negative pairs)
        target = torch.eye(vuln_patterns.size(0), device=vuln_patterns.device)
        
        # Calculate loss using margin ranking loss
        positive_loss = F.mse_loss(similarity * target, target)
        
        # For negative pairs, we want similarity to be less than margin
        # First, create a mask for negative pairs (off-diagonal elements)
        negative_mask = 1 - target
        
        # Calculate negative loss: max(0, similarity - margin) for negative pairs
        negative_similarity = similarity * negative_mask
        negative_loss = torch.mean(F.relu(negative_similarity - self.contrastive_margin) * negative_mask)
        
        # Total contrastive loss
        contrastive_loss = positive_loss + negative_loss
        
        return contrastive_loss
    
    def _calculate_pattern_similarity(self) -> torch.Tensor:
        """
        Calculate similarity between learned vulnerability and security patterns.
        
        Returns:
            Pattern similarity matrix
        """
        # Get learned patterns
        vuln_patterns = self.vuln_pattern_attention.get_patterns()
        security_patterns = self.security_pattern_attention.get_patterns()
        
        # Normalize patterns
        vuln_norm = F.normalize(vuln_patterns, p=2, dim=1)
        security_norm = F.normalize(security_patterns, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity = torch.mm(vuln_norm, security_norm.transpose(0, 1))
        
        return similarity
    
    def update_pattern_repository(self) -> None:
        """
        Update the pattern repository with current learned patterns.
        """
        # Get learned patterns
        vuln_patterns = self.vuln_pattern_attention.get_patterns().detach().cpu()
        security_patterns = self.security_pattern_attention.get_patterns().detach().cpu()
        
        # If repositories don't exist, create them
        if self.vuln_pattern_repository is None:
            self.vuln_pattern_repository = vuln_patterns
            self.security_pattern_repository = security_patterns
            
            # Calculate initial transformation rules
            self.transformation_rules = self._calculate_transformation_rules()
        else:
            # Concatenate with existing repositories
            self.vuln_pattern_repository = torch.cat([self.vuln_pattern_repository, vuln_patterns], dim=0)
            self.security_pattern_repository = torch.cat([self.security_pattern_repository, security_patterns], dim=0)
            
            # Update transformation rules
            self.transformation_rules = self._calculate_transformation_rules()
        
        # Prune repositories if they get too large
        self._prune_repositories()
    
    def _calculate_transformation_rules(self) -> torch.Tensor:
        """
        Calculate transformation rules from vulnerability to security patterns.
        
        Returns:
            Transformation rules tensor
        """
        # This is a simplified approach - in a real implementation,
        # you might use a more sophisticated method
        
        # Get pattern embeddings
        vuln_patterns = self.vuln_pattern_repository
        security_patterns = self.security_pattern_repository
        
        # Calculate pairwise similarities
        vuln_norm = F.normalize(vuln_patterns, p=2, dim=1)
        security_norm = F.normalize(security_patterns, p=2, dim=1)
        
        similarity = torch.mm(vuln_norm, security_norm.transpose(0, 1))
        
        # Get top matches for each vulnerability pattern
        _, top_matches = torch.topk(similarity, k=min(5, similarity.size(1)), dim=1)
        
        # Calculate transformation as the average vector from vulnerability to security pattern
        transformations = []
        for i in range(vuln_patterns.size(0)):
            # Get top matching security patterns
            matches = top_matches[i]
            
            # Calculate transformation vectors
            trans_vectors = []
            for j in matches:
                trans_vector = security_patterns[j] - vuln_patterns[i]
                trans_vectors.append(trans_vector)
            
            # Average transformation
            avg_transform = torch.stack(trans_vectors).mean(dim=0)
            transformations.append(avg_transform)
        
        # Stack transformations
        transformation_rules = torch.stack(transformations)
        
        return transformation_rules
    
    def _prune_repositories(self, max_patterns: int = 1000) -> None:
        """
        Prune pattern repositories if they get too large.
        
        Args:
            max_patterns: Maximum number of patterns to keep
        """
        # Skip if repositories are small enough
        if self.vuln_pattern_repository.size(0) <= max_patterns:
            return
        
        # Calculate pattern diversity (using pairwise similarities)
        vuln_norm = F.normalize(self.vuln_pattern_repository, p=2, dim=1)
        similarity = torch.mm(vuln_norm, vuln_norm.transpose(0, 1))
        
        # Set diagonal to 0 to ignore self-similarity
        similarity.fill_diagonal_(0)
        
        # Calculate diversity score for each pattern (lower similarity is more diverse)
        diversity_scores = -torch.sum(similarity, dim=1)
        
        # Get indices of the most diverse patterns
        _, diverse_indices = torch.topk(diversity_scores, k=max_patterns)
        
        # Keep only the diverse patterns
        self.vuln_pattern_repository = self.vuln_pattern_repository[diverse_indices]
        self.security_pattern_repository = self.security_pattern_repository[diverse_indices]
        
        # Recalculate transformation rules
        self.transformation_rules = self._calculate_transformation_rules()
    
    def extract_patterns_from_graph(
        self, 
        node_embeddings: torch.Tensor,
        is_vulnerable: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patterns from a graph's node embeddings.
        
        Args:
            node_embeddings: Node embeddings
            is_vulnerable: Whether this is a vulnerable code graph
            
        Returns:
            Tuple of (pattern embeddings, attention weights)
        """
        # Select the appropriate pattern attention module
        if is_vulnerable:
            pattern_attn = self.vuln_pattern_attention
        else:
            pattern_attn = self.security_pattern_attention
        
        # Apply pattern attention
        pattern_embeddings, attention_weights = pattern_attn(node_embeddings)
        
        # Project to pattern space
        patterns = self.pattern_projection(pattern_embeddings)
        
        return patterns, attention_weights
    
    def get_transformation_for_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """
        Get the most relevant transformation rule for a pattern.
        
        Args:
            pattern: Pattern embedding to transform
            
        Returns:
            Transformation vector
        """
        # Normalize pattern
        pattern_norm = F.normalize(pattern, p=2, dim=0)
        
        # Get repository patterns
        vuln_patterns = self.vuln_pattern_repository
        vuln_norm = F.normalize(vuln_patterns, p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mv(vuln_norm, pattern_norm)
        
        # Get top matches
        _, top_matches = torch.topk(similarities, k=min(3, similarities.size(0)))
        
        # Weighted average of transformation rules
        weighted_transform = torch.zeros_like(self.transformation_rules[0])
        total_weight = 0.0
        
        for idx in top_matches:
            weight = similarities[idx].item()
            weighted_transform += weight * self.transformation_rules[idx]
            total_weight += weight
        
        if total_weight > 0:
            weighted_transform /= total_weight
        
        return weighted_transform