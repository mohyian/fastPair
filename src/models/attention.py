"""
Attention Mechanisms

This module provides custom attention mechanisms for heterogeneous graph neural networks,
focusing on the vulnerability detection task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class EdgeTypeAttention(nn.Module):
    """
    Attention mechanism that learns to weight different edge types differently.
    """
    
    def __init__(self, num_edge_types: int, hidden_dim: int):
        """
        Initialize the edge type attention module.
        
        Args:
            num_edge_types: Number of edge types to weight
            hidden_dim: Dimension of node embeddings
        """
        super(EdgeTypeAttention, self).__init__()
        
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim
        
        # Query and key transformations
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge type embeddings
        self.edge_type_embeddings = nn.Embedding(num_edge_types, hidden_dim)
        
        # Attention calculation
        self.attention = nn.Linear(3 * hidden_dim, 1)
    
    def forward(
        self, 
        src_embeddings: torch.Tensor, 
        dst_embeddings: torch.Tensor, 
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention weights for edges.
        
        Args:
            src_embeddings: Source node embeddings [num_edges, hidden_dim]
            dst_embeddings: Destination node embeddings [num_edges, hidden_dim]
            edge_types: Edge type indices [num_edges]
            
        Returns:
            Attention weights for edges [num_edges, 1]
        """
        # Calculate query from destination nodes
        queries = self.query(dst_embeddings)
        
        # Calculate keys from source nodes
        keys = self.key(src_embeddings)
        
        # Get edge type embeddings
        edge_emb = self.edge_type_embeddings(edge_types)
        
        # Concatenate source, destination, and edge type embeddings
        combined = torch.cat([queries, keys, edge_emb], dim=1)
        
        # Calculate attention weights
        attention_weights = self.attention(combined)
        attention_weights = F.leaky_relu(attention_weights)
        
        return attention_weights

class MultiHeadEdgeAttention(nn.Module):
    """
    Multi-head attention mechanism for edges in a heterogeneous graph.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        edge_types: List[str],
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        """
        Initialize the multi-head edge attention module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            edge_types: List of edge types
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadEdgeAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_types = edge_types
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Check that hidden_dim is divisible by num_heads
        assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        # Parameter dictionaries for each edge type
        self.q_projections = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim) 
            for edge_type in edge_types
        })
        
        self.k_projections = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim) 
            for edge_type in edge_types
        })
        
        self.v_projections = nn.ModuleDict({
            edge_type: nn.Linear(hidden_dim, hidden_dim) 
            for edge_type in edge_types
        })
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout layer
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        node_embeddings: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply multi-head attention to node embeddings based on graph structure.
        
        Args:
            node_embeddings: Dictionary of node embeddings for each node type
            edge_indices: Dictionary of edge indices for each edge type
            
        Returns:
            Updated node embeddings
        """
        # Initialize output dictionary
        output_embeddings = {
            node_type: torch.zeros_like(embed) 
            for node_type, embed in node_embeddings.items()
        }
        
        # Process each edge type
        for edge_type, edge_index in edge_indices.items():
            src_type, rel_type, dst_type = edge_type
            
            # Skip if this edge type is not in our list
            if rel_type not in self.edge_types:
                continue
            
            # Get source and destination node embeddings
            src_embeds = node_embeddings[src_type]
            dst_embeds = node_embeddings[dst_type]
            
            # Get edges
            src, dst = edge_index
            
            # Apply projections
            q = self.q_projections[rel_type](dst_embeds)
            k = self.k_projections[rel_type](src_embeds)
            v = self.v_projections[rel_type](src_embeds)
            
            # Reshape for multi-head attention
            batch_size, _ = q.shape
            q = q.view(batch_size, self.num_heads, self.head_dim)
            k = k.view(batch_size, self.num_heads, self.head_dim)
            v = v.view(batch_size, self.num_heads, self.head_dim)
            
            # Get queries for destination nodes
            dst_queries = q[dst]
            
            # Get keys and values for source nodes
            src_keys = k[src]
            src_values = v[src]
            
            # Calculate attention scores
            scores = torch.sum(dst_queries * src_keys, dim=2) / (self.head_dim ** 0.5)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(scores, dim=1)
            attention_weights = self.attn_dropout(attention_weights)
            
            # Apply attention weights to values
            attended_values = torch.bmm(
                attention_weights.unsqueeze(1),
                src_values
            ).squeeze(1)
            
            # Reshape back
            attended_values = attended_values.view(attended_values.size(0), -1)
            
            # Project to output dimension
            output = self.output_projection(attended_values)
            
            # Update output embeddings for destination nodes
            output_embeddings[dst_type][dst] += output
        
        return output_embeddings

class PatternAttention(nn.Module):
    """
    Attention mechanism that focuses on potential vulnerability patterns.
    """
    
    def __init__(self, hidden_dim: int, num_patterns: int):
        """
        Initialize the pattern attention module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            num_patterns: Number of learnable patterns
        """
        super(PatternAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        
        # Learnable patterns
        self.pattern_embeddings = nn.Parameter(torch.randn(num_patterns, hidden_dim))
        
        # Attention mechanism
        self.pattern_attention = nn.Linear(2 * hidden_dim, 1)
        
        # Pattern transformation
        self.pattern_transform = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply pattern attention to node embeddings.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            
        Returns:
            Tuple of (pattern-attended node embeddings, attention weights)
        """
        num_nodes = node_embeddings.size(0)
        
        # Calculate attention between each node and each pattern
        attention_scores = torch.zeros(num_nodes, self.num_patterns, device=node_embeddings.device)
        
        for p in range(self.num_patterns):
            # Get pattern embedding
            pattern = self.pattern_embeddings[p:p+1].expand(num_nodes, -1)
            
            # Concatenate node embeddings with pattern
            concat = torch.cat([node_embeddings, pattern], dim=1)
            
            # Calculate attention score
            score = self.pattern_attention(concat)
            attention_scores[:, p] = score.squeeze()
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Calculate weighted sum of patterns
        weighted_patterns = torch.mm(attention_weights, self.pattern_embeddings)
        
        # Transform pattern-based embeddings
        pattern_embeddings = self.pattern_transform(weighted_patterns)
        
        # Combine original embeddings with pattern embeddings
        combined_embeddings = node_embeddings + pattern_embeddings
        
        return combined_embeddings, attention_weights
    
    def get_patterns(self) -> torch.Tensor:
        """
        Get the learned vulnerability patterns.
        
        Returns:
            Tensor of pattern embeddings
        """
        return self.pattern_embeddings