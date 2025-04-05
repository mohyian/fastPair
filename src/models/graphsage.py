"""
GraphSAGE Implementation

This module provides an implementation of the GraphSAGE algorithm for heterogeneous graphs
with attention mechanisms. The implementation is based on PyTorch Geometric and adapted
for the vulnerability detection task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, GATConv
from torch_geometric.nn import HeteroConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Union, Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class HeteroGraphSAGE(nn.Module):
    """
    GraphSAGE implementation for heterogeneous graphs with attention mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphSAGE model.
        
        Args:
            config: Configuration dictionary
        """
        super(HeteroGraphSAGE, self).__init__()
        
        self.config = config
        
        # Extract model configuration
        self.hidden_dim = config["model"]["hidden_channels"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout"]
        self.use_attention = config["model"]["use_attention"]
        self.aggregation = config["model"]["aggregation"]
        self.edge_types = config["model"]["edge_types"]
        
        # Node embedding dimensions
        self.node_embedding_dim = config["model"]["node_features"]["embedding_dim"]
        
        # Define node types
        self.node_types = ["vuln", "patch"]
        
        # Create initial node type embeddings
        self.node_type_embeddings = nn.ModuleDict({
            node_type: nn.Linear(self.node_embedding_dim, self.hidden_dim)
            for node_type in self.node_types
        })
        
        # Create list of edge types including alignment edges
        self.full_edge_types = []
        for src in self.node_types:
            for dst in self.node_types:
                if src == dst:  # Same node type edges
                    for edge_type in self.edge_types:
                        self.full_edge_types.append((src, edge_type, dst))
                else:  # Different node types - alignment edges
                    self.full_edge_types.append((src, "aligned", dst))
        
        # Create convolution layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            # Input dimension for the first layer, hidden dim for subsequent layers
            in_dim = self.hidden_dim
            
            # Create heterogeneous convolution
            conv_dict = {}
            
            for edge_type in self.full_edge_types:
                if self.use_attention:
                    # Determine whether to add self-loops based on edge type
                    # Don't add self-loops for alignment edges between different node types
                    add_self_loops = (edge_type[0] == edge_type[2])
                    
                    # Use Graph Attention Network convolution
                    conv_dict[edge_type] = GATConv(
                        in_dim, 
                        self.hidden_dim // 8,  # Divide by number of attention heads
                        heads=8,  # Number of attention heads
                        dropout=self.dropout,
                        add_self_loops=add_self_loops  # Set based on edge type
                    )
                else:
                    # Use standard GraphSAGE convolution
                    conv_dict[edge_type] = SAGEConv(
                        in_dim, 
                        self.hidden_dim
                    )
            
            self.convs.append(HeteroConv(conv_dict, aggr=self.aggregation))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.BatchNorm1d(self.hidden_dim)
                for node_type in self.node_types
            })
            for _ in range(self.num_layers)
        ])
        
        # Output layer to predict vulnerability
        self.linear = nn.Linear(self.hidden_dim * 2, 1)  # *2 because we concatenate vuln and patch embeddings
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE model.
        
        Args:
            data: Heterogeneous graph data
            
        Returns:
            Predictions for vulnerability
        """
        # Apply initial linear transformation to node features
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types:
                x_dict[node_type] = self.node_type_embeddings[node_type](data[node_type].x)
        
        # Apply graph convolutions
        for i in range(self.num_layers):
            # Store the inputs for residual connections
            x_dict_prev = {k: x.clone() for k, x in x_dict.items()}
            
            # Apply convolution
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            
            # Apply batch normalization
            for node_type in x_dict:
                x_dict[node_type] = self.batch_norms[i][node_type](x_dict[node_type])
            
            # Apply activation and dropout
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
            
            # Add residual connection after the first layer
            if i > 0:
                for node_type in x_dict:
                    if node_type in x_dict_prev:
                        x_dict[node_type] = x_dict[node_type] + x_dict_prev[node_type]
        
        # Global pooling for each node type
        global_embeddings = {}
        for node_type in self.node_types:
            if node_type in x_dict:
                # Get node embeddings
                node_embeddings = x_dict[node_type]
                
                # Get batch assignment (create if not present)
                if hasattr(data[node_type], 'batch'):
                    batch = data[node_type].batch
                else:
                    batch = torch.zeros(data[node_type].x.size(0), dtype=torch.long, device=data[node_type].x.device)
                
                # Apply global pooling
                if self.aggregation == 'mean':
                    global_embeddings[node_type] = global_mean_pool(node_embeddings, batch)
                elif self.aggregation == 'max':
                    global_embeddings[node_type] = global_max_pool(node_embeddings, batch)
                elif self.aggregation == 'add':
                    global_embeddings[node_type] = global_add_pool(node_embeddings, batch)
                else:
                    global_embeddings[node_type] = global_mean_pool(node_embeddings, batch)
        
        # Concatenate vulnerable and patched graph embeddings
        if "vuln" in global_embeddings and "patch" in global_embeddings:
            concat_embedding = torch.cat([global_embeddings["vuln"], global_embeddings["patch"]], dim=1)
        else:
            # Handle cases where one or both node types are missing
            missing_embedding = torch.zeros(1, self.hidden_dim, device=data[list(data.node_types)[0]].x.device)
            
            if "vuln" not in global_embeddings:
                global_embeddings["vuln"] = missing_embedding
            
            if "patch" not in global_embeddings:
                global_embeddings["patch"] = missing_embedding
            
            concat_embedding = torch.cat([global_embeddings["vuln"], global_embeddings["patch"]], dim=1)
        
        # Apply final linear layer to predict vulnerability
        output = self.linear(concat_embedding)
        output = torch.sigmoid(output)
        
        return output
    
    def get_node_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Get node embeddings from the model.
        
        Args:
            data: Heterogeneous graph data
            
        Returns:
            Dictionary of node embeddings for each node type
        """
        # Apply initial linear transformation to node features
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types:
                x_dict[node_type] = self.node_type_embeddings[node_type](data[node_type].x)
        
        # Apply graph convolutions
        for i in range(self.num_layers):
            # Store the inputs for residual connections
            x_dict_prev = {k: x.clone() for k, x in x_dict.items()}
            
            # Apply convolution
            x_dict = self.convs[i](x_dict, data.edge_index_dict)
            
            # Apply batch normalization
            for node_type in x_dict:
                x_dict[node_type] = self.batch_norms[i][node_type](x_dict[node_type])
            
            # Apply activation and dropout
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.dropout, training=self.training)
            
            # Add residual connection after the first layer
            if i > 0:
                for node_type in x_dict:
                    if node_type in x_dict_prev:
                        x_dict[node_type] = x_dict[node_type] + x_dict_prev[node_type]
        
        return x_dict