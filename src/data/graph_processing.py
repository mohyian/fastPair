"""
Graph Processing Module

This module handles the processing of pre-parsed code graphs into the format 
required for heterogeneous graph neural networks.

It assumes that the input graphs are already parsed from C/C++ code into
a basic graph representation, and this module enriches those graphs with
additional information and transforms them into PyTorch Geometric HeteroData objects.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, NormalizeFeatures

from src.utils.logger import get_logger

logger = get_logger(__name__)

class GraphProcessor:
    """
    Processes raw code graphs into heterogeneous graph representations
    for vulnerability detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphProcessor with configuration settings.
        
        Args:
            config: Configuration dictionary with graph processing parameters
        """
        self.config = config
        self.node_feature_dim = config["model"]["node_features"]["embedding_dim"]
        self.edge_types = config["model"]["edge_types"]
        self.use_positional_encoding = config["model"]["node_features"]["use_positional_encoding"]
        self.use_ast_type_features = config["model"]["node_features"]["use_ast_type_features"]
        self.use_code_token_features = config["model"]["node_features"]["use_code_token_features"]
        
        # Initialize tokenizer for code tokens if needed
        if self.use_code_token_features:
            self._initialize_code_tokenizer()
    
    def _initialize_code_tokenizer(self):
        """
        Initialize the tokenizer for code tokens.
        This could be a simple tokenizer or a more advanced one based on the requirements.
        """
        # This is a placeholder for a more sophisticated tokenizer
        # In a real implementation, you might use a pre-trained tokenizer
        self.token_to_idx = {}
        self.next_token_idx = 0
    
    def get_token_idx(self, token: str) -> int:
        """
        Get the index for a token, creating a new entry if it doesn't exist.
        
        Args:
            token: The code token to encode
            
        Returns:
            The token index
        """
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.next_token_idx
            self.next_token_idx += 1
        return self.token_to_idx[token]
    
    def process_graph_pair(self, 
                        vulnerable_graph: nx.DiGraph, 
                        patched_graph: nx.DiGraph, 
                        node_alignment: Dict[int, int] = None) -> Tuple[HeteroData, HeteroData]:
        """
        Process a pair of vulnerable and patched graphs into HeteroData objects.
        
        Args:
            vulnerable_graph: NetworkX DiGraph of the vulnerable code
            patched_graph: NetworkX DiGraph of the patched code
            node_alignment: Dictionary mapping nodes from vulnerable to patched graph
            
        Returns:
            Tuple of (processed vulnerable graph, processed patched graph)
        """
        vuln_hetero = self._convert_to_hetero(vulnerable_graph, is_vulnerable=True)
        patch_hetero = self._convert_to_hetero(patched_graph, is_vulnerable=False)
        
        if node_alignment:
            # Add alignment information to the graphs
            vuln_hetero.alignment = {}
            patch_hetero.alignment = {}
            
            for vuln_node, patch_node in node_alignment.items():
                if vuln_node < len(vuln_hetero["code"].x) and patch_node < len(patch_hetero["code"].x):
                    vuln_hetero.alignment[vuln_node] = patch_node
                    patch_hetero.alignment[patch_node] = vuln_node
        
        return vuln_hetero, patch_hetero
    
    def _convert_to_hetero(self, graph: nx.DiGraph, is_vulnerable: bool) -> HeteroData:
        """
        Convert a NetworkX graph to a PyTorch Geometric HeteroData object.
        
        Args:
            graph: NetworkX DiGraph of the code
            is_vulnerable: Whether this graph represents vulnerable code
            
        Returns:
            HeteroData object representing the heterogeneous graph
        """
        data = HeteroData()
        
        # Add node features
        node_features = self._extract_node_features(graph)
        data["code"].x = node_features
        
        # Add global graph label (vulnerable or not)
        data["code"].y = torch.tensor([1 if is_vulnerable else 0], dtype=torch.float)
        
        # Add different edge types
        for edge_type in self.edge_types:
            edges = self._extract_edges_by_type(graph, edge_type)
            if edges and len(edges[0]) > 0:
                src, dst = edges
                data["code", edge_type, "code"].edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return data
    
    def _extract_node_features(self, graph: nx.DiGraph) -> torch.Tensor:
        """
        Extract features for all nodes in the graph.
        
        Args:
            graph: NetworkX DiGraph of the code
            
        Returns:
            Tensor of node features
        """
        num_nodes = graph.number_of_nodes()
        features = torch.zeros((num_nodes, self.node_feature_dim))
        
        for node_id in range(num_nodes):
            if node_id in graph.nodes:
                node_data = graph.nodes[node_id]
                
                # Extract features from node data
                feature_vector = self._get_node_feature_vector(node_id, node_data, graph)
                
                # Ensure the feature vector has the correct dimension
                if feature_vector.size(0) < self.node_feature_dim:
                    padding = torch.zeros(self.node_feature_dim - feature_vector.size(0))
                    feature_vector = torch.cat([feature_vector, padding])
                elif feature_vector.size(0) > self.node_feature_dim:
                    feature_vector = feature_vector[:self.node_feature_dim]
                
                features[node_id] = feature_vector
        
        return features
    
    def _get_node_feature_vector(self, node_id: int, node_data: Dict, graph: nx.DiGraph) -> torch.Tensor:
        """
        Create a feature vector for a single node.
        
        Args:
            node_id: ID of the node
            node_data: Dictionary of node attributes
            graph: The full graph (for context-based features)
            
        Returns:
            Tensor of node features
        """
        features = []
        
        # Basic features that should be available for all nodes
        node_type = node_data.get('type', 'unknown')
        node_type_onehot = self._one_hot_encode_node_type(node_type)
        features.append(node_type_onehot)
        
        # AST type features if enabled
        if self.use_ast_type_features and 'ast_type' in node_data:
            ast_features = self._encode_ast_type(node_data['ast_type'])
            features.append(ast_features)
        
        # Code token features if enabled
        if self.use_code_token_features and 'code' in node_data:
            token_features = self._encode_code_token(node_data['code'])
            features.append(token_features)
        
        # Positional encoding if enabled
        if self.use_positional_encoding:
            pos_features = self._positional_encoding(node_id, graph.number_of_nodes())
            features.append(pos_features)
        
        # Combine all features
        combined_features = torch.cat(features)
        return combined_features
    
    def _one_hot_encode_node_type(self, node_type: str) -> torch.Tensor:
        """
        One-hot encode the node type.
        
        Args:
            node_type: Type of the node
            
        Returns:
            One-hot encoded tensor
        """
        # Define common node types in code graphs
        common_types = [
            'function', 'variable', 'constant', 'parameter', 
            'declaration', 'expression', 'statement', 'control', 
            'operator', 'call', 'return', 'condition', 'loop',
            'unknown'
        ]
        
        if node_type not in common_types:
            node_type = 'unknown'
        
        idx = common_types.index(node_type)
        one_hot = torch.zeros(len(common_types))
        one_hot[idx] = 1.0
        
        return one_hot
    
    def _encode_ast_type(self, ast_type: str) -> torch.Tensor:
        """
        Encode the AST type information.
        
        Args:
            ast_type: AST type of the node
            
        Returns:
            Encoded tensor
        """
        # This is a simplified encoding - in a real implementation,
        # you might use a more sophisticated approach or a pre-trained encoding
        common_ast_types = [
            'FunctionDecl', 'VarDecl', 'ParmVarDecl', 'CompoundStmt',
            'DeclStmt', 'BinaryOperator', 'CallExpr', 'IfStmt', 
            'ForStmt', 'WhileStmt', 'ReturnStmt', 'UnaryOperator',
            'unknown'
        ]
        
        if ast_type not in common_ast_types:
            ast_type = 'unknown'
        
        idx = common_ast_types.index(ast_type)
        encoding = torch.zeros(len(common_ast_types))
        encoding[idx] = 1.0
        
        return encoding
    
    def _encode_code_token(self, code: str) -> torch.Tensor:
        """
        Encode the code token.
        
        Args:
            code: Code snippet or token
            
        Returns:
            Encoded tensor
        """
        # In a real implementation, you might use a more sophisticated approach
        # such as subword tokenization or a pre-trained code embedding
        token_idx = self.get_token_idx(code)
        encoding_dim = 16  # Simplified dimension for this example
        
        # Simple embedding: convert token index to binary and pad
        binary = format(token_idx, '016b')
        encoding = torch.tensor([int(bit) for bit in binary], dtype=torch.float)
        
        return encoding
    
    def _positional_encoding(self, node_id: int, graph_size: int) -> torch.Tensor:
        """
        Create a positional encoding for a node.
        
        Args:
            node_id: ID of the node
            graph_size: Total number of nodes in the graph
            
        Returns:
            Positional encoding tensor
        """
        # Simple positional encoding: normalized position + sine/cosine encoding
        pos_dim = 16  # Simplified dimension for this example
        position = float(node_id) / max(1, graph_size - 1)  # Normalized position [0, 1]
        
        encoding = torch.zeros(pos_dim)
        # Set the first value to the normalized position
        encoding[0] = position
        
        # Add sine/cosine encodings at different frequencies
        for i in range(1, pos_dim // 2):
            encoding[2*i-1] = np.sin(position * (2 ** i))
            encoding[2*i] = np.cos(position * (2 ** i))
        
        return encoding
    
    def _extract_edges_by_type(self, graph: nx.DiGraph, edge_type: str) -> Tuple[List[int], List[int]]:
        """
        Extract edges of a specific type from the graph.
        
        Args:
            graph: NetworkX DiGraph of the code
            edge_type: Type of edge to extract
            
        Returns:
            Tuple of (source nodes, destination nodes)
        """
        src_nodes = []
        dst_nodes = []
        
        for u, v, data in graph.edges(data=True):
            # Check if the edge has the desired type
            if data.get('type') == edge_type:
                src_nodes.append(u)
                dst_nodes.append(v)
        
        return src_nodes, dst_nodes