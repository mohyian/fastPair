"""
HeteroGraph Module

This module provides the implementation for heterogeneous graph representation
of code, including vulnerable/patched code pairs. It handles the creation and
manipulation of heterogeneous graphs for vulnerability detection.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set

import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, NormalizeFeatures

from src.utils.logger import get_logger

logger = get_logger(__name__)

class HeteroGraphBuilder:
    """
    Builder class for creating and manipulating heterogeneous graphs
    for vulnerability detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the HeteroGraphBuilder with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.edge_types = config["model"]["edge_types"]
    
    def build_vuln_patch_graph_pair(
        self, 
        vuln_graph: HeteroData, 
        patch_graph: HeteroData
    ) -> HeteroData:
        """
        Construct a combined graph containing both vulnerable and patched code
        connected by alignment edges.
        
        Args:
            vuln_graph: HeteroData representing the vulnerable code
            patch_graph: HeteroData representing the patched code
            
        Returns:
            Combined heterogeneous graph
        """
        combined = HeteroData()
        
        # Add vulnerable code nodes with type "vuln"
        combined["vuln"].x = vuln_graph["code"].x
        combined["vuln"].y = vuln_graph["code"].y
        
        # Add patched code nodes with type "patch"
        combined["patch"].x = patch_graph["code"].x
        combined["patch"].y = patch_graph["code"].y
        
        # Add vulnerable-vulnerable edges (internal connections in vulnerable code)
        for edge_type in self.edge_types:
            edge_name = f"code, {edge_type}, code"
            if edge_name in vuln_graph.edge_types:
                combined["vuln", edge_type, "vuln"].edge_index = vuln_graph[edge_name].edge_index
        
        # Add patch-patch edges (internal connections in patched code)
        for edge_type in self.edge_types:
            edge_name = f"code, {edge_type}, code"
            if edge_name in patch_graph.edge_types:
                combined["patch", edge_type, "patch"].edge_index = patch_graph[edge_name].edge_index
        
        # Add alignment edges between vulnerable and patched code
        self._add_alignment_edges(combined, vuln_graph, patch_graph)
        
        return combined
    
    def _add_alignment_edges(
        self, 
        combined: HeteroData, 
        vuln_graph: HeteroData, 
        patch_graph: HeteroData
    ) -> None:
        """
        Add alignment edges between vulnerable and patched nodes.
        
        Args:
            combined: The combined graph to update
            vuln_graph: The vulnerable code graph
            patch_graph: The patched code graph
        """
        if not hasattr(vuln_graph, "alignment") or not hasattr(patch_graph, "alignment"):
            logger.warning("Alignment information is missing. Skipping alignment edges.")
            return
        
        vuln_to_patch_edges_src = []
        vuln_to_patch_edges_dst = []
        patch_to_vuln_edges_src = []
        patch_to_vuln_edges_dst = []
        
        # Create bidirectional alignment edges
        for vuln_node, patch_node in vuln_graph.alignment.items():
            vuln_to_patch_edges_src.append(vuln_node)
            vuln_to_patch_edges_dst.append(patch_node)
            
            patch_to_vuln_edges_src.append(patch_node)
            patch_to_vuln_edges_dst.append(vuln_node)
        
        # Add edges to the combined graph
        if vuln_to_patch_edges_src:
            combined["vuln", "aligned", "patch"].edge_index = torch.tensor(
                [vuln_to_patch_edges_src, vuln_to_patch_edges_dst], 
                dtype=torch.long
            )
            
            combined["patch", "aligned", "vuln"].edge_index = torch.tensor(
                [patch_to_vuln_edges_src, patch_to_vuln_edges_dst], 
                dtype=torch.long
            )
    
    def add_semantic_edges(self, graph: HeteroData, semantic_relations: List[Tuple[int, int, str]]) -> HeteroData:
        """
        Add semantic edges to a graph based on provided semantic relations.
        
        Args:
            graph: HeteroData graph to update
            semantic_relations: List of (source, destination, relation_type) tuples
            
        Returns:
            Updated graph with semantic edges
        """
        # Group relations by type
        relation_groups = {}
        for src, dst, rel_type in semantic_relations:
            if rel_type not in relation_groups:
                relation_groups[rel_type] = {"src": [], "dst": []}
            
            relation_groups[rel_type]["src"].append(src)
            relation_groups[rel_type]["dst"].append(dst)
        
        # Add each relation type as a separate edge type
        for rel_type, edges in relation_groups.items():
            edge_type = f"semantic_{rel_type}"
            
            # For "code" node type (single node type graph)
            if "code" in graph.node_types:
                graph["code", edge_type, "code"].edge_index = torch.tensor(
                    [edges["src"], edges["dst"]], 
                    dtype=torch.long
                )
            
            # For "vuln" and "patch" node types (combined graph)
            elif "vuln" in graph.node_types and "patch" in graph.node_types:
                # Add relations only within the same node type (vuln-vuln or patch-patch)
                # In a real implementation, you might need more complex logic
                
                # For vulnerable code nodes
                vuln_src = []
                vuln_dst = []
                
                # For patched code nodes
                patch_src = []
                patch_dst = []
                
                for i in range(len(edges["src"])):
                    src, dst = edges["src"][i], edges["dst"][i]
                    
                    # This is a simplified approach - in a real implementation,
                    # you would need to determine which nodes belong to which part
                    if src < len(graph["vuln"].x) and dst < len(graph["vuln"].x):
                        vuln_src.append(src)
                        vuln_dst.append(dst)
                    elif (src >= len(graph["vuln"].x) and dst >= len(graph["vuln"].x) and
                          src < len(graph["vuln"].x) + len(graph["patch"].x) and
                          dst < len(graph["vuln"].x) + len(graph["patch"].x)):
                        # Adjust indices for patch nodes
                        patch_src.append(src - len(graph["vuln"].x))
                        patch_dst.append(dst - len(graph["vuln"].x))
                
                if vuln_src:
                    graph["vuln", edge_type, "vuln"].edge_index = torch.tensor(
                        [vuln_src, vuln_dst], 
                        dtype=torch.long
                    )
                
                if patch_src:
                    graph["patch", edge_type, "patch"].edge_index = torch.tensor(
                        [patch_src, patch_dst], 
                        dtype=torch.long
                    )
        
        return graph

    def get_subgraph(self, graph: HeteroData, node_indices: List[int], node_type: str = "code") -> HeteroData:
        """
        Extract a subgraph containing only the specified nodes.
        
        Args:
            graph: Original HeteroData graph
            node_indices: List of node indices to keep
            node_type: Type of nodes (default: "code")
            
        Returns:
            Subgraph as HeteroData
        """
        subgraph = HeteroData()
        
        # Create node id mapping from original to subgraph
        node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_indices)}
        
        # Add nodes
        subgraph[node_type].x = graph[node_type].x[node_indices]
        
        # If the original graph has a label, copy it
        if hasattr(graph[node_type], 'y'):
            subgraph[node_type].y = graph[node_type].y
        
        # Handle each edge type
        for edge_type in graph.edge_types:
            src_type, rel_type, dst_type = edge_type
            
            # Only process edges if both source and destination are the same node type
            if src_type == node_type and dst_type == node_type:
                edge_index = graph[edge_type].edge_index
                src, dst = edge_index[0], edge_index[1]
                
                # Filter edges that connect nodes in the subgraph
                new_src = []
                new_dst = []
                
                for i in range(src.size(0)):
                    s, d = src[i].item(), dst[i].item()
                    if s in node_id_map and d in node_id_map:
                        new_src.append(node_id_map[s])
                        new_dst.append(node_id_map[d])
                
                if new_src:
                    subgraph[edge_type].edge_index = torch.tensor(
                        [new_src, new_dst], 
                        dtype=torch.long
                    )
        
        return subgraph

    def merge_graphs(self, graphs: List[HeteroData]) -> HeteroData:
        """
        Merge multiple heterogeneous graphs into a single batch.
        
        Args:
            graphs: List of HeteroData graphs to merge
            
        Returns:
            Merged graph
        """
        if not graphs:
            return HeteroData()
        
        # Use PyTorch Geometric's built-in batching
        from torch_geometric.data import Batch
        
        # For HeteroData, we need to ensure all graphs have the same structure
        node_types = set()
        edge_types = set()
        
        for graph in graphs:
            node_types.update(graph.node_types)
            edge_types.update(graph.edge_types)
        
        # Add missing node and edge types to all graphs
        for graph in graphs:
            for node_type in node_types:
                if node_type not in graph.node_types:
                    # Add empty node features
                    # This assumes all graphs have the same feature dimensions
                    if node_type in graphs[0].node_types:
                        dim = graphs[0][node_type].x.size(1)
                        graph[node_type].x = torch.zeros((0, dim))
            
            for edge_type in edge_types:
                if edge_type not in graph.edge_types:
                    # Add empty edge indices
                    graph[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Now we can batch the graphs
        try:
            return Batch.from_data_list(graphs)
        except Exception as e:
            logger.error(f"Error merging graphs: {e}")
            # Fallback: create a new graph and manually merge
            merged = HeteroData()
            
            # Track the number of nodes for each type to adjust edge indices
            node_counts = {node_type: 0 for node_type in node_types}
            
            # Process each graph
            for graph in graphs:
                for node_type in node_types:
                    if node_type in graph.node_types:
                        # Initialize if this is the first graph with this node type
                        if node_type not in merged.node_types:
                            merged[node_type].x = graph[node_type].x
                            if hasattr(graph[node_type], 'y'):
                                merged[node_type].y = graph[node_type].y
                        else:
                            merged[node_type].x = torch.cat([merged[node_type].x, graph[node_type].x], dim=0)
                            if hasattr(graph[node_type], 'y') and hasattr(merged[node_type], 'y'):
                                merged[node_type].y = torch.cat([merged[node_type].y, graph[node_type].y], dim=0)
                
                # Update node counts before processing edges
                for node_type in node_types:
                    if node_type in graph.node_types:
                        node_counts[node_type] += graph[node_type].x.size(0)
                
                # Process edges for each type
                for edge_type in edge_types:
                    if edge_type in graph.edge_types:
                        src_type, rel_type, dst_type = edge_type
                        
                        # Skip if the graph doesn't have nodes of the required types
                        if src_type not in graph.node_types or dst_type not in graph.node_types:
                            continue
                        
                        edge_index = graph[edge_type].edge_index
                        
                        # Adjust indices based on the number of nodes already in the merged graph
                        src_offset = node_counts[src_type] - graph[src_type].x.size(0)
                        dst_offset = node_counts[dst_type] - graph[dst_type].x.size(0)
                        
                        new_edge_index = edge_index.clone()
                        new_edge_index[0] += src_offset
                        new_edge_index[1] += dst_offset
                        
                        # Add edges to the merged graph
                        if edge_type not in merged.edge_types:
                            merged[edge_type].edge_index = new_edge_index
                        else:
                            merged[edge_type].edge_index = torch.cat(
                                [merged[edge_type].edge_index, new_edge_index], dim=1
                            )
            
            return merged