"""
Visualization Utility Module

This module provides functions for visualizing graph structures, patterns,
and vulnerability detection results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx

from src.utils.logger import get_logger

logger = get_logger(__name__)

def visualize_heterograph(
    graph: HeteroData,
    node_colors: Dict[str, List[str]] = None,
    node_sizes: Dict[str, List[float]] = None,
    output_path: str = None,
    title: str = "Heterogeneous Graph Visualization"
) -> plt.Figure:
    """
    Visualize a heterogeneous graph.
    
    Args:
        graph: Heterogeneous graph to visualize
        node_colors: Dictionary mapping node types to color lists
        node_sizes: Dictionary mapping node types to size lists
        output_path: Path to save the visualization (optional)
        title: Title for the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Convert each node type to a NetworkX graph
    nx_graphs = {}
    for node_type in graph.node_types:
        nx_graphs[node_type] = to_networkx(
            graph, 
            node_attrs=["x"], 
            edge_attrs=[],
            to_undirected=True,
            node_type=node_type
        )
    
    # Combine the graphs
    G = nx.Graph()
    
    # Add nodes for each type with different colors
    default_colors = {
        "code": "lightblue",
        "vuln": "red",
        "patch": "green"
    }
    
    default_sizes = {
        "code": 300,
        "vuln": 300,
        "patch": 300
    }
    
    # Track node positions by type
    pos_by_type = {}
    
    # Add nodes for each type
    for node_type, nx_graph in nx_graphs.items():
        # Get colors for this node type
        if node_colors and node_type in node_colors:
            colors = node_colors[node_type]
        else:
            default_color = default_colors.get(node_type, "gray")
            colors = [default_color] * nx_graph.number_of_nodes()
        
        # Get sizes for this node type
        if node_sizes and node_type in node_sizes:
            sizes = node_sizes[node_type]
        else:
            default_size = default_sizes.get(node_type, 300)
            sizes = [default_size] * nx_graph.number_of_nodes()
        
        # Add nodes to the combined graph
        for i, node in enumerate(nx_graph.nodes()):
            G.add_node(
                f"{node_type}_{node}",
                node_type=node_type,
                original_id=node,
                color=colors[i] if i < len(colors) else colors[-1],
                size=sizes[i] if i < len(sizes) else sizes[-1]
            )
    
    # Add edges for each type
    for edge_type in graph.edge_types:
        src_type, rel_type, dst_type = edge_type
        
        # Get the edge indices
        edge_index = graph[edge_type].edge_index
        src, dst = edge_index
        
        # Add edges to the combined graph
        for i in range(src.size(0)):
            s, d = src[i].item(), dst[i].item()
            G.add_edge(
                f"{src_type}_{s}",
                f"{dst_type}_{d}",
                edge_type=rel_type
            )
    
    # Calculate layout for each node type separately
    pos = {}
    offset = {
        "code": (0, 0),
        "vuln": (-1, 0),
        "patch": (1, 0)
    }
    
    for node_type in nx_graphs.keys():
        # Get nodes of this type
        nodes = [n for n, attr in G.nodes(data=True) if attr["node_type"] == node_type]
        
        if not nodes:
            continue
        
        # Create a subgraph for this node type
        subgraph = G.subgraph(nodes)
        
        # Calculate layout for this subgraph
        subgraph_pos = nx.spring_layout(subgraph)
        
        # Apply offset if there are multiple node types
        if len(nx_graphs) > 1:
            type_offset = offset.get(node_type, (0, 0))
            subgraph_pos = {node: (x + type_offset[0], y + type_offset[1]) 
                           for node, (x, y) in subgraph_pos.items()}
        
        # Add to overall positions
        pos.update(subgraph_pos)
    
    # Draw nodes for each type
    for node_type in nx_graphs.keys():
        # Get nodes of this type
        nodes = [n for n, attr in G.nodes(data=True) if attr["node_type"] == node_type]
        
        if not nodes:
            continue
        
        # Get colors and sizes
        node_attrs = [G.nodes[n] for n in nodes]
        colors = [attr["color"] for attr in node_attrs]
        sizes = [attr["size"] for attr in node_attrs]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, 
            pos, 
            nodelist=nodes, 
            node_color=colors,
            node_size=sizes,
            label=node_type,
            ax=ax
        )
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(
        G, 
        pos, 
        labels={n: attr["original_id"] for n, attr in G.nodes(data=True)},
        font_size=8,
        ax=ax
    )
    
    # Set title
    ax.set_title(title)
    
    # Remove axis
    ax.axis("off")
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=default_colors.get(node_type, "gray"), 
                  markersize=10, label=node_type)
        for node_type in nx_graphs.keys()
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved graph visualization to {output_path}")
    
    return fig

def visualize_pattern_attention(
    attention_weights: torch.Tensor,
    pattern_id: int,
    top_k: int = 10,
    output_path: str = None,
    title: str = "Pattern Attention Visualization"
) -> plt.Figure:
    """
    Visualize pattern attention weights.
    
    Args:
        attention_weights: Pattern attention weights [num_nodes, num_patterns]
        pattern_id: ID of the pattern to visualize
        top_k: Number of top nodes to highlight
        output_path: Path to save the visualization (optional)
        title: Title for the visualization
        
    Returns:
        Matplotlib figure
    """
    # Get attention weights for the specified pattern
    pattern_attention = attention_weights[:, pattern_id].cpu().numpy()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create barplot
    x = np.arange(len(pattern_attention))
    bar_colors = ['lightgray'] * len(pattern_attention)
    
    # Highlight top-k nodes
    if top_k > 0:
        top_indices = pattern_attention.argsort()[-top_k:][::-1]
        for idx in top_indices:
            bar_colors[idx] = 'red'
    
    # Plot the bars
    ax.bar(x, pattern_attention, color=bar_colors)
    
    # Set labels
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f"{title} - Pattern {pattern_id}")
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved pattern attention visualization to {output_path}")
    
    return fig

def visualize_pattern_similarity(
    similarity_matrix: torch.Tensor,
    output_path: str = None,
    title: str = "Pattern Similarity Matrix"
) -> plt.Figure:
    """
    Visualize pattern similarity matrix.
    
    Args:
        similarity_matrix: Pattern similarity matrix
        output_path: Path to save the visualization (optional)
        title: Title for the visualization
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    similarity = similarity_matrix.cpu().numpy()
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(similarity, cmap='viridis')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_xlabel('Security Patterns')
    ax.set_ylabel('Vulnerability Patterns')
    ax.set_title(title)
    
    # Add text annotations
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            text = ax.text(j, i, f"{similarity[i, j]:.2f}",
                          ha="center", va="center", color="w" if similarity[i, j] < 0.7 else "black")
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved pattern similarity visualization to {output_path}")
    
    return fig

def visualize_vulnerability_scores(
    scores: List[float],
    thresholds: List[float] = [0.3, 0.7],
    labels: List[str] = None,
    output_path: str = None,
    title: str = "Vulnerability Scores"
) -> plt.Figure:
    """
    Visualize vulnerability scores for multiple samples.
    
    Args:
        scores: List of vulnerability scores
        thresholds: List of threshold values [low, high]
        labels: List of labels for the samples (optional)
        output_path: Path to save the visualization (optional)
        title: Title for the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create labels if not provided
    if labels is None:
        labels = [f"Sample {i+1}" for i in range(len(scores))]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors based on thresholds
    colors = []
    for score in scores:
        if score < thresholds[0]:
            colors.append('green')
        elif score < thresholds[1]:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Create barplot
    x = np.arange(len(scores))
    ax.bar(x, scores, color=colors)
    
    # Add threshold lines
    ax.axhline(y=thresholds[0], linestyle='--', color='green', alpha=0.7)
    ax.axhline(y=thresholds[1], linestyle='--', color='red', alpha=0.7)
    
    # Set labels
    ax.set_xlabel('Samples')
    ax.set_ylabel('Vulnerability Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'Low Risk (<{thresholds[0]})'),
        Patch(facecolor='orange', label=f'Medium Risk ({thresholds[0]}-{thresholds[1]})'),
        Patch(facecolor='red', label=f'High Risk (>{thresholds[1]})')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved vulnerability scores visualization to {output_path}")
    
    return fig

def visualize_training_metrics(
    metrics: Dict[str, List[float]],
    output_path: str = None,
    title: str = "Training Metrics"
) -> plt.Figure:
    """
    Visualize training metrics over epochs.
    
    Args:
        metrics: Dictionary mapping metric names to lists of values
        output_path: Path to save the visualization (optional)
        title: Title for the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create a figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)), sharex=True)
    
    # If there's only one metric, axes is not a list
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[i]
        epochs = range(1, len(values) + 1)
        
        ax.plot(epochs, values, 'o-', linewidth=2, markersize=6)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} over Epochs")
        ax.grid(True)
    
    # Set x-axis label for the bottom plot
    axes[-1].set_xlabel('Epochs')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved training metrics visualization to {output_path}")
    
    return fig