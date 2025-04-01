import torch
import argparse
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def inspect_raw_files(data_dir):
    """Inspect the raw PyTorch files containing graph data."""
    raw_dir = os.path.join(data_dir, 'raw')
    
    # Load the files
    vuln_graphs_path = os.path.join(raw_dir, 'vulnerable_graphs.pt')
    patch_graphs_path = os.path.join(raw_dir, 'patched_graphs.pt')
    alignments_path = os.path.join(raw_dir, 'alignments.pt')
    
    # Check if files exist
    if not os.path.exists(vuln_graphs_path):
        print(f"Error: {vuln_graphs_path} does not exist")
        return
    if not os.path.exists(patch_graphs_path):
        print(f"Error: {patch_graphs_path} does not exist")
        return
    if not os.path.exists(alignments_path):
        print(f"Error: {alignments_path} does not exist")
        return
    
    # Load the data
    try:
        vuln_graphs = torch.load(vuln_graphs_path)
        patch_graphs = torch.load(patch_graphs_path)
        alignments = torch.load(alignments_path)
        
        print(f"\n=== Raw Files Summary ===")
        print(f"Vulnerable Graphs: {len(vuln_graphs)} graphs")
        print(f"Patched Graphs: {len(patch_graphs)} graphs")
        print(f"Alignments: {len(alignments)} mappings")
        
        # Inspect the first graph
        if vuln_graphs:
            print("\n=== First Vulnerable Graph ===")
            inspect_hetero_data(vuln_graphs[0])
        
        if patch_graphs:
            print("\n=== First Patched Graph ===")
            inspect_hetero_data(patch_graphs[0])
        
        if alignments:
            print("\n=== First Alignment Mapping ===")
            print(f"Number of aligned nodes: {len(alignments[0])}")
            # Print a few alignment examples
            alignment_items = list(alignments[0].items())
            for i, (vuln_node, patch_node) in enumerate(alignment_items[:5]):
                print(f"  {vuln_node} -> {patch_node}")
            if len(alignment_items) > 5:
                print(f"  ... and {len(alignment_items) - 5} more")
        
    except Exception as e:
        print(f"Error loading files: {e}")

def inspect_processed_files(data_dir):
    """Inspect the processed PyTorch files containing combined graph data."""
    processed_dir = os.path.join(data_dir, 'processed')
    data_path = os.path.join(processed_dir, 'data.pt')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist")
        return
    
    try:
        processed_data = torch.load(data_path)
        
        print(f"\n=== Processed Data Summary ===")
        print(f"Number of combined graphs: {len(processed_data)}")
        
        if processed_data:
            print("\n=== First Combined Graph ===")
            inspect_combined_hetero_data(processed_data[0])
            
    except Exception as e:
        print(f"Error loading processed data: {e}")

def inspect_hetero_data(data):
    """Inspect a PyTorch Geometric HeteroData object."""
    if not isinstance(data, HeteroData):
        print(f"Object is not a HeteroData: {type(data)}")
        return
    
    # Print node types and counts
    node_types = [key for key in data.node_types]
    print(f"Node types: {node_types}")
    
    for node_type in node_types:
        if hasattr(data[node_type], 'x'):
            num_nodes = data[node_type].x.size(0)
            feature_dim = data[node_type].x.size(1)
            print(f"  {node_type}: {num_nodes} nodes with {feature_dim} features each")
        else:
            print(f"  {node_type}: No features found")
    
    # Print edge types and counts
    print("Edge types:")
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index'):
            num_edges = data[edge_type].edge_index.size(1)
            print(f"  {edge_type}: {num_edges} edges")
        else:
            print(f"  {edge_type}: No edge indices found")
    
    # Check for graph-level attributes
    for node_type in node_types:
        if hasattr(data[node_type], 'y'):
            print(f"  Graph label for {node_type}: {data[node_type].y.item()}")

def inspect_combined_hetero_data(data):
    """Inspect a combined HeteroData object with both vulnerable and patched graphs."""
    if not isinstance(data, HeteroData):
        print(f"Object is not a HeteroData: {type(data)}")
        return
    
    # Print node types and counts
    node_types = [key for key in data.node_types]
    print(f"Node types: {node_types}")
    
    for node_type in node_types:
        if hasattr(data[node_type], 'x'):
            num_nodes = data[node_type].x.size(0)
            feature_dim = data[node_type].x.size(1)
            print(f"  {node_type}: {num_nodes} nodes with {feature_dim} features each")
        else:
            print(f"  {node_type}: No features found")
    
    # Print edge types and counts
    print("Edge types:")
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_index'):
            num_edges = data[edge_type].edge_index.size(1)
            print(f"  {edge_type}: {num_edges} edges")
        else:
            print(f"  {edge_type}: No edge indices found")
    
    # Check for alignment edges
    if ('vuln', 'aligned', 'patch') in data.edge_types:
        num_aligned = data[('vuln', 'aligned', 'patch')].edge_index.size(1)
        print(f"\nAlignment edges: {num_aligned} nodes aligned between vulnerable and patched graphs")

def visualize_graph(data_dir, index=0, graph_type='vuln'):
    """Visualize a graph from the dataset for inspection."""
    # Load the graph
    raw_dir = os.path.join(data_dir, 'raw')
    
    if graph_type == 'vuln':
        graphs_path = os.path.join(raw_dir, 'vulnerable_graphs.pt')
    else:
        graphs_path = os.path.join(raw_dir, 'patched_graphs.pt')
    
    if not os.path.exists(graphs_path):
        print(f"Error: {graphs_path} does not exist")
        return
    
    try:
        graphs = torch.load(graphs_path)
        
        if index >= len(graphs):
            print(f"Error: Index {index} is out of range (max: {len(graphs)-1})")
            return
        
        graph = graphs[index]
        
        # Convert to NetworkX for visualization
        G = convert_to_networkx(graph, graph_type)
        
        # Visualize
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=100)
        
        # Draw edges by type
        edge_colors = {'control_flow': 'red', 'data_flow': 'blue', 'semantic': 'green'}
        
        for edge_type, color in edge_colors.items():
            edge_list = [(u, v) for u, v, d in G.edges(data=True) if d['edge_type'] == edge_type]
            if edge_list:
                nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color=color, width=1.0)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(f"{graph_type.title()} Graph (Index: {index})")
        plt.legend(edge_colors.keys())
        plt.tight_layout()
        plt.axis('off')
        
        # Save the visualization
        output_file = f"{graph_type}_graph_{index}.png"
        plt.savefig(output_file, dpi=300)
        print(f"Visualization saved to {output_file}")
        plt.close()
        
    except Exception as e:
        print(f"Error during visualization: {e}")

def convert_to_networkx(data, graph_type):
    """Convert a PyTorch Geometric HeteroData object to a NetworkX graph for visualization."""
    G = nx.DiGraph()
    
    # Add nodes
    if hasattr(data[graph_type], 'x'):
        for i in range(data[graph_type].x.size(0)):
            G.add_node(i)
    
    # Add edges by type
    edge_types = [(graph_type, edge_type, graph_type) for edge_type in ['control_flow', 'data_flow', 'semantic']]
    
    for edge_type in edge_types:
        if edge_type in data.edge_types and hasattr(data[edge_type], 'edge_index'):
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                source, target = edge_index[0, i], edge_index[1, i]
                G.add_edge(source.item(), target.item(), edge_type=edge_type[1])
    
    return G

def main():
    parser = argparse.ArgumentParser(description='Inspect PyTorch .pt files containing graph data')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data')
    parser.add_argument('--visualize', action='store_true', help='Visualize a graph')
    parser.add_argument('--index', type=int, default=0, help='Index of graph to visualize')
    parser.add_argument('--graph_type', type=str, default='vuln', choices=['vuln', 'patch'], 
                        help='Type of graph to visualize')
    
    args = parser.parse_args()
    
    # Inspect the raw files
    inspect_raw_files(args.data_dir)
    
    # Inspect the processed files
    inspect_processed_files(args.data_dir)
    
    # Visualize a graph if requested
    if args.visualize:
        visualize_graph(args.data_dir, args.index, args.graph_type)

if __name__ == "__main__":
    main()