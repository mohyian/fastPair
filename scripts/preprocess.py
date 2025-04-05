#!/usr/bin/env python
"""
Data Preprocessing Script

This script processes the vulnerability dataset in log format and converts it to the format
required by the graph-based vulnerability detection system.
"""

import os
import sys
import re
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml
import shutil
from clang.cindex import TokenKind
import clang.cindex

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.graph_processing import GraphProcessor
from src.utils.logger import initialize_logging, get_logger

# Initialize logging
initialize_logging()
logger = get_logger(__name__)

# Edge type mapping from dataset to our model
EDGE_TYPE_MAPPING = {
    'AST': 'AST',
    'DDG': 'data_flow',
    'CFG': 'control_flow',
    'CDG': 'control_flow',  # Map CDG to control_flow
}

# Node type info
NODE_TYPE_INFO = {
    'C': 'context',   # Context node (unchanged)
    'D': 'context',   # Dependency node (unchanged)
    '-': 'vulnerable', # Pre-patch node (vulnerable)
    '+': 'patched'     # Post-patch node (patched)
}

def parse_log_file(log_file):
    """
    Parse a log file containing graph data.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary with graph data
    """
    logger.info(f"Parsing log file: {log_file}")
    
    # Read the log file
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split content by sections using delimiters
    parts = content.split('---------------------------')
    
    if len(parts) < 3:
        logger.warning(f"Invalid log file format (not enough sections): {log_file}")
        return None
    
    # First part contains all edges and nodes
    all_part = parts[0]
    all_sections = all_part.split('===========================')
    
    if len(all_sections) < 2:
        logger.warning(f"Invalid log file format (missing all_sections): {log_file}")
        return None
    
    all_edges = parse_edges_section(all_sections[0])
    all_nodes = parse_nodes_section(all_sections[1])
    
    # Second part contains pre-patch data
    pre_part = parts[1]
    pre_sections = pre_part.split('===========================')
    
    if len(pre_sections) < 2:
        logger.warning(f"Invalid log file format (missing pre_sections): {log_file}")
        return None
    
    pre_edges = parse_edges_section(pre_sections[0])
    pre_nodes = parse_nodes_section(pre_sections[1])
    
    # Third part contains post-patch data
    post_part = parts[2]
    post_sections = post_part.split('===========================')
    
    if len(post_sections) < 2:
        logger.warning(f"Invalid log file format (missing post_sections): {log_file}")
        return None
    
    post_edges = parse_edges_section(post_sections[0])
    post_nodes = parse_nodes_section(post_sections[1])
    
    return {
        'all_edges': all_edges,
        'all_nodes': all_nodes,
        'pre_edges': pre_edges,
        'pre_nodes': pre_nodes,
        'post_edges': post_edges,
        'post_nodes': post_nodes
    }

def parse_edges_section(section):
    """
    Parse the edges section of a log file.
    
    Args:
        section: Edge section content
        
    Returns:
        List of edge tuples (src, dst, type, version)
    """
    edges = []
    lines = section.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
        
        # Parse edge tuple (src, dst, type, version)
        match = re.match(r'\((-?\d+),\s*(-?\d+),\s*\'?([\w]+)\'?,\s*(-?\d+)\)', line)
        if match:
            src = int(match.group(1))
            dst = int(match.group(2))
            edge_type = match.group(3)
            version = int(match.group(4))
            
            # Map the edge type to our model's edge types
            mapped_type = None
            for prefix, model_type in EDGE_TYPE_MAPPING.items():
                if edge_type.startswith(prefix):
                    mapped_type = model_type
                    break
            
            if mapped_type:
                edges.append((src, dst, mapped_type, version))
    
    return edges

def parse_nodes_section(section):
    """
    Parse the nodes section of a log file.
    
    Args:
        section: Node section content
        
    Returns:
        List of node tuples (id, label, type, distance, line_num, code)
    """
    nodes = []
    lines = section.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
        
        # Simplified parsing for nodes
        try:
            # Remove parentheses
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                line = line[1:-1]
            
            # Extract node ID, label, type, distance
            parts = line.split(',', 5)  # Split into 6 parts
            if len(parts) < 6:
                continue
                
            node_id = int(parts[0].strip())
            label = int(parts[1].strip())
            
            # Extract node type (remove quotes)
            node_type = parts[2].strip()
            if node_type.startswith("'") and node_type.endswith("'"):
                node_type = node_type[1:-1]
            elif node_type.startswith('"') and node_type.endswith('"'):
                node_type = node_type[1:-1]
            
            distance = int(parts[3].strip())
            
            # Extract line number (remove quotes)
            line_num = parts[4].strip()
            if line_num.startswith("'") and line_num.endswith("'"):
                line_num = line_num[1:-1]
            elif line_num.startswith('"') and line_num.endswith('"'):
                line_num = line_num[1:-1]
            
            # Extract code (remove quotes)
            code = parts[5].strip()
            if code.startswith("'") and code.endswith("'"):
                code = code[1:-1]
            elif code.startswith('"') and code.endswith('"'):
                code = code[1:-1]
            
            nodes.append((node_id, label, node_type, distance, line_num, code))
        except Exception as e:
            logger.debug(f"Error parsing node line: {line}, error: {e}")
            continue
    
    return nodes

def create_node_features(code, node_type, label):
    """
    Create a 20-dimensional feature vector for a node based on its code.
    This implements a simplified version of the feature extraction from GraphSPD.
    
    Args:
        code: Code in the node
        node_type: Type of the node
        label: Label of the node (0 for patched, 1 for vulnerable)
        
    Returns:
        20-dimensional feature vector
    """
    # Initialize feature vector
    features = np.zeros(20)
    
    # First feature is the label
    features[0] = label
    
    # Extract code features
    if code:
        # Check for conditionals
        if 'if' in code or 'switch' in code or 'case' in code:
            features[1] = 1
        
        # Check for loops
        if 'for' in code or 'while' in code:
            features[2] = 1
        
        # Check for jumps
        if 'return' in code or 'break' in code or 'continue' in code or 'goto' in code:
            features[3] = 1
        
        # Function calls (simplified)
        if re.search(r'\w+\s*\(', code):
            features[4] = 1
        
        # Variables
        var_count = len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code))
        features[5] = min(var_count, 5)  # Cap at 5 for normalization
        
        # Constants
        const_count = len(re.findall(r'\b\d+\b', code))
        features[6] = min(const_count, 5)  # Cap at 5
        
        # String literals
        str_count = len(re.findall(r'\".*?\"', code))
        features[7] = min(str_count, 3)  # Cap at 3
        
        # Arithmetic operators
        arith_count = len(re.findall(r'[\+\-\*\/\%]', code))
        features[8] = min(arith_count, 5)  # Cap at 5
        
        # Relational operators
        rel_count = len(re.findall(r'[=!<>]=?', code))
        features[9] = min(rel_count, 3)  # Cap at 3
        
        # Logical operators
        log_count = len(re.findall(r'&&|\|\||!', code))
        features[10] = min(log_count, 3)  # Cap at 3
        
        # Bitwise operators
        bit_count = len(re.findall(r'[&\|\^~]|<<|>>', code))
        features[11] = min(bit_count, 3)  # Cap at 3
        
        # Code length (character count)
        features[12] = min(len(code), 100) / 100  # Normalize by 100
        
        # Memory-related keywords
        if any(word in code.lower() for word in 
               ['malloc', 'free', 'alloc', 'realloc', 'calloc', 'mem', 'memory',
                'new', 'delete', 'sizeof']):
            features[13] = 1
        
        # String-related functions
        if any(word in code.lower() for word in 
               ['strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp',
                'strlen', 'strstr', 'strchr', 'strrchr']):
            features[14] = 1
        
        # Locks/synchronization
        if any(word in code.lower() for word in 
               ['lock', 'mutex', 'spin', 'atomic', 'sync']):
            features[15] = 1
        
        # Pointer operations
        ptr_count = len(re.findall(r'->', code)) + len(re.findall(r'\*', code))
        features[16] = min(ptr_count, 3)  # Cap at 3
        
        # Array operations
        if '[' in code and ']' in code:
            features[17] = 1
        
        # NULL checks
        if 'NULL' in code or 'null' in code.lower() or '== 0' in code or '!= 0' in code:
            features[18] = 1
        
        # API calls (simplified)
        if any(word in code.lower() for word in 
               ['get', 'set', 'put', 'init', 'create', 'destroy', 'open', 'close',
                'read', 'write', 'send', 'recv', 'connect']):
            features[19] = 1
    
    return features

def create_networkx_graph(nodes, edges):
    """
    Create a NetworkX graph from nodes and edges, remapping negative IDs to positive.
    
    Args:
        nodes: List of node tuples
        edges: List of edge tuples
        
    Returns:
        NetworkX DiGraph and node ID mapping dictionary
    """
    G = nx.DiGraph()
    
    # Create a mapping from original IDs to sequential positive IDs
    original_ids = set(node[0] for node in nodes)
    for edge in edges:
        original_ids.add(edge[0])
        original_ids.add(edge[1])
    
    id_mapping = {original_id: i for i, original_id in enumerate(sorted(original_ids))}
    
    # Add nodes with attributes and remapped IDs
    for node in nodes:
        original_id, label, node_type, distance, line_num, code = node
        
        # Map the original ID to a new positive ID
        new_id = id_mapping[original_id]
        
        # Map node type to our model's node types
        mapped_type = NODE_TYPE_INFO.get(node_type, 'context')
        is_changed = True if mapped_type in ['vulnerable', 'patched'] else False
        
        # Create node features
        features = create_node_features(code, node_type, 1 if mapped_type == 'vulnerable' else 0)
        
        G.add_node(
            new_id,
            type=mapped_type,
            code=code,
            line=line_num,
            is_changed=is_changed,
            features=features,
            label=label,
            original_id=original_id  # Keep the original ID for reference
        )
    
    # Add edges with attributes and remapped IDs
    for edge in edges:
        src, dst, edge_type, version = edge
        
        # Map original IDs to new positive IDs
        new_src = id_mapping.get(src)
        new_dst = id_mapping.get(dst)
        
        # Only add edge if both nodes exist in the graph
        if new_src is not None and new_dst is not None:
            G.add_edge(
                new_src,
                new_dst,
                type=edge_type,
                version=version
            )
    
    return G, id_mapping

def create_alignment_mapping(pre_nodes, post_nodes, pre_id_mapping, post_id_mapping):
    """
    Create an alignment mapping between pre-patch and post-patch nodes using remapped IDs.
    
    Args:
        pre_nodes: List of pre-patch node tuples
        post_nodes: List of post-patch node tuples
        pre_id_mapping: Mapping from original to new IDs for pre-patch
        post_id_mapping: Mapping from original to new IDs for post-patch
        
    Returns:
        Dictionary mapping pre-patch node IDs to post-patch node IDs using new IDs
    """
    alignment = {}
    
    # Create dictionaries for quick access
    pre_dict = {node[0]: node for node in pre_nodes}
    post_dict = {node[0]: node for node in post_nodes}
    
    # First, map based on identical nodes (context nodes)
    for pre_original_id, pre_node in pre_dict.items():
        pre_label = pre_node[1]
        pre_type = pre_node[2]
        pre_line = pre_node[4]
        pre_code = pre_node[5]
        
        if pre_label == 0 and pre_type in ['C', 'D']:  # Context nodes
            for post_original_id, post_node in post_dict.items():
                post_label = post_node[1]
                post_type = post_node[2]
                post_line = post_node[4]
                post_code = post_node[5]
                
                if (post_label == 0 and post_type in ['C', 'D'] and 
                    pre_line == post_line and pre_code == post_code):
                    # Use the new remapped IDs for alignment
                    pre_new_id = pre_id_mapping.get(pre_original_id)
                    post_new_id = post_id_mapping.get(post_original_id)
                    
                    if pre_new_id is not None and post_new_id is not None:
                        alignment[pre_new_id] = post_new_id
                    break
    
    # For the remaining unaligned nodes, try to map based on line numbers
    # and similarity in the node IDs
    pre_unaligned = set(pre_id_mapping.values()) - set(alignment.keys())
    post_unaligned = set(post_id_mapping.values()) - set(alignment.values())
    
    # Create reverse mappings
    pre_reverse_mapping = {v: k for k, v in pre_id_mapping.items()}
    post_reverse_mapping = {v: k for k, v in post_id_mapping.items()}
    
    for pre_new_id in pre_unaligned:
        pre_original_id = pre_reverse_mapping.get(pre_new_id)
        pre_node = pre_dict.get(pre_original_id)
        
        if pre_node is None:
            continue
            
        pre_line = pre_node[4].replace('-', '')
        
        best_match = None
        best_score = -1
        
        for post_new_id in post_unaligned:
            post_original_id = post_reverse_mapping.get(post_new_id)
            post_node = post_dict.get(post_original_id)
            
            if post_node is None:
                continue
                
            post_line = post_node[4].replace('+', '')
            
            # Skip if line numbers don't match
            if pre_line != post_line:
                continue
            
            # Calculate similarity score
            similar_id = abs(pre_original_id) == abs(post_original_id)
            similar_code = similar_text(pre_node[5], post_node[5])
            score = int(similar_id) + similar_code
            
            if score > best_score:
                best_score = score
                best_match = post_new_id
        
        if best_match is not None:
            alignment[pre_new_id] = best_match
            post_unaligned.remove(best_match)
    
    return alignment

def similar_text(a, b):
    """
    Calculate text similarity as a value between 0 and 1.
    
    Args:
        a: First text
        b: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not a or not b:
        return 0
    
    # Simple character-based similarity
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    
    # Count matching characters
    matches = sum(c1 == c2 for c1, c2 in zip(shorter, longer))
    
    return matches / max(len(a), len(b))

def process_log_file(log_file, output_dir, config):
    """
    Process a log file and save the processed data.
    
    Args:
        log_file: Path to the log file
        output_dir: Directory to save the processed data
        config: Configuration dictionary
        
    Returns:
        Path to the processed data file
    """
    # Parse the log file
    data = parse_log_file(log_file)
    if not data:
        return None
    
    # Extract CVE ID from file path
    cve_id = extract_cve_id(log_file)
    
    # Create NetworkX graphs with remapped node IDs
    pre_graph, pre_id_mapping = create_networkx_graph(data['pre_nodes'], data['pre_edges'])
    post_graph, post_id_mapping = create_networkx_graph(data['post_nodes'], data['post_edges'])
    
    # Create alignment mapping with remapped IDs
    alignment = create_alignment_mapping(data['pre_nodes'], data['post_nodes'], pre_id_mapping, post_id_mapping)
    
    # Initialize graph processor
    graph_processor = GraphProcessor(config)
    
    # Convert NetworkX graphs to HeteroData
    vuln_hetero, patch_hetero = graph_processor.process_graph_pair(pre_graph, post_graph, alignment)
    
    # Set labels
    vuln_hetero['code'].y = torch.tensor([1.0])  # Vulnerable
    patch_hetero['code'].y = torch.tensor([0.0])  # Patched
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data
    output_file = os.path.join(output_dir, f"{cve_id}.pt")
    torch.save({
        'vuln_graph': vuln_hetero,
        'patch_graph': patch_hetero,
        'alignment': alignment
    }, output_file)
    
    return output_file

def extract_cve_id(file_path):
    """
    Extract CVE ID from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        CVE ID or a generated ID
    """
    # Try to extract CVE ID using regex
    match = re.search(r'CVE-\d+-\d+', file_path)
    if match:
        return match.group(0)
    
    # If no CVE ID is found, use the directory name
    dir_name = os.path.basename(os.path.dirname(file_path))
    
    # Clean up the directory name
    dir_name = re.sub(r'[^\w\-]', '_', dir_name)
    
    return dir_name

def split_dataset(processed_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split the processed dataset into training, validation, and test sets.
    
    Args:
        processed_dir: Directory with processed data files
        output_dir: Directory to save the split dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        Paths to the train, val, and test directories
    """
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all processed files
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    # Shuffle files
    np.random.shuffle(processed_files)
    
    # Calculate split indices
    n_files = len(processed_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    # Split the files
    train_files = processed_files[:n_train]
    val_files = processed_files[n_train:n_train+n_val]
    test_files = processed_files[n_train+n_val:]
    
    # Copy files to their respective directories
    for f in train_files:
        shutil.copy(os.path.join(processed_dir, f), os.path.join(train_dir, f))
    
    for f in val_files:
        shutil.copy(os.path.join(processed_dir, f), os.path.join(val_dir, f))
    
    for f in test_files:
        shutil.copy(os.path.join(processed_dir, f), os.path.join(test_dir, f))
    
    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    return train_dir, val_dir, test_dir

def create_raw_data(processed_dir, raw_dir):
    """
    Create raw data format expected by the model from processed data.
    
    Args:
        processed_dir: Directory with processed data files
        raw_dir: Directory to save the raw data
        
    Returns:
        Path to the raw data directory
    """
    # Create raw data directories
    os.makedirs(os.path.join(raw_dir, 'positives'), exist_ok=True)
    os.makedirs(os.path.join(raw_dir, 'negatives'), exist_ok=True)
    
    # Process all files in the processed directory
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for f in os.listdir(split_dir):
            if not f.endswith('.pt'):
                continue
            
            # Load the processed data
            data = torch.load(os.path.join(split_dir, f), weights_only=False)
            vuln_graph = data['vuln_graph']
            patch_graph = data['patch_graph']
            alignment = data['alignment']
            
            # Create raw data filenames
            vuln_file = os.path.join(raw_dir, 'positives', f)
            patch_file = os.path.join(raw_dir, 'negatives', f)
            
            # Save the raw data
            torch.save(vuln_graph, vuln_file)
            torch.save(patch_graph, patch_file)
    
    logger.info(f"Raw data created in {raw_dir}")
    
    return raw_dir

def main(args):
    """
    Main function for data preprocessing.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process dataset
    processed_files = []
    log_files = []
    
    # Collect all log files
    if os.path.isfile(args.input):
        log_files = [args.input]
    else:
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith('.log'):
                    log_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(log_files)} log files")
    
    # Process each log file
    for log_file in tqdm(log_files, desc="Processing log files"):
        processed_file = process_log_file(log_file, args.processed_dir, config)
        if processed_file:
            processed_files.append(processed_file)
    
    logger.info(f"Processed {len(processed_files)} log files")
    
    # Split dataset
    train_dir, val_dir, test_dir = split_dataset(args.processed_dir, args.output_dir)
    
    # Create raw data format
    raw_dir = os.path.join(args.output_dir, 'raw')
    create_raw_data(args.output_dir, raw_dir)
    
    logger.info("Data preprocessing complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess vulnerability dataset')
    parser.add_argument('--input', type=str, required=True, help='Input dataset directory or file')
    parser.add_argument('--processed_dir', type=str, default='data/processed', help='Directory to save processed data')
    parser.add_argument('--output_dir', type=str, default='data/splits', help='Directory to save the split dataset')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    
    args = parser.parse_args()
    
    main(args)