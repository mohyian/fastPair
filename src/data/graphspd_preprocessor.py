# src/data/graphspd_preprocessor.py
import os
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, HeteroData
from collections import defaultdict

class GraphSPDPreprocessor:
    """
    Preprocessor for GraphSPD data to convert it to a format compatible with PatchPairVul.
    """
    
    def __init__(self, root_dir, output_dir):
        """
        Initialize the preprocessor.
        
        Args:
            root_dir (str): Root directory containing GraphSPD data
            output_dir (str): Directory to save processed data
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.edge_types = ["AST", "DDG", "CFG"]  # Using edge types from GraphSPD
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def process_all(self):
        """Process all CVE directories in the root directory."""
        # Find all subdirectories
        cve_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        
        processed_data = []
        skipped = 0
        
        for cve_dir in tqdm(cve_dirs, desc="Processing CVE directories"):
            cve_path = os.path.join(self.root_dir, cve_dir)
            
            try:
                # Process the CVE directory
                graph_pair = self.process_directory(cve_path)
                if graph_pair:
                    processed_data.append(graph_pair)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error processing {cve_dir}: {e}")
                skipped += 1
        
        print(f"Processed {len(processed_data)} directories, skipped {skipped} directories")
        
        # Save processed data
        self._save_processed_data(processed_data)
        
        return processed_data
    
    def process_directory(self, cve_path):
        """
        Process a single CVE directory.
        
        Args:
            cve_path (str): Path to the CVE directory
            
        Returns:
            tuple: Tuple of (pre-patch graph, post-patch graph, metadata)
        """
        # Check if required files exist
        required_files = ['cpg_a.txt', 'cpg_b.txt', 'diff.txt']
        required_dirs = ['outA', 'outB', 'a', 'b']
        
        for f in required_files:
            if not os.path.exists(os.path.join(cve_path, f)):
                print(f"Missing required file: {f} in {cve_path}")
                return None
        
        for d in required_dirs:
            if not os.path.exists(os.path.join(cve_path, d)):
                print(f"Missing required directory: {d} in {cve_path}")
                return None
        
        # Parse the diff file to identify changed lines
        changed_lines = self._parse_diff_file(os.path.join(cve_path, 'diff.txt'))
        
        # Process pre-patch (A) and post-patch (B) graphs
        pre_graph = self._process_cpg(os.path.join(cve_path, 'outA'), changed_lines, is_vulnerable=True)
        post_graph = self._process_cpg(os.path.join(cve_path, 'outB'), changed_lines, is_vulnerable=False)
        
        if pre_graph is None or post_graph is None:
            return None
        
        # Extract metadata
        metadata = {
            'cve_dir': os.path.basename(cve_path),
            'changed_lines': len(changed_lines)
        }
        
        return (pre_graph, post_graph, metadata)
    
    def _parse_diff_file(self, diff_path):
        """
        Parse the diff file to identify changed lines.
        
        Args:
            diff_path (str): Path to the diff file
            
        Returns:
            dict: Mapping of filenames to changed line numbers
        """
        changed_lines = {}
        
        with open(diff_path, 'r', errors='ignore') as f:
            content = f.read()
        
        # Split by diff headers
        diff_sections = re.split(r'diff -brN', content)
        
        for section in diff_sections:
            # Skip empty sections
            if not section.strip():
                continue
            
            # Extract filenames
            file_match = re.search(r'(\S+)\s+(\S+)\s*$', section.split('\n')[0], re.MULTILINE)
            if file_match:
                file_a = file_match.group(1)
                file_b = file_match.group(2)
                
                # Extract line numbers from hunks
                hunk_matches = re.finditer(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', section)
                
                for hunk in hunk_matches:
                    a_start = int(hunk.group(1))
                    a_count = int(hunk.group(2)) if hunk.group(2) else 1
                    b_start = int(hunk.group(3))
                    b_count = int(hunk.group(4)) if hunk.group(4) else 1
                    
                    # Add changed lines
                    if file_a not in changed_lines:
                        changed_lines[file_a] = set()
                    if file_b not in changed_lines:
                        changed_lines[file_b] = set()
                    
                    for i in range(a_count):
                        changed_lines[file_a].add(a_start + i)
                    
                    for i in range(b_count):
                        changed_lines[file_b].add(b_start + i)
        
        return changed_lines
    
    def _process_cpg(self, cpg_dir, changed_lines, is_vulnerable):
        """
        Process code property graphs from a directory.
        
        Args:
            cpg_dir (str): Directory containing CPG files
            changed_lines (dict): Dictionary of changed line numbers
            is_vulnerable (bool): Whether this is a vulnerable version
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        all_nodes = []
        all_edges = []
        
        # Find all CPG dot files
        dot_files = [f for f in os.listdir(cpg_dir) if f.endswith('-cpg.dot') or f.endswith('.dot')]
        
        if not dot_files:
            print(f"No CPG files found in {cpg_dir}")
            return None
        
        # Process each CPG file
        for dot_file in dot_files:
            file_path = os.path.join(cpg_dir, dot_file)
            
            try:
                # Read the file content
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read()
                
                # Split into sections
                sections = content.split('--------')
                
                if len(sections) < 3:
                    continue
                
                # Extract graph name
                graph_name = sections[0].strip().replace('digraph ', '').strip()
                
                # Extract nodes
                node_section = sections[1].strip()
                nodes = []
                
                for line in node_section.split('\n'):
                    if not line.strip():
                        continue
                    
                    # Extract node ID and attributes
                    match = re.match(r'(\d+),\((.*)\)', line)
                    if match:
                        node_id = int(match.group(1))
                        attrs_str = match.group(2)
                        
                        # Parse attributes
                        attrs = {}
                        attrs['id'] = node_id
                        
                        # Extract attribute pairs
                        for attr_pair in re.finditer(r'([^,()]+),([^,()]+)', attrs_str):
                            key = attr_pair.group(1).strip()
                            value = attr_pair.group(2).strip()
                            
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            
                            attrs[key] = value
                        
                        nodes.append(attrs)
                
                # Extract edges
                edge_section = sections[2].strip()
                edges = []
                
                for line in edge_section.split('\n'):
                    if not line.strip():
                        continue
                    
                    # Extract source, target, and edge type
                    match = re.match(r'(\d+),(\d+),(\w+)', line)
                    if match:
                        source = int(match.group(1))
                        target = int(match.group(2))
                        edge_type = match.group(3)
                        
                        edges.append({
                            'source': source,
                            'target': target,
                            'type': edge_type
                        })
                
                # Add to collections
                all_nodes.extend(nodes)
                all_edges.extend(edges)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Create the graph if we have nodes and edges
        if all_nodes and all_edges:
            return self._create_graph(all_nodes, all_edges, changed_lines, is_vulnerable)
        
        return None
    
    def _create_graph(self, nodes, edges, changed_lines, is_vulnerable):
        """
        Create a PyTorch Geometric Data object from nodes and edges.
        
        Args:
            nodes (list): List of node dictionaries
            edges (list): List of edge dictionaries
            changed_lines (dict): Dictionary of changed line numbers
            is_vulnerable (bool): Whether this graph represents vulnerable code
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        if not nodes:
            return None
        
        # Create a mapping from original node IDs to new consecutive indices
        node_mapping = {}
        for i, node in enumerate(nodes):
            node_id = node['id']
            node_mapping[node_id] = i
        
        # Create node features
        node_features = []
        for node in nodes:
            features = self._extract_node_features(node, changed_lines)
            node_features.append(features)
        
        node_features_tensor = torch.tensor(node_features, dtype=torch.float)
        
        # Create edge indices for each edge type
        edge_index_dict = {}
        
        for edge_type in self.edge_types:
            type_edges = [e for e in edges if e['type'] == edge_type]
            
            if type_edges:
                src_indices = []
                dst_indices = []
                
                for edge in type_edges:
                    src_id = edge['source']
                    dst_id = edge['target']
                    
                    # Only include edges if both endpoints are in the node mapping
                    if src_id in node_mapping and dst_id in node_mapping:
                        src_indices.append(node_mapping[src_id])
                        dst_indices.append(node_mapping[dst_id])
                
                if src_indices and dst_indices:  # Only create edge index if we have edges
                    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                    edge_index_dict[edge_type] = edge_index
        
        # Create label (1 for vulnerable, 0 for patched)
        y = torch.tensor([1 if is_vulnerable else 0], dtype=torch.float)
        
        # Create graph data
        data = Data(
            x=node_features_tensor,
            y=y,
            edge_index_dict=edge_index_dict,
            num_nodes=len(nodes)
        )
        
        return data
    
    def _extract_node_features(self, node, changed_lines):
        """
        Extract features from a node.
        
        Args:
            node (dict): Node dictionary
            changed_lines (dict): Dictionary of changed line numbers
            
        Returns:
            list: Feature vector
        """
        features = []
        
        # Basic node information
        node_type = node.get('TYPE_FULL_NAME', '')
        
        # Node type features (one-hot encoded)
        is_function = 1.0 if 'METHOD' in node or 'FUNCTION' in node_type.upper() else 0.0
        is_variable = 1.0 if 'VAR' in node_type.upper() or 'IDENTIFIER' in node else 0.0
        is_parameter = 1.0 if 'PARAM' in node or 'PARAMETER' in node_type.upper() else 0.0
        is_literal = 1.0 if 'LITERAL' in node else 0.0
        is_return = 1.0 if 'RETURN' in node or 'RET' in node else 0.0
        is_call = 1.0 if 'CALL' in node or 'DISPATCH' in node else 0.0
        is_condition = 1.0 if 'IF' in node or 'CONDITION' in node else 0.0
        is_loop = 1.0 if 'LOOP' in node or 'FOR' in node or 'WHILE' in node else 0.0
        
        # Source code features
        code = node.get('CODE', '')
        has_code = 1.0 if code else 0.0
        code_length = min(len(code) / 100.0, 1.0)  # Normalized and capped
        
        # Control flow indicators
        has_condition = 1.0 if 'if' in code.lower() or 'switch' in code.lower() else 0.0
        has_loop = 1.0 if 'for' in code.lower() or 'while' in code.lower() else 0.0
        has_return = 1.0 if 'return' in code.lower() else 0.0
        
        # Data flow indicators
        has_assignment = 1.0 if '=' in code and '==' not in code else 0.0
        has_arithmetic = 1.0 if any(op in code for op in ['+', '-', '*', '/', '%']) else 0.0
        has_comparison = 1.0 if any(op in code for op in ['==', '!=', '>', '<', '>=', '<=']) else 0.0
        
        # Memory operation indicators (common vulnerability sources)
        has_pointer = 1.0 if '*' in code or '->' in code else 0.0
        has_array = 1.0 if '[' in code and ']' in code else 0.0
        has_memory_alloc = 1.0 if any(op in code.lower() for op in ['malloc', 'calloc', 'realloc', 'new']) else 0.0
        has_memory_free = 1.0 if any(op in code.lower() for op in ['free', 'delete']) else 0.0
        
        # Combine basic features
        basic_features = [
            is_function, is_variable, is_parameter, is_literal, is_return, is_call, is_condition, is_loop,
            has_code, code_length, has_condition, has_loop, has_return, 
            has_assignment, has_arithmetic, has_comparison,
            has_pointer, has_array, has_memory_alloc, has_memory_free
        ]
        
        features.extend(basic_features)
        
        return features
    
    def _save_processed_data(self, processed_data):
        """
        Save processed data to output directory.
        
        Args:
            processed_data (list): List of (pre_graph, post_graph, metadata) tuples
        """
        pre_patch_graphs = []
        post_patch_graphs = []
        metadata_list = []
        
        for data_tuple in processed_data:
            if data_tuple is None:
                continue
                
            pre_graph, post_graph, metadata = data_tuple
            
            if pre_graph is not None and post_graph is not None:
                pre_patch_graphs.append(pre_graph)
                post_patch_graphs.append(post_graph)
                metadata_list.append(metadata)
        
        # Save to files
        torch.save(pre_patch_graphs, os.path.join(self.output_dir, 'pre_patch_graphs.pt'))
        torch.save(post_patch_graphs, os.path.join(self.output_dir, 'post_patch_graphs.pt'))
        torch.save(metadata_list, os.path.join(self.output_dir, 'metadata.pt'))
        
        print(f"Saved {len(pre_patch_graphs)} graph pairs to {self.output_dir}")