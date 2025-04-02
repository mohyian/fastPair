# src/data/utils.py
import torch
import numpy as np
import re

def extract_node_features(node, changed_lines):
    """
    Extract features from a node, excluding direct vulnerability indicators.
    
    Args:
        node (dict): Node dictionary with attributes
        changed_lines (dict): Dictionary of changed line numbers
        
    Returns:
        list: Feature vector
    """
    features = []
    
    # Node type features
    node_type = node.get('TYPE_FULL_NAME', '')
    is_function = 1.0 if 'METHOD' in node or 'FUNCTION' in node_type else 0.0
    is_param = 1.0 if 'PARAM' in node or 'PARAMETER' in node_type else 0.0
    is_block = 1.0 if 'BLOCK' in node else 0.0
    is_return = 1.0 if 'RETURN' in node else 0.0
    is_call = 1.0 if 'CALL' in node_type else 0.0
    
    # Code structure features
    has_code = 1.0 if 'CODE' in node and node['CODE'] else 0.0
    code_length = len(node.get('CODE', '')) / 100.0  # Normalize
    
    # Line information
    line_num = node.get('LINE_NUMBER', '0')
    if not isinstance(line_num, (int, float)):
        line_num = int(re.findall(r'\d+', str(line_num))[0]) if re.findall(r'\d+', str(line_num)) else 0
    
    # Check if the node is in a changed line (important for vulnerability detection)
    # Note: We keep this information since it's derived from the diff, not directly labeled
    is_changed_line = 0.0
    filename = node.get('FILENAME', '')
    if filename and filename in changed_lines:
        if line_num in changed_lines[filename]:
            is_changed_line = 1.0
    
    # AST features
    ast_parent_type = 1.0 if 'AST_PARENT_TYPE' in node else 0.0
    
    # Combine features
    features.extend([
        is_function, is_param, is_block, is_return, is_call,
        has_code, code_length, line_num / 1000.0,  # Normalize line number
        is_changed_line, ast_parent_type
    ])
    
    # Control flow indicators
    is_if = 1.0 if 'if ' in node.get('CODE', '').lower() else 0.0
    is_loop = 1.0 if any(kw in node.get('CODE', '').lower() for kw in ['for ', 'while ']) else 0.0
    is_switch = 1.0 if 'switch' in node.get('CODE', '').lower() else 0.0
    
    features.extend([is_if, is_loop, is_switch])
    
    # Data flow indicators
    has_assignment = 1.0 if '=' in node.get('CODE', '') else 0.0
    has_arithmetic = 1.0 if any(op in node.get('CODE', '') for op in ['+', '-', '*', '/', '%']) else 0.0
    has_comparison = 1.0 if any(op in node.get('CODE', '') for op in ['==', '!=', '>', '<', '>=', '<=']) else 0.0
    
    features.extend([has_assignment, has_arithmetic, has_comparison])
    
    # Common vulnerability indicators (from the code, not direct labels)
    has_ptr = 1.0 if '*' in node.get('CODE', '') else 0.0
    has_array = 1.0 if '[' in node.get('CODE', '') and ']' in node.get('CODE', '') else 0.0
    has_malloc = 1.0 if 'malloc' in node.get('CODE', '').lower() else 0.0
    has_free = 1.0 if 'free' in node.get('CODE', '').lower() else 0.0
    
    features.extend([has_ptr, has_array, has_malloc, has_free])
    
    return features

def parse_edge_types(edges):
    """
    Parse edge types from the edges.
    
    Args:
        edges (list): List of edge dictionaries
        
    Returns:
        set: Set of unique edge types
    """
    return set(edge['type'] for edge in edges if 'type' in edge)