#!/usr/bin/env python3
"""
Debug script to inspect the PatchPairDataset structure.
Run this as a standalone script to examine the dataset format.
"""
import os
import sys
from pathlib import Path
import torch
from collections import Counter

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.dataset import PatchPairDataset

def inspect_dataset(data_dir):
    """Thoroughly inspect the dataset structure and content."""
    pre_patch_path = os.path.join(data_dir, 'pre_patch_graphs.pt')
    post_patch_path = os.path.join(data_dir, 'post_patch_graphs.pt')
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    
    print(f"Loading dataset from {data_dir}")
    print(f"Pre-patch path exists: {os.path.exists(pre_patch_path)}")
    print(f"Post-patch path exists: {os.path.exists(post_patch_path)}")
    print(f"Metadata path exists: {os.path.exists(metadata_path)}")
    
    dataset = PatchPairDataset(pre_patch_path, post_patch_path, metadata_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Inspect the dataset structure
    print("\n=== Dataset Structure ===")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    # For each key, inspect the structure
    for key in sample.keys():
        print(f"\nInspecting '{key}' component:")
        component = sample[key]
        if isinstance(component, dict):
            print(f"  Type: dict with keys {list(component.keys())}")
            for k, v in component.items():
                print(f"    {k}: {type(v)}")
                if hasattr(v, 'shape'):
                    print(f"      Shape: {v.shape}")
                elif hasattr(v, 'size'):
                    print(f"      Size: {v.size()}")
        else:
            print(f"  Type: {type(component)}")
            if hasattr(component, 'keys'):
                print(f"  Available attributes: {list(component.keys())}")
            for attr_name in dir(component):
                if not attr_name.startswith('_') and not callable(getattr(component, attr_name)):
                    attr = getattr(component, attr_name)
                    print(f"  {attr_name}: {type(attr)}")
                    if hasattr(attr, 'shape'):
                        print(f"    Shape: {attr.shape}")
                    elif hasattr(attr, 'size') and callable(attr.size):
                        try:
                            print(f"    Size: {attr.size()}")
                        except:
                            print(f"    Size: [Error retrieving size]")
    
    # Check label distribution
    labels = []
    for i in range(len(dataset)):
        try:
            label = dataset[i]['vuln'].y.item()
            labels.append(label)
        except (KeyError, AttributeError) as e:
            print(f"Error accessing label for sample {i}: {e}")
            # Try alternative ways to access the label
            if 'y' in dataset[i]:
                print(f"Sample {i} has 'y' at top level: {dataset[i]['y']}")
            else:
                print(f"Cannot find label for sample {i}")
    
    # Calculate distribution
    label_counter = Counter(labels)
    print("\n=== Label Distribution ===")
    for label, count in label_counter.items():
        print(f"Label {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    # Look for potential problems
    print("\n=== Potential Problems ===")
    if len(label_counter) <= 1:
        print("WARNING: All samples have the same label!")
    
    # Check for identical samples
    print("Checking for identical pre/post patch pairs...")
    identical_count = 0
    for i in range(min(100, len(dataset))):  # Check first 100 samples to save time
        sample = dataset[i]
        if 'pre_patch' in sample and 'post_patch' in sample:
            pre = sample['pre_patch']
            post = sample['post_patch']
            
            # Compare basic properties
            are_identical = True
            for attr in ['edge_index', 'edge_type', 'x', 'num_nodes']:
                if hasattr(pre, attr) and hasattr(post, attr):
                    pre_attr = getattr(pre, attr)
                    post_attr = getattr(post, attr)
                    
                    if isinstance(pre_attr, torch.Tensor) and isinstance(post_attr, torch.Tensor):
                        if not torch.equal(pre_attr, post_attr):
                            are_identical = False
                            break
            
            if are_identical:
                identical_count += 1
                if identical_count <= 5:  # Print details for first 5 identical pairs
                    print(f"Sample {i}: Pre-patch and post-patch appear identical")
    
    if identical_count > 0:
        print(f"Found {identical_count} samples with identical pre/post patches out of 100 checked")
        if identical_count > 50:
            print("WARNING: Most samples have identical pre/post patches. This likely indicates a data issue!")
    
    return dataset

if __name__ == "__main__":
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else './processed_data'
    inspect_dataset(data_dir)