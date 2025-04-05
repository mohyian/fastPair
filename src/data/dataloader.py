"""
DataLoader Module

This module provides data loading and batch creation functionality for training
and evaluating the vulnerability detection model. It handles loading graphs from
disk, creating batches, and splitting data into training/validation/test sets.
"""

import os
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils.logger import get_logger
from src.data.hetero_graph import HeteroGraphBuilder

logger = get_logger(__name__)

class VulnerabilityPairDataset(InMemoryDataset):
    """
    Dataset for vulnerability-patch pairs represented as heterogeneous graphs.
    """
    
    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory where the processed data will be stored
            transform: Transform to apply at loading time
            pre_transform: Transform to apply at processing time
            pre_filter: Filter to apply at processing time
            config: Configuration dictionary
        """
        self.config = config or {}
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self) -> List[str]:
        """
        List of raw file names.
        """
        return ['vulnerable_graphs.pt', 'patched_graphs.pt', 'alignments.pt']
    
    @property
    def processed_file_names(self) -> List[str]:
        """
        List of processed file names.
        """
        return ['vulnerability_pairs.pt']
    
    def process(self):
        """
        Process the raw data into the format required for training.
        """
        # Load raw data
        try:
            vulnerable_graphs = torch.load(os.path.join(self.raw_dir, 'vulnerable_graphs.pt'), weights_only=False)
            patched_graphs = torch.load(os.path.join(self.raw_dir, 'patched_graphs.pt'), weights_only=False)
            alignments = torch.load(os.path.join(self.raw_dir, 'alignments.pt'), weights_only=False)
            
            logger.info(f"Loaded {len(vulnerable_graphs)} vulnerable graphs and "
                        f"{len(patched_graphs)} patched graphs")
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise
        
        # Check that the number of graphs match
        if len(vulnerable_graphs) != len(patched_graphs) or len(vulnerable_graphs) != len(alignments):
            raise ValueError(f"Mismatch in number of graphs: {len(vulnerable_graphs)} vulnerable, "
                            f"{len(patched_graphs)} patched, {len(alignments)} alignments")
        
        # Create graph builder
        graph_builder = HeteroGraphBuilder(self.config)
        
        # Create combined vulnerability-patch pairs
        data_list = []
        for i in range(len(vulnerable_graphs)):
            vuln_graph = vulnerable_graphs[i]
            patch_graph = patched_graphs[i]
            alignment = alignments[i]
            
            # Add alignment information to the graphs
            vuln_graph.alignment = alignment
            patch_graph.alignment = {v: k for k, v in alignment.items()}  # Reverse mapping
            
            # Build combined graph
            combined_graph = graph_builder.build_vuln_patch_graph_pair(vuln_graph, patch_graph)
            
            # Apply pre-transform if defined
            if self.pre_transform is not None:
                combined_graph = self.pre_transform(combined_graph)
            
            # Apply pre-filter if defined
            if self.pre_filter is not None:
                if not self.pre_filter(combined_graph):
                    continue
            
            data_list.append(combined_graph)
        
        logger.info(f"Created {len(data_list)} combined vulnerability-patch pairs")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def get(self, idx: int) -> HeteroData:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            HeteroData object representing a vulnerability-patch pair
        """
        data = super().get(idx)
        return data

class VulnerabilityDataLoader:
    """
    Data loader for vulnerability detection that handles data splitting,
    batch creation, and provides iterators for training, validation, and testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processed_dir = config["data"]["processed_dir"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]
        self.train_ratio = config["data"]["train_ratio"]
        self.val_ratio = config["data"]["val_ratio"]
        self.test_ratio = config["data"]["test_ratio"]
        
        # Check that ratios add up to 1
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            logger.warning("Train, validation, and test ratios do not sum to 1. Normalizing.")
            total = self.train_ratio + self.val_ratio + self.test_ratio
            self.train_ratio /= total
            self.val_ratio /= total
            self.test_ratio /= total
    
    def load_dataset(self) -> VulnerabilityPairDataset:
        """
        Load the vulnerability pair dataset.
        
        Returns:
            VulnerabilityPairDataset
        """
        dataset = VulnerabilityPairDataset(
            root=self.processed_dir,
            config=self.config
        )
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        return dataset
    
    def create_data_loaders(self) -> Tuple[PyGDataLoader, PyGDataLoader, PyGDataLoader]:
        """
        Create data loaders for training, validation, and testing.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        dataset = self.load_dataset()
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self._split_dataset(dataset)
        
        # Create data loaders
        train_loader = PyGDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = PyGDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = PyGDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        logger.info(f"Created data loaders: {len(train_loader)} training batches, "
                   f"{len(val_loader)} validation batches, {len(test_loader)} test batches")
        
        return train_loader, val_loader, test_loader
    
    def _split_dataset(
        self, 
        dataset: VulnerabilityPairDataset
    ) -> Tuple[VulnerabilityPairDataset, VulnerabilityPairDataset, VulnerabilityPairDataset]:
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            dataset: Full dataset
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Calculate split indices
        n = len(dataset)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        # Create a permutation for random splitting
        indices = torch.randperm(n).tolist()
        
        # Split the dataset
        from torch_geometric.data import Subset
        train_dataset = Subset(dataset, indices[:train_end])
        val_dataset = Subset(dataset, indices[train_end:val_end])
        test_dataset = Subset(dataset, indices[val_end:])
        
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation, "
                   f"{len(test_dataset)} test samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_class_weights(self, dataset: VulnerabilityPairDataset) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced data.
        
        Args:
            dataset: Dataset to calculate weights for
            
        Returns:
            Tensor of class weights
        """
        # Count number of samples in each class
        class_counts = torch.zeros(2)  # Binary classification: vulnerable or not
        
        for i in range(len(dataset)):
            data = dataset[i]
            label = data["vuln"].y.item()  # Assuming binary classification
            class_counts[int(label)] += 1
        
        # Calculate weights: higher weight for less frequent class
        weights = 1.0 / class_counts
        
        # Normalize weights
        weights = weights / weights.sum() * 2.0
        
        logger.info(f"Class weights: {weights}")
        
        return weights