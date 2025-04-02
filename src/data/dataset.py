# src/data/dataset.py
import os
import torch
from torch_geometric.data import Dataset, Data, HeteroData

class PatchPairDataset(Dataset):
    """
    Dataset for patch pairs, suitable for PatchPairVul model.
    """
    
    def __init__(self, pre_patch_path, post_patch_path, metadata_path=None, 
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize the dataset.
        
        Args:
            pre_patch_path (str): Path to pre-patch graph file
            post_patch_path (str): Path to post-patch graph file
            metadata_path (str, optional): Path to metadata file
            transform (callable, optional): Transform to be applied on each data object
            pre_transform (callable, optional): Transform to be applied on each data object before saving
            pre_filter (callable, optional): Function to filter out unwanted data objects
        """
        self.pre_patch_path = pre_patch_path
        self.post_patch_path = post_patch_path
        self.metadata_path = metadata_path
        
        super(PatchPairDataset, self).__init__(None, transform, pre_transform, pre_filter)
        
        # Load data
        self.pre_patch_graphs = torch.load(pre_patch_path, weights_only=False)
        self.post_patch_graphs = torch.load(post_patch_path, weights_only=False)
        
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path, weights_only=False)
        else:
            self.metadata = [{} for _ in range(len(self.pre_patch_graphs))]
    
    def len(self):
        """Number of examples in the dataset."""
        return len(self.pre_patch_graphs)
    
    def get(self, idx):
        """Get a single data object."""
        pre_patch_graph = self.pre_patch_graphs[idx]
        post_patch_graph = self.post_patch_graphs[idx]
        
        # Create a HeteroData object for the pair
        data = HeteroData()
        
        # Add pre-patch graph data
        data['vuln'].x = pre_patch_graph.x
        data['vuln'].y = pre_patch_graph.y
        
        # Add post-patch graph data
        data['patch'].x = post_patch_graph.x
        data['patch'].y = post_patch_graph.y
        
        # Add edges for each edge type
        for edge_type in pre_patch_graph.edge_index_dict:
            edge_index = pre_patch_graph.edge_index_dict[edge_type]
            data['vuln', edge_type, 'vuln'].edge_index = edge_index
        
        for edge_type in post_patch_graph.edge_index_dict:
            edge_index = post_patch_graph.edge_index_dict[edge_type]
            data['patch', edge_type, 'patch'].edge_index = edge_index
        
        # Add metadata
        data.metadata = self.metadata[idx]
        
        # Apply transform if specified
        if self.transform is not None:
            data = self.transform(data)
        
        return data