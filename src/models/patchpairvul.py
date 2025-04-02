# src/models/patchpairvul.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class PatchPairVul(torch.nn.Module):
    """
    Graph Neural Network model for vulnerability detection in patch pairs.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.5,
                 edge_types=None, use_attention=True):
        """
        Initialize the model.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            num_layers (int): Number of GNN layers
            dropout (float): Dropout probability
            edge_types (list): List of edge types
            use_attention (bool): Whether to use attention or not
        """
        super(PatchPairVul, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_types = edge_types or ["AST", "DDG", "CFG"]
        self.use_attention = use_attention
        
        # Input projection for each node type
        self.input_proj = nn.ModuleDict({
            'vuln': nn.Linear(input_dim, hidden_dim),
            'patch': nn.Linear(input_dim, hidden_dim)
        })
        
        # GNN layers
        self.convs = nn.ModuleList()
        
        for _ in range(num_layers):
            # Create heterogeneous convolution layer
            conv_dict = {}
            
            # Add convolution for each node type and edge type
            for node_type in ['vuln', 'patch']:
                for edge_type in self.edge_types:
                    edge_tuple = (node_type, edge_type, node_type)
                    
                    if use_attention:
                        # Graph Attention Network
                        conv_dict[edge_tuple] = GATConv(
                            hidden_dim, hidden_dim // 8, heads=8, dropout=dropout
                        )
                    else:
                        # GraphSAGE Network
                        conv_dict[edge_tuple] = SAGEConv(
                            hidden_dim, hidden_dim
                        )
            
            # Add heterogeneous convolution to list
            hetero_conv = HeteroConv(conv_dict, aggr='mean')
            self.convs.append(hetero_conv)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.ModuleDict({
                'vuln': nn.BatchNorm1d(hidden_dim),
                'patch': nn.BatchNorm1d(hidden_dim)
            }))
        
        # Output layers
        self.graph_proj = nn.Linear(hidden_dim * 4, hidden_dim)  # 4 = 2 (max+mean) * 2 (vuln+patch)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric HeteroData object
            
        Returns:
            Tensor: Binary classification output (0-1)
        """
        # Initialize node embeddings
        x_dict = {}
        for node_type in ['vuln', 'patch']:
            if node_type in data.node_types:
                x_dict[node_type] = self.input_proj[node_type](data[node_type].x)
        
        # Get edge indices for each edge type
        edge_index_dict = {}
        for node_type in ['vuln', 'patch']:
            if node_type in data.node_types:
                for edge_type in self.edge_types:
                    key = (node_type, edge_type, node_type)
                    if key in data.edge_types:
                        edge_index_dict[key] = data[key].edge_index
        
        # Apply GNN layers
        for i in range(self.num_layers):
            # Get new embeddings
            x_dict_new = self.convs[i](x_dict, edge_index_dict)
            
            # Apply batch normalization
            for node_type in x_dict_new.keys():
                if node_type in self.batch_norms[i]:
                    x_dict_new[node_type] = self.batch_norms[i][node_type](x_dict_new[node_type])
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = F.dropout(x_dict_new[node_type], p=self.dropout, training=self.training)
            
            # Residual connection
            for node_type in x_dict.keys():
                if node_type in x_dict_new:
                    x_dict_new[node_type] = x_dict_new[node_type] + x_dict[node_type]
            
            # Update embeddings
            x_dict = x_dict_new
        
        # Global pooling
        vuln_graph_emb = None
        patch_graph_emb = None
        
        if 'vuln' in x_dict:
            vuln_batch = data['vuln'].batch if hasattr(data['vuln'], 'batch') else None
            if vuln_batch is None:
                vuln_batch = torch.zeros(data['vuln'].x.size(0), dtype=torch.long, device=data['vuln'].x.device)
            vuln_mean = global_mean_pool(x_dict['vuln'], vuln_batch)
            vuln_max = global_max_pool(x_dict['vuln'], vuln_batch)
            vuln_graph_emb = torch.cat([vuln_mean, vuln_max], dim=1)
        
        if 'patch' in x_dict:
            patch_batch = data['patch'].batch if hasattr(data['patch'], 'batch') else None
            if patch_batch is None:
                patch_batch = torch.zeros(data['patch'].x.size(0), dtype=torch.long, device=data['patch'].x.device)
            patch_mean = global_mean_pool(x_dict['patch'], patch_batch)
            patch_max = global_max_pool(x_dict['patch'], patch_batch)
            patch_graph_emb = torch.cat([patch_mean, patch_max], dim=1)
        
        # Handle cases where one of the node types is missing
        if vuln_graph_emb is None:
            vuln_graph_emb = torch.zeros_like(patch_graph_emb)
        
        if patch_graph_emb is None:
            patch_graph_emb = torch.zeros_like(vuln_graph_emb)
        
        # Combine embeddings
        graph_emb = torch.cat([vuln_graph_emb, patch_graph_emb], dim=1)
        
        # Project and classify
        graph_emb = self.graph_proj(graph_emb)
        graph_emb = F.relu(graph_emb)
        graph_emb = F.dropout(graph_emb, p=self.dropout, training=self.training)
        
        out = self.classifier(graph_emb)
        out = torch.sigmoid(out)
        
        return out