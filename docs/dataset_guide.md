# Dataset Integration Guide

This guide explains how to feed your existing dataset into the vulnerability detection system, including the handling of pre-patch and post-patch graph pairs, alignment mappings, and implementation details for researchers and developers.

## Table of Contents

1. [Dataset Format and Requirements](#dataset-format-and-requirements)
2. [Alignment Mapping Explained](#alignment-mapping-explained)
3. [Working with Existing Datasets](#working-with-existing-datasets)
4. [Data Processing Internals](#data-processing-internals)
5. [Research Extensions](#research-extensions)

## Dataset Format and Requirements

### Expected Format

The system expects three main components:

1. **Pre-patch (Vulnerable) Graphs**: Code graphs representing vulnerable code
2. **Post-patch (Fixed) Graphs**: Code graphs representing fixed/patched code
3. **Alignment Mappings**: Mappings between nodes in the pre-patch and post-patch graphs

### File Structure

The dataset should be structured as follows:

```
data/
├── raw/
│   ├── vulnerable_graphs.pt    # Pre-patch graphs
│   ├── patched_graphs.pt       # Post-patch graphs
│   └── alignments.pt           # Alignment mappings
└── processed/                  # Generated during preprocessing
```

### File Formats

Each file is a PyTorch serialized object:

- `vulnerable_graphs.pt`: A list of NetworkX `DiGraph` objects or PyTorch Geometric `HeteroData` objects
- `patched_graphs.pt`: A list of NetworkX `DiGraph` objects or PyTorch Geometric `HeteroData` objects
- `alignments.pt`: A list of dictionaries, each mapping pre-patch node IDs to post-patch node IDs

Example of loading these files:

```python
import torch
import networkx as nx

# Load the data files
vulnerable_graphs = torch.load('data/raw/vulnerable_graphs.pt')
patched_graphs = torch.load('data/raw/patched_graphs.pt')
alignments = torch.load('data/raw/alignments.pt')

# Verify the format
assert isinstance(vulnerable_graphs, list)
assert isinstance(patched_graphs, list)
assert isinstance(alignments, list)
assert len(vulnerable_graphs) == len(patched_graphs) == len(alignments)

# Check the first graph and alignment
vuln_graph = vulnerable_graphs[0]
patch_graph = patched_graphs[0]
alignment = alignments[0]

assert isinstance(vuln_graph, nx.DiGraph) or hasattr(vuln_graph, 'edge_index')
assert isinstance(patch_graph, nx.DiGraph) or hasattr(patch_graph, 'edge_index')
assert isinstance(alignment, dict)
```

## Alignment Mapping Explained

### What is an Alignment Mapping?

An alignment mapping is a dictionary that maps node IDs from the pre-patch graph to corresponding node IDs in the post-patch graph. This mapping identifies which nodes represent the same code elements before and after the patch.

Example alignment mapping:
```python
alignment = {
    0: 0,    # Node 0 in pre-patch maps to node 0 in post-patch
    1: 1,    # Node 1 in pre-patch maps to node 1 in post-patch
    2: 2,    # Node 2 in pre-patch maps to node 2 in post-patch
    # Node 3 in pre-patch has no mapping (deleted in the patch)
    4: 3,    # Node 4 in pre-patch maps to node 3 in post-patch
    # Node 4 in post-patch has no mapping (added in the patch)
}
```

### Why Are Alignments Important?

Alignments are crucial for the pattern learning component of the system. They allow the model to:

1. **Learn Transformation Patterns**: Understand how vulnerable code transforms into secure code
2. **Focus on Relevant Changes**: Identify which parts of the code are modified in the patch
3. **Create Contrastive Examples**: Generate positive and negative examples for contrastive learning

### Creating Alignment from Changed Lines

If your dataset has pre-patch and post-patch graphs with attributes for changed lines, you can generate alignment mappings automatically. Here's a pseudocode example:

```python
def create_alignment_from_changed_lines(pre_patch_graph, post_patch_graph):
    alignment = {}
    
    # Get nodes with line information
    pre_nodes = {node: data.get('line_number') for node, data in pre_patch_graph.nodes(data=True) 
                if 'line_number' in data and not data.get('is_changed', False)}
    post_nodes = {node: data.get('line_number') for node, data in post_patch_graph.nodes(data=True) 
                if 'line_number' in data and not data.get('is_changed', False)}
    
    # Create reverse mapping from line numbers to node IDs
    pre_lines_to_nodes = {}
    for node, line in pre_nodes.items():
        if line not in pre_lines_to_nodes:
            pre_lines_to_nodes[line] = []
        pre_lines_to_nodes[line].append(node)
    
    post_lines_to_nodes = {}
    for node, line in post_nodes.items():
        if line not in post_lines_to_nodes:
            post_lines_to_nodes[line] = []
        post_lines_to_nodes[line].append(node)
    
    # Create alignment based on matching line numbers
    for line, pre_line_nodes in pre_lines_to_nodes.items():
        if line in post_lines_to_nodes:
            post_line_nodes = post_lines_to_nodes[line]
            
            # Simple case: one-to-one mapping
            if len(pre_line_nodes) == 1 and len(post_line_nodes) == 1:
                alignment[pre_line_nodes[0]] = post_line_nodes[0]
            # More complex cases can be handled with node type, code text, etc.
    
    return alignment
```

This is a simplified approach. In practice, you would want to:

1. Use node types, AST structure, or code text to improve matching
2. Handle line shifts caused by insertions/deletions
3. Consider code similarity metrics for ambiguous cases

### Is an Explicit Alignment Mapping Required?

**Short answer: Yes, the alignment mapping is important for optimal performance.**

While the system can technically run without alignment mappings (by treating each graph independently), the alignment is crucial for:

1. The pattern learning component to function effectively
2. Learning the relationship between vulnerable and patched code
3. Generating meaningful transformation rules

Without alignments, the model would:
1. Learn to classify graphs as vulnerable/non-vulnerable
2. But struggle to learn how vulnerability patterns transform into secure patterns
3. Provide less precise explanations and fix suggestions

## Working with Existing Datasets

### Converting Datasets with Changed Line Attributes

If your dataset has pre-patch and post-patch graphs with attributes for changed lines, you can convert it using the following steps:

1. **Define a Conversion Function**:

```python
def convert_existing_dataset(pre_patch_graphs, post_patch_graphs):
    aligned_graphs = []
    
    for pre_graph, post_graph in zip(pre_patch_graphs, post_patch_graphs):
        # Extract changed lines
        pre_changed_lines = set()
        post_changed_lines = set()
        
        for node, data in pre_graph.nodes(data=True):
            if data.get('is_changed', False):
                pre_changed_lines.add(data.get('line_number'))
        
        for node, data in post_graph.nodes(data=True):
            if data.get('is_changed', False):
                post_changed_lines.add(data.get('line_number'))
        
        # Create alignment mapping
        alignment = {}
        
        # Align unchanged nodes
        for pre_node, pre_data in pre_graph.nodes(data=True):
            if 'line_number' in pre_data and pre_data['line_number'] not in pre_changed_lines:
                pre_line = pre_data['line_number']
                
                # Find matching node in post-patch graph
                for post_node, post_data in post_graph.nodes(data=True):
                    if 'line_number' in post_data and post_data['line_number'] == pre_line:
                        alignment[pre_node] = post_node
                        break
        
        aligned_graphs.append((pre_graph, post_graph, alignment))
    
    return aligned_graphs
```

2. **Implement a custom DataLoader**:

```python
class CustomVulnerabilityDataset(torch.utils.data.Dataset):
    def __init__(self, pre_patch_graphs, post_patch_graphs, graph_processor):
        self.graph_processor = graph_processor
        self.data_pairs = self.process_graph_pairs(pre_patch_graphs, post_patch_graphs)
    
    def process_graph_pairs(self, pre_patch_graphs, post_patch_graphs):
        data_pairs = []
        
        for pre_graph, post_graph in zip(pre_patch_graphs, post_patch_graphs):
            # Create alignment mapping from changed lines
            alignment = create_alignment_from_changed_lines(pre_graph, post_graph)
            
            # Process using the system's graph processor
            vuln_hetero, patch_hetero = self.graph_processor.process_graph_pair(
                pre_graph, post_graph, alignment
            )
            
            data_pairs.append((vuln_hetero, patch_hetero))
        
        return data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        vuln_graph, patch_graph = self.data_pairs[idx]
        return vuln_graph, patch_graph
```

3. **Register your custom dataset with the system**:

```python
from src.data.graph_processing import GraphProcessor

# Initialize the graph processor
graph_processor = GraphProcessor(config)

# Create your custom dataset
custom_dataset = CustomVulnerabilityDataset(
    pre_patch_graphs,
    post_patch_graphs,
    graph_processor
)

# Create DataLoader
from torch_geometric.loader import DataLoader
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
```

### Alternative: Generating Alignment Mappings

If generating alignments from changed line attributes proves difficult, you can try these alternative approaches:

1. **AST Differencing**: Use tools like GumTree to perform AST differencing and generate mappings between AST nodes.

2. **Code Clone Detection**: Use code clone detection techniques to identify similar code fragments between pre-patch and post-patch graphs.

3. **Graph Matching Algorithms**: Apply graph matching algorithms like the Hungarian algorithm to find optimal node mappings.

4. **Neural Alignment**: Train a separate model to predict node alignments based on node features and context.

### Data Preprocessing Example

Here's a complete example of preprocessing your dataset:

```python
import torch
import networkx as nx
from src.data.graph_processing import GraphProcessor

def preprocess_dataset(pre_patch_graphs, post_patch_graphs, config):
    # Initialize the graph processor
    graph_processor = GraphProcessor(config)
    
    vulnerable_graphs = []
    patched_graphs = []
    alignments = []
    
    for i, (pre_graph, post_graph) in enumerate(zip(pre_patch_graphs, post_patch_graphs)):
        # Create alignment mapping from changed lines
        alignment = create_alignment_from_changed_lines(pre_graph, post_graph)
        
        # Store the processed graphs and alignment
        vulnerable_graphs.append(pre_graph)
        patched_graphs.append(post_graph)
        alignments.append(alignment)
    
    # Save the processed dataset
    torch.save(vulnerable_graphs, 'data/raw/vulnerable_graphs.pt')
    torch.save(patched_graphs, 'data/raw/patched_graphs.pt')
    torch.save(alignments, 'data/raw/alignments.pt')
    
    print(f"Preprocessed {len(vulnerable_graphs)} graph pairs")
    
    return vulnerable_graphs, patched_graphs, alignments
```

## Data Processing Internals

To better understand how the system processes your data internally, here's a detailed explanation of the data flow:

### 1. Data Loading

When the system loads your dataset, it follows these steps:

```python
# In src/data/dataloader.py, VulnerabilityPairDataset.process()
def process(self):
    # Load raw data
    vulnerable_graphs = torch.load(os.path.join(self.raw_dir, 'vulnerable_graphs.pt'))
    patched_graphs = torch.load(os.path.join(self.raw_dir, 'patched_graphs.pt'))
    alignments = torch.load(os.path.join(self.raw_dir, 'alignments.pt'))
```

### 2. Graph Processing

Each graph pair is then processed to create heterogeneous graph representations:

```python
# In src/data/dataloader.py, VulnerabilityPairDataset.process()
graph_builder = HeteroGraphBuilder(self.config)
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
    data_list.append(combined_graph)
```

### 3. Heterogeneous Graph Construction

The `HeteroGraphBuilder` combines the pre-patch and post-patch graphs into a single heterogeneous graph with alignment edges:

```python
# In src/data/hetero_graph.py, HeteroGraphBuilder.build_vuln_patch_graph_pair()
def build_vuln_patch_graph_pair(self, vuln_graph, patch_graph):
    combined = HeteroData()
    
    # Add vulnerable code nodes with type "vuln"
    combined["vuln"].x = vuln_graph["code"].x
    combined["vuln"].y = vuln_graph["code"].y
    
    # Add patched code nodes with type "patch"
    combined["patch"].x = patch_graph["code"].x
    combined["patch"].y = patch_graph["code"].y
    
    # Add vulnerable-vulnerable edges
    for edge_type in self.edge_types:
        edge_name = f"code, {edge_type}, code"
        if edge_name in vuln_graph.edge_types:
            combined["vuln", edge_type, "vuln"].edge_index = vuln_graph[edge_name].edge_index
    
    # Add patch-patch edges
    for edge_type in self.edge_types:
        edge_name = f"code, {edge_type}, code"
        if edge_name in patch_graph.edge_types:
            combined["patch", edge_type, "patch"].edge_index = patch_graph[edge_name].edge_index
    
    # Add alignment edges between vulnerable and patched code
    self._add_alignment_edges(combined, vuln_graph, patch_graph)
    
    return combined
```

### 4. Alignment Edge Creation

Alignment edges are created using the alignment mapping:

```python
# In src/data/hetero_graph.py, HeteroGraphBuilder._add_alignment_edges()
def _add_alignment_edges(self, combined, vuln_graph, patch_graph):
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
```

### 5. Data Batching and Training

During training, the system:

1. Loads batches of heterogeneous graphs
2. Passes them through the GraphSAGE model
3. Gets node embeddings for both vulnerable and patched code
4. Applies the pattern learning module
5. Computes contrastive learning losses
6. Updates the model parameters

```python
# In scripts/train.py, train_epoch()
for batch_idx, batch in enumerate(dataloader):
    # Move batch to device
    batch = batch.to(device)
    
    # Forward pass through the model
    optimizer.zero_grad()
    
    # Get node embeddings
    node_embeddings = model.get_node_embeddings(batch)
    
    # Get vulnerability prediction
    prediction = model(batch)
    
    # Get ground truth
    if 'vuln' in batch.node_types:
        y = batch['vuln'].y
    else:
        y = batch['code'].y
    
    # Calculate classification loss
    classification_loss = nn.BCELoss()(prediction, y)
    
    # Pattern learning
    contrastive_loss, reconstruction_loss, pattern_similarity, _ = pattern_module(
        node_embeddings, node_embeddings
    )
    
    # Combine losses
    total_loss = (
        classification_weight * classification_loss +
        contrastive_weight * contrastive_loss +
        reconstruction_weight * reconstruction_loss
    )
    
    # Backward pass and optimization
    total_loss.backward()
    optimizer.step()
```

## Research Extensions

As the researcher and developer of this project, you might want to extend it in various ways. Here are some suggestions:

### 1. Custom Pattern Learning Algorithms

The pattern learning module (`src/models/pattern_learning.py`) is a key component for research. You can:

- Modify the contrastive learning loss function
- Implement different pattern extraction techniques
- Experiment with various transformation rule learning approaches

Example of a custom contrastive loss function:
```python
def custom_contrastive_loss(vuln_patterns, security_patterns, margin=0.5, temperature=0.1):
    """
    Custom contrastive loss function with temperature scaling.
    """
    # Normalize patterns
    vuln_norm = F.normalize(vuln_patterns, p=2, dim=1)
    security_norm = F.normalize(security_patterns, p=2, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.mm(vuln_norm, security_norm.transpose(0, 1)) / temperature
    
    # Get positive and negative pairs
    positive_pairs = torch.diag(similarity)
    mask = torch.eye(similarity.size(0), device=similarity.device) == 0
    negative_pairs = similarity[mask].view(similarity.size(0), -1)
    
    # Calculate positive and negative losses
    positive_loss = -torch.mean(positive_pairs)
    negative_loss = torch.mean(torch.logsumexp(negative_pairs, dim=1))
    
    # Combined loss
    loss = positive_loss + negative_loss
    
    return loss
```

### 2. Graph Neural Network Architectures

You can experiment with different GNN architectures by modifying the `src/models/graphsage.py` file:

- Replace GraphSAGE with GAT, GCN, or other architectures
- Implement heterogeneous versions of advanced GNNs
- Try different aggregation functions and attention mechanisms

Example of implementing a GAT-based model:
```python
class HeteroGAT(nn.Module):
    """
    Heterogeneous Graph Attention Network implementation.
    """
    
    def __init__(self, config):
        super(HeteroGAT, self).__init__()
        
        self.config = config
        self.hidden_dim = config["model"]["hidden_channels"]
        self.num_layers = config["model"]["num_layers"]
        self.dropout = config["model"]["dropout"]
        self.num_heads = config["model"].get("num_heads", 8)
        
        # Define node types and edge types
        self.node_types = ["vuln", "patch"]
        self.edge_types = config["model"]["edge_types"]
        
        # Create initial node type embeddings
        self.node_type_embeddings = nn.ModuleDict({
            node_type: nn.Linear(config["model"]["node_features"]["embedding_dim"], self.hidden_dim)
            for node_type in self.node_types
        })
        
        # Create full edge types list
        self.full_edge_types = []
        for src in self.node_types:
            for dst in self.node_types:
                if src == dst:  # Same node type edges
                    for edge_type in self.edge_types:
                        self.full_edge_types.append((src, edge_type, dst))
                else:  # Different node types - alignment edges
                    self.full_edge_types.append((src, "aligned", dst))
        
        # Create GAT layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            conv_dict = {}
            
            for edge_type in self.full_edge_types:
                conv_dict[edge_type] = GATConv(
                    self.hidden_dim, 
                    self.hidden_dim // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
        
        # Output layer
        self.linear = nn.Linear(self.hidden_dim * 2, 1)
    
    def forward(self, data):
        # Implementation similar to HeteroGraphSAGE but with GAT layers
        # ...
```

### 3. Alignment Learning

Instead of relying on explicit alignment mappings, you could research methods to learn alignments automatically:

- Implement a graph matching neural network
- Use attention mechanisms to learn soft alignments
- Develop a separate model for alignment prediction

Example of an alignment learning module:
```python
class AlignmentLearner(nn.Module):
    """
    Module for learning alignments between vulnerable and patched code.
    """
    
    def __init__(self, hidden_dim):
        super(AlignmentLearner, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Score function for computing alignment scores
        self.score_func = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, vuln_embeddings, patch_embeddings):
        """
        Compute soft alignment matrix between vulnerable and patched code embeddings.
        
        Args:
            vuln_embeddings: Tensor of shape [num_vuln_nodes, hidden_dim]
            patch_embeddings: Tensor of shape [num_patch_nodes, hidden_dim]
            
        Returns:
            Soft alignment matrix of shape [num_vuln_nodes, num_patch_nodes]
        """
        num_vuln_nodes = vuln_embeddings.size(0)
        num_patch_nodes = patch_embeddings.size(0)
        
        # Compute all pairwise combinations
        vuln_expanded = vuln_embeddings.unsqueeze(1).expand(-1, num_patch_nodes, -1)
        patch_expanded = patch_embeddings.unsqueeze(0).expand(num_vuln_nodes, -1, -1)
        
        # Concatenate embeddings for each pair
        pairs = torch.cat([vuln_expanded, patch_expanded], dim=2)
        
        # Reshape for score computation
        pairs_flat = pairs.view(-1, self.hidden_dim * 2)
        
        # Compute alignment scores
        scores_flat = self.score_func(pairs_flat)
        scores = scores_flat.view(num_vuln_nodes, num_patch_nodes)
        
        # Apply softmax to get soft alignments
        alignments = F.softmax(scores, dim=1)
        
        return alignments
```

### 4. Custom Loss Functions and Metrics

You can implement custom loss functions and metrics for your specific research goals:

- Develop specialized loss functions for security pattern learning
- Create metrics for evaluating pattern quality
- Implement explainability metrics for vulnerability detection

Example of a custom metric:
```python
def pattern_diversity_score(patterns, threshold=0.8):
    """
    Compute the diversity of learned patterns.
    
    Args:
        patterns: Tensor of shape [num_patterns, pattern_dim]
        threshold: Similarity threshold for considering patterns as duplicates
        
    Returns:
        Diversity score between 0 and 1
    """
    # Normalize patterns
    patterns_norm = F.normalize(patterns, p=2, dim=1)
    
    # Compute pairwise similarities
    similarity = torch.mm(patterns_norm, patterns_norm.t())
    
    # Remove self-similarities
    similarity.fill_diagonal_(0)
    
    # Count similar pattern pairs
    similar_pairs = (similarity > threshold).sum().item() / 2  # Divide by 2 to avoid counting twice
    
    # Maximum possible similar pairs
    max_pairs = (patterns.size(0) * (patterns.size(0) - 1)) / 2
    
    # Compute diversity score (1 - percentage of similar pairs)
    diversity_score = 1 - (similar_pairs / max_pairs)
    
    return diversity_score
```

### 5. Debugging and Visualization Tools

Implement additional debugging and visualization tools for research:

- Pattern visualization tools
- Attention heatmaps
- Graph structure exploration
- Learning curve analysis

Example of a debugging tool:
```python
def analyze_pattern_learning(model, pattern_module, dataloader, num_samples=5):
    """
    Analyze pattern learning behavior on specific samples.
    
    Args:
        model: Trained model
        pattern_module: Trained pattern module
        dataloader: DataLoader with samples
        num_samples: Number of samples to analyze
    """
    model.eval()
    pattern_module.eval()
    
    samples = []
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break
        
        # Get node embeddings
        with torch.no_grad():
            node_embeddings = model.get_node_embeddings(batch)
            _, _, pattern_similarity, pattern_outputs = pattern_module(
                node_embeddings, node_embeddings
            )
        
        # Get attention weights
        vuln_attention = pattern_outputs["vuln_attention"]
        security_attention = pattern_outputs["security_attention"]
        
        # Find top nodes for each pattern
        top_vuln_nodes = torch.topk(vuln_attention, k=5, dim=0)[1]
        top_sec_nodes = torch.topk(security_attention, k=5, dim=0)[1]
        
        samples.append({
            "batch": batch,
            "pattern_similarity": pattern_similarity,
            "top_vuln_nodes": top_vuln_nodes,
            "top_sec_nodes": top_sec_nodes
        })
    
    return samples
```

By understanding these internal details and extension possibilities, you'll be better equipped to adapt and extend the system for your research needs.