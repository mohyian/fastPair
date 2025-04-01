# Processed Data Directory

This directory contains processed data ready for training and evaluation.

## Expected Files

- `vulnerability_pairs.pt`: PyTorch file containing processed HeteroData objects with combined vulnerable-patched pairs

## Data Structure

The processed data consists of PyTorch Geometric HeteroData objects with the following structure:

### Node Types

- `vuln`: Nodes from vulnerable code graphs
- `patch`: Nodes from patched code graphs

### Edge Types

- `(vuln, control_flow, vuln)`: Control flow edges within vulnerable code
- `(vuln, data_flow, vuln)`: Data flow edges within vulnerable code
- `(vuln, semantic, vuln)`: Semantic relationship edges within vulnerable code
- `(patch, control_flow, patch)`: Control flow edges within patched code
- `(patch, data_flow, patch)`: Data flow edges within patched code
- `(patch, semantic, patch)`: Semantic relationship edges within patched code
- `(vuln, aligned, patch)`: Alignment edges from vulnerable to patched nodes
- `(patch, aligned, vuln)`: Alignment edges from patched to vulnerable nodes

### Node Features

Each node has a feature vector with:

- `x`: Node feature matrix
- `y`: Graph-level label (1 for vulnerable, 0 for secure)

### Alignment Information

The `alignment` attribute on each graph contains the mapping between vulnerable and patched nodes.

## Data Processing

The raw data is processed using the `VulnerabilityPairDataset` class in `src/data/dataloader.py`. The processing steps include:

1. Loading the raw vulnerable graphs, patched graphs, and alignments
2. Creating heterogeneous graph representations
3. Adding alignment information and edges
4. Applying any transformations or filters
5. Collating the data into a format suitable for PyTorch Geometric

## Data Usage

This processed data is used directly by the DataLoader for training, validation, and testing. To use the processed data, create a `VulnerabilityPairDataset` instance and a DataLoader:

```python
from src.data.dataloader import VulnerabilityDataLoader

data_loader = VulnerabilityDataLoader(config)
train_loader, val_loader, test_loader = data_loader.create_data_loaders()
```