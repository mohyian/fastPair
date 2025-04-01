# Data Directory

This directory contains data files for the vulnerability detection system.

## Directory Structure

- `raw/`: Raw input data
- `processed/`: Processed data ready for training and evaluation

## Raw Data Format

The raw data should be organized as follows:

- `vulnerable_graphs.pt`: List of PyTorch Geometric HeteroData objects representing vulnerable code
- `patched_graphs.pt`: List of PyTorch Geometric HeteroData objects representing patched/secure code
- `alignments.pt`: List of dictionaries mapping nodes between vulnerable and patched graphs

### Graph Structure

Each graph contains the following elements:

1. Nodes with features:
   - `x`: Node feature matrix
   - `y`: Graph-level label (1 for vulnerable, 0 for secure)

2. Edges of different types:
   - `control_flow`: Control flow edges
   - `data_flow`: Data flow edges
   - `semantic`: Semantic relationship edges

## Processed Data Format

The processed data is organized as HeteroData objects with the following structure:

- Node types: `vuln` and `patch`
- Edge types: `(vuln, control_flow, vuln)`, `(vuln, data_flow, vuln)`, `(vuln, semantic, vuln)`, `(patch, control_flow, patch)`, `(patch, data_flow, patch)`, `(patch, semantic, patch)`, `(vuln, aligned, patch)`, `(patch, aligned, vuln)`

## Preparing Your Own Data

To prepare your own data, follow these steps:

1. Parse your source code into graph representations using the appropriate language adapter
2. Convert the graphs into PyTorch Geometric HeteroData format
3. Save the graphs as `vulnerable_graphs.pt`, `patched_graphs.pt`, and alignments as `alignments.pt`
4. Place these files in the `raw/` directory
5. Run the data processing script to generate the processed data

For more details on data processing, refer to the documentation.