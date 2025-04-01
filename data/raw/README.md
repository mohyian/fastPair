# Raw Data Directory

This directory contains raw input data for the vulnerability detection system.

## Expected Files

- `vulnerable_graphs.pt`: PyTorch file containing a list of NetworkX graphs or PyTorch Geometric HeteroData objects representing vulnerable code
- `patched_graphs.pt`: PyTorch file containing a list of NetworkX graphs or PyTorch Geometric HeteroData objects representing patched/secure code
- `alignments.pt`: PyTorch file containing a list of dictionaries mapping nodes between vulnerable and patched graphs

## Graph Structure

Each graph should represent a parsed code snippet with the following components:

1. Nodes representing code entities such as:
   - Functions
   - Variables
   - Operations
   - Control structures
   
2. Node features including:
   - Type information
   - AST type
   - Code tokens
   - Line numbers
   
3. Edges of different types:
   - Control flow edges (representing execution flow)
   - Data flow edges (representing data dependencies)
   - Semantic edges (representing semantic relationships)

## Alignment Format

The alignment dictionaries map nodes from vulnerable to patched graphs:

```python
{
    vulnerable_node_id_1: patched_node_id_1,
    vulnerable_node_id_2: patched_node_id_2,
    ...
}
```

## Data Sources

You can obtain raw data from:

1. Code repositories with vulnerability fixes
2. Vulnerability databases (e.g., CVEs, NVD)
3. Security patch commits
4. Synthetic examples generated for specific vulnerability types

## Data Preparation

To prepare your own raw data, you can use the language adapters provided in `src/extensions/language_adapters/`.