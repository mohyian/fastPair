# Architecture Overview

This document outlines the architecture of the Graph-Based Vulnerability Detection System.

## System Components

The system consists of the following main components:

1. **Graph Processing**: Converts code into heterogeneous graph representations
2. **Message Passing Network**: Implements GraphSAGE with heterogeneous graph support
3. **Pattern Learning Layer**: Learns vulnerability and security patterns
4. **Inference Engine**: Detects vulnerabilities in new code
5. **Extensions Framework**: Provides extensibility points for languages and patterns

## Architecture Diagram

```
┌─────────────────┐
│                 │
│    Source Code  │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Language        │
│  Adapters        │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │
│  Graph Processing│
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │
│  HeteroGraph    │
│  Representation │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │
│  Message Passing│◄───┤  Pattern Learning│
│  Network        │    │  Layer          │
│  (GraphSAGE)    │    │                 │
└────────┬────────┘    └─────────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │
│  Vulnerability  │    │  Pattern        │
│  Scoring        │    │  Repository     │
│                 │    │                 │
└────────┬────────┘    └─────────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │
│  Reporting      │    │  Fix Suggestion │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

## Component Details

### Graph Processing

This component is responsible for parsing code into graph representations and enriching them with additional features. It consists of:

- `GraphProcessor`: Processes raw code graphs into heterogeneous graph format
- `HeteroGraphBuilder`: Builds heterogeneous graphs with multiple node and edge types

### Message Passing Network

This component implements the graph neural network that processes the heterogeneous graphs. Key elements include:

- `HeteroGraphSAGE`: GraphSAGE implementation for heterogeneous graphs
- Attention mechanisms for focusing on important parts of the graph
- Message passing operations for different edge types

### Pattern Learning Layer

This component learns vulnerability and security patterns from the data. It includes:

- `PatternLearningModule`: Implements pattern learning using contrastive learning
- Pattern extraction mechanisms for identifying vulnerability patterns
- Transformation rules for learning how to convert vulnerable code to secure code

### Inference Engine

This component performs inference on new code to detect vulnerabilities. It consists of:

- `VulnerabilityScorer`: Scores code graphs for potential vulnerabilities
- `PatternMatcher`: Matches learned patterns against new code
- `VulnerabilityReporter`: Generates reports and visualizations

### Extensions Framework

This framework provides extensibility points for adding support for new languages, patterns, and vulnerability types:

- `BaseLanguageAdapter`: Base class for language-specific adapters
- `BasePatternPlugin`: Base class for vulnerability pattern plugins
- `BaseVulnerabilityHandler`: Base class for vulnerability type handlers

## Data Flow

1. Source code is parsed by the appropriate language adapter
2. The parsed code is converted to a heterogeneous graph representation
3. The graph is processed by the message passing network
4. The pattern learning layer extracts vulnerability patterns
5. The inference engine scores the code for vulnerabilities
6. Reports and fix suggestions are generated

## Integration Points

The system provides several integration points for extension:

1. **Language Adapters**: For adding support for new programming languages
2. **Pattern Plugins**: For detecting specific vulnerability patterns
3. **Vulnerability Handlers**: For handling specific types of vulnerabilities
4. **Configuration**: For customizing the system's behavior

## Training Pipeline

The training pipeline consists of the following steps:

1. Load and preprocess the data
2. Initialize the model and pattern learning module
3. Train the model using the training data
4. Evaluate the model on the validation data
5. Save the best model based on validation metrics
6. Test the final model on the test data

## Inference Pipeline

The inference pipeline consists of the following steps:

1. Load the trained model and pattern module
2. Parse the input code using the appropriate language adapter
3. Convert the code to a heterogeneous graph representation
4. Process the graph using the model to get node embeddings
5. Score the graph for vulnerabilities
6. Match vulnerability patterns
7. Generate reports and fix suggestions