# Usage Guide

This guide provides instructions for using the Graph-Based Vulnerability Detection System.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Vulnerability Detection](#vulnerability-detection)
6. [Extending the System](#extending-the-system)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vulnerability-detection.git
cd vulnerability-detection
```

2. Create a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Or using conda
conda create -n vulndet python=3.8
conda activate vulndet
```

3. Install PyTorch Geometric dependencies:
```bash
# Install PyTorch first (example for CUDA 11.3)
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# Install PyTorch Geometric
pip install torch-geometric
```

4. Install project dependencies:
```bash
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Configuration

The system is configured using YAML configuration files in the `config/` directory:

### Main Configuration (`config/config.yaml`)

The main configuration file contains settings for:
- Data loading and processing
- Model architecture
- Training parameters
- Inference settings
- Extension settings

Example configuration:

```yaml
# Data settings
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  batch_size: 32
  num_workers: 4

# Model settings
model:
  name: "GraphSAGE"
  hidden_channels: 128
  num_layers: 3
  dropout: 0.2
  # ... more model settings ...

# Training settings
training:
  lr: 0.001
  weight_decay: 0.0001
  epochs: 100
  # ... more training settings ...

# Inference settings
inference:
  confidence_threshold: 0.7
  # ... more inference settings ...
```

### Logging Configuration (`config/logging.yaml`)

The logging configuration file specifies logging behavior:
- Log levels
- Output formats
- File locations

## Training

### Preparing Training Data

1. Place raw data files in the `data/raw/` directory:
   - `vulnerable_graphs.pt`: Vulnerable code graphs
   - `patched_graphs.pt`: Patched code graphs
   - `alignments.pt`: Node alignments between vulnerable and patched graphs

2. The system will process these files automatically during training.

### Training the Model

To train the model, use the `train.py` script:

```bash
python scripts/train.py --config config/config.yaml
```

Additional training options:
```bash
python scripts/train.py --config config/config.yaml --data_dir /path/to/data --output_dir /path/to/output --device cuda --epochs 200
```

### Training Output

Training will produce:
- Model checkpoints in the specified output directory
- Training logs with metrics
- Visualizations of pattern similarity matrices
- Training progress graphs

## Evaluation

### Evaluating a Trained Model

To evaluate a trained model, use the `evaluate.py` script:

```bash
python scripts/evaluate.py --config config/config.yaml --model_path models/saved_models/best_model.pt
```

Additional evaluation options:
```bash
python scripts/evaluate.py --config config/config.yaml --model_path models/saved_models/best_model.pt --test_data data/test --output_dir results/evaluation --device cuda --threshold 0.65
```

### Evaluation Output

Evaluation will produce:
- Evaluation metrics (accuracy, precision, recall, F1, etc.)
- ROC curve visualization
- Precision-Recall curve visualization
- Confusion matrix
- Pattern visualizations

## Vulnerability Detection

### Detecting Vulnerabilities in Code

To detect vulnerabilities in new code, use the `predict.py` script:

```bash
python scripts/predict.py --config config/config.yaml --model_path models/saved_models/best_model.pt --code_path path/to/code/
```

Additional prediction options:
```bash
python scripts/predict.py --config config/config.yaml --model_path models/saved_models/best_model.pt --code_path path/to/code/ --output_dir results/predictions --device cuda --threshold 0.7 --html_report
```

### Prediction Output

Vulnerability detection will produce:
- Vulnerability reports in JSON format
- HTML reports (if specified)
- Visualizations of detected patterns
- Summary report of all analyzed files

### Reading Vulnerability Reports

The vulnerability reports include:
- Overall vulnerability score
- Identified vulnerability patterns
- Affected code regions
- Suggested fixes
- Visualizations of vulnerable patterns

## Extending the System

### Adding a New Language Adapter

1. Create a new class that inherits from `BaseLanguageAdapter` in `src/extensions/language_adapters/`:

```python
from src.extensions.language_adapters.base import BaseLanguageAdapter

class MyLanguageAdapter(BaseLanguageAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.language_name = "my_language"
        self.supported_extensions = ["ext1", "ext2"]
        
    def parse_code(self, code):
        # Implementation for parsing code string
        
    def parse_file(self, file_path):
        # Implementation for parsing code file
        
    def extract_control_flow(self, graph):
        # Implementation for extracting control flow
        
    # ... implement other required methods ...
```

2. Update the configuration to enable the new language adapter:

```yaml
extensions:
  languages:
    - name: "my_language"
      enabled: true
      parser_path: "path/to/parser"
```

### Adding a New Pattern Plugin

1. Create a new class that inherits from `BasePatternPlugin` in `src/extensions/pattern_plugins/`:

```python
from src.extensions.pattern_plugins.base import BasePatternPlugin

class MyPatternPlugin(BasePatternPlugin):
    def __init__(self, config):
        super().__init__(config)
        self.pattern_name = "my_pattern"
        self.vulnerability_type = "my_vulnerability"
        
    def initialize_patterns(self):
        # Initialize pattern repositories
        
    def match_pattern(self, graph, node_embeddings=None):
        # Implementation for matching patterns
        
    def suggest_fix(self, graph, match_result):
        # Implementation for suggesting fixes
```

2. Update the configuration to enable the new pattern plugin.

### Adding a New Vulnerability Handler

1. Create a new class that inherits from `BaseVulnerabilityHandler` in `src/extensions/vulnerability_handlers/`:

```python
from src.extensions.vulnerability_handlers.base import BaseVulnerabilityHandler

class MyVulnerabilityHandler(BaseVulnerabilityHandler):
    def __init__(self, config):
        super().__init__(config)
        self.vulnerability_type = "my_vulnerability"
        
    def detect(self, graph, node_embeddings=None):
        # Implementation for detecting vulnerabilities
        
    def suggest_fix(self, graph, detection_result):
        # Implementation for suggesting fixes
        
    def explain(self, graph, detection_result):
        # Implementation for explaining vulnerabilities
```

2. Update the configuration to enable the new vulnerability handler.

## Troubleshooting

### Common Issues

1. **CUDA Memory Issues**

   If you encounter CUDA memory errors, try:
   ```bash
   # Reduce batch size
   python scripts/train.py --config config/config.yaml --batch_size 16
   
   # Use CPU instead of GPU
   python scripts/train.py --config config/config.yaml --device cpu
   ```

2. **Import Errors**

   If you encounter import errors, ensure that:
   - The virtual environment is activated
   - The package is installed in development mode (`pip install -e .`)
   - The project root is in the Python path

3. **Data Loading Issues**

   If data loading fails, check:
   - The data files are correctly placed in the raw data directory
   - The file formats match the expected formats
   - The data directory paths are correctly specified in the configuration

4. **Training Divergence**

   If training diverges or produces NaN values:
   - Reduce the learning rate
   - Add gradient clipping
   - Check for outliers in the training data

5. **Slow Performance**

   To improve performance:
   - Increase the number of data loading workers
   - Use GPU acceleration if available
   - Optimize the batch size for your hardware

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the logs for error messages
2. Look for similar issues in the issue tracker
3. Create a new issue with detailed information about the problem