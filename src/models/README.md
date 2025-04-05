# Models Directory

This directory contains saved model checkpoints for the vulnerability detection system.

## Directory Structure

- `saved_models/`: Trained model checkpoints

## Model Format

Model checkpoints are saved as PyTorch `.pt` files with the following format:

```python
{
    'epoch': <epoch_number>,
    'model_state_dict': <model_state_dict>,
    'pattern_module_state_dict': <pattern_module_state_dict>,
    'optimizer_state_dict': <optimizer_state_dict>,
    'metrics': <validation_metrics>,
    'config': <configuration_dict>
}
```

## Model Types

### GraphSAGE Model

The main model architecture is a heterogeneous GraphSAGE with attention mechanisms, defined in `src/models/graphsage.py`. This model processes the heterogeneous graph data and outputs vulnerability predictions.

### Pattern Learning Module

The pattern learning module, defined in `src/models/pattern_learning.py`, learns vulnerability and security patterns from the data using contrastive learning and transformation rules.

## Using Saved Models

To load and use a saved model:

```python
import torch
from src.models.graphsage import HeteroGraphSAGE
from src.models.pattern_learning import PatternLearningModule

# Load the checkpoint
checkpoint = torch.load('path/to/checkpoint.pt', weights_only=False)

# Create the model and pattern module
model = HeteroGraphSAGE(checkpoint['config'])
pattern_module = PatternLearningModule(checkpoint['config'])

# Load the state dictionaries
model.load_state_dict(checkpoint['model_state_dict'])
pattern_module.load_state_dict(checkpoint['pattern_module_state_dict'])

# Set to evaluation mode
model.eval()
pattern_module.eval()
```

## Training New Models

To train a new model, use the `scripts/train.py` script:

```bash
python scripts/train.py --config config/config.yaml
```

The best model and intermediate checkpoints will be saved to this directory.