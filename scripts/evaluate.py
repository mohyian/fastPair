#!/usr/bin/env python
"""
Evaluation Script

This script evaluates a trained vulnerability detection model on a test dataset.
It calculates and reports various metrics and generates visualizations.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, auc, roc_auc_score
)

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import VulnerabilityDataLoader
from src.models.graphsage import HeteroGraphSAGE
from src.models.pattern_learning import PatternLearningModule
from src.utils.logger import initialize_logging, get_logger
from src.utils.metrics import MetricsTracker
from src.utils.visualization import (
    visualize_pattern_similarity, visualize_vulnerability_scores,
    visualize_pattern_attention
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate vulnerability detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                      help='Path to test data (overrides config)')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Path to output directory for results')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for evaluation (overrides config)')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Classification threshold for binary predictions')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.test_data:
        config['data']['processed_dir'] = args.test_data
    
    if args.device:
        config['training']['device'] = args.device
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return config

def load_model(model_path, config, device):
    """Load the model from a checkpoint."""
    # Create model
    model = HeteroGraphSAGE(config)
    
    # Create pattern learning module
    pattern_module = PatternLearningModule(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model and pattern module state
    model.load_state_dict(checkpoint['model_state_dict'])
    pattern_module.load_state_dict(checkpoint['pattern_module_state_dict'])
    
    # Move to device
    model.to(device)
    pattern_module.to(device)
    
    # Set to evaluation mode
    model.eval()
    pattern_module.eval()
    
    return model, pattern_module, checkpoint

def evaluate(model, pattern_module, dataloader, device, threshold=0.5):
    """Evaluate the model on the test dataset."""
    model.eval()
    pattern_module.eval()
    
    all_predictions = []
    all_labels = []
    all_losses = []
    all_pattern_similarities = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = batch.to(device)
            
            # Get node embeddings
            node_embeddings = model.get_node_embeddings(batch)
            
            # Get vulnerability prediction
            prediction = model(batch)
            
            # Get ground truth
            if 'vuln' in batch.node_types:
                y = batch['vuln'].y
            else:
                y = batch['code'].y
            
            # Calculate loss
            loss = nn.BCELoss()(prediction, y)
            
            # Pattern learning
            _, _, pattern_similarity, pattern_outputs = pattern_module(
                node_embeddings, node_embeddings
            )
            
            # Store predictions, labels, and losses
            all_predictions.append(prediction.cpu())
            all_labels.append(y.cpu())
            all_losses.append(loss.item())
            all_pattern_similarities.append(pattern_similarity.cpu())
            
            # Store attention weights
            if 'vuln_attention' in pattern_outputs:
                all_attention_weights.append(pattern_outputs['vuln_attention'].cpu())
    
    # Concatenate predictions and labels
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, threshold)
    
    # Add average loss
    metrics['avg_loss'] = np.mean(all_losses)
    
    # Combine pattern similarities
    if all_pattern_similarities:
        avg_pattern_similarity = torch.mean(torch.stack(all_pattern_similarities), dim=0)
    else:
        avg_pattern_similarity = None
    
    # Combine attention weights
    if all_attention_weights:
        avg_attention_weights = torch.mean(torch.stack(all_attention_weights), dim=0)
    else:
        avg_attention_weights = None
    
    return metrics, predictions, labels, avg_pattern_similarity, avg_attention_weights

def calculate_metrics(predictions, labels, threshold=0.5):
    """Calculate evaluation metrics."""
    # Convert to numpy
    predictions_np = predictions.numpy()
    labels_np = labels.numpy()
    
    # Calculate binary predictions
    binary_preds = (predictions_np > threshold).astype(int)
    
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((binary_preds == 1) & (labels_np == 1))
    fp = np.sum((binary_preds == 1) & (labels_np == 0))
    tn = np.sum((binary_preds == 0) & (labels_np == 0))
    fn = np.sum((binary_preds == 0) & (labels_np == 1))
    
    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(labels_np, predictions_np)
    except:
        roc_auc = 0
    
    # Precision-Recall AUC
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(labels_np, predictions_np)
        pr_auc = auc(recall_curve, precision_curve)
    except:
        pr_auc = 0
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }
    
    return metrics

def plot_roc_curve(labels, predictions, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved ROC curve to {output_path}")
    
    return output_path

def plot_precision_recall_curve(labels, predictions, output_dir):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.axhline(y=sum(labels)/len(labels), color='navy', lw=2, linestyle='--', 
               label=f'Baseline (class balance = {sum(labels)/len(labels):.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved precision-recall curve to {output_path}")
    
    return output_path

def plot_confusion_matrix(labels, binary_preds, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, binary_preds)
    
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-vulnerable', 'Vulnerable'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.grid(False)
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")
    
    return output_path

def save_metrics_report(metrics, output_dir):
    """Save metrics report to a file."""
    report = f"""
Vulnerability Detection Evaluation Report
----------------------------------------

Classification Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}
- Specificity: {metrics['specificity']:.4f}
- ROC AUC: {metrics['roc_auc']:.4f}
- PR AUC: {metrics['pr_auc']:.4f}

Confusion Matrix:
- True Positives: {metrics['tp']}
- False Positives: {metrics['fp']}
- True Negatives: {metrics['tn']}
- False Negatives: {metrics['fn']}

Additional Metrics:
- Average Loss: {metrics.get('avg_loss', 'N/A')}
    """
    
    output_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {output_path}")
    
    return output_path

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # Initialize logging
    initialize_logging()
    
    # Set device
    device = torch.device(config['training']['device'] 
                        if torch.cuda.is_available() and config['training']['device'] == 'cuda' 
                        else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, pattern_module, checkpoint = load_model(args.model_path, config, device)
    
    # Load test data
    logger.info("Loading test data")
    data_loader = VulnerabilityDataLoader(config)
    _, _, test_loader = data_loader.create_data_loaders()
    
    # Evaluate model
    logger.info("Evaluating model")
    metrics, predictions, labels, pattern_similarity, attention_weights = evaluate(
        model, pattern_module, test_loader, device, args.threshold
    )
    
    # Log metrics
    logger.info(f"Evaluation Results - "
              f"Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"ROC AUC: {metrics['roc_auc']:.4f}, "
              f"PR AUC: {metrics['pr_auc']:.4f}")
    
    # Convert to numpy
    predictions_np = predictions.numpy()
    labels_np = labels.numpy()
    binary_preds = (predictions_np > args.threshold).astype(int)
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # ROC curve
    plot_roc_curve(labels_np, predictions_np, args.output_dir)
    
    # Precision-Recall curve
    plot_precision_recall_curve(labels_np, predictions_np, args.output_dir)
    
    # Confusion matrix
    plot_confusion_matrix(labels_np, binary_preds, args.output_dir)
    
    # Pattern similarity visualization
    if pattern_similarity is not None:
        visualize_pattern_similarity(
            pattern_similarity,
            output_path=os.path.join(args.output_dir, 'pattern_similarity.png'),
            title='Pattern Similarity Matrix'
        )
    
    # Pattern attention visualization
    if attention_weights is not None:
        # Visualize top patterns
        for pattern_idx in range(min(3, attention_weights.size(1))):
            visualize_pattern_attention(
                attention_weights,
                pattern_idx,
                top_k=10,
                output_path=os.path.join(args.output_dir, f'pattern_{pattern_idx}_attention.png'),
                title=f'Pattern {pattern_idx} Attention'
            )
    
    # Save metrics report
    save_metrics_report(metrics, args.output_dir)
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    # Get global logger
    logger = get_logger(__name__)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise