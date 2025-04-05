#!/usr/bin/env python
"""
Training Script

This script trains the vulnerability detection model using the provided configuration.
It handles data loading, model initialization, training, and evaluation.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import VulnerabilityDataLoader
from src.models.graphsage import HeteroGraphSAGE
from src.models.pattern_learning import PatternLearningModule
from src.utils.logger import initialize_logging, get_logger
from src.utils.metrics import MetricsTracker, calculate_pattern_metrics
from src.utils.visualization import visualize_training_metrics, visualize_pattern_similarity

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train vulnerability detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Path to data directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Path to output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use for training (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of training epochs (overrides config)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    if args.data_dir:
        config['data']['processed_dir'] = args.data_dir
    
    if args.output_dir:
        config['training']['save_dir'] = args.output_dir
    
    if args.device:
        config['training']['device'] = args.device
    
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    return config

def train_epoch(model, pattern_module, dataloader, optimizer, device, config):
    """Train the model for one epoch."""
    model.train()
    pattern_module.train()
    
    metrics = MetricsTracker()
    
    # Get loss weights from config
    loss_weights = config['training']['loss_weights']
    classification_weight = loss_weights.get('classification', 1.0)
    contrastive_weight = loss_weights.get('contrastive', 0.5)
    reconstruction_weight = loss_weights.get('reconstruction', 0.2)
    
    # Use tqdm for progress bar
    with tqdm(dataloader, desc="Training", leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            # Skip empty batches
            if batch is None:
                continue
            
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
            
            # Fix tensor shapes to ensure they match
            # Ensure both prediction and y have the same shape
            if prediction.dim() == 0:  # If prediction is a scalar
                prediction = prediction.unsqueeze(0)  # Make it [1]
            
            if y.dim() == 0:  # If y is a scalar
                y = y.unsqueeze(0)  # Make it [1]
            
            # If shapes still don't match, try to adapt
            if prediction.shape != y.shape:
                # If one is [batch_size, 1] and other is [batch_size]
                if prediction.dim() > y.dim():
                    prediction = prediction.squeeze(-1)
                elif y.dim() > prediction.dim():
                    prediction = prediction.unsqueeze(-1)
            
            # Ensure y is float type
            y = y.float()
            
            # Print shapes for debugging
            # print(f"Prediction shape: {prediction.shape}, y shape: {y.shape}")
            
            # Calculate classification loss
            try:
                classification_loss = nn.BCELoss()(prediction, y)
            except Exception as e:
                logger.error(f"BCELoss error: {e}, prediction shape: {prediction.shape}, y shape: {y.shape}")
                # Use a fallback loss calculation
                classification_loss = F.mse_loss(prediction, y)
            
            # Pattern learning
            try:
                contrastive_loss, reconstruction_loss, pattern_similarity, _ = pattern_module(
                    node_embeddings, node_embeddings
                )
            except Exception as e:
                logger.error(f"Pattern learning error: {e}")
                # Create dummy tensors for losses
                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
                reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=True)
                pattern_similarity = torch.zeros((1, 1), device=device)
            
            # Combine losses
            total_loss = (
                classification_weight * classification_loss +
                contrastive_weight * contrastive_loss +
                reconstruction_weight * reconstruction_loss
            )
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Apply gradient clipping
            if config['training'].get('gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['gradient_clipping']
                )
            
            optimizer.step()
            
            # Update metrics
            metrics.update(prediction.detach(), y.detach(), total_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'clf_loss': f"{classification_loss.item():.4f}",
                'cont_loss': f"{contrastive_loss.item():.4f}",
                'rec_loss': f"{reconstruction_loss.item():.4f}"
            })
            
            # Log metrics at specified frequency
            log_frequency = config['logging'].get('log_frequency', 10)
            if batch_idx % log_frequency == 0:
                logger.info(
                    f"Batch {batch_idx}/{len(dataloader)} - "
                    f"Loss: {total_loss.item():.4f}, "
                    f"CLF: {classification_loss.item():.4f}, "
                    f"CONT: {contrastive_loss.item():.4f}, "
                    f"REC: {reconstruction_loss.item():.4f}"
                )
    
    # End epoch and get metrics
    epoch_metrics = metrics.end_epoch()
    
    # Add pattern similarity metrics
    # Create a dummy alignment matrix (in a real implementation, this would come from the data)
    alignment = torch.eye(pattern_similarity.size(0), device=pattern_similarity.device)
    pattern_metrics = calculate_pattern_metrics(pattern_similarity, alignment)
    
    # Combine metrics
    epoch_metrics.update(pattern_metrics)
    
    return epoch_metrics, pattern_similarity

def validate(model, pattern_module, dataloader, device, config):
    """Validate the model on the validation set."""
    model.eval()
    pattern_module.eval()
    
    metrics = MetricsTracker()
    
    # Get loss weights from config
    loss_weights = config['training']['loss_weights']
    classification_weight = loss_weights.get('classification', 1.0)
    contrastive_weight = loss_weights.get('contrastive', 0.5)
    reconstruction_weight = loss_weights.get('reconstruction', 0.2)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
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
            
            # Calculate classification loss
            prediction = prediction.squeeze()
            classification_loss = nn.BCELoss()(prediction, y.float())
            
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
            
            # Update metrics
            metrics.update(prediction.detach(), y.detach(), total_loss.item())
    
    # End epoch and get metrics
    epoch_metrics = metrics.end_epoch()
    
    # Add pattern similarity metrics
    # Create a dummy alignment matrix (in a real implementation, this would come from the data)
    alignment = torch.eye(pattern_similarity.size(0), device=pattern_similarity.device)
    pattern_metrics = calculate_pattern_metrics(pattern_similarity, alignment)
    
    # Combine metrics
    epoch_metrics.update(pattern_metrics)
    
    return epoch_metrics, pattern_similarity

def save_checkpoint(model, pattern_module, optimizer, epoch, metrics, config, is_best=False):
    """Save a model checkpoint."""
    # Create save directory if it doesn't exist
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'pattern_module_state_dict': pattern_module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    # Save the checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # If this is the best model, save a copy
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")

def main():
    """Main training function."""
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
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    # Create data loader
    # IMPORTANT: Use our custom dataloader instead of the original one
    from src.data.custom_dataloader import VulnerabilityPairDataLoader
    data_loader = VulnerabilityPairDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    # Create model
    model = HeteroGraphSAGE(config)
    model.to(device)
    
    # Create pattern learning module
    pattern_module = PatternLearningModule(config)
    pattern_module.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(pattern_module.parameters()),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
        verbose=True
    )
    
    # Initialize best validation score
    best_val_score = 0.0
    best_epoch = 0
    
    # Initialize metrics trackers
    train_metrics_tracker = MetricsTracker()
    val_metrics_tracker = MetricsTracker()
    
    # Training loop
    num_epochs = config['training']['epochs']
    early_stopping_patience = config['training']['early_stopping_patience']
    early_stopping_counter = 0
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    # Use tqdm for epoch progress bar
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        
        # Train for one epoch
        start_time = time.time()
        train_metrics, train_pattern_similarity = train_epoch(
            model, pattern_module, train_loader, optimizer, device, config
        )
        train_metrics_tracker.epoch_metrics.append(train_metrics)
        train_time = time.time() - start_time
        
        # Validate the model
        val_metrics, val_pattern_similarity = validate(
            model, pattern_module, val_loader, device, config
        )
        val_metrics_tracker.epoch_metrics.append(val_metrics)
        
        # Log metrics
        logger.info(f"Epoch {epoch} - "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Train F1: {train_metrics['f1']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}, "
                  f"Time: {train_time:.2f}s")
        
        # Update learning rate scheduler
        scheduler.step(val_metrics['loss'])
        
        # Check if this is the best model
        current_val_score = val_metrics['f1']  # Use F1 score as the main metric
        is_best = current_val_score > best_val_score
        
        if is_best:
            best_val_score = current_val_score
            best_epoch = epoch
            early_stopping_counter = 0
            
            # Update pattern repository
            pattern_module.update_pattern_repository()
            
            # Visualize patterns
            visualize_pattern_similarity(
                val_pattern_similarity,
                output_path=os.path.join(config['training']['save_dir'], f'pattern_similarity_epoch_{epoch}.png'),
                title=f'Pattern Similarity Matrix - Epoch {epoch}'
            )
        else:
            early_stopping_counter += 1
        
        # Save checkpoint
        save_checkpoint(model, pattern_module, optimizer, epoch, val_metrics, config, is_best)
        
        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Visualize training metrics
    train_metrics_history = train_metrics_tracker.get_all_epoch_metrics()
    val_metrics_history = val_metrics_tracker.get_all_epoch_metrics()
    
    metrics_to_plot = {
        'Loss': (train_metrics_history.get('loss', []), val_metrics_history.get('loss', [])),
        'Accuracy': (train_metrics_history.get('accuracy', []), val_metrics_history.get('accuracy', [])),
        'F1 Score': (train_metrics_history.get('f1', []), val_metrics_history.get('f1', [])),
        'Precision': (train_metrics_history.get('precision', []), val_metrics_history.get('precision', [])),
        'Recall': (train_metrics_history.get('recall', []), val_metrics_history.get('recall', []))
    }
    
    # Create visualization of training metrics
    visualize_training_progress(
        metrics_to_plot,
        output_dir=config['training']['save_dir'],
        title='Training Progress'
    )
    
    logger.info(f"Training complete. Best model at epoch {best_epoch} with F1 score {best_val_score:.4f}")
    
    # Test the best model
    logger.info("Loading best model for testing")
    best_model_path = os.path.join(config['training']['save_dir'], 'best_model.pt')
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        pattern_module.load_state_dict(checkpoint['pattern_module_state_dict'])
        
        logger.info("Evaluating on test set")
        test_metrics, _ = validate(model, pattern_module, test_loader, device, config)
        
        logger.info(f"Test Results - "
                  f"Loss: {test_metrics['loss']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.4f}, "
                  f"Precision: {test_metrics['precision']:.4f}, "
                  f"Recall: {test_metrics['recall']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}, "
                  f"ROC AUC: {test_metrics['roc_auc']:.4f}, "
                  f"PR AUC: {test_metrics['pr_auc']:.4f}")
    else:
        logger.warning(f"Best model not found at {best_model_path}")

def visualize_training_progress(metrics_dict, output_dir, title='Training Progress'):
    """
    Visualize training metrics.
    
    Args:
        metrics_dict: Dictionary mapping metric names to (train_values, val_values) tuples
        output_dir: Directory to save visualizations
        title: Title for the visualizations
    """
    import matplotlib.pyplot as plt
    
    for metric_name, (train_values, val_values) in metrics_dict.items():
        if not train_values or not val_values:
            continue
        
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_values) + 1)
        
        plt.plot(epochs, train_values, 'b-', label=f'Train {metric_name}')
        plt.plot(epochs, val_values, 'r-', label=f'Validation {metric_name}')
        
        plt.title(f'{title} - {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        
        output_path = os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_progress.png')
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved {metric_name} progress visualization to {output_path}")

if __name__ == '__main__':
    # Get global logger
    logger = get_logger(__name__)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise