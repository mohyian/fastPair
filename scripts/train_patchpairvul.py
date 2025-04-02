#!/usr/bin/env python3
# scripts/train_patchpairvul.py
import os
import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.dataset import PatchPairDataset
from src.models.patchpairvul import PatchPairVul

def train(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    
    total_loss = 0
    all_targets = []
    all_outputs = []
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Get labels
        target = data['vuln'].y
        
        # Debug: Save first batch data for inspection
        if batch_idx == 0:
            print(f"\nDEBUG - First batch:")
            print(f"Output shape: {output.shape}, Target shape: {target.shape}")
            print(f"Output values (first 5): {output[:5].detach().cpu().tolist()}")
            print(f"Target values (first 5): {target[:5].detach().cpu().tolist()}")
            print(f"Target distribution: {target.sum().item()}/{len(target)} positive")
            
        # Collect all targets and outputs for analysis
        if len(output.shape) > 1:
            batch_output = output.squeeze().detach().cpu()
        else:
            batch_output = output.detach().cpu()
        all_targets.extend(target.detach().cpu().tolist())
        all_outputs.extend(batch_output.tolist())
        
        # Ensure output and target have the same shape
        if output.shape != target.shape:
            # If output is [batch_size, 1], squeeze it to [batch_size]
            if len(output.shape) > len(target.shape):
                output = output.squeeze()
            # If target is [batch_size] and output is [batch_size, 1]
            elif len(target.shape) < len(output.shape):
                target = target.unsqueeze(1)
        
        # Calculate loss
        loss = nn.BCELoss()(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
    
    # Overall training set analysis
    unique_targets = set(all_targets)
    target_counts = {val: all_targets.count(val) for val in unique_targets}
    
    # Check if targets are all the same
    if len(unique_targets) <= 1:
        print("\nWARNING: All training targets have the same value!")
    
    print(f"\nTraining set statistics:")
    print(f"Target distribution: {target_counts}")
    print(f"Prediction range: min={min(all_outputs):.4f}, max={max(all_outputs):.4f}, mean={sum(all_outputs)/len(all_outputs):.4f}")
    
    # Check for random predictions
    if max(all_outputs) - min(all_outputs) < 0.1:
        print("WARNING: Predictions have very small range! Model might not be learning.")
    
    return total_loss / len(train_loader.dataset)

def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    
    y_true = []
    y_pred = []
    raw_outputs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get labels
            target = data['vuln'].y
            
            # Ensure output has the same shape as target for metrics calculation
            if output.shape != target.shape:
                if len(output.shape) > len(target.shape):
                    output = output.squeeze()
            
            # Collect predictions, raw outputs, and true labels
            y_true.extend(target.cpu().numpy())
            y_pred.extend((output > 0.5).float().cpu().numpy())
            raw_outputs.extend(output.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Debug information
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    true_counts = {val: y_true.count(val) for val in unique_true}
    pred_counts = {val: y_pred.count(val) for val in unique_pred}
    
    print(f"\nValidation set statistics:")
    print(f"Labels distribution: {true_counts}")
    print(f"Predictions distribution: {pred_counts}")
    print(f"Raw output range: min={min(raw_outputs):.4f}, max={max(raw_outputs):.4f}, mean={sum(raw_outputs)/len(raw_outputs):.4f}")
    
    # Check for potential issues
    if len(unique_true) <= 1:
        print("WARNING: All validation targets have the same value!")
    
    if len(unique_pred) <= 1:
        print("WARNING: All validation predictions are the same! Model might not be learning.")
    
    # Add confusion matrix for binary classification
    true_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    false_pos = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    true_neg = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    false_neg = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    
    print("\nConfusion Matrix:")
    print(f"True Positives: {true_pos}, False Positives: {false_pos}")
    print(f"True Negatives: {true_neg}, False Negatives: {false_neg}")
    
    return accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Train PatchPairVul model')
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Directory containing processed data')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--use_attention', action='store_true',
                        help='Use graph attention networks instead of GraphSAGE')
    parser.add_argument('--debug', action='store_true',
                        help='Enable additional debugging output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    pre_patch_path = os.path.join(args.data_dir, 'pre_patch_graphs.pt')
    post_patch_path = os.path.join(args.data_dir, 'post_patch_graphs.pt')
    metadata_path = os.path.join(args.data_dir, 'metadata.pt')
    
    print(f"Loading dataset from {args.data_dir}")
    dataset = PatchPairDataset(pre_patch_path, post_patch_path, metadata_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Examine dataset before splitting
    print(f"\nExamining dataset before split:")
    # Check the first few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        print(f"  Label: {sample['vuln'].y.item()}")
        
        # Safely inspect graph structures by printing available keys first
        print(f"  Pre-patch graph available attributes: {list(sample['pre_patch'].keys())}")
        print(f"  Post-patch graph available attributes: {list(sample['post_patch'].keys())}")
        
        # Print node and edge information if available
        if hasattr(sample['pre_patch'], 'num_nodes'):
            print(f"  Pre-patch graph: nodes={sample['pre_patch'].num_nodes}")
        if hasattr(sample['pre_patch'], 'edge_index') and sample['pre_patch'].edge_index is not None:
            print(f"  Pre-patch graph: edges={sample['pre_patch'].edge_index.size(1) if sample['pre_patch'].edge_index.numel() > 0 else 0}")
            
        if hasattr(sample['post_patch'], 'num_nodes'):
            print(f"  Post-patch graph: nodes={sample['post_patch'].num_nodes}")
        if hasattr(sample['post_patch'], 'edge_index') and sample['post_patch'].edge_index is not None:
            print(f"  Post-patch graph: edges={sample['post_patch'].edge_index.size(1) if sample['post_patch'].edge_index.numel() > 0 else 0}")
    
    # Check label distribution
    labels = [dataset[i]['vuln'].y.item() for i in range(len(dataset))]
    positive_count = sum(labels)
    print(f"Label distribution: {positive_count}/{len(labels)} positive ({positive_count/len(labels)*100:.2f}%)")
    
    # Check for potential data problems
    if positive_count == 0 or positive_count == len(labels):
        print("WARNING: All samples have the same label! This will cause learning problems.")
    
    # Split dataset with fixed seed for reproducibility
    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    # Use a generator with fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Verify split label distributions
    train_labels = [train_dataset[i]['vuln'].y.item() for i in range(len(train_dataset))]
    val_labels = [val_dataset[i]['vuln'].y.item() for i in range(len(val_dataset))]
    train_positive = sum(train_labels)
    val_positive = sum(val_labels)
    
    print(f"Train set: {train_positive}/{len(train_labels)} positive ({train_positive/len(train_labels)*100:.2f}%)")
    print(f"Val set: {val_positive}/{len(val_labels)} positive ({val_positive/len(val_labels)*100:.2f}%)")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Determine input dimension from the first sample
    sample = dataset[0]
    # Safely determine input dimension
    print("Examining 'vuln' graph structure:")
    print(f"  Available keys in vuln: {list(sample['vuln'].keys())}")
    
    # Check if x attribute exists and get its dimension
    if hasattr(sample['vuln'], 'x') and sample['vuln'].x is not None:
        input_dim = sample['vuln'].x.size(1)
        print(f"Node feature dimension from vuln.x: {input_dim}")
    else:
        # Fallback to a default value or inspect other attributes
        print("WARNING: vuln.x not found, trying to determine input dimension from other sources")
        
        # Try to find feature dimensions from other graphs in the sample
        if hasattr(sample['pre_patch'], 'x') and sample['pre_patch'].x is not None:
            input_dim = sample['pre_patch'].x.size(1)
            print(f"Using node feature dimension from pre_patch.x: {input_dim}")
        elif hasattr(sample['post_patch'], 'x') and sample['post_patch'].x is not None:
            input_dim = sample['post_patch'].x.size(1)
            print(f"Using node feature dimension from post_patch.x: {input_dim}")
        else:
            # Default value as last resort
            input_dim = 20  # Using the value from the error message
            print(f"No feature dimensions found in graphs, using default: {input_dim}")
    
    print(f"Node feature dimension: {input_dim}")
    
    # Create model
    model = PatchPairVul(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        edge_types=["AST", "DDG", "CFG"],  # Using edge types from GraphSPD
        use_attention=args.use_attention
    ).to(device)
    
    # Debug model architecture
    if args.debug:
        print("\nModel architecture:")
        print(model)
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Examine model and dataset before training
    if args.debug:
        # Check the model output for a batch to understand prediction distribution
        sample_batch = next(iter(train_loader))
        sample_batch = sample_batch.to(device)
        model.eval()  # Set to eval mode temporarily
        with torch.no_grad():
            sample_output = model(sample_batch)
            sample_target = sample_batch['vuln'].y
            
            print("\nModel debug information:")
            print(f"Model output shape: {sample_output.shape}")
            print(f"Sample predictions (first 5): {sample_output[:5].squeeze().cpu().tolist()}")
            print(f"Sample targets (first 5): {sample_target[:5].cpu().tolist()}")
            
            # Check for constant predictions
            if torch.allclose(sample_output, sample_output[0]):
                print("WARNING: Model is producing identical outputs for all samples!")
        
        model.train()  # Set back to train mode
    
    # Start training
    print(f"\nStarting training for {args.epochs} epochs")
    start_time = time.time()
    
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train(model, train_loader, optimizer, device)
        
        # Evaluate
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f'Epoch {epoch}/{args.epochs} [{epoch_time:.2f}s] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, '
              f'Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            print(f'Saved new best model (F1: {val_f1:.4f})')
            best_val_f1 = val_f1
            best_epoch = epoch
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best model saved at epoch {best_epoch} with F1 score {best_val_f1:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_model.pt'))
    print('Final model saved')

if __name__ == "__main__":
    main()