"""
Metrics Utility Module

This module provides functions for calculating and tracking metrics for evaluating
the vulnerability detection system's performance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

import torch
from torch_geometric.data import HeteroData

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MetricsTracker:
    """
    Class for tracking and calculating various evaluation metrics.
    """
    
    def __init__(self):
        """
        Initialize the metrics tracker.
        """
        self.reset()
    
    def reset(self) -> None:
        """
        Reset all tracked metrics.
        """
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        self.losses = []
        self.predictions = []
        self.ground_truth = []
        
        self.epoch_metrics = []
    
    def update(
        self, 
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        loss: float = None,
        threshold: float = 0.5
    ) -> None:
        """
        Update metrics with new batch of predictions and ground truth.
        
        Args:
            predictions: Predicted vulnerability scores
            ground_truth: Ground truth labels
            loss: Loss value for this batch (optional)
            threshold: Classification threshold for binary predictions
        """
        # Convert to numpy arrays
        pred_np = predictions.detach().cpu().numpy()
        gt_np = ground_truth.detach().cpu().numpy()
        
        # Store for later calculations
        self.predictions.extend(pred_np.flatten())
        self.ground_truth.extend(gt_np.flatten())
        
        # Add loss if provided
        if loss is not None:
            self.losses.append(loss)
        
        # Calculate binary predictions using threshold
        binary_preds = (pred_np > threshold).astype(int)
        binary_gt = (gt_np > threshold).astype(int)
        
        # Update confusion matrix counters
        self.true_positives += np.sum((binary_preds == 1) & (binary_gt == 1))
        self.false_positives += np.sum((binary_preds == 1) & (binary_gt == 0))
        self.true_negatives += np.sum((binary_preds == 0) & (binary_gt == 0))
        self.false_negatives += np.sum((binary_preds == 0) & (binary_gt == 1))
    
    def get_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate and return current metrics.
        
        Args:
            threshold: Classification threshold for binary predictions
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Calculate average loss if available
        if self.losses:
            metrics['loss'] = np.mean(self.losses)
        
        # Skip if there are no predictions
        if not self.predictions:
            return metrics
        
        # Convert to numpy arrays
        y_pred = np.array(self.predictions)
        y_true = np.array(self.ground_truth)
        
        # Calculate binary predictions using threshold
        binary_preds = (y_pred > threshold).astype(int)
        binary_gt = (y_true > threshold).astype(int)
        
        # Calculate metrics
        metrics['accuracy'] = accuracy_score(binary_gt, binary_preds)
        
        # Handle potential division by zero
        if np.sum(binary_preds) > 0:
            metrics['precision'] = precision_score(binary_gt, binary_preds, zero_division=0)
        else:
            metrics['precision'] = 0.0
        
        if np.sum(binary_gt) > 0:
            metrics['recall'] = recall_score(binary_gt, binary_preds, zero_division=0)
        else:
            metrics['recall'] = 0.0
        
        metrics['f1'] = f1_score(binary_gt, binary_preds, zero_division=0)
        
        # Calculate ROC AUC if there are both positive and negative samples
        if len(np.unique(binary_gt)) > 1:
            metrics['roc_auc'] = roc_auc_score(binary_gt, y_pred)
            
            # Calculate PR AUC
            precision, recall, _ = precision_recall_curve(binary_gt, y_pred)
            metrics['pr_auc'] = auc(recall, precision)
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
        # Calculate confusion matrix metrics
        tp = self.true_positives
        fp = self.false_positives
        tn = self.true_negatives
        fn = self.false_negatives
        
        # Precision and recall from confusion matrix
        if tp + fp > 0:
            metrics['precision_cm'] = tp / (tp + fp)
        else:
            metrics['precision_cm'] = 0.0
        
        if tp + fn > 0:
            metrics['recall_cm'] = tp / (tp + fn)
        else:
            metrics['recall_cm'] = 0.0
        
        # F1 score from confusion matrix
        if metrics['precision_cm'] + metrics['recall_cm'] > 0:
            metrics['f1_cm'] = 2 * (metrics['precision_cm'] * metrics['recall_cm']) / (metrics['precision_cm'] + metrics['recall_cm'])
        else:
            metrics['f1_cm'] = 0.0
        
        # Calculate specificity and fall-out
        if tn + fp > 0:
            metrics['specificity'] = tn / (tn + fp)
            metrics['fallout'] = fp / (tn + fp)
        else:
            metrics['specificity'] = 0.0
            metrics['fallout'] = 0.0
        
        return metrics
    
    def end_epoch(self) -> Dict[str, float]:
        """
        End the current epoch and store the metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        # Calculate metrics for the epoch
        metrics = self.get_metrics()
        
        # Store metrics
        self.epoch_metrics.append(metrics)
        
        # Reset metrics for the next epoch (except epoch_metrics)
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
        self.losses = []
        self.predictions = []
        self.ground_truth = []
        
        return metrics
    
    def get_epoch_metrics(self, epoch: int = -1) -> Dict[str, float]:
        """
        Get metrics for a specific epoch.
        
        Args:
            epoch: Epoch index (default: last epoch)
            
        Returns:
            Dictionary of metric names and values
        """
        if not self.epoch_metrics:
            return {}
        
        if epoch < 0:
            # Get the last epoch
            return self.epoch_metrics[epoch]
        
        if epoch >= len(self.epoch_metrics):
            logger.warning(f"Requested epoch {epoch} but only {len(self.epoch_metrics)} epochs available")
            return self.epoch_metrics[-1]
        
        return self.epoch_metrics[epoch]
    
    def get_all_epoch_metrics(self) -> Dict[str, List[float]]:
        """
        Get all metrics across all epochs.
        
        Returns:
            Dictionary mapping metric names to lists of values
        """
        if not self.epoch_metrics:
            return {}
        
        # Initialize result with empty lists
        result = {metric: [] for metric in self.epoch_metrics[0].keys()}
        
        # Add values for each epoch
        for epoch_metric in self.epoch_metrics:
            for metric, value in epoch_metric.items():
                result[metric].append(value)
        
        return result
    
    def get_best_epoch(self, metric: str = 'f1') -> Tuple[int, Dict[str, float]]:
        """
        Get the epoch with the best value for a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (best epoch index, metrics dictionary)
        """
        if not self.epoch_metrics:
            return -1, {}
        
        # Check if the metric exists
        if metric not in self.epoch_metrics[0]:
            logger.warning(f"Metric '{metric}' not found in epoch metrics. Using 'loss' instead.")
            metric = 'loss' if 'loss' in self.epoch_metrics[0] else list(self.epoch_metrics[0].keys())[0]
        
        # Find the best epoch
        if metric == 'loss':
            # For loss, lower is better
            best_epoch = np.argmin([m.get(metric, float('inf')) for m in self.epoch_metrics])
        else:
            # For other metrics, higher is better
            best_epoch = np.argmax([m.get(metric, -float('inf')) for m in self.epoch_metrics])
        
        return best_epoch, self.epoch_metrics[best_epoch]

def calculate_node_classification_metrics(
    node_predictions: torch.Tensor,
    node_labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate metrics for node classification.
    
    Args:
        node_predictions: Predicted vulnerability scores for nodes
        node_labels: Ground truth labels for nodes
        threshold: Classification threshold for binary predictions
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy arrays
    pred_np = node_predictions.detach().cpu().numpy()
    gt_np = node_labels.detach().cpu().numpy()
    
    # Calculate binary predictions using threshold
    binary_preds = (pred_np > threshold).astype(int)
    binary_gt = (gt_np > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(binary_gt, binary_preds),
        'precision': precision_score(binary_gt, binary_preds, zero_division=0),
        'recall': recall_score(binary_gt, binary_preds, zero_division=0),
        'f1': f1_score(binary_gt, binary_preds, zero_division=0)
    }
    
    # Calculate ROC AUC if there are both positive and negative samples
    if len(np.unique(binary_gt)) > 1:
        metrics['roc_auc'] = roc_auc_score(binary_gt, pred_np)
        
        # Calculate PR AUC
        precision, recall, _ = precision_recall_curve(binary_gt, pred_np)
        metrics['pr_auc'] = auc(recall, precision)
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics

def calculate_pattern_metrics(
    pattern_similarity: torch.Tensor,
    pattern_alignment: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate metrics for pattern learning.
    
    Args:
        pattern_similarity: Pattern similarity matrix
        pattern_alignment: Ground truth pattern alignment matrix
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy arrays
    sim_np = pattern_similarity.detach().cpu().numpy()
    align_np = pattern_alignment.detach().cpu().numpy()
    
    # Calculate mean similarity for aligned patterns
    aligned_sim = sim_np[align_np > 0]
    non_aligned_sim = sim_np[align_np == 0]
    
    metrics = {
        'aligned_similarity_mean': np.mean(aligned_sim) if len(aligned_sim) > 0 else 0.0,
        'non_aligned_similarity_mean': np.mean(non_aligned_sim) if len(non_aligned_sim) > 0 else 0.0,
        'aligned_similarity_std': np.std(aligned_sim) if len(aligned_sim) > 0 else 0.0,
        'non_aligned_similarity_std': np.std(non_aligned_sim) if len(non_aligned_sim) > 0 else 0.0
    }
    
    # Calculate separation metric
    metrics['separation'] = metrics['aligned_similarity_mean'] - metrics['non_aligned_similarity_mean']
    
    # Calculate alignment accuracy
    # For each row, check if the highest similarity is for the aligned pattern
    row_max_indices = np.argmax(sim_np, axis=1)
    col_max_indices = np.argmax(sim_np, axis=0)
    
    # For each vulnerability pattern (row), get the index of the aligned security pattern
    row_aligned_indices = [np.where(align_np[i, :] > 0)[0] for i in range(align_np.shape[0])]
    
    # Calculate row accuracy
    row_correct = 0
    for i in range(len(row_max_indices)):
        if len(row_aligned_indices[i]) > 0 and row_max_indices[i] in row_aligned_indices[i]:
            row_correct += 1
    
    metrics['row_alignment_accuracy'] = row_correct / len(row_max_indices) if len(row_max_indices) > 0 else 0.0
    
    return metrics