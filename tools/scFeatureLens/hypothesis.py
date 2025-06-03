"""
Hypothesis generation and testing with SAE features.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from scipy import stats


def predict_embedding_change(
    embeddings: torch.Tensor,
    cell_indices: Union[List[int], np.ndarray],
    feature_idx: int,
    sae_model: torch.nn.Module,
    activations: Optional[torch.Tensor] = None,
    target_activation: Optional[float] = None,
    target_percentile: Optional[float] = None,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Predict embedding changes for a given set of cells when perturbing a specific SAE feature.
    
    Args:
        embeddings: Original embeddings tensor
        cell_indices: Indices of cells to perturb
        feature_idx: Index of the SAE feature to perturb
        sae_model: Trained SAE model
        activations: Pre-computed activations (optional)
        target_activation: Target activation value for the feature
        target_percentile: Target percentile for activation (alternative to target_activation)
        device: Computing device
        
    Returns:
        Perturbed embeddings for the selected cells
    """
    # Move tensors to device
    embeddings = embeddings.to(device)
    sae_model = sae_model.to(device)
    
    # Handle indices
    if isinstance(cell_indices, list):
        cell_indices = np.array(cell_indices)
        
    # Get embeddings for selected cells
    cell_embeddings = embeddings[cell_indices]
    
    # Compute activations if not provided
    if activations is None:
        with torch.no_grad():
            _, activs = sae_model(embeddings)
    else:
        activs = activations.to(device)
    
    # Get activations for the selected cells
    cell_activations = activs[cell_indices]
    
    # Determine target activation
    if target_activation is None and target_percentile is not None:
        target_activation = torch.quantile(
            activs[:, feature_idx], 
            target_percentile / 100.0
        ).item()
    elif target_activation is None:
        # Use average activation * 1.5 as default
        target_activation = activs[:, feature_idx].mean().item() * 1.5
        
    # Clone activations and modify the target feature
    perturbed_activations = cell_activations.clone()
    perturbed_activations[:, feature_idx] = target_activation
    
    # Compute perturbed embeddings
    with torch.no_grad():
        perturbed_embeddings = sae_model.decoder(perturbed_activations)
    
    return perturbed_embeddings


def get_differential_features(
    embeddings: torch.Tensor,
    cell_indices_1: Union[List[int], np.ndarray],
    cell_indices_2: Union[List[int], np.ndarray],
    sae_model: torch.nn.Module,
    activations: Optional[torch.Tensor] = None,
    n_top_features: int = 10,
    p_value_threshold: float = 0.01,
    min_effect_size: float = 0.5,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Find top SAE features with significant activation differences between two cell sets.
    
    Args:
        embeddings: Embeddings tensor
        cell_indices_1: Indices for first cell set
        cell_indices_2: Indices for second cell set
        sae_model: Trained SAE model
        activations: Pre-computed activations (optional)
        n_top_features: Number of top features to return
        p_value_threshold: Significance threshold for t-test
        min_effect_size: Minimum Cohen's d effect size threshold
        device: Computing device
        
    Returns:
        DataFrame with top differential features and statistics
    """
    # Move tensors to device
    embeddings = embeddings.to(device)
    sae_model = sae_model.to(device)
    
    # Convert indices to numpy arrays if they're lists
    if isinstance(cell_indices_1, list):
        cell_indices_1 = np.array(cell_indices_1)
    if isinstance(cell_indices_2, list):
        cell_indices_2 = np.array(cell_indices_2)
    
    # Compute activations if not provided
    if activations is None:
        with torch.no_grad():
            _, activs = sae_model(embeddings)
    else:
        activs = activations.to(device)
    
    # Move activations to CPU for numpy operations
    activs_np = activs.cpu().numpy()
    
    # Get activations for both cell sets
    activs_1 = activs_np[cell_indices_1]
    activs_2 = activs_np[cell_indices_2]
    
    # Calculate statistics for each feature
    results = []
    n_features = activs.shape[1]
    
    for i in range(n_features):
        # Calculate means
        mean_1 = np.mean(activs_1[:, i])
        mean_2 = np.mean(activs_2[:, i])
        diff = mean_2 - mean_1
        
        # T-test
        t_stat, p_val = stats.ttest_ind(activs_1[:, i], activs_2[:, i], equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(cell_indices_1) - 1) * np.var(activs_1[:, i]) + 
             (len(cell_indices_2) - 1) * np.var(activs_2[:, i])) / 
            (len(cell_indices_1) + len(cell_indices_2) - 2)
        )
        
        if pooled_std > 0:
            effect_size = abs(mean_1 - mean_2) / pooled_std
        else:
            effect_size = 0
            
        results.append({
            'feature': i,
            'mean_group1': mean_1,
            'mean_group2': mean_2,
            'diff': diff,
            'abs_diff': abs(diff),
            't_statistic': t_stat,
            'p_value': p_val,
            'effect_size': effect_size
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Filter by significance and effect size
    significant_df = results_df[
        (results_df['p_value'] < p_value_threshold) & 
        (results_df['effect_size'] > min_effect_size)
    ]
    
    # Sort by absolute difference and effect size
    significant_df = significant_df.sort_values(
        by=['abs_diff', 'effect_size'], 
        ascending=False
    )
    
    # Return top features
    return significant_df.head(n_top_features)


def get_feature_activation_stats(
    activations: torch.Tensor,
    feature_idx: int,
    cell_indices: Optional[Union[List[int], np.ndarray]] = None
) -> dict:
    """
    Get detailed statistics for a specific feature's activations.
    
    Args:
        activations: Feature activations tensor
        feature_idx: Index of the feature to analyze
        cell_indices: Optional subset of cells to analyze
        
    Returns:
        Dictionary with activation statistics
    """
    if cell_indices is not None:
        if isinstance(cell_indices, list):
            cell_indices = np.array(cell_indices)
        feature_acts = activations[cell_indices, feature_idx].cpu().numpy()
    else:
        feature_acts = activations[:, feature_idx].cpu().numpy()
    
    stats_dict = {
        'mean': np.mean(feature_acts),
        'median': np.median(feature_acts),
        'std': np.std(feature_acts),
        'min': np.min(feature_acts),
        'max': np.max(feature_acts),
        'percentile_25': np.percentile(feature_acts, 25),
        'percentile_75': np.percentile(feature_acts, 75),
        'percentile_95': np.percentile(feature_acts, 95),
        'percentile_99': np.percentile(feature_acts, 99),
        'fraction_active': np.mean(feature_acts > 0),
        'n_cells': len(feature_acts)
    }
    
    return stats_dict
