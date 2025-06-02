#!/usr/bin/env python3
"""
Example script demonstrating how to use the SC MechInterp tool
with the provided Geneformer embeddings.
"""

import torch
import numpy as np
from pathlib import Path
from sc_mechinterp import SCMechInterpPipeline, AnalysisConfig

def run_geneformer_example():
    """Run analysis on the provided Geneformer activations."""
    
    # Check if the Geneformer activations file exists
    activations_path = "sae_geneformer_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt"
    if not Path(activations_path).exists():
        print(f"Error: {activations_path} not found!")
        print("Make sure you're running this script from the correct directory.")
        return
    
    print("Loading Geneformer SAE activations...")
    activations = torch.load(activations_path)
    print(f"Loaded activations with shape: {activations.shape}")
    
    # Since we already have activations, we'll create a simplified config
    # that skips the SAE training step
    config = AnalysisConfig(
        embeddings_path=activations_path,  # We'll treat activations as embeddings
        output_dir="geneformer_results",
        train_sae=False,  # Skip SAE training since we have activations
        sae_model_path=None,  # Not needed since we're not training
        activation_percentile=99,
        min_active_samples=100,
        deg_p_threshold=1e-5,
        deg_fold_change_threshold=2.0,
        go_category="biological_process",
        verbose=True,
        random_seed=42
    )
    
    # Create pipeline but modify it to work with pre-computed activations
    pipeline = SCMechInterpPipeline(config)
    
    # Manually set the activations since we already have them
    pipeline.activations = activations
    pipeline.logger.info(f"Using pre-computed activations with shape: {activations.shape}")
    
    # Select active features
    from analysis_functions import filter_active_features
    
    active_features = filter_active_features(
        activations,
        min_active_samples=config.min_active_samples,
        max_active_samples=config.max_active_samples
    )
    
    pipeline.logger.info(f"Found {len(active_features)} active features")
    
    # For this example, let's analyze just the first few features
    n_features_to_analyze = min(5, len(active_features))
    features_to_analyze = active_features[:n_features_to_analyze]
    
    pipeline.logger.info(f"Analyzing first {n_features_to_analyze} features as example")
    
    # Example: Show feature activation statistics
    for i, feat_idx in enumerate(features_to_analyze):
        feat_activations = activations[:, feat_idx]
        n_active = (feat_activations > 0).sum().item()
        mean_activation = feat_activations[feat_activations > 0].mean().item()
        max_activation = feat_activations.max().item()
        
        print(f"Feature {feat_idx}: {n_active} active samples, "
              f"mean activation: {mean_activation:.4f}, max: {max_activation:.4f}")
    
    # Save feature statistics
    feature_stats = []
    for feat_idx in active_features:
        feat_activations = activations[:, feat_idx]
        stats = {
            'feature_id': feat_idx.item(),
            'n_active_samples': (feat_activations > 0).sum().item(),
            'mean_activation': feat_activations[feat_activations > 0].mean().item() if (feat_activations > 0).sum() > 0 else 0,
            'max_activation': feat_activations.max().item(),
            'std_activation': feat_activations[feat_activations > 0].std().item() if (feat_activations > 0).sum() > 1 else 0
        }
        feature_stats.append(stats)
    
    import pandas as pd
    feature_stats_df = pd.DataFrame(feature_stats)
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    feature_stats_df.to_csv(output_dir / "feature_statistics.csv", index=False)
    pipeline.logger.info(f"Feature statistics saved to {output_dir / 'feature_statistics.csv'}")
    
    # Save configuration
    pipeline.save_config()
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total features: {activations.shape[1]}")
    print(f"Active features: {len(active_features)}")
    print(f"Total samples: {activations.shape[0]}")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return pipeline, active_features, feature_stats_df


def create_minimal_working_example():
    """Create a minimal example with synthetic data."""
    
    print("Creating minimal working example with synthetic data...")
    
    # Create synthetic embeddings
    n_samples = 1000
    n_features = 100
    
    # Generate embeddings with some structure
    np.random.seed(42)
    torch.manual_seed(42)
    
    embeddings = torch.randn(n_samples, n_features)
    
    # Add some structured patterns
    embeddings[:200, :10] += 2.0  # First 10 features active in first 200 samples
    embeddings[300:500, 10:20] += 1.5  # Next 10 features active in samples 300-500
    
    # Save synthetic embeddings
    torch.save(embeddings, "synthetic_embeddings.pt")
    
    # Create configuration
    config = AnalysisConfig(
        embeddings_path="synthetic_embeddings.pt",
        output_dir="synthetic_results",
        train_sae=True,
        sae_hidden_size=50,  # Smaller for this example
        sae_epochs=100,  # Fewer epochs for this example
        activation_percentile=95,  # Lower percentile for more active features
        min_active_samples=50,
        verbose=True
    )
    
    # Run pipeline
    pipeline = SCMechInterpPipeline(config)
    
    # Load embeddings
    pipeline.load_embeddings()
    
    # Train SAE
    pipeline.train_sae()
    
    # Extract activations
    pipeline.extract_activations()
    
    # Select features
    active_features = pipeline.select_active_features()
    
    print(f"Found {len(active_features)} active features in synthetic data")
    
    # Save results
    pipeline.save_config()
    
    return pipeline


if __name__ == "__main__":
    print("SC MechInterp Tool - Example Usage")
    print("="*50)
    
    # Try to run with Geneformer data first
    try:
        pipeline, active_features, stats = run_geneformer_example()
        print("Successfully completed Geneformer example!")
    except FileNotFoundError:
        print("Geneformer data not found, running synthetic example instead...")
        pipeline = create_minimal_working_example()
        print("Successfully completed synthetic example!")
