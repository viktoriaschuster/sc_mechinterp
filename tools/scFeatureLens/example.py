#!/usr/bin/env python3
"""
Example usage of scFeatureLens for single-cell mechanistic interpretability analysis.

This script demonstrates how to use the scFeatureLens pipeline to:
1. Load cell embeddings from a foundation model
2. Train a sparse autoencoder to extract interpretable features
3. Perform differential gene expression analysis on feature-active cells
4. Run gene set enrichment analysis to interpret features

Author: Viktoria Schuster
"""

import os
import numpy as np
import pandas as pd
import torch
import anndata as ad
from pathlib import Path

# Import scFeatureLens components
from .pipeline import SCFeatureLensPipeline, AnalysisConfig
from .sae import SparseAutoencoder


def create_example_data(n_cells: int = 1000, n_genes: int = 2000, embedding_dim: int = 512, output_dir: str = "example_data"):
    """
    Create synthetic example data for testing scFeatureLens.
    
    Args:
        n_cells: Number of cells to simulate
        n_genes: Number of genes to simulate  
        embedding_dim: Dimensionality of cell embeddings
        output_dir: Directory to save example data
    """
    print("Creating synthetic example data...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic cell embeddings (e.g., from Geneformer)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create embeddings with some structure
    # Simulate 3 cell types with different patterns
    n_per_type = n_cells // 3
    embeddings = []
    cell_types = []
    
    for i in range(3):
        # Each cell type has a different mean embedding
        mean_embedding = np.random.randn(embedding_dim) * 0.5
        type_embeddings = np.random.randn(n_per_type, embedding_dim) * 0.3 + mean_embedding
        embeddings.append(type_embeddings)
        cell_types.extend([f"CellType_{i+1}"] * n_per_type)
    
    # Handle remainder
    remainder = n_cells - 3 * n_per_type
    if remainder > 0:
        mean_embedding = np.random.randn(embedding_dim) * 0.5
        type_embeddings = np.random.randn(remainder, embedding_dim) * 0.3 + mean_embedding
        embeddings.append(type_embeddings)
        cell_types.extend([f"CellType_1"] * remainder)
    
    embeddings = np.vstack(embeddings)
    
    # Save embeddings
    embeddings_path = Path(output_dir) / "cell_embeddings.npy"
    np.save(embeddings_path, embeddings.astype(np.float32))
    print(f"Saved embeddings to {embeddings_path}")
    
    # Generate synthetic gene expression data
    # Create expression matrix with some correlation to cell types
    gene_names = [f"Gene_{i+1:04d}" for i in range(n_genes)]
    
    # Simulate count data with different expression patterns per cell type
    expression_data = []
    for i, cell_type in enumerate(cell_types):
        type_idx = int(cell_type.split("_")[1]) - 1
        
        # Base expression levels
        base_expression = np.random.negative_binomial(n=5, p=0.3, size=n_genes)
        
        # Add cell-type specific effects for some genes
        n_specific_genes = n_genes // 10  # 10% of genes are cell-type specific
        specific_genes = np.random.choice(n_genes, n_specific_genes, replace=False)
        for gene_idx in specific_genes:
            if (gene_idx + type_idx) % 3 == 0:  # Different patterns per type
                base_expression[gene_idx] *= np.random.uniform(2, 5)  # Upregulated
        
        expression_data.append(base_expression)
    
    expression_matrix = np.vstack(expression_data)
    
    # Create AnnData object
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({
            'cell_type': cell_types,
            'n_counts': expression_matrix.sum(axis=1),
            'n_genes': (expression_matrix > 0).sum(axis=1)
        }),
        var=pd.DataFrame({
            'gene_name': gene_names,
            'n_cells': (expression_matrix > 0).sum(axis=0)
        })
    )
    
    # Save gene expression data
    expression_path = Path(output_dir) / "gene_expression.h5ad"
    adata.write_h5ad(expression_path)
    print(f"Saved gene expression data to {expression_path}")
    
    # Create a simple custom gene set file
    gene_sets_path = Path(output_dir) / "custom_gene_sets.yaml"
    custom_gene_sets = {
        "CellType1_Markers": gene_names[:50],  # First 50 genes
        "CellType2_Markers": gene_names[50:100],  # Next 50 genes
        "CellType3_Markers": gene_names[100:150],  # Next 50 genes
        "Housekeeping_Genes": gene_names[1500:1600],  # Some housekeeping genes
        "Stress_Response": gene_names[1600:1650],  # Stress response genes
    }
    
    import yaml
    with open(gene_sets_path, 'w') as f:
        yaml.dump(custom_gene_sets, f)
    print(f"Saved custom gene sets to {gene_sets_path}")
    
    return {
        'embeddings_path': str(embeddings_path),
        'expression_path': str(expression_path),
        'gene_sets_path': str(gene_sets_path),
        'n_cells': n_cells,
        'n_genes': n_genes,
        'embedding_dim': embedding_dim
    }


def run_basic_example():
    """
    Run a basic example of the scFeatureLens pipeline.
    """
    print("=" * 60)
    print("scFeatureLens Basic Example")
    print("=" * 60)
    
    # Create example data
    data_info = create_example_data(n_cells=500, n_genes=1000, embedding_dim=256)
    
    # Configure the analysis
    config = AnalysisConfig(
        # Input data
        embeddings_path=data_info['embeddings_path'],
        gene_expression_data_path=data_info['expression_path'],
        output_dir="example_results",
        
        # SAE parameters
        sae_hidden_size=1000,  # Sparse hidden layer
        sae_l1_penalty=1e-3,
        sae_learning_rate=1e-4,
        sae_epochs=100,  # Reduced for example
        train_sae=True,
        
        # Feature selection
        activation_percentile=95,  # Top 5% most active features
        min_active_samples=20,  # Need at least 20 cells to be active
        
        # Differential expression
        deg_p_threshold=1e-3,
        deg_fold_change_threshold=1.5,
        
        # Gene set analysis
        gene_sets_path=data_info['gene_sets_path'],
        gene_set_type="custom",
        min_genes_per_set=5,
        max_genes_per_set=200,
        
        # Computational
        device="auto",
        batch_size=64,
        verbose=True,
        random_seed=42
    )
    
    # Initialize pipeline
    pipeline = SCFeatureLensPipeline(config)
    
    # Run the full analysis
    results = pipeline.run_analysis()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    # Print summary of results
    if results:
        print(f"Results saved to: {config.output_dir}")
        print(f"Number of features analyzed: {len(results.get('feature_results', []))}")
        print(f"Number of significant DEGs found: {sum(len(r.get('significant_degs', [])) for r in results.get('feature_results', []))}")
        print(f"Number of enriched gene sets: {sum(len(r.get('enriched_sets', [])) for r in results.get('feature_results', []))}")
    
    return results


def run_pretrained_example():
    """
    Example using a pre-trained SAE model.
    """
    print("=" * 60)
    print("scFeatureLens Pre-trained Model Example")
    print("=" * 60)
    
    # Create example data
    data_info = create_example_data(n_cells=300, n_genes=800, embedding_dim=256)
    
    # First, train and save an SAE model
    print("Training SAE model...")
    config_train = AnalysisConfig(
        embeddings_path=data_info['embeddings_path'],
        output_dir="example_sae_training",
        sae_hidden_size=800,
        sae_epochs=50,
        train_sae=True,
        verbose=False
    )
    
    pipeline_train = SCFeatureLensPipeline(config_train)
    pipeline_train.load_embeddings()
    sae_model = pipeline_train.train_sae()
    
    # Save the trained model
    sae_model_path = Path(config_train.output_dir) / "trained_sae.pt"
    torch.save(sae_model.state_dict(), sae_model_path)
    print(f"Saved trained SAE to: {sae_model_path}")
    
    # Now use the pre-trained model for analysis
    config_analysis = AnalysisConfig(
        embeddings_path=data_info['embeddings_path'],
        gene_expression_data_path=data_info['expression_path'],
        sae_model_path=str(sae_model_path),
        output_dir="example_pretrained_results",
        
        # Use pre-trained model
        train_sae=False,
        sae_hidden_size=800,  # Must match the saved model
        
        # Analysis parameters
        activation_percentile=90,
        min_active_samples=15,
        deg_p_threshold=1e-2,
        
        # Gene sets
        gene_sets_path=data_info['gene_sets_path'],
        gene_set_type="custom",
        
        verbose=True
    )
    
    pipeline_analysis = SCFeatureLensPipeline(config_analysis)
    results = pipeline_analysis.run_analysis()
    
    print("Pre-trained model analysis complete!")
    return results


def run_go_enrichment_example():
    """
    Example using GO (Gene Ontology) enrichment analysis.
    Note: This requires GO data files to be available.
    """
    print("=" * 60)
    print("scFeatureLens GO Enrichment Example")
    print("=" * 60)
    
    # Create example data with real gene names (subset of human genes)
    real_gene_names = [
        "ACTB", "GAPDH", "TUBB", "RPL13A", "RPS18", "EEF1A1", "PPIA", "LDHA",
        "ALDOA", "ENO1", "PGAM1", "TPI1", "GAPDHS", "PGK1", "PFKM", "HK1",
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "IL2RA", "FOXP3",
        "GZMB", "PRF1", "IFNG", "TNF", "IL2", "IL4", "IL5", "IL13",
        "MS4A1", "CD19", "CD20", "CD79A", "CD79B", "IGHM", "IGHA1", "IGLC1",
        "ALB", "AFP", "APOE", "TTR", "RBP4", "SERPINA1", "FGA", "FGB",
        "COL1A1", "COL1A2", "COL3A1", "ELN", "FBN1", "LAMA1", "LAMB1", "LAMC1"
    ]
    
    # Extend with synthetic names if needed
    n_genes = 500
    if len(real_gene_names) < n_genes:
        synthetic_names = [f"Gene_{i+1:04d}" for i in range(len(real_gene_names), n_genes)]
        gene_names = real_gene_names + synthetic_names
    else:
        gene_names = real_gene_names[:n_genes]
    
    # Create expression data with these gene names
    output_dir = "example_go_data"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate embeddings
    n_cells = 400
    embedding_dim = 256
    embeddings = np.random.randn(n_cells, embedding_dim).astype(np.float32)
    embeddings_path = Path(output_dir) / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    
    # Generate expression data
    expression_matrix = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({
            'n_counts': expression_matrix.sum(axis=1),
            'n_genes': (expression_matrix > 0).sum(axis=1)
        }),
        var=pd.DataFrame({'gene_name': gene_names})
    )
    
    expression_path = Path(output_dir) / "expression.h5ad"
    adata.write_h5ad(expression_path)
    
    # Configure for GO analysis
    config = AnalysisConfig(
        embeddings_path=str(embeddings_path),
        gene_expression_data_path=str(expression_path),
        output_dir="example_go_results",
        
        # SAE parameters
        sae_hidden_size=600,
        sae_epochs=30,
        train_sae=True,
        
        # GO analysis (Note: requires GO data files)
        gene_set_type="go",
        go_category="biological_process",
        go_obo_path="data/go-basic.obo",  # Download from http://geneontology.org/
        go_gaf_path="data/goa_human.gaf",  # Download from ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/
        min_genes_per_set=5,
        max_genes_per_set=200,
        
        # Analysis parameters
        activation_percentile=95,
        min_active_samples=20,
        
        verbose=True
    )
    
    # Check if GO files exist
    if not (Path(config.go_obo_path).exists() and Path(config.go_gaf_path).exists()):
        print("WARNING: GO data files not found. Please download:")
        print("1. go-basic.obo from http://geneontology.org/docs/download-ontology/")
        print("2. goa_human.gaf from ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/")
        print("Falling back to custom gene sets...")
        
        # Create simple custom gene sets as fallback
        custom_sets = {
            "Metabolic_Process": gene_names[:30],
            "Immune_Response": [g for g in gene_names if any(marker in g for marker in ["CD", "IL", "TNF", "IFN"])],
            "Structural": [g for g in gene_names if any(marker in g for marker in ["COL", "ELN", "FBN", "LAM"])],
        }
        
        import yaml
        custom_path = Path(output_dir) / "fallback_sets.yaml"
        with open(custom_path, 'w') as f:
            yaml.dump(custom_sets, f)
        
        config.gene_set_type = "custom"
        config.gene_sets_path = str(custom_path)
    
    pipeline = SCFeatureLensPipeline(config)
    results = pipeline.run_analysis()
    
    print("GO enrichment analysis complete!")
    return results


def main():
    """Main function to run examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="scFeatureLens Examples")
    parser.add_argument(
        "--example", 
        choices=["basic", "pretrained", "go"], 
        default="basic",
        help="Which example to run"
    )
    
    args = parser.parse_args()
    
    if args.example == "basic":
        results = run_basic_example()
    elif args.example == "pretrained":
        results = run_pretrained_example()
    elif args.example == "go":
        results = run_go_enrichment_example()
    
    print("\nExample completed successfully!")
    print("Check the output directories for detailed results.")


if __name__ == "__main__":
    main()