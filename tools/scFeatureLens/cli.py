"""
Command-line interface for scFeatureLens.
"""

import argparse
import yaml
from pathlib import Path
from .pipeline import SCFeatureLensPipeline, AnalysisConfig


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="scFeatureLens: Single-Cell Feature Lens for Mechanistic Interpretability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Geneformer activations
  python -m scFeatureLens.cli sae_geneformer_activations.pt

  # Run with custom config
  python -m scFeatureLens.cli embeddings.pt --config my_config.yaml
  
  # Quick analysis with default settings
  python -m scFeatureLens.cli embeddings.pt --output-dir results --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "embeddings_path",
        type=str,
        help="Path to embeddings file (.pt, .npy, .csv, or .h5ad)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--sae-model-path",
        type=str,
        help="Path to pre-trained SAE model file"
    )
    
    parser.add_argument(
        "--no-train-sae",
        action="store_true",
        help="Don't train SAE, use existing model"
    )
    
    parser.add_argument(
        "--gene-expression-data",
        type=str,
        help="Path to gene expression data for DEG analysis"
    )
    
    parser.add_argument(
        "--go-category",
        type=str,
        choices=["biological_process", "molecular_function", "cellular_component"],
        default="biological_process",
        help="GO category for enrichment analysis (default: biological_process)"
    )
    
    parser.add_argument(
        "--min-active-samples",
        type=int,
        default=100,
        help="Minimum number of active samples per feature (default: 100)"
    )
    
    parser.add_argument(
        "--activation-percentile",
        type=float,
        default=99,
        help="Percentile for high activation threshold (default: 99)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computing device: auto, cpu, cuda, or cuda:X (default: auto)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="scFeatureLens 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = AnalysisConfig(**config_dict)
    else:
        config = AnalysisConfig(embeddings_path=args.embeddings_path)
    
    # Override config with command-line arguments
    if args.output_dir != "results":
        config.output_dir = args.output_dir
    if args.sae_model_path:
        config.sae_model_path = args.sae_model_path
    if args.no_train_sae:
        config.train_sae = False
    if args.gene_expression_data:
        config.gene_expression_data_path = args.gene_expression_data
    if args.go_category != "biological_process":
        config.go_category = args.go_category
    if args.min_active_samples != 100:
        config.min_active_samples = args.min_active_samples
    if args.activation_percentile != 99:
        config.activation_percentile = args.activation_percentile
    if args.device != "auto":
        config.device = args.device
    if args.verbose:
        config.verbose = True
    
    print("scFeatureLens: Single-Cell Feature Lens for Mechanistic Interpretability")
    print("=" * 70)
    print(f"Input: {config.embeddings_path}")
    print(f"Output: {config.output_dir}")
    print(f"Device: {config.device}")
    print("=" * 70)
    
    # Run analysis
    pipeline = SCFeatureLensPipeline(config)
    pipeline.save_config()
    pipeline.run_analysis()
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
