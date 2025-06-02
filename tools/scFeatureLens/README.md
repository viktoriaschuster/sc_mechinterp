# scFeatureLens

**Extract meaningful features from single-cell RNA-seq data model embeddings using sparse autoencoders and perform biological interpretation.**

scFeatureLens is a mechanistic interpretability tool designed to understand what foundation models like Geneformer learn from single-cell RNA-seq data. It uses sparse autoencoders to identify interpretable features in model embeddings and provides biological context through differential gene expression analysis and gene set enrichment.

## 🎯 What scFeatureLens Does

### Core Functionality
- **🧠 Sparse Autoencoder Training**: Train sparse autoencoders on embeddings to extract interpretable features
- **🔍 Feature Analysis**: Identify which features are active for different cell types or conditions
- **📊 Differential Expression**: Compare gene expression between feature-active and feature-inactive cells
- **🧬 Gene Set Enrichment**: Analyze enriched biological pathways and GO terms for each feature
- **📈 Biological Interpretation**: Understand what biological processes each latent feature represents

### Supported Input Formats
- **Model Embeddings**: `.pt`, `.npy`, `.csv`, `.h5ad` files
- **Foundation Models**: Geneformer, multiDGD, or any custom embeddings
- **Gene Expression Data**: AnnData (`.h5ad`) or CSV files for biological interpretation

## 🚀 Quick Start

### Basic Example
```bash
# Run with example data
python -m tools.scFeatureLens.example --example basic

# Analyze your own embeddings
python -m tools.scFeatureLens.cli your_embeddings.pt --output-dir results
```

### Python API
```python
from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    n_features=1000,
    sparsity_penalty=1e-3,
    n_epochs=100
)

# Run analysis
pipeline = SCFeatureLensPipeline(config)
results = pipeline.run_analysis("embeddings.pt", output_dir="results")
```

## 📋 Requirements

### Input Data
- **Embeddings**: Model embeddings from any foundation model (required)
- **Gene Expression**: Single-cell gene expression data (optional, for biological interpretation)
- **Gene Sets**: GO terms (automatically downloaded) or custom gene sets (optional)

### System Requirements
- Python 3.8+
- PyTorch
- 8GB+ RAM (16GB+ recommended for large datasets)
- GPU support (optional, but recommended for large models)

## 🔧 Installation & Setup

scFeatureLens is part of the sc_mechinterp toolkit. For installation instructions, see the main repository [setup guide](../../README.md#installation).

Quick setup:
```bash
# From the repository root
./setup_env.sh

# Verify installation
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Ready!')"
```

## 📖 Usage Guide

### Command Line Interface

#### Basic Usage
```bash
# Analyze embeddings with default settings
python -m tools.scFeatureLens.cli embeddings.pt

# Specify output directory
python -m tools.scFeatureLens.cli embeddings.pt --output-dir my_results

# Use custom configuration
python -m tools.scFeatureLens.cli embeddings.pt --config config.yaml
```

#### Advanced Options
```bash
# Full analysis with gene expression data
python -m tools.scFeatureLens.cli embeddings.pt \
    --gene-expression data.h5ad \
    --custom-gene-sets gene_sets.yaml \
    --output-dir results \
    --n-features 1000 \
    --sparsity-penalty 1e-3 \
    --verbose

# Get help
python -m tools.scFeatureLens.cli --help
```

### Python API

#### Basic Analysis
```python
from tools.scFeatureLens import SCFeatureLensPipeline

# Simple analysis
pipeline = SCFeatureLensPipeline()
results = pipeline.run_analysis("embeddings.pt")
```

#### Advanced Configuration
```python
from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig

# Custom configuration
config = AnalysisConfig(
    n_features=1000,              # Number of SAE features
    sparsity_penalty=1e-3,        # L1 penalty strength
    n_epochs=100,                 # Training epochs
    learning_rate=1e-3,           # Learning rate
    batch_size=512,               # Batch size
    top_k_features=50,            # Features to analyze
    activation_threshold=0.5      # Feature activation threshold
)

# Run with gene expression data
pipeline = SCFeatureLensPipeline(config)
results = pipeline.run_analysis(
    embeddings_path="embeddings.pt",
    gene_expression_path="data.h5ad",
    custom_gene_sets_path="gene_sets.yaml",
    output_dir="results"
)

# Access results
print(f"Active features: {len(results['active_features'])}")
print(f"DEGs found: {results['n_significant_degs']}")
print(f"Enriched gene sets: {results['n_enriched_gene_sets']}")
```

## 📊 Output & Results

### Generated Files
```
results/
├── config.yaml                 # Analysis configuration
├── sae_model.pt                # Trained sparse autoencoder
├── sae_activations.pt          # Feature activations per cell
├── analysis_summary.json       # Summary statistics
├── differential_expression/    # DE analysis results
│   ├── feature_001_degs.csv
│   └── ...
├── gene_set_enrichment/        # Enrichment analysis
│   ├── feature_001_enrichment.csv
│   └── ...
└── analysis.log               # Detailed logs
```

### Key Results
- **Feature Activations**: Which cells activate each sparse autoencoder feature
- **Differential Expression**: Genes up/downregulated in feature-active vs inactive cells
- **Gene Set Enrichment**: Biological pathways and GO terms enriched for each feature
- **Feature Interpretations**: Biological meaning of each learned feature

## 🎯 Examples & Tutorials

### Example Data & Scripts
- **📁 [Examples Directory](../../examples/scFeatureLens/)**: Complete examples with data
- **🔬 [Basic Example](../../examples/scFeatureLens/example.py)**: Start here for your first analysis
- **⚙️ [Configuration Examples](../../examples/scFeatureLens/config_example.yaml)**: Sample configuration files

### Tutorials
```bash
# Run basic tutorial
python -m tools.scFeatureLens.example --example basic

# Analyze Geneformer activations (if available)
python -m tools.scFeatureLens.example --example geneformer

# Custom analysis tutorial
python -m tools.scFeatureLens.example --example custom
```

## 🔧 Configuration

### Configuration File Format
```yaml
# config_example.yaml
sae:
  n_features: 1000
  sparsity_penalty: 0.001
  learning_rate: 0.001
  n_epochs: 100
  batch_size: 512

analysis:
  top_k_features: 50
  activation_threshold: 0.5
  min_cells_per_group: 10

differential_expression:
  test_method: "wilcoxon"
  p_value_threshold: 0.05
  log_fold_change_threshold: 0.25

gene_set_enrichment:
  use_go_terms: true
  p_value_threshold: 0.05
  min_gene_set_size: 5
  max_gene_set_size: 500
```

### Key Parameters
- **`n_features`**: Number of sparse autoencoder features (default: 1000)
- **`sparsity_penalty`**: L1 regularization strength (default: 1e-3)
- **`top_k_features`**: Number of most active features to analyze (default: 50)
- **`activation_threshold`**: Threshold for feature activation (default: 0.5)

## 📚 Understanding the Method

### Sparse Autoencoders for Interpretability
1. **Training**: Train a sparse autoencoder on model embeddings to learn interpretable features
2. **Sparsity**: L1 penalty encourages features to activate only for specific cell types/states
3. **Interpretation**: Each feature should represent a coherent biological concept

### Biological Analysis Pipeline
1. **Feature Selection**: Identify most active and interpretable features
2. **Cell Grouping**: Split cells into feature-active vs feature-inactive groups
3. **Differential Expression**: Find genes that differ between groups
4. **Enrichment Analysis**: Identify enriched biological pathways and GO terms
5. **Interpretation**: Understand what biological process each feature represents

### Example Interpretations
- **Feature 42**: Active in T cells → DEGs include CD3, CD8 → Enriched for "T cell activation"
- **Feature 137**: Active in stressed cells → DEGs include heat shock proteins → Enriched for "stress response"
- **Feature 299**: Active in cycling cells → DEGs include cyclins, CDKs → Enriched for "cell cycle"

## 🔬 Advanced Usage

### Custom Gene Sets
```python
# Define custom gene sets
custom_gene_sets = {
    "my_pathway": ["GENE1", "GENE2", "GENE3"],
    "stress_response": ["HSP70", "HSP90", "HSPA1A"]
}

# Save as YAML
with open("custom_gene_sets.yaml", "w") as f:
    yaml.dump(custom_gene_sets, f)

# Use in analysis
python -m tools.scFeatureLens.cli embeddings.pt --custom-gene-sets custom_gene_sets.yaml
```

### Integration with Scanpy
```python
import scanpy as sc
import anndata as ad
from tools.scFeatureLens import SCFeatureLensPipeline

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Extract embeddings (example with a hypothetical embedding)
embeddings = adata.obsm["X_geneformer"]  # or your embedding key

# Run scFeatureLens analysis
pipeline = SCFeatureLensPipeline()
results = pipeline.run_analysis(
    embeddings,
    gene_expression_path=adata,  # Can pass AnnData directly
    output_dir="results"
)
```

## 🐛 Troubleshooting

### Common Issues

#### Memory Errors
```bash
# Reduce batch size and number of features
python -m tools.scFeatureLens.cli embeddings.pt \
    --config config.yaml \
    --batch-size 256 \
    --n-features 500
```

#### Import Errors
```bash
# Verify installation
python -c "from tools.scFeatureLens import SCFeatureLensPipeline"

# Run environment validation
python ../../setup/validate_environment.py
```

#### No Significant Results
- Check activation threshold (try lowering to 0.1-0.3)
- Increase number of features to analyze
- Verify gene expression data quality
- Check that embeddings contain meaningful signal

### Getting Help
- **📖 Documentation**: See [main repository docs](../../docs/)
- **🔧 Setup Issues**: Check [environment setup guide](../../docs/ENVIRONMENT_SETUP.md)
- **🐳 Docker**: See [Docker guide](../../docs/DOCKER_GUIDE.md) for containerized analysis

## 📄 Citation

If you use scFeatureLens in your research, please cite:

```bibtex
@misc{schuster2025sparseautoencodersmakesense,
    title={Can sparse autoencoders make sense of latent representations?}, 
    author={Viktoria Schuster},
    year={2025},
    eprint={2410.11468},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2410.11468}, 
}
```

---

**Next Steps:**
1. 🎯 Try the [basic example](../../examples/scFeatureLens/example.py)
2. 📚 Read the [main repository documentation](../../README.md)
3. 🔧 Set up your [analysis environment](../../docs/ENVIRONMENT_SETUP.md)
4. 🚀 Start analyzing your own embeddings!
