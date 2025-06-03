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
- **Model Embeddings**: `.pt`, `.npy`, or `.csv` files
- **Foundation Models**: Geneformer, multiDGD, or any custom embeddings
- **Gene Expression Data**: AnnData (`.h5ad`) for biological interpretation (more formats coming soon)

## 🚀 Quick Start

### Command Line Interface
```bash
# Analyze your own embeddings (simplified)
python -m tools.scFeatureLens.cli your_embeddings.pt --output-dir results
```

### Python API
```python
from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    n_features=10000,
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

# Use a pre-trained SAE model
python -m tools.scFeatureLens.cli embeddings.pt --sae-model-path model.pt --no-train-sae

# Only train the SAE model without running analysis
python -m tools.scFeatureLens.cli embeddings.pt --train-only True
```

#### Advanced Options
```bash
# Full analysis with gene expression data
python -m tools.scFeatureLens.cli embeddings.pt \
    --data data.h5ad \
    --output-dir results \
    --epochs 500 \
    --go-category biological_process \
    --min-active-samples 100 \
    --activation-percentile 99 \
    --predictions predictions.npy \ # path to model predictions of data to be used in DEG (optional)
    --dispersions dispersions.npy \ # path to model dispersions of data to be used in enrichment (optional)
    --verbose

# Use custom configuration file
python -m tools.scFeatureLens.cli embeddings.pt --config config.yaml

# Specify device for computation
python -m tools.scFeatureLens.cli embeddings.pt --device cuda:0

# Get help and see all options
python -m tools.scFeatureLens.cli --help
```

### Python API

#### Basic Analysis
```python
from tools.scFeatureLens import SCFeatureLensPipeline

# Simple analysis
# Custom configuration
config = AnalysisConfig() # Default parameters can be adjusted here
pipeline = SCFeatureLensPipeline(config)
results = pipeline.run_analysis("embeddings.pt")
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

### Tutorials

- **[Basic Analysis Example](examples/scFeatureLens/feature_analysis_demo.py)**: Run a simple analysis on model embeddings

## 🔧 Configuration


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