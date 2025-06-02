# Quick Start Guide for sc_mechinterp

This guide will get you up and running with sc_mechinterp in less than 5 minutes.

## 🚀 Quick Setup (Automated)

```bash
# Clone or navigate to the repository
cd sc_mechinterp

# Run the automated setup script
./setup_env.sh
```

The script will:
1. Detect your available environment managers (conda, poetry, venv)
2. Create an isolated environment
3. Install all dependencies
4. Verify the installation

## 🔧 Manual Setup (If you prefer control)

### Option 1: Using conda (Recommended)

```bash
# Create environment
conda create -n sc_mechinterp python=3.10
conda activate sc_mechinterp

# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### Option 2: Using venv

```bash
# Create and activate environment
python -m venv sc_mechinterp_env
source sc_mechinterp_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## ✅ Verify Installation

```bash
# Test imports
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Ready to go!')"

# Run basic test
python tests/test_basic.py
```

## 🎯 Run Your First Analysis

```bash
# Run the basic example (creates synthetic data)
python -m scFeatureLens.example --example basic

# Or use the CLI with provided data
python -m scFeatureLens.cli sae_geneformer_10000_l1-1e-3_lr-1e-4_500epochs_activations.pt --output-dir my_results
```

## 📁 Project Structure

```
sc_mechinterp/
├── scFeatureLens/          # Main tool package
│   ├── pipeline.py         # Core analysis pipeline
│   ├── sae.py             # Sparse autoencoder implementation
│   ├── analysis_functions.py  # Statistical analysis functions
│   ├── cli.py             # Command-line interface
│   └── example.py         # Usage examples
├── data/                  # Data files (GO terms, gene sets)
├── scripts/               # Utility scripts
├── tests/                 # Test files
└── docs/                  # Documentation
```

## 🔬 Basic Usage

### Python API

```python
from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig

# Configure analysis
config = AnalysisConfig(
    embeddings_path="my_embeddings.pt",
    output_dir="results",
    sae_hidden_size=1000,
    train_sae=True
)

# Run analysis
pipeline = SCFeatureLensPipeline(config)
results = pipeline.run_analysis()
```

### Command Line

```bash
# Basic usage
scfeaturelens my_embeddings.pt --output-dir results

# With custom configuration
scfeaturelens my_embeddings.pt --config my_config.yaml

# With gene expression data
scfeaturelens my_embeddings.pt \
    --gene-expression-data my_expression.h5ad \
    --output-dir results
```

## 🗂️ Data Setup

```bash
# Download GO data files
python scripts/setup_data.py --species human --example-sets

# This creates:
# - data/go-basic.obo (GO ontology)
# - data/goa_human.gaf (GO annotations)
# - data/custom_gene_sets/ (example gene sets)
```

## 📖 What's Next?

1. **Read the full documentation**: `ENVIRONMENT_SETUP.md`
2. **Explore examples**: Check out different examples in `scFeatureLens/example.py`
3. **Customize your analysis**: Modify `scFeatureLens/config_example.yaml`
4. **Join the community**: Report issues and contribute on GitHub

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure your environment is activated
2. **PyTorch issues**: Visit [pytorch.org](https://pytorch.org/) for installation help
3. **Memory errors**: Reduce `batch_size` in your configuration
4. **Missing data**: Run `python scripts/setup_data.py` to download required files

### Get Help

- Check the documentation: `ENVIRONMENT_SETUP.md`
- Run tests: `python tests/test_basic.py`
- File an issue on GitHub

## 🎉 You're Ready!

You now have a fully functional sc_mechinterp environment. Start exploring mechanistic interpretability in single-cell data!

```bash
# Try the basic example
python -m scFeatureLens.example --example basic
```
