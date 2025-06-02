# Reproducibility and Isolation Guide for scFeatureLens

This guide provides comprehensive instructions for setting up a completely isolated, reproducible environment for scFeatureLens analysis.

## 🎯 Objectives

1. **Complete Isolation**: Ensure the environment doesn't interfere with other projects
2. **Reproducibility**: Guarantee consistent results across different systems
3. **Documentation**: Provide clear setup and usage instructions
4. **Validation**: Include verification steps to ensure everything works

## 🚀 Quick Start (Recommended)

For most users, the automated setup is the fastest way to get started:

```bash
# Clone or navigate to the repository
cd sc_mechinterp

# Run automated setup
./setup_env.sh

# Follow the prompts to choose your preferred environment manager
# The script will handle everything automatically
```

## 📋 Manual Setup Options

### Option 1: Conda Environment (Recommended for Data Science)

#### Complete Fresh Setup

```bash
# 1. Create isolated conda environment
conda create -n sc_mechinterp python=3.10 -y

# 2. Activate the environment
conda activate sc_mechinterp

# 3. Install PyTorch (choose CPU or GPU)
# For CPU:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# For GPU (adjust CUDA version as needed):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Install scFeatureLens in development mode
pip install -e .

# 6. Verify installation
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Installation successful')"
```

#### Using Environment File (Exact Reproduction)

```bash
# Create environment from exact specifications
conda env create -f environment.yml

# Activate environment
conda activate sc_mechinterp

# Install package
pip install -e .
```

### Option 2: Virtual Environment (Standard Python)

#### Complete Fresh Setup

```bash
# 1. Create virtual environment
python -m venv sc_mechinterp_env

# 2. Activate environment
# On macOS/Linux:
source sc_mechinterp_env/bin/activate
# On Windows:
sc_mechinterp_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (adjust URL for your system)
pip install torch torchvision torchaudio

# 5. Install dependencies
pip install -r requirements.txt

# 6. Install package in development mode
pip install -e .

# 7. Save exact environment state
pip freeze > requirements-frozen.txt
```

### Option 3: Poetry (Modern Dependency Management)

```bash
# 1. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies and create virtual environment
poetry install

# 3. Activate shell
poetry shell

# 4. Verify installation
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Installation successful')"
```

## 🔍 Verification Steps

After setting up your environment, run these verification steps:

```bash
# 1. Check Python environment
python --version
which python

# 2. Test core imports
python -c "
import torch
import numpy as np
import pandas as pd
import scipy
import yaml
print('✓ All core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
"

# 3. Test scFeatureLens imports
python -c "
from tools.scFeatureLens import SCFeatureLensPipeline, SparseAutoencoder
from tools.scFeatureLens.analysis_functions import perform_differential_expression
print('✓ scFeatureLens imported successfully')
"

# 4. Run basic functionality test
python tests/test_basic.py

# 5. Run example pipeline (creates synthetic data)
python -m scFeatureLens.example --example basic

# 6. Test CLI interface
python -m scFeatureLens.cli --help
```

## 📁 Data Setup

### Download Required Data Files

```bash
# Set up data directories and download required files
python scripts/setup_data.py --example-sets

# This will create:
# examples/scFeatureLens/gene_ontology/     - GO term files  
# examples/scFeatureLens/custom_gene_sets/  - Custom gene set examples
# examples/scFeatureLens/example_data/      - Synthetic example data
```

### Custom Data Integration

```bash
# Directory structure for your data:
examples/scFeatureLens/
├── embeddings/          # Your embedding files (.npy, .pt)
├── gene_expression/     # Gene expression data (.h5ad, .csv)
├── gene_sets/          # Custom gene sets (.yaml, .json)
└── results/            # Analysis output directory
```

## 🔧 Configuration Management

### Create Analysis Configuration

```bash
# Copy example configuration
cp scFeatureLens/config_example.yaml my_analysis_config.yaml

# Edit for your specific analysis
# Key sections:
# - sae_config: Sparse autoencoder parameters
# - analysis_config: Analysis settings
# - output_config: Output preferences
```

### Environment Variables (Optional)

```bash
# Add to ~/.bashrc or ~/.zshrc for persistent settings
export SC_MECHINTERP_DATA_DIR="/path/to/your/data"
export SC_MECHINTERP_CACHE_DIR="/path/to/cache"
export SC_MECHINTERP_CONFIG="/path/to/default/config.yaml"

# Or create project-specific .env file
echo "SC_MECHINTERP_DATA_DIR=/Users/yourusername/sc_data" > .env
```

## 🧪 Development Setup

### For Contributors and Advanced Users

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy jupyter

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run full test suite
pytest tests/ -v

# Code formatting
black scFeatureLens/
flake8 scFeatureLens/

# Type checking
mypy scFeatureLens/
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./sc_mechinterp_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Click gear icon → Add → Existing environment
3. Select Python executable from your environment

## 📊 Usage Examples

### Basic Analysis

```bash
# Using synthetic data (for testing)
python -m scFeatureLens.example --example basic

# Using real data
python -m scFeatureLens.cli your_embeddings.npy \
    --gene-expression your_gene_data.h5ad \
    --config your_config.yaml \
    --output-dir results/
```

### Advanced Analysis

```bash
# With custom gene sets and GO enrichment
python -m tools.scFeatureLens.cli embeddings.npy \
    --gene-expression data.h5ad \
    --custom-gene-sets custom_sets.yaml \
    --use-go-terms \
    --go-data-dir examples/scFeatureLens/gene_ontology/ \
    --output-dir results/ \
    --n-features 1000 \
    --sparsity-penalty 1e-3
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure package is installed in development mode
pip install -e .

# Check if you're in the correct environment
which python
echo $CONDA_DEFAULT_ENV  # for conda
echo $VIRTUAL_ENV        # for venv
```

#### Memory Issues
```bash
# Reduce batch size and number of features for large datasets
python -m scFeatureLens.cli embeddings.npy \
    --batch-size 32 \
    --n-features 500 \
    --output-dir results/
```

#### Conda Activation Issues
```bash
# Initialize conda for your shell
conda init

# Or manually source conda
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate sc_mechinterp
```

### Environment Reset

If something goes wrong, completely reset:

```bash
# For conda
conda env remove -n sc_mechinterp
conda env create -f environment.yml

# For venv
rm -rf sc_mechinterp_env
python -m venv sc_mechinterp_env
source sc_mechinterp_env/bin/activate
pip install -r requirements.txt
pip install -e .

# For poetry
poetry env remove python
poetry install
```

## 📈 Performance Optimization

### For Large Datasets

```bash
# Use GPU if available
export CUDA_VISIBLE_DEVICES=0

# Optimize memory usage
python -m scFeatureLens.cli embeddings.npy \
    --batch-size 16 \
    --n-workers 4 \
    --memory-efficient \
    --output-dir results/
```

### Parallel Processing

```bash
# Utilize multiple CPU cores
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Run analysis with multiprocessing
python -m scFeatureLens.cli embeddings.npy \
    --n-workers 8 \
    --output-dir results/
```

## 📝 Logging and Monitoring

### Enable Detailed Logging

```bash
# Set logging level
export SC_MECHINTERP_LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitor Resource Usage

```bash
# Monitor CPU and memory during analysis
python -c "
import psutil
from tools.scFeatureLens import SCFeatureLensPipeline
# Your analysis code here
print(f'CPU usage: {psutil.cpu_percent()}%')
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"
```

## 🔄 Environment Deactivation

```bash
# Conda
conda deactivate

# Virtual environment
deactivate

# Poetry
exit  # (from poetry shell)
```

## 📚 Additional Resources

- [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) - Detailed environment setup
- [QUICKSTART.md](QUICKSTART.md) - 5-minute quick start
- [DEVELOPMENT_CHECKLIST.md](DEVELOPMENT_CHECKLIST.md) - Development verification
- [README.md](README.md) - Project overview and basic usage
- [examples/scFeatureLens/data_setup.md](../examples/scFeatureLens/data_setup.md) - Data organization guide

## 🆘 Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Verify your environment setup with the checklist
3. Run the verification commands
4. Check the example outputs
5. Create an issue with detailed error messages and environment info
