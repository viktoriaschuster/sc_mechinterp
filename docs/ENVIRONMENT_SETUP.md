# Environment Setup for sc_mechinterp

This guide will help you set up a clean, isolated Python environment for the sc_mechinterp tools.

## Option 1: Using conda (Recommended)

### Create a new conda environment

```bash
# Create environment with Python 3.10
conda create -n sc_mechinterp python=3.10

# Activate the environment
conda activate sc_mechinterp

# Install PyTorch (adjust for your system - CPU vs GPU)
# For CPU:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For GPU (CUDA 11.8):
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Export environment (for sharing)

```bash
# Export exact package versions
conda env export > environment.yml

# Export minimal requirements (cross-platform)
conda env export --from-history > environment-minimal.yml
```

### Restore environment from file

```bash
# From exact export
conda env create -f environment.yml

# From minimal requirements
conda env create -f environment-minimal.yml
```

## Option 2: Using venv (Standard Library)

### Create virtual environment

```bash
# Create virtual environment
python -m venv sc_mechinterp_env

# Activate on macOS/Linux
source sc_mechinterp_env/bin/activate

# Activate on Windows
sc_mechinterp_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust URL for your system)
pip install torch torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Save requirements

```bash
# Save exact versions
pip freeze > requirements-frozen.txt
```

## Option 3: Using Poetry (Modern Dependency Management)

### Install Poetry (if not already installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Set up project with Poetry

```bash
# Initialize poetry (if not done)
poetry init

# Install dependencies
poetry install

# Add new dependencies
poetry add torch torchvision torchaudio
poetry add numpy pandas scipy scikit-learn

# Install in development mode
poetry install

# Activate shell
poetry shell
```

## Verification

After setting up your environment, verify the installation:

```bash
# Test basic imports
python -c "import torch, numpy, pandas, scipy; print('✓ Core packages imported successfully')"

# Test scFeatureLens
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ scFeatureLens imported successfully')"

# Run basic tests
python tests/test_basic.py

# Run example (creates synthetic data)
python -m scFeatureLens.example --example basic
```

## Development Setup

For development work, also install development tools:

```bash
# Development dependencies
pip install pytest pytest-cov black flake8 mypy jupyter notebook

# Pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## IDE Configuration

### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./sc_mechinterp_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

1. Go to File → Settings → Project → Python Interpreter
2. Click gear icon → Add → Existing environment
3. Select the Python executable from your environment

## Common Issues and Solutions

### Import Errors

If you get import errors:

```bash
# Make sure you're in the right environment
which python

# Install in development mode
pip install -e .

# Check if package is installed
pip list | grep sc-mechinterp
```

### PyTorch Installation

For PyTorch issues, visit [pytorch.org](https://pytorch.org/get-started/locally/) and use their installation command generator.

### Memory Issues

For large datasets, you may need to adjust batch sizes in your configuration:

```yaml
# In your config file
batch_size: 32  # Reduce if you get out-of-memory errors
n_processes: 4  # Adjust based on your system
```

## Environment Variables

You can set environment variables for consistent behavior:

```bash
# Add to your .bashrc or .zshrc
export SC_MECHINTERP_DATA_DIR="/path/to/your/data"
export SC_MECHINTERP_CACHE_DIR="/path/to/cache"

# Or create a .env file in the project root
echo "SC_MECHINTERP_DATA_DIR=/path/to/your/data" > .env
```

## Jupyter Notebook Setup

To use the tools in Jupyter notebooks:

```bash
# Install Jupyter in your environment
pip install jupyter ipykernel

# Create kernel for this environment
python -m ipykernel install --user --name=sc_mechinterp --display-name="sc_mechinterp"

# Start Jupyter
jupyter notebook
```

Then select the "sc_mechinterp" kernel when creating new notebooks.

## Deactivating Environment

```bash
# For conda
conda deactivate

# For venv
deactivate

# For poetry
exit  # (from poetry shell)
```
