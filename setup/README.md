# Setup & Installation

This directory contains all files needed to set up isolated, reproducible environments for sc_mechinterp tools.

## Quick Start

**Automated Setup (Recommended):**
```bash
./setup_env.sh
```

The script will detect your system and guide you through setting up the optimal environment.

## Setup Files

### Environment Management
- **`setup_env.sh`** - Automated environment setup script with conda activation
- **`environment.yml`** - Conda environment specification
- **`requirements.txt`** - Python package dependencies
- **`pyproject.toml`** - Poetry configuration with development tools
- **`setup.py`** - Package installation configuration

### Deployment
- **`Dockerfile`** - Container setup for ultimate isolation

### Validation
- **`validate_environment.py`** - Comprehensive environment testing (9 test categories)

## Environment Options

Choose your preferred isolated environment:

1. **Conda** (Recommended for data science)
   ```bash
   conda env create -f environment.yml
   conda activate sc_mechinterp
   ```

2. **Virtual Environment** (Standard Python)
   ```bash
   python -m venv sc_mechinterp
   source sc_mechinterp/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

3. **Poetry** (Modern dependency management)
   ```bash
   poetry install
   poetry shell
   ```

4. **Docker** (Ultimate isolation)
   ```bash
   docker build -t sc_mechinterp .
   docker run -it sc_mechinterp
   ```

## Validation

After setup, validate your environment:

```bash
python validate_environment.py
```

This runs 9 comprehensive tests to ensure everything is working correctly.

## Troubleshooting

- **Import errors**: Ensure you're in the correct environment
- **Package conflicts**: Use isolated environments (conda/venv/docker)
- **Permission issues**: Check directory permissions and virtual environment activation

For detailed setup instructions, see the documentation in `../docs/`.
