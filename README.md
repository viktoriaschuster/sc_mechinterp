# sc_mechinterp

A collection of mechanistic interpretability tools for single-cell data analysis.

## Overview

This repository contains various tools for mechanistic interpretability analysis of single-cell RNA-seq data and foundation models. Each tool focuses on different aspects of understanding and interpreting the learned representations in single-cell models.

**🔬 Available Tools**

- [scFeatureLens](tools/scFeatureLens/)

*More tools coming in the future! The repository structure is designed to easily accommodate additional mechanistic interpretability tools.*

## 📁 Repository Structure

```
sc_mechinterp/
├── 📄 README.md              # This file
├── 📄 LICENSE                # MIT license
├── 📄 setup.py               # Package installation
├── 📄 setup_env.sh           # Quick environment setup
├── 📁 setup/                 # Environment & installation files
├── 📁 docs/                  # Documentation
├── 📁 tools/                 # All analysis tools
├── 📁 examples/              # Usage examples & demos
├── 📁 tests/                 # Test suites  
└── 📁 scripts/               # Utility scripts
```

This structure ensures:
- **🎯 Clean organization**: Core tools separated from setup and examples
- **📈 Scalability**: Easy to add new tools without cluttering the root
- **🔧 Easy maintenance**: Setup files organized in dedicated directories
- **📚 Clear documentation**: All guides in the `docs/` directory

## 🔬 Environment & Reproducibility

### Isolated Environment Setup

scFeatureLens provides multiple options for creating isolated, reproducible environments:

- **🔄 Complete Isolation**: Each setup method creates a dedicated environment that won't interfere with other projects
- **📋 Reproducibility**: Pin exact dependency versions for consistent results across systems  
- **🛡️ Safety**: No conflicts with existing Python installations or packages
- **📦 Portability**: Easy to share and deploy across different machines

### Available Setup Methods

| Method | Best For | Isolation Level | Command |
|--------|----------|-----------------|---------|
| **Automated Script** | Quick start, auto-detection | High | `./setup_env.sh` |
| **Conda** | Data science workflows | High | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Virtual Environment** | Standard Python projects | Medium | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Poetry** | Modern dependency management | High | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Docker** | Ultimate isolation & deployment | Maximum | See [`docs/DOCKER_GUIDE.md`](docs/DOCKER_GUIDE.md) |

### Quick Environment Check

```bash
# Verify your environment is properly isolated
python --version          # Should show Python 3.8+
which python              # Should point to your environment
echo $CONDA_DEFAULT_ENV   # Should show 'sc_mechinterp' (if using conda)

# Test package imports
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Isolated environment ready')"
```

## 🔬 Tools

### scFeatureLens

A tool for extracting meaningful features from single-cell RNA-seq data model embeddings using sparse autoencoders and performing biological interpretation through differential gene expression analysis and gene set enrichment.

**Features:**
- Train sparse autoencoders on embeddings from any foundation model
- Extract and analyze feature activations
- Perform differential expression analysis on feature-active vs inactive cells
- Gene set enrichment analysis (GO terms, pathways, custom gene sets)
- Support for multiple embedding formats (Geneformer, multiDGD, custom)

**Quick Start:**
```bash
# Run basic example
python -m tools.scFeatureLens.example --example basic

# CLI usage
python -m tools.scFeatureLens.cli --help

# Custom analysis
python -m tools.scFeatureLens.cli my_embeddings.pt --config my_config.yaml
```

**More Details:** See [`tools/scFeatureLens/`](tools/scFeatureLens/) and [`examples/scFeatureLens/`](examples/scFeatureLens/)

## 🚀 Quick Start

**New to the project? Start here:**

### Automated Setup (Recommended)

```bash
# Clone or navigate to the repository
git clone https://github.com/yourusername/sc_mechinterp.git
cd sc_mechinterp

# Run automated environment setup
./setup_env.sh

# Follow prompts to choose your preferred environment manager
# The script handles isolation and dependency management automatically

# Test installation
python -m tools.scFeatureLens.example --example basic
```

### Environment Options

Choose your preferred isolated environment setup:

1. **🔬 Automated Setup**: [`./setup_env.sh`](setup_env.sh) - One-command setup with auto-detection
2. **⚡ Quick Setup**: [`docs/QUICKSTART.md`](docs/QUICKSTART.md) - 5-minute manual setup  
3. **📚 Detailed Setup**: [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) - Comprehensive environment guide
4. **🔄 Reproducibility**: [`docs/REPRODUCIBILITY_GUIDE.md`](docs/REPRODUCIBILITY_GUIDE.md) - Complete isolation and reproducibility
5. **🐳 Docker Setup**: [`docs/DOCKER_GUIDE.md`](docs/DOCKER_GUIDE.md) - Ultimate isolation with containers

### Verification

After setup, verify your installation:

```bash
# Test core functionality
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Installation successful')"

# Run validation suite
python setup/validate_environment.py

# Run full example pipeline
python -m tools.scFeatureLens.example --example basic

# Check CLI interface
python -m tools.scFeatureLens.cli --help
```

## Installation

### Automated (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/sc_mechinterp.git
cd sc_mechinterp

# Run automated setup
./setup_env.sh
```

### Manual Installation

```bash
# Create isolated environment (conda recommended)
conda create -n sc_mechinterp python=3.10
conda activate sc_mechinterp

# Install PyTorch (adjust for your system)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install dependencies and package
pip install -r setup/requirements.txt
pip install -e .
```

**⚠️ Important**: Always use an isolated environment to avoid conflicts with other projects.

## Data Requirements

### For scFeatureLens

- **Embeddings**: Model embeddings in `.pt`, `.npy`, `.csv`, or `.h5ad` format
- **Gene Expression Data** (optional): For differential expression analysis
- **Gene Sets**: GO terms (automatically downloaded) or custom gene sets

### Example Data

The repository includes example data in `examples/scFeatureLens/`:
- Pre-trained SAE activations
- Synthetic datasets for testing
- Configuration examples

## 📖 Documentation

All documentation is organized in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - Detailed installation guide  
- **[Reproducibility Guide](docs/REPRODUCIBILITY_GUIDE.md)** - Ensuring consistent results
- **[Docker Guide](docs/DOCKER_GUIDE.md)** - Container deployment
- **[Development Checklist](docs/DEVELOPMENT_CHECKLIST.md)** - For contributors

## Configuration

Each tool can be configured using YAML files. See `examples/scFeatureLens/config_example.yaml` for an example configuration.

## Development

To add a new tool to the collection:

1. Create a new directory under `tools/`
2. Implement the tool following the established patterns (see `tools/README.md`)
3. Add CLI interface and configuration support
4. Add examples in `examples/YourTool/`
5. Update this README with tool documentation
6. Add tests in the `tests/` directory

See [`docs/DEVELOPMENT_CHECKLIST.md`](docs/DEVELOPMENT_CHECKLIST.md) for detailed guidelines.

## Contributing

Contributions are welcome! Please see the development section above and check the documentation in `docs/` for guidelines.

## License

MIT License - see `LICENSE` file for details.

## Citation

If you use scFeatureLens your research, please cite:

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