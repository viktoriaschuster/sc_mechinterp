# sc_mechinterp

A collection of mechanistic interpretability tools for single-cell data analysis. *A lot of setup code is experimental and may change in the future. Please let me know if you have any issues or suggestions.*

## 🚀 Quick Navigation

**🔬 [Available Tools](#-available-tools)** | **⚡ [Quick Start](#-quick-start)** | **📚 [Documentation](docs/)** | **🎯 [Examples](examples/)**

---

## Overview

This repository is for mechanistic interpretability analysis of single-cell RNA-seq data and foundation models. Each tool focuses on different aspects of understanding and interpreting the learned representations in single-cell models.

## 🔬 Available Tools

### 🧬 [scFeatureLens](tools/scFeatureLens/) - Sparse Autoencoder Feature Extraction

**Extract meaningful features from single-cell RNA-seq model embeddings using sparse autoencoders**

- 🧠 **Train sparse autoencoders** on embeddings from any foundation model (Geneformer, multiDGD, etc.)
- 🔍 **Feature analysis** - identify which features are active for different cell types or conditions  
- 📊 **Differential expression** - compare gene expression between feature-active and inactive cells
- 🧬 **Gene set enrichment** - analyze enriched biological pathways and GO terms
- 📈 **Biological interpretation** - understand what biological processes each feature represents

📚 **[Complete Documentation & API Guide →](tools/scFeatureLens/README.md)**  
🎯 **[Examples & Tutorials →](examples/scFeatureLens/)**

---

*🚀 More mechanistic interpretability tools coming soon! The repository structure is designed to easily accommodate additional tools.*

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
```

## 🔬 Environment & Reproducibility

scFeatureLens provides multiple options for creating isolated, reproducible environments:

- **🔄 Complete Isolation**: Each setup method creates a dedicated environment that won't interfere with other projects
- **📋 Reproducibility**: Pin exact dependency versions for consistent results across systems  
- **🛡️ Safety**: No conflicts with existing Python installations or packages
- **📦 Portability**: Easy to share and deploy across different machines

### Available Setup Methods

| Method | Command |
|--------|----------|
| **Automated Script** | `./setup_env.sh` |
| **Conda** | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Virtual Environment** | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Poetry** | See [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) |
| **Docker** | See [`docs/DOCKER_GUIDE.md`](docs/DOCKER_GUIDE.md) |

I have only been working with conda, so let me know if you have any issues with the other setups.

### Quick Environment Check

```bash
# Verify your environment is properly isolated
python --version          # Should show Python 3.8+
which python              # Should point to your environment
echo $CONDA_DEFAULT_ENV   # Should show 'sc_mechinterp' (if using conda)

# Test package imports
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Isolated environment ready')"
```

## 🚀 Quick Start

**New to the project? Start here:**

### Automated Setup (Recommended)

```bash
# Clone or navigate to the repository
git clone https://github.com/yourusername/sc_mechinterp.git
cd sc_mechinterp

# Run automated environment setup
./setup_env.sh
```

### Verification

After setup, verify your installation:

```bash
# Test core functionality
python -c "from tools.scFeatureLens import SCFeatureLensPipeline; print('✓ Installation successful')"

# Run validation suite
python setup/validate_environment.py

# Check CLI interface
python -m tools.scFeatureLens.cli --help
```

## Data Requirements

### For scFeatureLens

- **Embeddings**: Model embeddings in `.pt`, `.npy`, or `.csv` format
- **Gene Expression Data** (optional): For downstream analysis. Currently supported: `.h5ad`. Coming soon: `.loom`, `.csv`, `.h5mu`.
- **Gene Sets**: GO terms (automatically downloaded) or custom gene sets (coming soon)

### Example Data

The repository includes example data in `examples/scFeatureLens/` from the paper.

## 📖 Documentation

All documentation is organized in the [`docs/`](docs/) directory:

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - Detailed installation guide  

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