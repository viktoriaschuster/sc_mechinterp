# Examples

This directory contains example usage of all tools in the sc_mechinterp collection.

## 🧬 scFeatureLens Examples

The `scFeatureLens/` directory contains comprehensive examples for the scFeatureLens tool:

### 🚀 Quick Start Examples

1. **🎯 Basic Example** (`--example basic`)
   - Complete end-to-end analysis with synthetic data
   - Train SAE from scratch and run full analysis pipeline
   - Perfect for first-time users and testing

2. **⚡ Pre-trained Example** (`--example pretrained`)
   - Use pre-trained SAE model to skip training
   - Faster workflow focusing on analysis and interpretation
   - Great for exploring features without training time

3. **🧬 GO Enrichment Example** (`--example go`)
   - Full Gene Ontology enrichment analysis
   - Real biological pathway interpretation
   - Advanced feature interpretation and biological insights

### 🏃‍♂️ Running Examples

```bash
# Navigate to the repository root
cd sc_mechinterp

# Set up environment (if not already done)
./setup_env.sh

# 🎯 Run basic example (recommended first step)
python -m tools.scFeatureLens.example --example basic

# ⚡ Run pre-trained example  
python -m tools.scFeatureLens.example --example pretrained

# 🧬 Run GO enrichment example
python -m tools.scFeatureLens.example --example go
```

### Example Data

- `example_data/`: Synthetic datasets for testing
- `example_results/`: Output from basic example
- `example_pretrained_results/`: Output from pre-trained example
- `example_sae_training/`: SAE models from training

### Configuration

- `config_example.yaml`: Example configuration file showing all available options

## Adding New Examples

When adding examples for new tools:

1. Create a new subdirectory under `examples/`
2. Include example data, scripts, and configuration
3. Update this README with usage instructions
4. Ensure examples are self-contained and documented
