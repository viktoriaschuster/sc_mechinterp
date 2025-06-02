# Examples

This directory contains example usage of all tools in the sc_mechinterp collection.

## scFeatureLens Examples

The `scFeatureLens/` directory contains comprehensive examples for the scFeatureLens tool:

### Available Examples

1. **Basic Example** (`example.py --example basic`)
   - Train SAE from scratch
   - Extract features and run analysis
   - Generate synthetic data demonstration

2. **Pre-trained Example** (`example.py --example pretrained`)
   - Use pre-trained SAE model
   - Skip training and focus on analysis
   - Faster workflow for testing

3. **GO Enrichment Example** (`example.py --example go`)
   - Full Gene Ontology enrichment analysis
   - Real biological pathway interpretation
   - Advanced feature interpretation

### Running Examples

```bash
# Navigate to the repository root
cd sc_mechinterp

# Set up environment (if not already done)
./setup/setup_env.sh

# Run basic example
python -m tools.scFeatureLens.example --example basic

# Run pre-trained example  
python -m tools.scFeatureLens.example --example pretrained

# Run GO enrichment example
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
