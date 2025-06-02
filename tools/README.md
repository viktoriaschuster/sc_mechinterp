# Tools

This directory contains all the mechanistic interpretability tools for single-cell data analysis.

## Available Tools

### scFeatureLens

**Purpose**: Extract meaningful features from single-cell RNA-seq data model embeddings using sparse autoencoders and perform biological interpretation.

**Location**: `scFeatureLens/`

**Key Features**:
- Train sparse autoencoders on embeddings from foundation models (Geneformer, multiDGD, etc.)
- Extract interpretable features from SAE activations
- Differential gene expression analysis on feature-active vs inactive cells
- Gene set enrichment analysis (GO terms, pathways, custom gene sets)
- Generalizable across different embeddings and biological contexts

**Usage**:
```bash
# Command line interface
python -m tools.scFeatureLens.sc_mechinterp --embeddings data.npy --output results/

# Python API
from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig
config = AnalysisConfig(embeddings_path="data.npy")
pipeline = SCFeatureLensPipeline(config)
pipeline.run_full_analysis()
```

**Documentation**: See `../examples/scFeatureLens/` for usage examples

## Tool Structure

Each tool follows this structure:
```
tool_name/
├── __init__.py          # Package initialization and exports
├── pipeline.py          # Main pipeline orchestrator
├── analysis_functions.py # Core analysis methods
├── cli.py              # Command-line interface
├── config_example.yaml  # Example configuration
└── README.md           # Tool-specific documentation
```

## Adding New Tools

When adding a new tool:

1. Create a new directory under `tools/`
2. Follow the established structure pattern
3. Implement CLI interface and configuration support
4. Add examples in `../examples/`
5. Update this README with tool documentation
6. Add tests in `../tests/`

## Integration

Tools are designed to work together and can be chained for complex analysis workflows. Each tool maintains its own configuration and can be used independently or as part of larger pipelines.
