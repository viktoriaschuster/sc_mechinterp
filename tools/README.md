# 🔬 Mechanistic Interpretability Tools

This directory contains all the mechanistic interpretability tools for single-cell data analysis.

## 🧬 Available Tools

### [scFeatureLens](scFeatureLens/) - Sparse Autoencoder Feature Extraction

**Extract meaningful features from single-cell RNA-seq model embeddings using sparse autoencoders**

🎯 **Purpose**: Understand what foundation models learn by training sparse autoencoders on embeddings and performing biological interpretation through differential expression and gene set enrichment analysis.

📁 **Location**: [`scFeatureLens/`](scFeatureLens/)

✨ **Key Capabilities**:
- 🧠 Train sparse autoencoders on embeddings from any foundation model (Geneformer, multiDGD, etc.)
- 🔍 Extract interpretable features from SAE activations
- 📊 Differential gene expression analysis on feature-active vs inactive cells
- 🧬 Gene set enrichment analysis (GO terms, pathways, custom gene sets)
- 🌐 Generalizable across different embeddings and biological contexts

🚀 **Quick Start**:
```bash
# Run basic example
python -m tools.scFeatureLens.example --example basic

# Analyze your embeddings
python -m tools.scFeatureLens.cli your_embeddings.pt --output-dir results
```

📚 **Documentation**: 
- **[Complete Guide →](scFeatureLens/README.md)** - Full documentation, API reference, and advanced usage
- **[Examples →](../examples/scFeatureLens/)** - Tutorials and example analyses

---

*🚀 More tools coming soon! Each new tool will follow the same structure and quality standards.*

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
