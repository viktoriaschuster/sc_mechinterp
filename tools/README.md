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