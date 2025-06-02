sc_mechinterp Documentation
===========================

**sc_mechinterp** is a collection of mechanistic interpretability tools for single-cell data analysis.

The package provides various tools for mechanistic interpretability analysis of single-cell RNA-seq data and foundation models. Each tool focuses on different aspects of understanding and interpreting the learned representations in single-cell models.

.. note::
   This package is currently in active development. If you encounter any issues, please 
   report them on our `GitHub <https://github.com/yourusername/sc_mechinterp>`_.

Available Tools
---------------

scFeatureLens
~~~~~~~~~~~~~

**Extract meaningful features from single-cell RNA-seq model embeddings using sparse autoencoders**

scFeatureLens uses sparse autoencoders to identify interpretable features in model embeddings and provides biological context through differential gene expression analysis and gene set enrichment.

**Key Features:**

• Train sparse autoencoders on embeddings from any foundation model
• Extract interpretable features from SAE activations  
• Differential gene expression analysis on feature-active vs inactive cells
• Gene set enrichment analysis (GO terms, pathways, custom gene sets)
• Support for multiple embedding formats (Geneformer, multiDGD, custom)

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   usage/installation
   usage/quickstart
   usage/tutorials

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   usage/guide
   usage/scfeaturelens
   usage/advanced

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   API Reference <api_reference>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Development

   development/contributing
   development/architecture

