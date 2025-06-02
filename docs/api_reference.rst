.. _api_reference:

API Reference
===============

This page provides a complete API reference for all classes and functions in sc_mechinterp.

.. note::
   The API is organized by modules. Click on any class or function name to see detailed documentation.

scFeatureLens
-------------

The main module for sparse autoencoder feature extraction and analysis.

Main Pipeline Class
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tools.scFeatureLens.pipeline.SCFeatureLensPipeline
   :members:
   :show-inheritance:

   The main pipeline class for running complete scFeatureLens analyses.

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tools.scFeatureLens.pipeline.AnalysisConfig
   :members:
   :show-inheritance:

   Configuration class for analysis parameters.

Sparse Autoencoder
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tools.scFeatureLens.sae.SparseAutoencoder
   :members:
   :show-inheritance:

   Sparse autoencoder implementation for feature extraction.

CLI Interface
~~~~~~~~~~~~~

.. autofunction:: tools.scFeatureLens.cli.main

   Main command-line interface function.

Examples
~~~~~~~~

.. autofunction:: tools.scFeatureLens.example.main

   Main example function that demonstrates all features.

Analysis Functions
------------------

Statistical analysis and biological interpretation functions.

Core Analysis Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: tools.scFeatureLens.analysis_functions.differential_expression_analysis

Usage Examples
--------------

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from tools.scFeatureLens.pipeline import SCFeatureLensPipeline, AnalysisConfig

   # Create configuration
   config = AnalysisConfig(
       embeddings_path="embeddings.pt",
       output_dir="results",
       sae_hidden_size=10000,
       sae_l1_penalty=1e-3,
       sae_epochs=500
   )

   # Run analysis
   pipeline = SCFeatureLensPipeline(config)
   pipeline.run_analysis()
