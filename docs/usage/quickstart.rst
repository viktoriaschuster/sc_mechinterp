Quick Start
===========

Get up and running with sc_mechinterp in 5 minutes! This guide will walk you through 
your first analysis using scFeatureLens with example data.

Prerequisites
-------------

Make sure you have sc_mechinterp installed. If not, see the :doc:`installation` guide.

Your First Analysis
-------------------

Let's run a complete mechanistic interpretability analysis on single-cell embeddings:

.. code-block:: bash

   # Navigate to your sc_mechinterp directory
   cd sc_mechinterp

   # Run the basic example
   python -m tools.scFeatureLens.example --example basic

This command will:

1. **Generate synthetic data** - Creates example embeddings and gene expression data
2. **Train a sparse autoencoder** - Learns interpretable features from the embeddings
3. **Extract features** - Identifies which cells activate each feature
4. **Analyze differential expression** - Finds genes that differ between feature-active and inactive cells
5. **Generate results** - Creates a complete analysis report

Expected Output
~~~~~~~~~~~~~~~

You should see output similar to this:

.. code-block:: text

   🧬 Running scFeatureLens Basic Example
   =====================================
   
   📊 Generating synthetic data...
   ✓ Created 1000 cells × 512 embeddings
   ✓ Created 1000 cells × 2000 genes expression data
   
   🧠 Training sparse autoencoder...
   Training SAE: 100%|████████████| 100/100 [00:30<00:00,  3.33it/s]
   ✓ SAE training complete
   
   🔍 Analyzing features...
   ✓ Found 50 active features
   ✓ Differential expression analysis complete
   
   📈 Results saved to: example_results/
   
   🎉 Analysis complete! Check the results directory.

Exploring Results
-----------------

The analysis creates several output files:

.. code-block:: text

   example_results/
   ├── config.yaml                    # Analysis configuration
   ├── sae_model.pt                   # Trained sparse autoencoder
   ├── sae_activations.pt            # Feature activations per cell
   ├── analysis_summary.json         # Summary statistics
   ├── differential_expression/      # DE analysis results
   │   ├── feature_001_degs.csv
   │   └── ...
   ├── gene_set_enrichment/          # Enrichment analysis
   │   ├── feature_001_enrichment.csv
   │   └── ...
   └── analysis.log                 # Detailed logs

Key Results to Examine
~~~~~~~~~~~~~~~~~~~~~~

**1. Analysis Summary**

.. code-block:: bash

   cat example_results/analysis_summary.json

This shows overall statistics like number of active features, significant DEGs, and enriched pathways.

**2. Feature Activations**

.. code-block:: python

   import torch
   activations = torch.load('example_results/sae_activations.pt')
   print(f"Shape: {activations.shape}")  # [n_cells, n_features]

**3. Differential Expression Results**

.. code-block:: bash

   head example_results/differential_expression/feature_001_degs.csv

Shows genes up/downregulated in cells where feature 1 is active.

Your Own Data
-------------

Now let's analyze your own embeddings:

Basic Analysis
~~~~~~~~~~~~~~

.. code-block:: bash

   # Analyze your embeddings
   python -m tools.scFeatureLens.cli your_embeddings.pt --output-dir my_results

The tool supports multiple formats:
- PyTorch tensors (``.pt``)
- NumPy arrays (``.npy``)
- CSV files (``.csv``)
- AnnData objects (``.h5ad``)

With Gene Expression Data
~~~~~~~~~~~~~~~~~~~~~~~~~

For biological interpretation, provide gene expression data:

.. code-block:: bash

   python -m tools.scFeatureLens.cli embeddings.pt \
       --gene-expression data.h5ad \
       --output-dir results

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Create a configuration file for reproducible analyses:

.. code-block:: yaml

   # my_config.yaml
   sae:
     n_features: 1000
     sparsity_penalty: 0.001
     n_epochs: 100
   
   analysis:
     top_k_features: 50
     activation_threshold: 0.5

.. code-block:: bash

   python -m tools.scFeatureLens.cli embeddings.pt --config my_config.yaml

Python API
----------

For more control, use the Python API:

.. code-block:: python

   from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig

   # Configure analysis
   config = AnalysisConfig(
       n_features=1000,
       sparsity_penalty=1e-3,
       n_epochs=100
   )

   # Run analysis
   pipeline = SCFeatureLensPipeline(config)
   results = pipeline.run_analysis(
       embeddings_path="embeddings.pt",
       output_dir="results"
   )

   # Access results
   print(f"Active features: {len(results['active_features'])}")
   print(f"Significant DEGs: {results['n_significant_degs']}")

Understanding Results
---------------------

Feature Interpretation
~~~~~~~~~~~~~~~~~~~~~~

Each sparse autoencoder feature should represent a coherent biological concept:

- **Feature 42**: Active in T cells → DEGs include CD3, CD8 → Enriched for "T cell activation"
- **Feature 137**: Active in stressed cells → DEGs include heat shock proteins → Enriched for "stress response"  
- **Feature 299**: Active in cycling cells → DEGs include cyclins, CDKs → Enriched for "cell cycle"

Quality Metrics
~~~~~~~~~~~~~~~

Look for these indicators of good feature learning:

- **Sparsity**: Each feature should activate in only a subset of cells
- **Biological coherence**: DEGs should make biological sense
- **Enrichment significance**: p-values < 0.05 for pathway enrichment
- **Effect sizes**: Log fold changes > 0.25 for meaningful differences

Common Issues
-------------

**No Significant Results**

Try adjusting these parameters:

.. code-block:: yaml

   analysis:
     activation_threshold: 0.1    # Lower threshold
     top_k_features: 100          # Analyze more features

**Memory Errors**

Reduce computational requirements:

.. code-block:: yaml

   sae:
     batch_size: 256              # Smaller batches
     n_features: 500              # Fewer features

**Poor Feature Quality**

Adjust training parameters:

.. code-block:: yaml

   sae:
     sparsity_penalty: 0.01       # Increase sparsity
     learning_rate: 0.0001        # Lower learning rate
     n_epochs: 200                # More training

Next Steps
----------

Now that you've run your first analysis:

1. **Explore the tutorials**: :doc:`tutorials` for more detailed examples
2. **Read the user guide**: :doc:`../usage/scfeaturelens` for comprehensive documentation
3. **Check the API reference**: :doc:`../api_reference` for detailed function documentation
4. **Join the community**: :doc:`../community/support` for help and discussions

Advanced Usage
--------------

For advanced analyses, sc_mechinterp supports:

- **Custom gene sets**: Define your own pathway databases
- **Batch processing**: Analyze multiple datasets automatically  
- **Integration with scanpy**: Seamless workflow with existing pipelines
- **Reproducible environments**: Docker containers for exact reproducibility

See the :doc:`../usage/advanced` guide for details on these features.
