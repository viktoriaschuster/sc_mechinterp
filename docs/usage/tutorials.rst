Tutorials
=========

This page provides step-by-step tutorials for using sc_mechinterp.

Basic Tutorial
-------------

This tutorial guides you through a basic analysis with scFeatureLens.

Advanced Tutorial
----------------

This tutorial covers more advanced features of scFeatureLens.
Tutorials
=========

Learn how to use sc_mechinterp with hands-on examples and step-by-step tutorials.

.. note::
   All tutorials use example data included in the repository. You can run them immediately after installation.

Tutorial Overview
-----------------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - Description
     - Duration
     - Prerequisites
   * - :ref:`basic-tutorial`
     - Complete workflow with synthetic data
     - 10 minutes
     - Basic installation
   * - :ref:`pretrained-tutorial`
     - Using pre-trained models
     - 5 minutes
     - Basic installation
   * - :ref:`go-tutorial`
     - Gene Ontology enrichment analysis
     - 15 minutes
     - Basic installation
   * - :ref:`geneformer-tutorial`
     - Analyzing Geneformer embeddings
     - 20 minutes
     - Geneformer data
   * - :ref:`custom-tutorial`
     - Custom datasets and configurations
     - 25 minutes
     - Your own data

.. _basic-tutorial:

Tutorial 1: Basic Analysis Workflow
-----------------------------------

This tutorial covers the complete scFeatureLens workflow using synthetic data.

**What you'll learn:**
- How to train a sparse autoencoder
- Feature extraction and analysis
- Differential expression analysis
- Result interpretation

**Step 1: Run the Basic Example**

.. code-block:: bash

   cd sc_mechinterp
   python -m tools.scFeatureLens.example --example basic

**Step 2: Examine the Configuration**

.. code-block:: bash

   cat example_results/config.yaml

This shows all parameters used in the analysis:

.. code-block:: yaml

   sae:
     n_features: 1000
     sparsity_penalty: 0.001
     learning_rate: 0.001
     n_epochs: 100
     batch_size: 512
   
   analysis:
     top_k_features: 50
     activation_threshold: 0.5
     min_cells_per_group: 10

**Step 3: Analyze Feature Activations**

Let's examine which features are most active:

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt

   # Load feature activations
   activations = torch.load('example_results/sae_activations.pt')
   
   # Calculate feature activity
   feature_activity = (activations > 0.5).float().mean(dim=0)
   
   # Plot most active features
   plt.figure(figsize=(10, 6))
   plt.hist(feature_activity, bins=50)
   plt.xlabel('Fraction of cells where feature is active')
   plt.ylabel('Number of features')
   plt.title('Feature Activity Distribution')
   plt.show()

**Step 4: Examine Differential Expression Results**

.. code-block:: python

   import pandas as pd

   # Load DE results for most active feature
   de_results = pd.read_csv('example_results/differential_expression/feature_001_degs.csv')
   
   # Show top upregulated genes
   top_up = de_results[de_results['log_fold_change'] > 0].head(10)
   print("Top upregulated genes:")
   print(top_up[['gene', 'log_fold_change', 'p_value_adj']])

**Expected Output:**

.. code-block:: text

   Top upregulated genes:
         gene  log_fold_change  p_value_adj
   0    GENE1             2.34        0.001
   1    GENE5             1.98        0.002
   2   GENE12             1.76        0.003
   ...

**Step 5: Interpret Results**

The basic example creates interpretable features that correspond to:
- Cell type markers
- Stress response genes
- Cell cycle signatures
- Metabolic pathways

.. _pretrained-tutorial:

Tutorial 2: Using Pre-trained Models
------------------------------------

Skip training time by using pre-trained sparse autoencoders.

**What you'll learn:**
- Loading pre-trained SAE models
- Feature analysis without training
- Comparing different model architectures

**Step 1: Run Pre-trained Example**

.. code-block:: bash

   python -m tools.scFeatureLens.example --example pretrained

This uses a pre-trained SAE model included in the repository.

**Step 2: Load and Examine the Model**

.. code-block:: python

   from tools.scFeatureLens import SAE
   import torch

   # Load pre-trained model
   model = SAE.load('examples/scFeatureLens/sae_geneformer_human-bonemarrow_Luecken_activations.pt')
   
   print(f"Model architecture:")
   print(f"Input dimension: {model.input_dim}")
   print(f"Hidden dimension: {model.hidden_dim}")
   print(f"Number of features: {model.n_features}")

**Step 3: Analyze Feature Quality**

.. code-block:: python

   # Load activations from pre-trained analysis
   activations = torch.load('example_pretrained_results/sae_activations.pt')
   
   # Calculate sparsity metrics
   sparsity = (activations == 0).float().mean()
   print(f"Overall sparsity: {sparsity:.3f}")
   
   # Feature-wise sparsity
   feature_sparsity = (activations == 0).float().mean(dim=0)
   print(f"Mean feature sparsity: {feature_sparsity.mean():.3f}")

**Step 4: Compare with Basic Example**

Compare the quality of features between basic and pre-trained examples:

.. code-block:: python

   import pandas as pd
   
   # Load summary statistics
   basic_summary = pd.read_json('example_results/analysis_summary.json', typ='series')
   pretrained_summary = pd.read_json('example_pretrained_results/analysis_summary.json', typ='series')
   
   comparison = pd.DataFrame({
       'Basic': basic_summary,
       'Pre-trained': pretrained_summary
   })
   print(comparison)

.. _go-tutorial:

Tutorial 3: Gene Ontology Enrichment Analysis
---------------------------------------------

Perform comprehensive biological interpretation using Gene Ontology terms.

**What you'll learn:**
- Gene Ontology enrichment analysis
- Interpreting biological pathways
- Custom gene set analysis

**Step 1: Run GO Enrichment Example**

.. code-block:: bash

   python -m tools.scFeatureLens.example --example go

**Step 2: Examine Enrichment Results**

.. code-block:: python

   import pandas as pd

   # Load enrichment results for top feature
   enrichment = pd.read_csv('example_results/gene_set_enrichment/feature_001_enrichment.csv')
   
   # Show top enriched pathways
   top_pathways = enrichment.head(10)
   print("Top enriched pathways:")
   print(top_pathways[['term_name', 'p_value', 'fold_enrichment', 'gene_count']])

**Expected Output:**

.. code-block:: text

   Top enriched pathways:
                           term_name   p_value  fold_enrichment  gene_count
   0                  T cell activation  0.00001             3.45          23
   1           immune system process  0.00002             2.87          45
   2               cell proliferation  0.00005             2.34          34
   ...

**Step 3: Visualize Enrichment Results**

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Create enrichment plot
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
   
   # Plot 1: P-values
   ax1.barh(range(10), -np.log10(top_pathways['p_value']))
   ax1.set_yticks(range(10))
   ax1.set_yticklabels(top_pathways['term_name'], fontsize=10)
   ax1.set_xlabel('-log10(p-value)')
   ax1.set_title('Pathway Significance')
   
   # Plot 2: Fold enrichment
   ax2.barh(range(10), top_pathways['fold_enrichment'])
   ax2.set_yticks(range(10))
   ax2.set_yticklabels(top_pathways['term_name'], fontsize=10)
   ax2.set_xlabel('Fold Enrichment')
   ax2.set_title('Enrichment Strength')
   
   plt.tight_layout()
   plt.show()

**Step 4: Analyze Feature Coherence**

.. code-block:: python

   # Calculate feature coherence based on enrichment
   coherence_scores = []
   
   for feature_id in range(1, 51):  # Top 50 features
       try:
           enrich_file = f'example_results/gene_set_enrichment/feature_{feature_id:03d}_enrichment.csv'
           enrich_data = pd.read_csv(enrich_file)
           
           # Coherence = number of significant pathways
           n_significant = (enrich_data['p_value'] < 0.05).sum()
           coherence_scores.append(n_significant)
       except FileNotFoundError:
           coherence_scores.append(0)
   
   # Plot coherence distribution
   plt.figure(figsize=(10, 6))
   plt.hist(coherence_scores, bins=20)
   plt.xlabel('Number of significantly enriched pathways')
   plt.ylabel('Number of features')
   plt.title('Feature Biological Coherence')
   plt.show()

.. _geneformer-tutorial:

Tutorial 4: Analyzing Geneformer Embeddings
-------------------------------------------

Analyze real embeddings from the Geneformer foundation model.

**Prerequisites:**
- Geneformer embeddings (download instructions below)
- Gene expression data

**Step 1: Download Geneformer Embeddings**

.. code-block:: bash

   # Download example Geneformer embeddings
   cd examples/scFeatureLens
   wget https://example.com/geneformer_embeddings.pt

**Step 2: Run Analysis**

.. code-block:: bash

   python -m tools.scFeatureLens.cli geneformer_embeddings.pt \
       --gene-expression gene_expression.h5ad \
       --output-dir geneformer_results \
       --n-features 2000 \
       --sparsity-penalty 0.005

**Step 3: Analyze Geneformer-Specific Features**

.. code-block:: python

   import pandas as pd
   import numpy as np

   # Load Geneformer analysis results
   summary = pd.read_json('geneformer_results/analysis_summary.json', typ='series')
   
   print("Geneformer Analysis Summary:")
   print(f"Total features: {summary['total_features']}")
   print(f"Active features: {summary['active_features']}")
   print(f"Features with significant DEGs: {summary['features_with_degs']}")
   print(f"Features with pathway enrichment: {summary['features_with_enrichment']}")

**Step 4: Compare Cell Type Signatures**

.. code-block:: python

   # Analyze cell type specific features
   import anndata as ad
   
   # Load gene expression data
   adata = ad.read_h5ad('gene_expression.h5ad')
   
   # Load feature activations
   activations = torch.load('geneformer_results/sae_activations.pt')
   
   # Identify cell type specific features
   cell_types = adata.obs['cell_type'].unique()
   
   for cell_type in cell_types:
       cell_mask = adata.obs['cell_type'] == cell_type
       type_activations = activations[cell_mask].mean(dim=0)
       
       # Find most active features for this cell type
       top_features = torch.topk(type_activations, k=5).indices
       print(f"\n{cell_type} top features: {top_features.tolist()}")

.. _custom-tutorial:

Tutorial 5: Custom Datasets and Advanced Configuration
------------------------------------------------------

Learn to analyze your own datasets with custom configurations.

**What you'll learn:**
- Data preprocessing for sc_mechinterp
- Advanced configuration options
- Batch processing multiple datasets

**Step 1: Prepare Your Data**

.. code-block:: python

   import numpy as np
   import torch
   import anndata as ad

   # Example: Convert your embeddings to supported format
   
   # From numpy array
   embeddings = np.load('your_embeddings.npy')
   torch.save(torch.tensor(embeddings), 'embeddings.pt')
   
   # From AnnData obsm
   adata = ad.read_h5ad('your_data.h5ad')
   embeddings = adata.obsm['X_embedding']  # or your embedding key
   torch.save(torch.tensor(embeddings), 'embeddings.pt')

**Step 2: Create Custom Configuration**

.. code-block:: yaml

   # custom_config.yaml
   sae:
     n_features: 2000              # More features for complex data
     sparsity_penalty: 0.01        # Higher sparsity
     learning_rate: 0.0005         # Lower learning rate
     n_epochs: 200                 # More training epochs
     batch_size: 256               # Smaller batches for memory
   
   analysis:
     top_k_features: 100           # Analyze more features
     activation_threshold: 0.3     # Lower threshold
     min_cells_per_group: 20       # Require more cells per group
   
   differential_expression:
     test_method: "wilcoxon"       # Statistical test
     p_value_threshold: 0.01       # Stricter significance
     log_fold_change_threshold: 0.5  # Higher effect size
   
   gene_set_enrichment:
     use_go_terms: true
     p_value_threshold: 0.01
     min_gene_set_size: 10
     max_gene_set_size: 300

**Step 3: Run Custom Analysis**

.. code-block:: bash

   python -m tools.scFeatureLens.cli embeddings.pt \
       --gene-expression gene_expression.h5ad \
       --config custom_config.yaml \
       --output-dir custom_results \
       --verbose

**Step 4: Batch Process Multiple Datasets**

.. code-block:: python

   from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig
   import os

   # Define datasets
   datasets = [
       {'embeddings': 'dataset1_embeddings.pt', 'expression': 'dataset1_expr.h5ad'},
       {'embeddings': 'dataset2_embeddings.pt', 'expression': 'dataset2_expr.h5ad'},
       {'embeddings': 'dataset3_embeddings.pt', 'expression': 'dataset3_expr.h5ad'},
   ]

   # Configure analysis
   config = AnalysisConfig(
       n_features=1500,
       sparsity_penalty=0.005,
       n_epochs=150
   )

   # Process each dataset
   for i, dataset in enumerate(datasets):
       print(f"Processing dataset {i+1}/{len(datasets)}")
       
       output_dir = f"batch_results/dataset_{i+1}"
       os.makedirs(output_dir, exist_ok=True)
       
       pipeline = SCFeatureLensPipeline(config)
       results = pipeline.run_analysis(
           embeddings_path=dataset['embeddings'],
           gene_expression_path=dataset['expression'],
           output_dir=output_dir
       )
       
       print(f"Completed: {results['n_active_features']} active features found")

**Step 5: Compare Results Across Datasets**

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt

   # Collect summary statistics
   summaries = []
   for i in range(1, len(datasets) + 1):
       summary = pd.read_json(f'batch_results/dataset_{i}/analysis_summary.json', typ='series')
       summary['dataset'] = f'Dataset {i}'
       summaries.append(summary)

   # Create comparison DataFrame
   comparison_df = pd.DataFrame(summaries)
   
   # Plot comparison
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   axes[0,0].bar(comparison_df['dataset'], comparison_df['active_features'])
   axes[0,0].set_title('Active Features')
   axes[0,0].set_ylabel('Count')
   
   axes[0,1].bar(comparison_df['dataset'], comparison_df['features_with_degs'])
   axes[0,1].set_title('Features with DEGs')
   axes[0,1].set_ylabel('Count')
   
   axes[1,0].bar(comparison_df['dataset'], comparison_df['n_significant_degs'])
   axes[1,0].set_title('Total Significant DEGs')
   axes[1,0].set_ylabel('Count')
   
   axes[1,1].bar(comparison_df['dataset'], comparison_df['features_with_enrichment'])
   axes[1,1].set_title('Features with Enrichment')
   axes[1,1].set_ylabel('Count')
   
   plt.tight_layout()
   plt.show()

Next Steps
----------

After completing these tutorials:

1. **Deep dive into scFeatureLens**: Read the :doc:`../usage/scfeaturelens` guide
2. **Explore advanced features**: Check the :doc:`../usage/advanced` documentation  
3. **API reference**: Browse the complete :doc:`../api_reference` documentation
4. **Join the community**: Visit :doc:`../community/support` for help and discussions

Troubleshooting
---------------

If you encounter issues with any tutorial:

1. Check that your environment is properly set up: ``python setup/validate_environment.py``
2. Verify you have the latest version: ``git pull origin main``
3. Review the :doc:`../community/support` page for common solutions
4. Open an issue on GitHub with your specific error message
