#!/usr/bin/env python3
"""
Pipeline for scFeatureLens: Mechanistic interpretability of single-cell RNA-seq 
foundation models using sparse autoencoders.

This module implements the complete analysis pipeline, including:
- SAE training
- Feature extraction
- DEG analysis
- GO enrichment
"""

import torch
import numpy as np
import pandas as pd
import anndata as ad
import multiprocessing as mp
import logging
import math
import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

# Statistical packages
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

# GO analysis packages (with error handling if not installed)
try:
    from goatools.obo_parser import GODag
    from goatools.anno.gaf_reader import GafReader
    GO_AVAILABLE = True
except ImportError:
    GO_AVAILABLE = False

# Import the SparseAutoencoder from our package
from tools.scFeatureLens.sae import SparseAutoencoder, train_sae
from tools.scFeatureLens.utils import AnalysisConfig, load_embeddings, load_expression_data, load_predictions, load_dispersions, prep_go_sets
from tools.scFeatureLens.analysis_functions import DEG_analysis_unpaired, go_analysis

class SCFeatureLensPipeline:
    """Pipeline for Sparse Autoencoder analysis of embeddings."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize the pipeline with the given configuration."""
        self.config = config
        self.logger = self._setup_logger()
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        # Initialize storage for models and data
        self.embeddings = None
        self.expression_data = None
        self.sae_model = None
        self.prediction_model = None
        self.activations = None
        self.dispersions = None  # For DEG analysis
        self.active_features = None
        self.gene_expression = None  # Either real or predicted
        self.gene_names = None
        
        # GO analysis data
        self.go_dag = None
        self.go_assoc = None
        self.go_data = None
        
        # Results storage
        self.deg_results = {}
        self.go_results = {}
        self.feature_stats = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("scFeatureLens")
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def load_embeddings(self):
        """Load embeddings from file."""
        self.logger.info(f"Loading embeddings from {self.config.embeddings_path}")

        try:
            self.embeddings = load_embeddings(self.config.embeddings_path)
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def load_expression_data(self):
        """Load gene expression data if needed for DEG analysis."""
        if not self.config.expression_data_path:
            self.logger.info("No expression data path specified, skipping")
            return
        
        self.logger.info(f"Loading expression data from {self.config.expression_data_path}")
        try:
            self.expression_data, self.gene_names, self.gene_expression = load_expression_data(self.config.expression_data_path)
            self.logger.info(f"Loaded expression data with {self.expression_data.shape[0]} cells and {self.expression_data.shape[1]} genes")
        except Exception as e:
            self.logger.error(f"Failed to load expression data: {e}")
            raise
    
    def load_predictions(self):
        """Load predictions from file for DEG analysis instead of real data."""
        if not hasattr(self.config, "predictions_path") or not self.config.predictions_path:
            self.logger.info("No predictions path provided, will use expression data")
            return
        
        self.logger.info(f"Loading predictions from {self.config.predictions_path}")
        
        try:
            self.gene_expression = load_predictions(self.config.predictions_path)
            self.using_predictions = True
            
        except Exception as e:
            self.logger.error(f"Failed to load predictions: {e}")
            raise

    def load_dispersions(self):
        """Load dispersion factors for negative binomial model in DEG analysis."""
        if not hasattr(self.config, "dispersions_path") or not self.config.dispersions_path:
            self.logger.info("No dispersions path provided, will use default dispersions")
            return
        
        self.logger.info(f"Loading dispersions from {self.config.dispersions_path}")
        
        try:
            self.dispersions = load_dispersions(self.config.dispersions_path)
            self.logger.info(f"Loaded dispersions with shape {self.dispersions.shape}")
            
            # Flag that we have dispersions for NB model
            self.using_nb_model = True
            
        except Exception as e:
            self.logger.error(f"Failed to load dispersions: {e}")
            raise
    
    def load_sae_model(self):
        """Load a pre-trained SAE model."""
        if not self.config.sae_model_path:
            self.logger.warning("No SAE model path specified, cannot load model")
            return
        
        self.logger.info(f"Loading SAE model from {self.config.sae_model_path}")
        try:
            # Create model instance
            input_size = self.embeddings.shape[1]
            hidden_size = self.config.sae_hidden_size
            self.sae_model = SparseAutoencoder(input_size, hidden_size)
            
            # Load state dict
            self.sae_model.load_state_dict(torch.load(self.config.sae_model_path))
            self.sae_model.to(self.config.device)
            self.logger.info(f"Successfully loaded SAE model")
        except Exception as e:
            self.logger.error(f"Failed to load SAE model: {e}")
            raise
    
    def train_sae(self):

        self.sae_model = train_sae(self.embeddings, self.config, self.logger)
    
    def extract_activations(self):
        """Extract activations from the trained SAE model."""
        if self.sae_model is None:
            self.logger.error("No SAE model available for activation extraction")
            return
        
        self.logger.info("Extracting SAE activations...")
        
        # Extract in batches
        batch_size = self.config.sae_batch_size
        device = torch.device(self.config.device)
        activations = []
        
        with torch.no_grad():
            for i in range(0, self.embeddings.shape[0], batch_size):
                batch = self.embeddings[i:i+batch_size].to(device)
                _, batch_activations = self.sae_model(batch)
                activations.append(batch_activations.cpu())
        
        self.activations = torch.cat(activations, dim=0)
        
        # Save activations
        activations_path = os.path.join(self.config.output_dir, "activations.pt")
        torch.save(self.activations, activations_path)
        self.logger.info(f"Extracted activations with shape {self.activations.shape}, saved to {activations_path}")
    
    def select_active_features(self):
        """Select active features based on activation patterns."""
        if self.activations is None:
            self.logger.error("No activations available for feature selection")
            return []
        
        self.logger.info("Selecting active features...")

        active_feature_ids = torch.where(self.activations.sum(dim=0) > 0)[0]
        n_zeros_per_feature = (self.activations == 0).sum(dim=0)
        nonzero_feats = torch.Tensor(list(
            set(torch.where(n_zeros_per_feature > self.activations.shape[0] - 100)[0].numpy()).union(
            set(torch.where(n_zeros_per_feature < (self.activations.shape[0] - 100))[0].numpy()))
        ))
        active_feature_ids = torch.Tensor(list(set(active_feature_ids.numpy()).intersection(set(nonzero_feats.numpy())))).long()
        
        self.active_features = active_feature_ids
        self.logger.info(f"Selected {len(self.active_features)} active features")
        
        # Calculate feature statistics
        if len(self.active_features) > 0:
            self._calculate_feature_statistics()
        
        return self.active_features
    
    def _calculate_feature_statistics(self):
        """Calculate statistics for each active feature."""
        if self.active_features is None or len(self.active_features) == 0:
            return
        
        feature_stats = []
        for feat_idx in self.active_features:
            feat_activations = self.activations[:, feat_idx]
            stats = {
                'feature_id': feat_idx.item(),
                'n_active_samples': (feat_activations > 0).sum().item(),
                'mean_activation': feat_activations[feat_activations > 0].mean().item() if (feat_activations > 0).sum() > 0 else 0,
                'max_activation': feat_activations.max().item(),
                'std_activation': feat_activations[feat_activations > 0].std().item() if (feat_activations > 0).sum() > 1 else 0
            }
            feature_stats.append(stats)
        
        self.feature_stats = pd.DataFrame(feature_stats)
        stats_path = os.path.join(self.config.output_dir, "feature_statistics.csv")
        self.feature_stats.to_csv(stats_path, index=False)
        self.logger.info(f"Feature statistics saved to {stats_path}")
    
    def save_config(self):
        """Save the configuration to a JSON file."""
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        self.logger.info(f"Configuration saved to {config_path}")
    
    def save_results(self):
        """Save all results to the output directory."""
        # Config already saved in other methods
        
        # Save feature statistics if not already saved
        if self.feature_stats is not None:
            stats_path = os.path.join(self.config.output_dir, "feature_statistics.csv")
            self.feature_stats.to_csv(stats_path, index=False)
        
        # Save summary statistics
        summary = {
            'total_embeddings': self.embeddings.shape[0] if self.embeddings is not None else 0,
            'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'active_features': len(self.active_features) if self.active_features is not None else 0,
            'total_genes': self.gene_expression.shape[1] if self.gene_expression is not None else 0,
        }
        
        summary_path = os.path.join(self.config.output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.save_config()

        # saving 
        self.results_df.to_csv(os.path.join(self.config.output_dir, "go_enrichment_results.csv"), index=False)
        np.save(os.path.join(self.config.output_dir, "gene_hits_per_feature.npy"), self.gene_hits_per_feature)
        
        return summary
    
    def perform_differential_expression_and_go_enrichment(self):
        # if no gene sets were given, use GO
        adata_go_filtered = prep_go_sets(self.gene_names, self.logger)
        gene_hits_per_feature = np.zeros((len(self.active_features), adata_go_filtered.X.shape[1]))

        # run the analysis
        # modify the tqdm message
        import time
        # move some things outside for more efficiency
        gene_name_to_ref_idx_map = {name: i for i, name in enumerate(adata_go_filtered.var['name'])}
        pool = mp.Pool(mp.cpu_count())
        result_columns = ['n_hits', 'expected', 'binom_pval', 'binom_direction', 'binom_fold_change', 'fdr', 'z_score', 'mw_pval', 'effect_size']
        results_df = []

        import tqdm
        pbar = tqdm.tqdm(range(len(self.active_features)))
        for i in pbar:
            start_time = time.monotonic()
            feat = self.active_features[i]

            ###
            # create feature-specific sample sets for DEG
            ###

            # get the top 1% of the feature
            top_pct = np.percentile(self.activations[:,feat].numpy(), self.config.activation_percentile)
            top_pct_indices = torch.where(self.activations[:,feat] > top_pct)[0]
            bottom_pct = np.percentile(self.activations[:,feat].numpy(), 100 - self.config.activation_percentile)
            # get the reps
            y_pos = self.gene_expression[top_pct_indices, :]
            bottom_indices = torch.where(self.activations[:,feat] == 0.0)[0]
            # if there are max_active_samples, limit the bottom indices
            if self.config.max_active_samples is not None and len(bottom_indices) > self.config.max_active_samples:
                bottom_indices = bottom_indices[:self.config.max_active_samples]
            bottom_pct_indices = torch.where(self.activations[:,feat] < bottom_pct)[0]
            if self.config.max_active_samples is not None and len(bottom_indices) > self.config.max_active_samples:
                bottom_pct_indices = bottom_pct_indices[:self.config.max_active_samples]
            bottom_indices = torch.cat((bottom_pct_indices, bottom_indices))

            y_neg = self.gene_expression[bottom_indices, :]

            ###
            # DEG analysis
            ###
            #current_time1 = time.monotonic()
            gene_p_values = DEG_analysis_unpaired(y_pos, y_neg, self.gene_names, pool)
            #current_time2 = time.monotonic()
            #print(f"Finished: DEG. Took: {current_time2 - current_time1:.4f} seconds.")

            gene_p_values_ranked = gene_p_values.sort_values(by='fold_change', ascending=True)
            gene_p_values_ranked['rank'] = range(1, len(gene_p_values_ranked)+1)
            # now sort them by the adata_go matrix
            #gene_p_values_ranked['ref_idx'] = [list(adata_go.var['name']).index(gene) for gene in gene_p_values_ranked['gene']]
            gene_p_values_ranked['ref_idx'] = [gene_name_to_ref_idx_map.get(gene) for gene in gene_p_values_ranked['gene']]
            max_fold_change = gene_p_values_ranked['fold_change'].max()
            min_p_value = gene_p_values_ranked['adj_p_value'].min()
            # adjust the thresholds for GO analysis to prevent empty lists for binomial analysis
            current_p_threshold = self.config.deg_p_threshold
            if min_p_value > current_p_threshold:
                current_p_threshold = 0.05
            current_fold_change_threshold = self.config.deg_fold_change_threshold
            if max_fold_change < current_fold_change_threshold:
                current_fold_change_threshold = None
            
            gene_df_temp = gene_p_values_ranked[(gene_p_values_ranked['adj_p_value'] < current_p_threshold)]
            if current_fold_change_threshold is not None:
                gene_df_temp = gene_df_temp[(gene_df_temp['fold_change'] > current_fold_change_threshold) | (gene_df_temp['fold_change'] < 1/current_fold_change_threshold)]
            n_top_genes = len(gene_df_temp)
            pbar.set_description(f"{i}: Feat {feat} w {n_top_genes} top genes")
            
            ###
            # GO analysis
            ###

            #current_time1 = time.monotonic()
            results = pool.starmap(go_analysis, [(gene_p_values_ranked, go_idx, adata_go_filtered, current_p_threshold, current_fold_change_threshold) for go_idx in range(adata_go_filtered.X.shape[0])])
            #current_time2 = time.monotonic()
            #print(f"Finished: GO analysis for feature {feat.item()}. Took: {current_time2 - current_time1:.4f} seconds.")
            feat_pos = np.where(self.active_features == feat)[0][0]
            gene_hits_per_feature[feat_pos,results[-1][-1]] = 1

            # make the results a dataframe
            results_df_temp = pd.DataFrame(results)
            results_df_temp = results_df_temp.iloc[:,:-1]
            results_df_temp.columns = result_columns
            results_df_temp['go_id'] = adata_go_filtered.obs['go_id'].values
            results_df_temp['go_name'] = adata_go_filtered.obs['go_name'].values
            results_df_temp['go_level'] = adata_go_filtered.obs['go_level'].values
            results_df_temp['feature'] = feat.item()
            results_df_temp['p_threshold'] = current_p_threshold
            results_df_temp['fold_change_threshold'] = current_fold_change_threshold
            results_df_temp['n_top_genes'] = n_top_genes
            end_time = time.monotonic()
            results_df_temp['time'] = end_time - start_time
            results_df.append(results_df_temp)
        pbar.close()
        pool.close()
        pool.join()
        self.logger.info("Finished DEG and GO enrichment analysis for all features.")
        self.results_df = pd.concat(results_df, ignore_index=True)
        self.gene_hits_per_feature = gene_hits_per_feature
    
    def run_analysis(self):
        """Run the complete analysis pipeline based on configuration."""
        # Load embeddings
        self.load_embeddings()
        
        # Load expression data if path is provided
        if hasattr(self.config, 'expression_data_path') and self.config.expression_data_path:
            self.load_expression_data()
        elif hasattr(self.config, 'gene_expression_data_path') and self.config.gene_expression_data_path:
            # Handle the cli.py naming convention
            self.config.expression_data_path = self.config.gene_expression_data_path
            self.load_expression_data()
        if hasattr(self.config, 'predictions_path') and self.config.predictions_path:
            self.load_predictions()
        if hasattr(self.config, 'dispersions_path') and self.config.dispersions_path:
            self.load_dispersions()
    
        # Early return if only training SAE
        if hasattr(self.config, 'train_only') and self.config.train_only:
            if self.config.train_sae:
                self.train_sae()
            self.logger.info("SAE training completed. Exiting early due to train_only flag.")
            return
        
        # Either load or train the SAE model
        if not self.config.train_sae and hasattr(self.config, 'sae_model_path') and self.config.sae_model_path:
            self.load_sae_model()
        else:
            self.train_sae()
        
        # Extract activations
        self.extract_activations()
        
        # Select active features
        self.select_active_features()
        
        # run deg and enrichment
        self.perform_differential_expression_and_go_enrichment()

        # Save results
        return self.save_results()