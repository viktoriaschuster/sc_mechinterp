#!/usr/bin/env python3
"""
scFeatureLens: Single-Cell Feature Lens for Mechanistic Interpretability

A tool for extracting meaningful features from single-cell RNA-seq data model embeddings
using sparse autoencoders and performing biological interpretation through differential
gene expression analysis and gene set enrichment.

Author: Viktoria Schuster
License: MIT
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import anndata as ad
import math
import os
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp
import time
from dataclasses import dataclass, asdict
import yaml

# Statistical packages
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats

# GO analysis packages
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader

# Progress tracking
import tqdm

# System info
import psutil

# Local imports
from .sae import SparseAutoencoder
from .analysis_functions import (
    differential_expression_analysis,
    gene_set_enrichment_analysis,
    filter_active_features,
    select_high_low_activation_samples
)

@dataclass
class AnalysisConfig:
    """Configuration for the analysis pipeline."""
    
    # Input/Output paths
    embeddings_path: str
    output_dir: str = "results"
    sae_model_path: Optional[str] = None
    
    # Data configuration
    gene_expression_data_path: Optional[str] = None
    library_size_column: str = "n_counts"
    gene_name_column: str = "gene_name"
    
    # SAE configuration
    sae_hidden_size: int = 10000
    sae_l1_penalty: float = 1e-3
    sae_learning_rate: float = 1e-4
    sae_epochs: int = 500
    train_sae: bool = True
    
    # Feature selection
    activation_percentile: float = 99
    min_active_samples: int = 100
    max_active_samples: Optional[int] = None
    
    # Differential expression
    deg_p_threshold: float = 1e-5
    deg_fold_change_threshold: float = 2.0
    
    # Gene set analysis
    gene_sets_path: Optional[str] = None
    gene_set_type: str = "go"  # "go" or "custom"
    go_category: str = "biological_process"  # biological_process, molecular_function, cellular_component
    go_obo_path: str = "data/go-basic.obo"
    go_gaf_path: str = "data/goa_human.gaf"
    min_genes_per_set: int = 10
    max_genes_per_set: int = 500
    
    # Enrichment analysis
    enrichment_p_threshold: float = 0.05
    
    # Computational
    device: str = "auto"
    batch_size: int = 128
    n_processes: Optional[int] = None
    verbose: bool = True
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.n_processes is None:
            self.n_processes = mp.cpu_count()
            
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class SCFeatureLensPipeline:
    """Main pipeline for single-cell mechanistic interpretability analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.embeddings = None
        self.sae_model = None
        self.activations = None
        self.gene_sets_data = None
        
        # Set random seeds
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.output_dir) / 'analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_device(self):
        """Setup computation device."""
        self.device = torch.device(self.config.device)
        self.logger.info(f"Using device: {self.device}")
        
        if self.config.verbose:
            physical_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            self.logger.info("--- System Information ---")
            self.logger.info(f"Physical CPU Cores: {physical_cores}")
            self.logger.info(f"Logical CPU Cores: {logical_cores}")
            self.logger.info(f"Total Memory: {total_memory_gb:.2f} GB")
            self.logger.info(f"Available Memory: {available_memory_gb:.2f} GB")
            self.logger.info(f"Number of processes: {self.config.n_processes}")
            self.logger.info("--------------------------")
    
    def load_embeddings(self) -> torch.Tensor:
        """Load embeddings from file."""
        self.logger.info(f"Loading embeddings from {self.config.embeddings_path}")
        
        embeddings_path = Path(self.config.embeddings_path)
        
        if embeddings_path.suffix == '.pt':
            embeddings = torch.load(embeddings_path, map_location='cpu')
        elif embeddings_path.suffix == '.npy':
            embeddings = torch.from_numpy(np.load(embeddings_path)).float()
        elif embeddings_path.suffix == '.csv':
            df = pd.read_csv(embeddings_path, index_col=0)
            embeddings = torch.from_numpy(df.values).float()
        elif embeddings_path.suffix == '.h5ad':
            adata = ad.read_h5ad(embeddings_path)
            embeddings = torch.from_numpy(adata.X).float()
        else:
            raise ValueError(f"Unsupported file format: {embeddings_path.suffix}")
        
        self.embeddings = embeddings
        self.logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def train_sae(self) -> SparseAutoencoder:
        """Train sparse autoencoder on embeddings."""
        if self.embeddings is None:
            raise ValueError("Embeddings must be loaded before training SAE")
        
        self.logger.info("Training Sparse Autoencoder...")
        
        input_size = self.embeddings.shape[1]
        hidden_size = self.config.sae_hidden_size
        
        model = SparseAutoencoder(input_size, hidden_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.sae_learning_rate)
        
        # Training loop
        model.train()
        batch_size = self.config.batch_size
        n_batches = math.ceil(self.embeddings.shape[0] / batch_size)
        
        for epoch in tqdm.tqdm(range(self.config.sae_epochs), desc="Training SAE"):
            total_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, self.embeddings.shape[0])
                batch = self.embeddings[start_idx:end_idx].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                decoded, encoded = model(batch)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(decoded, batch)
                
                # L1 sparsity penalty
                l1_loss = self.config.sae_l1_penalty * torch.mean(torch.abs(encoded))
                
                # Total loss
                loss = recon_loss + l1_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 50 == 0:
                avg_loss = total_loss / n_batches
                self.logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}")
        
        # Save model
        model_path = Path(self.config.output_dir) / "sae_model.pt"
        torch.save(model.state_dict(), model_path)
        self.logger.info(f"SAE model saved to {model_path}")
        
        self.sae_model = model
        return model
    
    def load_sae(self) -> SparseAutoencoder:
        """Load pre-trained SAE model."""
        if self.config.sae_model_path is None:
            raise ValueError("SAE model path must be specified")
        
        self.logger.info(f"Loading SAE model from {self.config.sae_model_path}")
        
        input_size = self.embeddings.shape[1]
        hidden_size = self.config.sae_hidden_size
        
        model = SparseAutoencoder(input_size, hidden_size)
        model.load_state_dict(torch.load(self.config.sae_model_path, map_location='cpu'))
        model.to(self.device)
        
        self.sae_model = model
        return model
    
    def extract_activations(self) -> torch.Tensor:
        """Extract SAE activations from embeddings."""
        if self.embeddings is None:
            raise ValueError("Embeddings must be loaded before extracting activations")
        if self.sae_model is None:
            raise ValueError("SAE model must be loaded/trained before extracting activations")
        
        self.logger.info("Extracting SAE activations...")
        
        self.sae_model.eval()
        activations = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in tqdm.tqdm(range(0, self.embeddings.shape[0], batch_size), desc="Extracting activations"):
                batch = self.embeddings[i:i + batch_size].to(self.device)
                _, encoded = self.sae_model(batch)
                activations.append(encoded.cpu())
        
        activations = torch.cat(activations, dim=0)
        
        # Save activations
        activations_path = Path(self.config.output_dir) / "sae_activations.pt"
        torch.save(activations, activations_path)
        self.logger.info(f"Activations saved to {activations_path}")
        
        self.activations = activations
        return activations
    
    def select_active_features(self) -> torch.Tensor:
        """Select features that are active and meet criteria."""
        if self.activations is None:
            raise ValueError("Activations must be extracted before selecting features")
        
        self.logger.info("Selecting active features...")
        
        valid_features = filter_active_features(
            self.activations,
            self.config.min_active_samples,
            self.config.max_active_samples
        )
        
        self.logger.info(f"Selected {len(valid_features)} active features")
        return valid_features
    
    def load_gene_sets(self):
        """Load gene sets for enrichment analysis."""
        self.logger.info("Loading gene sets...")
        
        if self.config.gene_set_type == "go":
            self._load_go_gene_sets()
        elif self.config.gene_set_type == "custom":
            self._load_custom_gene_sets()
        else:
            raise ValueError(f"Unsupported gene set type: {self.config.gene_set_type}")
    
    def _load_go_gene_sets(self):
        """Load GO gene sets."""
        # This is a simplified version - you'd need to implement the full GO loading logic
        # based on your existing code in automated_analysis_fast.py
        self.logger.info("Loading GO gene sets...")
        
        # Load GO DAG and annotations
        obodag = GODag(self.config.go_obo_path)
        ogaf = GafReader(self.config.go_gaf_path)
        ns2assc = ogaf.get_ns2assc()
        
        # Filter by category and size
        go_terms = [go_id for go_id in obodag.keys() 
                   if obodag[go_id].namespace == self.config.go_category]
        
        # Here you would implement the full GO gene set loading logic
        # This is a placeholder that should be replaced with your actual implementation
        self.gene_sets_data = {
            'go_terms': go_terms,
            'obodag': obodag,
            'associations': ns2assc
        }
        
        self.logger.info(f"Loaded {len(go_terms)} GO terms for {self.config.go_category}")
    
    def _load_custom_gene_sets(self):
        """Load custom gene sets from file."""
        if self.config.gene_sets_path is None:
            raise ValueError("Gene sets path must be specified for custom gene sets")
        
        self.logger.info(f"Loading custom gene sets from {self.config.gene_sets_path}")
        
        gene_sets_path = Path(self.config.gene_sets_path)
        
        if gene_sets_path.suffix.lower() in ['.yaml', '.yml']:
            with open(gene_sets_path, 'r') as f:
                gene_sets = yaml.safe_load(f)
        elif gene_sets_path.suffix.lower() == '.json':
            with open(gene_sets_path, 'r') as f:
                gene_sets = json.load(f)
        else:
            raise ValueError(f"Unsupported gene sets file format: {gene_sets_path.suffix}")
        
        # Filter gene sets by size
        filtered_sets = {}
        for set_name, genes in gene_sets.items():
            if isinstance(genes, list) and \
               self.config.min_genes_per_set <= len(genes) <= self.config.max_genes_per_set:
                filtered_sets[set_name] = genes
        
        self.gene_sets_data = {
            'custom_sets': filtered_sets
        }
        
        self.logger.info(f"Loaded {len(filtered_sets)} custom gene sets")
        return filtered_sets
        pass
    
    def run_deg_analysis(self, y_pos: np.ndarray, y_neg: np.ndarray, 
                        gene_names: List[str], pool) -> pd.DataFrame:
        """Run differential expression analysis."""
        return differential_expression_analysis(y_pos, y_neg, gene_names, pool)
    
    def run_enrichment_analysis(self, gene_df: pd.DataFrame, gene_set_id: str) -> Dict:
        """Run enrichment analysis for a specific gene set."""
        # This is a placeholder for the enrichment analysis logic
        # You would implement the actual GO analysis functions here
        # based on your existing code
        
        # Placeholder results
        return {
            'n_hits': 0,
            'expected': 0,
            'binom_pval': 1.0,
            'binom_direction': '+',
            'binom_fold_change': 1.0,
            'fdr': 1.0,
            'z_score': 0.0,
            'mw_pval': 1.0,
            'effect_size': 0.0
        }
    
    def run_analysis(self) -> Dict:
        """Run the complete analysis pipeline."""
        self.logger.info("Starting scFeatureLens analysis pipeline...")
        
        # Save configuration
        self.save_config()
        
        # Step 1: Load embeddings
        self.load_embeddings()
        
        # Step 2: Train or load SAE
        if self.config.train_sae or self.config.sae_model_path is None:
            self.train_sae()
        else:
            self.load_sae()
        
        # Step 3: Extract activations
        self.extract_activations()
        
        # Step 4: Select active features
        active_features = self.select_active_features()
        
        # Step 5: Load gene sets
        self.load_gene_sets()
        
        # Step 6: Run analysis based on available data
        results = {}
        
        # If the input is already activations (common case), we proceed with feature analysis
        if self._is_activations_input():
            results = self._run_feature_analysis_on_activations(active_features)
        else:
            # Step 7: Load gene expression data if provided
            if self.config.gene_expression_data_path:
                self.logger.info("Loading gene expression data...")
                self._load_gene_expression_data()
                results = self._run_full_analysis_with_expression_data(active_features)
            else:
                self.logger.warning("No gene expression data provided. Analysis will be limited to feature activation patterns.")
                results = self._run_feature_analysis_only(active_features)
        
        # Step 8: Save final results summary
        self._save_results_summary(results)
        
        self.logger.info("scFeatureLens analysis complete!")
        return results
    
    def _is_activations_input(self) -> bool:
        """Check if input is already SAE activations based on filename."""
        return "activations" in str(self.config.embeddings_path).lower()
    
    def _load_gene_expression_data(self):
        """Load gene expression data for differential expression analysis."""
        if self.config.gene_expression_data_path is None:
            raise ValueError("Gene expression data path must be specified")
        
        self.logger.info(f"Loading gene expression data from {self.config.gene_expression_data_path}")
        
        expr_path = Path(self.config.gene_expression_data_path)
        
        if expr_path.suffix == '.h5ad':
            adata = ad.read_h5ad(expr_path)
            self.gene_expression_data = adata
            self.gene_names = adata.var[self.config.gene_name_column].tolist()
            self.library_sizes = adata.obs[self.config.library_size_column].values
        elif expr_path.suffix == '.csv':
            df = pd.read_csv(expr_path, index_col=0)
            # Assume genes are columns, cells are rows
            self.gene_expression_data = df.values
            self.gene_names = df.columns.tolist()
            self.library_sizes = df.sum(axis=1).values
        else:
            raise ValueError(f"Unsupported gene expression file format: {expr_path.suffix}")
        
        self.logger.info(f"Loaded gene expression data: {len(self.gene_names)} genes, {len(self.library_sizes)} cells")
    
    def _run_feature_analysis_on_activations(self, active_features: torch.Tensor) -> Dict:
        """Run analysis when input is already SAE activations."""
        self.logger.info("Running feature analysis on SAE activations...")
        
        # Create multiprocessing pool
        pool = mp.Pool(self.config.n_processes)
        results_df = []
        feature_results = []
        
        try:
            # For each active feature, we would typically:
            # 1. Identify high vs low activation samples
            # 2. If we have gene expression data, run DEG analysis
            # 3. Run gene set enrichment analysis
            
            for i, feat_id in enumerate(tqdm.tqdm(active_features, desc="Analyzing features")):
                self.logger.debug(f"Processing feature {feat_id}")
                
                # Select high and low activation samples
                high_indices, low_indices = select_high_low_activation_samples(
                    self.activations, feat_id, self.config.activation_percentile
                )
                
                self.logger.debug(f"Feature {feat_id}: {len(high_indices)} high, {len(low_indices)} low activation samples")
                
                # Feature analysis results
                feature_result = {
                    'feature_id': feat_id.item(),
                    'n_high_activation': len(high_indices),
                    'n_low_activation': len(low_indices),
                    'activation_percentile': self.config.activation_percentile,
                    'significant_degs': [],  # Would be populated with gene expression data
                    'enriched_sets': []      # Would be populated with enrichment analysis
                }
                
                results_df.append(feature_result)
                feature_results.append(feature_result)
                
        finally:
            pool.close()
            pool.join()
        
        # Save feature analysis results
        results_df = pd.DataFrame(results_df)
        results_path = Path(self.config.output_dir) / "feature_analysis_results.csv"
        results_df.to_csv(results_path, index=False)
        self.logger.info(f"Feature analysis results saved to {results_path}")
        
        return {
            'feature_results': feature_results,
            'summary': {
                'n_features_analyzed': len(feature_results),
                'total_high_activation_samples': sum(r['n_high_activation'] for r in feature_results),
                'total_low_activation_samples': sum(r['n_low_activation'] for r in feature_results)
            }
        }
    
    def _run_full_analysis_with_expression_data(self, active_features: torch.Tensor) -> Dict:
        """Run full analysis including differential expression and enrichment."""
        self.logger.info("Running full analysis with gene expression data...")
        
        # This would implement the complete pipeline from your original code
        # including DEG analysis and GO enrichment
        # For now, return basic structure
        return {
            'feature_results': [],
            'summary': {
                'n_features_analyzed': len(active_features),
                'analysis_type': 'full_with_expression'
            }
        }
    
    def _run_feature_analysis_only(self, active_features: torch.Tensor) -> Dict:
        """Run analysis without gene expression data."""
        self.logger.info("Running feature analysis without gene expression data...")
        return self._run_feature_analysis_on_activations(active_features)
    
    def _save_results_summary(self, results: Dict):
        """Save analysis results summary."""
        summary_path = Path(self.config.output_dir) / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Analysis summary saved to {summary_path}")
    
    def save_config(self):
        """Save analysis configuration."""
        config_path = Path(self.config.output_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")


# For backwards compatibility
SCMechInterpPipeline = SCFeatureLensPipeline
