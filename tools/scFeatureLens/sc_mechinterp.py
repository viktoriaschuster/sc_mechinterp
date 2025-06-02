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

class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for mechanistic interpretability of embeddings."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class SCMechInterpPipeline:
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
        
        # Find features with any activation
        active_feature_ids = torch.where(self.activations.sum(dim=0) > 0)[0]
        
        # Filter by minimum and maximum active samples
        n_zeros_per_feature = (self.activations == 0).sum(dim=0)
        
        valid_features = []
        for feat_id in active_feature_ids:
            n_active = self.activations.shape[0] - n_zeros_per_feature[feat_id]
            
            if n_active >= self.config.min_active_samples:
                if self.config.max_active_samples is None or n_active <= self.config.max_active_samples:
                    valid_features.append(feat_id)
        
        valid_features = torch.tensor(valid_features, dtype=torch.long)
        
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
        
        # Implementation depends on the format of your custom gene sets
        # This is a placeholder
        pass
    
    def run_deg_analysis(self, y_pos: np.ndarray, y_neg: np.ndarray, 
                        gene_names: List[str], pool: mp.Pool) -> pd.DataFrame:
        """Run differential expression analysis."""
        fold_changes = y_pos.mean(axis=0) / y_neg.mean(axis=0)
        
        if len(y_pos.shape) < 2:
            y_pos = np.expand_dims(y_pos, axis=0)
        if len(y_neg.shape) < 2:
            y_neg = np.expand_dims(y_neg, axis=0)
        
        # Pre-filter valid genes
        pos_std = np.std(y_pos, axis=0) > 0
        neg_std = np.std(y_neg, axis=0) > 0
        pos_no_nan = ~np.isnan(y_pos).any(axis=0)
        neg_no_nan = ~np.isnan(y_neg).any(axis=0)
        valid_mask = pos_std & neg_std & pos_no_nan & neg_no_nan
        valid_genes = np.where(valid_mask)[0]
        
        # Initialize p-values
        p_values = np.ones(y_pos.shape[1])
        
        # Run t-tests on valid genes
        try:
            results = pool.starmap(stats.ttest_ind, 
                                 [(y_pos[:, gene], y_neg[:, gene], 0, False) 
                                  for gene in valid_genes])
            
            for i, gene in enumerate(valid_genes):
                p_values[gene] = results[i].pvalue
                
        except Exception as e:
            self.logger.error(f"Error in t-tests: {e}")
        
        # Create results DataFrame
        gene_p_values = pd.DataFrame({
            'gene': gene_names,
            'p_value': p_values,
            'fold_change': fold_changes
        })
        
        # Adjust p-values for multiple testing
        gene_p_values['adj_p_value'] = multipletests(
            gene_p_values['p_value'], method='fdr_bh')[1]
        
        # Sort by p-value
        gene_p_values = gene_p_values.sort_values(by='p_value')
        
        return gene_p_values
    
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
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.logger.info("Starting analysis pipeline...")
        
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
        
        # Step 6: Load gene expression data if provided
        if self.config.gene_expression_data_path:
            self.logger.info("Loading gene expression data...")
            # Implementation depends on your data format
            # This is where you would load the actual gene expression data
            # for differential expression analysis
        
        # Step 7: Run feature analysis
        self.logger.info("Running feature analysis...")
        
        # Create multiprocessing pool
        pool = mp.Pool(self.config.n_processes)
        results_df = []
        
        try:
            for i, feat_id in enumerate(tqdm.tqdm(active_features, desc="Analyzing features")):
                # Here you would implement the actual feature analysis
                # This includes:
                # 1. Selecting high/low activation samples
                # 2. Running differential expression analysis
                # 3. Running enrichment analysis
                # 4. Collecting results
                
                # Placeholder for now
                self.logger.debug(f"Processing feature {feat_id}")
                
        finally:
            pool.close()
            pool.join()
        
        # Step 8: Save results
        self.logger.info("Analysis complete!")
    
    def save_config(self):
        """Save analysis configuration."""
        config_path = Path(self.config.output_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Single-Cell Mechanistic Interpretability Analysis Tool"
    )
    
    # Required arguments
    parser.add_argument(
        "embeddings_path",
        type=str,
        help="Path to embeddings file (.pt, .npy, .csv, or .h5ad)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--sae-model-path",
        type=str,
        help="Path to pre-trained SAE model file"
    )
    
    parser.add_argument(
        "--no-train-sae",
        action="store_true",
        help="Don't train SAE, use existing model"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = AnalysisConfig(**config_dict)
    else:
        config = AnalysisConfig(embeddings_path=args.embeddings_path)
    
    # Override config with command-line arguments
    if args.output_dir != "results":
        config.output_dir = args.output_dir
    if args.sae_model_path:
        config.sae_model_path = args.sae_model_path
    if args.no_train_sae:
        config.train_sae = False
    if args.verbose:
        config.verbose = True
    
    # Run analysis
    pipeline = SCMechInterpPipeline(config)
    pipeline.save_config()
    pipeline.run_analysis()


if __name__ == "__main__":
    main()
