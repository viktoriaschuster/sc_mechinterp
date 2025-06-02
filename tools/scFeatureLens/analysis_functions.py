"""
Core analysis functions for single-cell mechanistic interpretability.

This module contains the statistical and biological analysis functions
extracted and generalized from the original pipeline.
"""

import numpy as np
import pandas as pd
import torch
import scipy.stats as stats
import math
from typing import Dict, List, Tuple, Optional
from statsmodels.stats.multitest import multipletests
import anndata as ad
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader


def binomial_test(n_study: int, n_hit: int, n_c: int, n: int) -> Tuple[float, float, str, float, float]:
    """
    Perform binomial test for gene set enrichment.
    
    Args:
        n_study: Number of genes in the study set
        n_hit: Number of hits (genes in both study set and gene set)
        n_c: Total number of genes in the gene set
        n: Total number of genes
    
    Returns:
        Tuple of (p_value, expected, direction, fold_enrichment, fdr)
    """
    p_c = n_c / n  # Expected probability of a hit
    results = stats.binomtest(n_hit, n_study, p_c)
    over_under = '+' if n_hit > (n_study * p_c) else '-'
    fold_enrichment = n_hit / (n_study * p_c) if n_study * p_c > 0 else 0
    fdr = (n_study - n_hit) / n_study if n_study > 0 else 0
    expected = n_study * p_c
    
    return results.pvalue, expected, over_under, fold_enrichment, fdr


def mann_whitney_u_test(ranks: np.ndarray, in_set: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform Mann-Whitney U test for gene set enrichment.
    
    Args:
        ranks: Array of gene ranks
        in_set: Boolean array indicating which genes are in the set
    
    Returns:
        Tuple of (z_score, p_value, effect_size)
    """
    n_s = sum(in_set)  # Number of genes in set
    n_t = len(ranks[~in_set])  # Number of genes not in set
    
    if n_s == 0 or n_t == 0:
        return 0.0, 1.0, 0.0
    
    r_s = sum(ranks[in_set])  # Sum of ranks for genes in set
    r_t = sum(ranks[~in_set])  # Sum of ranks for genes not in set
    
    u_s = n_s * n_t + ((n_s * (n_s + 1)) / 2) - r_s
    u_t = n_s * n_t + ((n_t * (n_t + 1)) / 2) - r_t
    u = min(u_s, u_t)
    
    # Calculate z-score
    mean_u = n_s * n_t / 2
    std_u = math.sqrt(n_s * n_t * (n_s + n_t + 1) / 12)
    
    if std_u == 0:
        return 0.0, 1.0, 0.0
    
    z_score = (u - mean_u) / std_u
    p_value = stats.norm.cdf(z_score)
    effect_size = u_s / (n_s * n_t)
    
    return z_score, p_value, effect_size


def differential_expression_analysis(
    y_pos: np.ndarray, 
    y_neg: np.ndarray, 
    gene_names: List[str],
    pool=None
) -> pd.DataFrame:
    """
    Perform differential expression analysis between two groups.
    
    Args:
        y_pos: Gene expression matrix for positive samples (samples x genes)
        y_neg: Gene expression matrix for negative samples (samples x genes)
        gene_names: List of gene names
        pool: Multiprocessing pool for parallel computation
    
    Returns:
        DataFrame with differential expression results
    """
    # Calculate fold changes
    fold_changes = y_pos.mean(axis=0) / (y_neg.mean(axis=0) + 1e-8)  # Add small epsilon
    
    # Ensure 2D arrays
    if len(y_pos.shape) < 2:
        y_pos = np.expand_dims(y_pos, axis=0)
    if len(y_neg.shape) < 2:
        y_neg = np.expand_dims(y_neg, axis=0)
    
    # Pre-filter valid genes to avoid statistical issues
    pos_std = np.std(y_pos, axis=0) > 0
    neg_std = np.std(y_neg, axis=0) > 0
    pos_no_nan = ~np.isnan(y_pos).any(axis=0)
    neg_no_nan = ~np.isnan(y_neg).any(axis=0)
    valid_mask = pos_std & neg_std & pos_no_nan & neg_no_nan
    valid_genes = np.where(valid_mask)[0]
    
    # Initialize p-values to 1.0 (no significance)
    p_values = np.ones(y_pos.shape[1])
    
    # Run t-tests on valid genes
    if pool is not None and len(valid_genes) > 0:
        try:
            results = pool.starmap(
                stats.ttest_ind, 
                [(y_pos[:, gene], y_neg[:, gene], 0, False) for gene in valid_genes]
            )
            
            for i, gene in enumerate(valid_genes):
                p_values[gene] = results[i].pvalue
                
        except Exception as e:
            print(f"Error in t-tests: {e}")
    
    # Create results DataFrame
    gene_p_values = pd.DataFrame({
        'gene': gene_names,
        'p_value': p_values,
        'fold_change': fold_changes
    })
    
    # Adjust p-values for multiple testing using Benjamini-Hochberg correction
    gene_p_values['adj_p_value'] = multipletests(
        gene_p_values['p_value'], method='fdr_bh'
    )[1]
    
    # Sort by p-value
    gene_p_values = gene_p_values.sort_values(by='p_value')
    
    return gene_p_values


def gene_set_enrichment_analysis(
    gene_df: pd.DataFrame,
    gene_set_genes: List[str],
    gene_name_to_idx: Dict[str, int],
    p_threshold: float = 1e-5,
    fold_change_threshold: Optional[float] = None
) -> Dict:
    """
    Perform gene set enrichment analysis.
    
    Args:
        gene_df: DataFrame with differential expression results
        gene_set_genes: List of genes in the gene set
        gene_name_to_idx: Mapping from gene names to indices
        p_threshold: P-value threshold for significance
        fold_change_threshold: Fold change threshold (optional)
    
    Returns:
        Dictionary with enrichment analysis results
    """
    # Mark genes that are in the gene set
    gene_df['in_set'] = gene_df['gene'].isin(gene_set_genes)
    
    # Filter significant genes
    gene_df_selected = gene_df[gene_df['adj_p_value'] < p_threshold].copy()
    
    if fold_change_threshold is not None:
        gene_df_selected = gene_df_selected[
            (gene_df_selected['fold_change'] > fold_change_threshold) | 
            (gene_df_selected['fold_change'] < 1/fold_change_threshold)
        ]
    
    # Calculate enrichment statistics
    if len(gene_df_selected) > 0:
        # Binomial test
        n_hits = sum(gene_df_selected['in_set'])
        try:
            binom_pval, binom_expected, binom_direction, binom_fold, binom_fdr = binomial_test(
                len(gene_df_selected),
                n_hits,
                len(gene_set_genes),
                len(gene_df)
            )
        except Exception as e:
            print(f'Problem with binomial test: {e}')
            binom_pval = 1.0
            binom_expected = None
            binom_direction = None
            binom_fold = None
            binom_fdr = None
    else:
        n_hits = 0
        binom_pval = 1.0
        binom_expected = None
        binom_direction = None
        binom_fold = None
        binom_fdr = None
    
    # Mann-Whitney U test
    # Create ranks based on fold change
    gene_df_ranked = gene_df.sort_values(by='fold_change', ascending=True)
    gene_df_ranked['rank'] = range(1, len(gene_df_ranked) + 1)
    
    z_score, mw_pval, effect_size = mann_whitney_u_test(
        gene_df_ranked['rank'].values,
        gene_df_ranked['in_set'].values
    )
    
    return {
        'n_hits': n_hits,
        'expected': binom_expected,
        'binom_pval': binom_pval,
        'binom_direction': binom_direction,
        'binom_fold_change': binom_fold,
        'fdr': binom_fdr,
        'z_score': z_score,
        'mw_pval': mw_pval,
        'effect_size': effect_size
    }


def load_go_gene_sets(
    go_obo_path: str,
    go_gaf_path: str,
    gene_names: List[str],
    go_category: str = 'biological_process',
    min_genes: int = 10,
    max_genes: int = 500
) -> Tuple[Dict, pd.DataFrame]:
    """
    Load GO gene sets and create gene-term association matrix.
    
    Args:
        go_obo_path: Path to GO OBO file
        go_gaf_path: Path to GO GAF file
        gene_names: List of gene names in the dataset
        go_category: GO category to use
        min_genes: Minimum number of genes per GO term
        max_genes: Maximum number of genes per GO term
    
    Returns:
        Tuple of (gene_sets_dict, go_info_df)
    """
    # Load GO DAG and annotations
    obodag = GODag(go_obo_path)
    ogaf = GafReader(go_gaf_path)
    ns2assc = ogaf.get_ns2assc()
    
    # Get GO terms for the specified category
    go_bp = [go_id for go_id in obodag.keys() 
             if obodag[go_id].namespace == go_category]
    
    # Create gene sets dictionary
    gene_sets = {}
    go_info = []
    
    for go_id in go_bp:
        if go_id in ns2assc[go_category]:
            # Get genes associated with this GO term
            associated_genes = ns2assc[go_category][go_id]
            
            # Filter for genes in our dataset
            genes_in_dataset = [gene for gene in associated_genes if gene in gene_names]
            
            # Filter by size constraints
            if min_genes <= len(genes_in_dataset) <= max_genes:
                gene_sets[go_id] = genes_in_dataset
                go_info.append({
                    'go_id': go_id,
                    'go_name': obodag[go_id].name,
                    'go_namespace': obodag[go_id].namespace,
                    'n_genes': len(genes_in_dataset)
                })
    
    go_info_df = pd.DataFrame(go_info)
    
    return gene_sets, go_info_df


def create_gene_term_matrix(
    gene_sets: Dict[str, List[str]], 
    gene_names: List[str]
) -> ad.AnnData:
    """
    Create a gene-term association matrix.
    
    Args:
        gene_sets: Dictionary mapping term IDs to lists of gene names
        gene_names: List of all gene names
    
    Returns:
        AnnData object with gene-term associations
    """
    # Create binary matrix
    n_terms = len(gene_sets)
    n_genes = len(gene_names)
    
    # Create mapping for efficient lookup
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    
    # Initialize matrix
    matrix = np.zeros((n_terms, n_genes), dtype=bool)
    term_info = []
    
    for term_idx, (term_id, genes) in enumerate(gene_sets.items()):
        term_info.append({'term_id': term_id})
        
        for gene in genes:
            if gene in gene_to_idx:
                gene_idx = gene_to_idx[gene]
                matrix[term_idx, gene_idx] = True
    
    # Create AnnData object
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(term_info),
        var=pd.DataFrame({'gene_name': gene_names})
    )
    
    return adata


def select_high_low_activation_samples(
    activations: torch.Tensor,
    feature_idx: int,
    percentile: float = 99
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select samples with high and low activation for a specific feature.
    
    Args:
        activations: Tensor of shape (n_samples, n_features)
        feature_idx: Index of the feature to analyze
        percentile: Percentile threshold for high activation
    
    Returns:
        Tuple of (high_activation_indices, low_activation_indices)
    """
    feature_activations = activations[:, feature_idx]
    
    # High activation: top percentile
    high_threshold = np.percentile(feature_activations.numpy(), percentile)
    high_indices = torch.where(feature_activations > high_threshold)[0]
    
    # Low activation: exactly zero
    low_indices = torch.where(feature_activations == 0.0)[0]
    
    return high_indices, low_indices


def filter_active_features(
    activations: torch.Tensor,
    min_active_samples: int = 100,
    max_active_samples: Optional[int] = None
) -> torch.Tensor:
    """
    Filter features based on activation criteria.
    
    Args:
        activations: Tensor of shape (n_samples, n_features)
        min_active_samples: Minimum number of samples with non-zero activation
        max_active_samples: Maximum number of samples with non-zero activation
    
    Returns:
        Tensor of valid feature indices
    """
    # Find features with any activation
    active_feature_ids = torch.where(activations.sum(dim=0) > 0)[0]
    
    # Count active samples per feature
    n_zeros_per_feature = (activations == 0).sum(dim=0)
    n_active_per_feature = activations.shape[0] - n_zeros_per_feature
    
    # Filter by activation criteria
    valid_features = []
    for feat_id in active_feature_ids:
        n_active = n_active_per_feature[feat_id]
        
        if n_active >= min_active_samples:
            if max_active_samples is None or n_active <= max_active_samples:
                valid_features.append(feat_id)
    
    return torch.tensor(valid_features, dtype=torch.long)
