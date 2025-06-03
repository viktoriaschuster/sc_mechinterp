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

def run_nb_model(y_pos, y_neg, gene_idx, conditions, pairing):
    # Combine the gene expression data for the current gene across both conditions
    gene_expression = np.concatenate([y_pos[:, gene_idx], y_neg[:, gene_idx]])

    # Design matrix: Intercept (ones), pairing, and condition
    X = np.column_stack([np.ones_like(conditions), pairing, conditions])

    # Fit a negative binomial model for the current gene
    glm_model = sm.GLM(gene_expression, X, family=NegativeBinomial(alpha=dispersion_factors[gene_idx]))
    result = glm_model.fit()

    # Extract p-value for the condition (perturbation effect)
    p_value = result.pvalues[2]  # The p-value for the "condition" variable
    fold_change = np.exp(result.params[2])  # Fold change is exp(beta)

    return p_value, fold_change

"""
def DEG_analysis_unpaired(y_pos, y_neg, gene_names):

    fold_changes = y_pos.mean(axis=0) / y_neg.mean(axis=0)

    # Perform unpaired t-tests for each gene
    pool = mp.Pool(mp.cpu_count())
    _, p_values = zip(*pool.starmap(stats.ttest_ind, [(y_pos[:, gene], y_neg[:, gene], 0, False) for gene in range(y_pos.shape[1])]))
    #p_values = []
    #for gene in range(y_pos.shape[1]):
    #    t_stat, p_val = stats.ttest_ind(y_pos[:,gene], y_neg[:,gene], equal_var=False)  # Unequal variance (Welch's t-test)
    #    p_values.append(p_val)

    # Convert the p-values into a numpy array for further processing
    p_values = np.array(p_values)

    # Convert p-values and fold changes into a DataFrame
    gene_p_values = pd.DataFrame({
        'gene': gene_names,  # Assuming you have gene names as your columns' index
        'p_value': p_values,
        'fold_change': fold_changes
    })

    # Adjust p-values for multiple testing using Benjamini-Hochberg correction
    gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]

    # Sort the results by p-value
    gene_p_values = gene_p_values.sort_values(by='p_value')

    return gene_p_values
"""

def DEG_analysis_unpaired(y_pos, y_neg, gene_names, pool):
    fold_changes = y_pos.mean(axis=0) / (y_neg.mean(axis=0) + 1e-8) # Avoid division by zero
    if len(y_pos.shape) < 2:
        y_pos = np.expand_dims(y_pos, axis=0)
    if len(y_neg.shape) < 2:
        y_neg = np.expand_dims(y_neg, axis=0)
    
    # Add error checking for t-tests
    p_values = np.ones(y_pos.shape[1])  # Default to 1.0
    
    # Pre-filter genes to avoid issues
    """
    valid_genes = []
    for gene in range(y_pos.shape[1]):
        pos_data = y_pos[:, gene]
        neg_data = y_neg[:, gene]
        
        # Check for valid data (non-constant, no NaNs)
        if (np.std(pos_data) > 0 and np.std(neg_data) > 0 and 
            not np.isnan(pos_data).any() and not np.isnan(neg_data).any()):
            valid_genes.append(gene)
    """
    # Vectorized filtering of valid genes
    pos_std = np.std(y_pos, axis=0) > 0
    neg_std = np.std(y_neg, axis=0) > 0
    pos_no_nan = ~np.isnan(y_pos).any(axis=0)
    neg_no_nan = ~np.isnan(y_neg).any(axis=0)
    # Combine all conditions
    valid_mask = pos_std & neg_std & pos_no_nan & neg_no_nan
    valid_genes = np.where(valid_mask)[0]
    
    # Only run t-tests on valid genes
    #pool = mp.Pool(mp.cpu_count())
    try:
        results = pool.starmap(stats.ttest_ind, 
                              [(y_pos[:, gene], y_neg[:, gene], 0, False) 
                               for gene in valid_genes])
        
        # Assign results back to the full p_values array
        for i, gene in enumerate(valid_genes):
            p_values[gene] = results[i].pvalue
            
    except Exception as e:
        print(f"Error in t-tests: {e}")
    #finally:
    #    pool.close()
    #    pool.join()

    # Convert p-values and fold changes into a DataFrame
    gene_p_values = pd.DataFrame({
        'gene': gene_names,
        'p_value': p_values,
        'fold_change': fold_changes
    })

    # Adjust p-values for multiple testing
    gene_p_values['adj_p_value'] = multipletests(gene_p_values['p_value'], method='fdr_bh')[1]

    # Sort the results by p-value
    gene_p_values = gene_p_values.sort_values(by='p_value')

    return gene_p_values

def run_enrichment(ranks, in_set, n_r, n_h, n, epsilon, i):
    p_hit = (sum((ranks[:i])[in_set[:i]]) + epsilon) / n_r
    p_miss = (len((ranks[:i])[~in_set[:i]]) + epsilon) / (n - n_h)
    return abs(p_hit - p_miss)

def binomial_test(n_study, n_hit, n_c, n):
    p_c = n_c / n # this is the expected probability of a hit
    results = stats.binomtest(n_hit, n_study, p_c)
    over_under = '+' if n_hit > (n_study * p_c) else '-'
    fold_enrichment = n_hit / (n_study * p_c)
    fdr = (n_study - n_hit) / n_study
    expected = n_study * p_c
    return results.pvalue, expected, over_under, fold_enrichment, fdr

def mann_whitney_u_test(ranks, in_set):
    # s stands for set, t for total
    n_s = sum(in_set)
    # according to wikipedia
    n_t = len(ranks[~in_set])
    r_s = sum(ranks[in_set])
    r_t = sum(ranks[~in_set])
    u_s = n_s * n_t + ((n_s * (n_s + 1)) / 2) - r_s
    u_t = n_s * n_t + ((n_t * (n_t + 1)) / 2) - r_t
    u = min(u_s, u_t)
    z_score = (u - (n_s * n_t / 2)) / math.sqrt(n_s * n_t * (n_s + n_t + 1) / 12)
    p_value = stats.norm.cdf(z_score)
    effect_size = u_s / (n_s * n_t)
    return z_score, p_value, effect_size

def go_analysis(gene_df, go_id, ref_data, p_threshold=1e-5, fold_change_threshold=None):
    pos_ids = ref_data.X[go_id, :].indices
    gene_df['in_set'] = [gene in pos_ids for gene in gene_df['ref_idx']]

    gene_df_selected = gene_df[(gene_df['adj_p_value'] < p_threshold)]
    if fold_change_threshold is not None:
        gene_df_selected = gene_df_selected[(gene_df_selected['fold_change'] > fold_change_threshold) | (gene_df_selected['fold_change'] < 1/fold_change_threshold)]
    if len(gene_df_selected) > 0:
        ###
        # binomial test
        ###
        hit_positions = gene_df_selected['ref_idx'].values
        n_hits = sum(gene_df_selected['in_set'])
        try:
            binom_pval, binom_expected, binom_direction, binom_fold, binom_fdr = binomial_test(
                len(gene_df_selected),
                sum(gene_df_selected['in_set']),
                len(pos_ids),
                len(gene_df)
            )
        except:
            print('problem with binom test', n_hits, len(gene_df_selected))
            binom_pval = 1.0
            binom_expected = None
            binom_direction = None
            binom_fold = None
            binom_fdr = None
    else:
        hit_positions = []
        n_hits = 0
        binom_pval = 1.0
        binom_expected = None
        binom_direction = None
        binom_fold = None
        binom_fdr = None
    
    ###
    # Mann-Whitney U test
    ###
    z_score, mw_pval, effect_size = mann_whitney_u_test(
        gene_df['rank'].values, 
        gene_df['in_set'].values
    )
    
    return n_hits, binom_expected, binom_pval, binom_direction, binom_fold, binom_fdr, z_score, mw_pval, effect_size, hit_positions
