"""
Visualization tools for SAE features and GO terms.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import textwrap
from typing import Optional, Union, List, Tuple, Dict
from sklearn.decomposition import PCA
from goatools.obo_parser import GODag
import os
from pathlib import Path
from ..scFeatureLens.hypothesis import predict_embedding_change


def compute_go_feature_matrix(
    go_df: pd.DataFrame,
    activations: torch.Tensor
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """
    Compute the GO-feature matrix based on GO term analysis.
    
    Args:
        go_df: DataFrame with GO term analysis results
        activations: Feature activations tensor
        
    Returns:
        Tuple containing:
        - go_feature_matrix: Binary matrix with GO terms as rows, features as columns
        - mtrx_feature_ids: Feature IDs used in the matrix
        - mtrx_go_names: GO term names used in the matrix
    """
    # Get unique GO IDs and names
    go_ids = go_df['go_id'].unique()
    go_names = [go_df[go_df['go_id'] == x]['go_name'].iloc[0] for x in go_ids]
    
    # Initialize matrix
    go_feature_matrix = torch.zeros((len(go_ids), activations.shape[1]))
    
    # Fill matrix with binary indicators
    for i, go_id in enumerate(go_ids):
        for feat in go_df[go_df['go_id'] == go_id]['feature']:
            go_feature_matrix[i, feat] = 1
    
    # Get features that are associated with at least one GO term
    mtrx_feature_ids = torch.where(go_feature_matrix.sum(dim=0) > 0)[0]
    
    # Filter matrix to include only features with GO associations
    filtered_matrix = go_feature_matrix[:, mtrx_feature_ids]
    
    return filtered_matrix, mtrx_feature_ids.numpy(), go_names


def add_parent_go_terms(
    go_df: pd.DataFrame, 
    obo_file_path: str
) -> pd.DataFrame:
    """
    Add parent GO terms to GO analysis results.
    
    Args:
        go_df: DataFrame with GO term analysis results
        obo_file_path: Path to GO basic OBO file
        
    Returns:
        DataFrame with added parent GO term columns
    """
    # Load GO DAG
    obodag = GODag(obo_file_path)
    
    # For each GO term, create a list of all parents
    go_terms = go_df['go_id'].unique()
    go_terms_ancestry = {}

    for go_term in go_terms:
        go_obj = obodag[go_term]
        level = go_obj.level
        out_parents = []
        out_levels = []
        while level > 1:
            parents = list(go_obj.parents)
            if len(parents) > 1:
                # Get the parent with level-1
                for parent in parents:
                    if parent.level == level - 1:
                        go_obj = parent
                        break
            else:
                go_obj = parents[0]
            out_parents.append(go_obj.id)
            out_levels.append(go_obj.level)
            level = go_obj.level
        go_terms_ancestry[go_term] = {'parents': out_parents, 'levels': out_levels}

    # Add parent GO terms to dataframe
    go_df['parent_go_id_level1'] = go_df['go_id'].apply(
        lambda x: go_terms_ancestry[x]['parents'][-1] if len(go_terms_ancestry[x]['parents']) > 0 else x
    )
    go_df['parent_go_name_level1'] = go_df['parent_go_id_level1'].apply(lambda x: obodag[x].name)
    
    go_df['parent_go_id_level2'] = go_df['go_id'].apply(
        lambda x: go_terms_ancestry[x]['parents'][-2] if len(go_terms_ancestry[x]['parents']) > 1 else x
    )
    go_df['parent_go_name_level2'] = go_df['parent_go_id_level2'].apply(
        lambda x: obodag[x].name if x != '' else ''
    )
    
    go_df['parent_go_id_level3'] = go_df['go_id'].apply(
        lambda x: go_terms_ancestry[x]['parents'][-3] if len(go_terms_ancestry[x]['parents']) > 2 else x
    )
    go_df['parent_go_name_level3'] = go_df['parent_go_id_level3'].apply(
        lambda x: obodag[x].name if x != '' else ''
    )
    
    go_df['parent_go_id_level4'] = go_df['go_id'].apply(
        lambda x: go_terms_ancestry[x]['parents'][-4] if len(go_terms_ancestry[x]['parents']) > 3 else x
    )
    go_df['parent_go_name_level4'] = go_df['parent_go_id_level4'].apply(
        lambda x: obodag[x].name if x != '' else ''
    )
    
    # Filter to significant terms
    return go_df


def visualize_go_feature_space(
    go_df: pd.DataFrame,
    activations: torch.Tensor,
    obo_file_path: str,
    method: str = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 42,
    spread: float = 1.0,
    figsize: Tuple[int, int] = (12, 10),
    output_path: Optional[str] = None,
    color_by: str = 'parent_go_name_level1'
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame, Union[umap.UMAP, PCA]]:
    """
    Visualize GO feature space using dimensionality reduction.
    
    Args:
        go_df: DataFrame with GO term analysis results
        activations: Feature activations tensor
        obo_file_path: Path to GO basic OBO file
        method: Dimensionality reduction method ('umap' or 'pca')
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed
        figsize: Figure size
        output_path: Path to save the figure
        color_by: Column to use for coloring points
        
    Returns:
        Tuple containing figure, axes, DataFrame with plot data, and the dimensionality reducer
    """
    # Add parent GO terms if not already present
    if 'parent_go_name_level1' not in go_df.columns:
        go_df = add_parent_go_terms(go_df, obo_file_path)
        
    # Compute GO feature matrix
    go_feature_matrix, mtrx_feature_ids, mtrx_go_names = compute_go_feature_matrix(go_df, activations)
    
    # Apply dimensionality reduction
    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=2,
            min_dist=min_dist,
            n_neighbors=n_neighbors,
            random_state=random_state,
            spread=spread
        )
    else:  # PCA
        reducer = PCA(n_components=2, random_state=random_state)
    
    # Fit and transform
    embedding = reducer.fit_transform(go_feature_matrix.T)
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame(embedding, columns=['Dimension 1', 'Dimension 2'])
    df_plot['feature'] = mtrx_feature_ids
    
    # Add GO term information
    level_1_terms = []
    for feat in mtrx_feature_ids:
        temp_go_term = go_df[go_df['feature'] == feat][color_by].value_counts().index[0]
        level_1_terms.append(temp_go_term)
    
    df_plot[color_by] = level_1_terms
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    unique_terms = df_plot[color_by].unique()
    
    # Create a color palette
    palette = sns.color_palette('Paired', n_colors=len(unique_terms))
    
    # Create scatter plot
    sns.scatterplot(
        data=df_plot,
        x='Dimension 1',
        y='Dimension 2',
        hue=color_by,
        palette=palette,
        s=5,
        alpha=0.7,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(f'GO Feature Space ({method.upper()})')
    ax.set_xlabel(f'{method.upper()} Dimension 1')
    ax.set_ylabel(f'{method.upper()} Dimension 2')
    
    # Adjust legend
    handles, labels = ax.get_legend_handles_labels()
    labels = [textwrap.fill(label, 20) for label in labels]
    ax.legend(
        handles, 
        labels, 
        title=color_by,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        frameon=True,
        markerscale=3
    )
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, ax, df_plot, go_feature_matrix, reducer


def probe_go_terms(
    go_df: pd.DataFrame,
    activations: torch.Tensor,
    terms: List[str],
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    method: str = 'umap',
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    random_state: int = 42,
    reducer: Optional[Union[umap.UMAP, PCA]] = None,
    go_feature_matrix: Optional[torch.Tensor] = None,
    mtrx_feature_ids: Optional[np.ndarray] = None,
    mtrx_go_names: Optional[List[str]] = None,
    embedding: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    Probe GO terms with specific keywords and visualize feature spaces.
    
    Args:
        go_df: DataFrame with GO term analysis results
        activations: Feature activations tensor
        terms: List of terms/keywords to probe
        n_cols: Number of columns in the grid
        figsize: Figure size (auto-calculated if None)
        output_path: Path to save the figure
        method: Dimensionality reduction method ('umap' or 'pca')
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed
        reducer: Pre-existing dimensionality reducer (optional)
        go_feature_matrix: Pre-computed GO feature matrix (optional)
        mtrx_feature_ids: Pre-computed feature IDs (optional)
        mtrx_go_names: Pre-computed GO names (optional)
        embedding: Pre-computed dimensionality reduction embedding (optional)
        
    Returns:
        Tuple containing figure and DataFrame with plot data
    """
    # Compute GO feature matrix if not provided
    if go_feature_matrix is None or mtrx_feature_ids is None or mtrx_go_names is None:
        go_feature_matrix, mtrx_feature_ids, mtrx_go_names = compute_go_feature_matrix(go_df, activations)
    
    # Apply dimensionality reduction if not provided
    if reducer is None:
        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=2,
                min_dist=min_dist,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
        
        # Fit and transform
        embedding = reducer.fit_transform(go_feature_matrix.T)
    elif (reducer is not None) and (embedding is None):
        # If reducer is provided but not embedding, use it to transform the matrix
        embedding = reducer.transform(go_feature_matrix.T)
    else:
        if method.lower() == 'umap':
            reducer = umap.UMAP(
                n_components=2,
                min_dist=min_dist,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
        else:  # PCA
            reducer = PCA(n_components=2, random_state=random_state)
        
        # Fit and transform
        embedding = reducer.fit_transform(go_feature_matrix.T)

    
    # Create dataframe for plotting
    df_plot = pd.DataFrame(embedding, columns=[f'{method.upper()} 1', f'{method.upper()} 2'])
    df_plot['feature'] = mtrx_feature_ids
    
    # Calculate number of rows needed
    n_rows = (len(terms) + n_cols - 1) // n_cols
    
    # Calculate figure size if not provided
    if figsize is None:
        # Use 4 inches per subplot as a reasonable default
        figsize = (4 * n_cols, 4 * n_rows)
    
    # Create figure and grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Define colors for binary indicators
    binary_colors = sns.color_palette('rocket_r', 100)
    binary_colors = [binary_colors[0], binary_colors[75]]
    
    # Plot each term
    for i, term in enumerate(terms):
        if i < len(axes):
            ax = axes[i]
            
            # Find GO terms containing the search term
            test_go_terms = np.array([1 if term.lower() in x.lower() else 0 for x in mtrx_go_names]).astype(bool)
            
            # Count features associated with matched GO terms
            test_counts = go_feature_matrix[test_go_terms, :].clone().sum(dim=0).int()
            test_counts = (test_counts > 0).float()
            
            # Add term indicator to dataframe
            df_temp = df_plot.copy()
            df_temp[term] = test_counts.numpy()
            
            # Sort by term indicator
            df_temp = df_temp.sort_values(by=term, ascending=True)
            
            # Plot
            sns.scatterplot(
                data=df_temp,
                x=f'{method.upper()} 1',
                y=f'{method.upper()} 2',
                hue=term,
                s=5,
                alpha=0.7,
                ec=None,
                palette=binary_colors,
                ax=ax
            )
            
            # Customize subplot
            ax.set_title(f'"{term}"')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend().remove()
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, df_plot


def visualize_feature_in_embedding(
    embeddings: torch.Tensor,
    feature_idx: Union[int, List[int]],
    activations: Optional[torch.Tensor] = None,
    sae_model: Optional[torch.nn.Module] = None,
    method: str = 'pca',
    n_components: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'rocket_r',
    random_state: int = 42,
    title: Optional[str] = None,
    device: str = 'cpu',
    output_path: Optional[str] = None,
    max_columns: int = 10
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]], pd.DataFrame]:
    """
    Visualize SAE feature activations in embedding space.
    
    Args:
        embeddings: Embeddings tensor
        feature_idx: Index or list of indices of the features to visualize
        activations: Pre-computed activations (optional)
        sae_model: Trained SAE model (required if activations not provided)
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Number of components for dimensionality reduction
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap for visualization
        random_state: Random seed
        title: Plot title (used only for single feature)
        device: Computing device
        output_path: Path to save the figure
        max_columns: Maximum number of columns for multiple feature plots
        
    Returns:
        Tuple containing figure, axes (single Axes or list of Axes), and DataFrame with plot data
    """
    # Handle single feature or list of features
    if isinstance(feature_idx, (list, tuple, np.ndarray)):
        features = list(feature_idx)
    else:
        features = [feature_idx]
    
    # Move tensors to device
    embeddings = embeddings.to(device)
    
    # Compute activations if not provided
    if activations is None:
        if sae_model is None:
            raise ValueError("Either activations or sae_model must be provided")
        sae_model = sae_model.to(device)
        with torch.no_grad():
            _, activs = sae_model(embeddings)
    else:
        activs = activations.to(device)
    
    # Apply dimensionality reduction
    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )
    else:  # PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
    
    # Fit and transform
    reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())
    
    # Create dataframe for plotting
    df_plot = pd.DataFrame(
        reduced_embeddings,
        columns=[f'{method.upper()} {i+1}' for i in range(n_components)]
    )
    
    # Multiple features case
    if len(features) > 1:
        # Calculate grid dimensions
        n_cols = min(len(features), max_columns)
        n_rows = (len(features) + n_cols - 1) // n_cols
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)
        
        # Create figure and axes
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]  # Convert to list for consistent handling
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        
        # Plot each feature
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                
                # Add feature activation to dataframe
                df_plot[f'Feature {feature}'] = activs[:, feature].cpu().numpy()
                
                # Sort by feature activation for better visualization
                df_feature = df_plot.sort_values(by=f'Feature {feature}', ascending=True)
                
                # Create colormap
                palette = sns.color_palette(cmap, as_cmap=True)
                
                # Plot
                sns.scatterplot(
                    data=df_feature,
                    x=f'{method.upper()} 1',
                    y=f'{method.upper()} 2',
                    hue=f'Feature {feature}',
                    palette=palette,
                    s=5,
                    alpha=0.7,
                    ax=ax
                )
                
                # Customize subplot
                feat_title = f'Feature {feature}' if title is None else f"{title} {feature}"
                ax.set_title(feat_title)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add colorbar
                norm = plt.Normalize(
                    df_feature[f'Feature {feature}'].min(),
                    df_feature[f'Feature {feature}'].max()
                )
                # remove the legend
                ax.legend().remove()
                sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax)
        
        # Hide unused subplots
        for j in range(len(features), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
    # Single feature case (original behavior)
    else:
        feature = features[0]
        
        # Add feature activation to dataframe
        df_plot[f'Feature {feature}'] = activs[:, feature].cpu().numpy()
        
        # Sort by feature activation for better visualization
        df_plot = df_plot.sort_values(by=f'Feature {feature}', ascending=True)
        
        # Calculate figure size if not provided
        if figsize is None:
            figsize = (10, 8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        axes = ax  # For consistent return
        
        # Create colormap
        palette = sns.color_palette(cmap, as_cmap=True)
        
        # Plot
        sns.scatterplot(
            data=df_plot,
            x=f'{method.upper()} 1',
            y=f'{method.upper()} 2',
            hue=f'Feature {feature}',
            palette=palette,
            s=5,
            alpha=0.7,
            ax=ax
        )
        
        # Customize plot
        if title is None:
            title = f'Feature {feature} Activations'
        ax.set_title(title)
        
        # Add colorbar
        norm = plt.Normalize(
            df_plot[f'Feature {feature}'].min(),
            df_plot[f'Feature {feature}'].max()
        )
        # remove the legend
        ax.legend().remove()
        sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(f'Feature {feature} Activation')
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, axes, df_plot


def visualize_feature_perturbation(
    embeddings: torch.Tensor,
    cell_indices: Union[List[int], np.ndarray],
    feature_idx: int,
    sae_model: torch.nn.Module,
    activations: Optional[torch.Tensor] = None,
    target_activation: Optional[float] = None,
    target_percentile: Optional[float] = None,
    method: str = 'pca',
    n_components: int = 2,
    figsize: Tuple[int, int] = (10, 8),
    random_state: int = 42,
    device: str = 'cpu',
    output_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Visualize the effect of perturbing an SAE feature in embedding space.
    
    Args:
        embeddings: Embeddings tensor
        cell_indices: Indices of cells to perturb
        feature_idx: Index of the feature to perturb
        sae_model: Trained SAE model
        activations: Pre-computed activations (optional)
        target_activation: Target activation value for the feature
        target_percentile: Target percentile for activation (alternative to target_activation)
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Number of components for dimensionality reduction
        figsize: Figure size
        random_state: Random seed
        device: Computing device
        output_path: Path to save the figure
        
    Returns:
        Tuple containing figure, axes, and DataFrame with plot data
    """
    # Move tensors to device
    embeddings = embeddings.to(device)
    sae_model = sae_model.to(device)
    
    # Convert indices to numpy array if it's a list
    if isinstance(cell_indices, list):
        cell_indices = np.array(cell_indices)
    
    # Get perturbed embeddings
    perturbed_embeddings = predict_embedding_change(
        embeddings=embeddings,
        cell_indices=cell_indices,
        feature_idx=feature_idx,
        sae_model=sae_model,
        activations=activations,
        target_activation=target_activation,
        target_percentile=target_percentile,
        device=device
    )
    
    # Apply dimensionality reduction
    if method.lower() == 'umap':
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state
        )
    else:  # PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
    
    # Fit reducer on all embeddings
    reduced_embeddings = reducer.fit_transform(embeddings.cpu().numpy())
    
    # Create dataframe for all points
    df_all = pd.DataFrame(
        reduced_embeddings,
        columns=[f'{method.upper()} {i+1}' for i in range(n_components)]
    )
    df_all['type'] = 'other'
    
    # Get original embeddings for selected cells
    cell_embeddings = embeddings[cell_indices].cpu().numpy()
    reduced_cell_embeddings = reducer.transform(cell_embeddings)
    
    # Create dataframe for original selected cells
    df_original = pd.DataFrame(
        reduced_cell_embeddings,
        columns=[f'{method.upper()} {i+1}' for i in range(n_components)]
    )
    df_original['type'] = 'original'
    
    # Get reduced perturbed embeddings
    reduced_perturbed = reducer.transform(perturbed_embeddings.cpu().numpy())
    
    # Create dataframe for perturbed cells
    df_perturbed = pd.DataFrame(
        reduced_perturbed,
        columns=[f'{method.upper()} {i+1}' for i in range(n_components)]
    )
    df_perturbed['type'] = 'perturbed'
    
    # Combine dataframes
    df_plot = pd.concat([df_all, df_original, df_perturbed], ignore_index=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define color palette - explicitly create a palette with enough colors
    # to avoid index out of range errors
    rocket_palette = sns.color_palette('rocket_r', n_colors=10)
    palette = {
        'other': 'lightgrey',
        'original': rocket_palette[3],
        'perturbed': rocket_palette[-2]  # Use second-to-last color to be safe
    }
    
    # Create scatter plot
    sns.scatterplot(
        data=df_plot,
        x=f'{method.upper()} 1',
        y=f'{method.upper()} 2',
        hue='type',
        palette=palette,
        s=5,
        alpha=0.7,
        ax=ax
    )
    
    # Customize plot
    ax.set_title(f'Effect of Perturbing Feature {feature_idx}')
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    return fig, ax, df_plot
