from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
import pandas as pd
import anndata as ad

@dataclass
class AnalysisConfig:
    """Configuration for the scFeatureLens analysis pipeline."""
    
    # Input/Output paths
    embeddings_path: str
    output_dir: str = "results"
    expression_data_path: Optional[str] = None
    
    # SAE configuration
    train_sae: bool = True
    sae_model_path: Optional[str] = None
    sae_hidden_size: int = 10000
    sae_l1_weight: float = 1e-3
    sae_learning_rate: float = 1e-4
    sae_epochs: int = 500
    sae_batch_size: int = 128
    
    # Feature selection
    activation_percentile: float = 99
    min_active_samples: int = 100
    max_active_samples: Optional[int] = 1000
    
    # DEG analysis
    deg_analysis: bool = True
    deg_p_threshold: float = 1e-5
    deg_fold_change_threshold: float = 2.0
    
    # Gene Ontology analysis
    go_analysis: bool = True
    go_category: str = "biological_process"  # biological_process, molecular_function, cellular_component
    go_min_genes: int = 10
    go_max_genes: int = 500
    
    # Use model for predictions
    use_model_predictions: bool = False
    model_path: Optional[str] = None
    
    # General settings
    verbose: bool = True
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def load_embeddings(embeddings_path: str):
    """Load embeddings from file."""
    #self.logger.info(f"Loading embeddings from {self.config.embeddings_path}")
    file_path = Path(embeddings_path)
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pt':
        # PyTorch format
        embeddings = torch.load(embeddings_path)
        if isinstance(embeddings, dict) and 'embeddings' in embeddings:
            embeddings = embeddings['embeddings']
        
    elif file_extension == '.npy':
        # NumPy format
        embeddings_np = np.load(embeddings_path)
        embeddings = torch.from_numpy(embeddings_np.astype(np.float32))
        
    elif file_extension == '.csv':
        # CSV format
        embeddings_df = pd.read_csv(embeddings_path)
        # check if there is an index column (by finding a column with 'index' in its name or 'Unnamed: 0')
        if 'index' in embeddings_df.columns or 'Unnamed: 0' in embeddings_df.columns:
            # remove the index column
            #self.logger.info("Removing index column from CSV")
            embeddings_df = embeddings_df.drop(columns=['index', 'Unnamed: 0'], errors='ignore')
        # check if all columns are numeric
        column_dtypes = embeddings_df.dtypes
        if not all(column_dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            #self.logger.warning("Non-numeric columns found in CSV, removing them")
            #self.logger.info("Columns before filtering: %s", len(embeddings_df.columns.tolist()))
            embeddings_df = embeddings_df.select_dtypes(include=[np.number])
            #self.logger.info("Columns after filtering: %s", len(embeddings_df.columns.tolist()))
        embeddings = torch.tensor(embeddings_df.values.astype(np.float32))
    
    #self.embeddings = embeddings
    #self.logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")

    return embeddings

def load_expression_data(expression_data_path: str):
    """Load gene expression data if needed for DEG analysis."""
    expression_data = ad.read_h5ad(expression_data_path)
    if hasattr(expression_data, 'var_names'):
        gene_names = expression_data.var_names.tolist()  
    else:
        gene_names = None
    # Extract gene expression values
    gene_expression = np.asarray(expression_data.X.todense()) if hasattr(expression_data.X, 'todense') else expression_data.X

    return expression_data, gene_names, gene_expression

def load_predictions(predictions_path: str):
    """Load predictions from file for DEG analysis instead of real data."""
    file_path = Path(predictions_path)
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pt':
        # PyTorch format
        predictions = torch.load(predictions_path)
        if isinstance(predictions, dict) and 'predictions' in predictions:
            predictions = predictions['predictions']
        predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
            
    elif file_extension == '.npy':
        # NumPy format
        predictions = np.load(predictions_path)
        
    else:
        raise ValueError(f"Unsupported file format for predictions: {file_extension}. "
                        f"Supported formats are: .pt, .npy")
        
    return predictions

def load_dispersions(dispersions_path: str):
    """Load dispersion factors for negative binomial model in DEG analysis."""
    file_path = Path(dispersions_path)
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pt':
        # PyTorch format
        dispersions = torch.load(dispersions_path)
        if isinstance(dispersions, dict) and 'dispersions' in dispersions:
            dispersions = dispersions['dispersions']
        dispersions = dispersions.numpy() if isinstance(dispersions, torch.Tensor) else dispersions
            
    elif file_extension == '.npy':
        # NumPy format
        dispersions = np.load(dispersions_path)

    else:
        raise ValueError(f"Unsupported file format for dispersions: {file_extension}. "
                         f"Supported formats are: .pt, .npy")

    return dispersions

def prep_go_sets(data_gene_names: str,
                logger,
                go_category: str = "biological_process", 
                min_genes: int = 10, 
                max_genes: int = 500):
    from goatools.obo_parser import GODag
    #from goatools.anno.gaf_reader import GafReader
    obodag = GODag("data/go-basic.obo")
    #ogaf = GafReader("01_data/goa_human.gaf")
    #ns2assc = ogaf.get_ns2assc()
    #prot2ensembl = pd.read_csv('01_data/protname2ensembl.tsv', sep='\t')
    #df_go_levels = pd.read_csv('01_data/go_term_levels.tsv', sep='\t')
    adata_go = ad.read_h5ad('data/go_gene_matrix.h5ad')
    adata_go.var['name'] = data_gene_names

    # get the ids that are within a usable range of associated genes and are biological processes
    go_bp = [go_id for go_id in obodag.keys() if obodag[go_id].namespace == go_category]
    # get the ids that are within this range
    go_ids_filtered = np.where((adata_go.X.sum(axis=1) >= min_genes) & (adata_go.X.sum(axis=1) <= max_genes))[0]
    logger.info(f"There are {len(go_ids_filtered)} go terms with at least {min_genes} gene associated")
    adata_go_filtered = adata_go[go_ids_filtered, :]
    # next further filter by the go_bp terms
    go_ids_filtered_bp = np.where(adata_go_filtered.obs['go_id'].isin(go_bp))[0]
    logger.info(f"There are {len(go_ids_filtered_bp)} go terms that are {go_category}")
    adata_go_filtered = adata_go_filtered[go_ids_filtered_bp, :]

    return adata_go_filtered