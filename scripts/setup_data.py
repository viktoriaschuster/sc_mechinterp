#!/usr/bin/env python3
"""
Download and setup data files for sc_mechinterp tools.

This script downloads necessary data files including GO ontology and annotations.
"""

import os
import sys
import urllib.request
import gzip
import shutil
from pathlib import Path
import argparse


def download_file(url: str, output_path: Path, decompress: bool = False):
    """Download a file from URL to output path."""
    print(f"Downloading {url}...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
        
        if decompress and output_path.suffix == '.gz':
            print(f"Decompressing {output_path}...")
            with gzip.open(output_path, 'rb') as f_in:
                decompressed_path = output_path.with_suffix('')
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(output_path)
            print(f"Decompressed to {decompressed_path}")
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False
    
    return True


def setup_go_data(data_dir: Path, species: str = "human"):
    """Setup Gene Ontology data files."""
    print("Setting up Gene Ontology data...")
    
    # Download GO ontology
    go_obo_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    go_obo_path = data_dir / "go-basic.obo"
    
    if not go_obo_path.exists():
        download_file(go_obo_url, go_obo_path)
    else:
        print(f"GO ontology already exists: {go_obo_path}")
    
    # Download annotations based on species
    if species.lower() == "human":
        gaf_url = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz"
        gaf_path = data_dir / "goa_human.gaf.gz"
        final_path = data_dir / "goa_human.gaf"
    elif species.lower() == "mouse":
        gaf_url = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/MGI/goa_mouse.gaf.gz"
        gaf_path = data_dir / "goa_mouse.gaf.gz"
        final_path = data_dir / "goa_mouse.gaf"
    else:
        print(f"Unsupported species: {species}")
        return False
    
    if not final_path.exists():
        download_file(gaf_url, gaf_path, decompress=True)
    else:
        print(f"GO annotations already exist: {final_path}")
    
    return True


def create_example_gene_sets(data_dir: Path):
    """Create example custom gene sets."""
    print("Creating example gene sets...")
    
    custom_dir = data_dir / "custom_gene_sets"
    custom_dir.mkdir(exist_ok=True)
    
    # Cell type markers
    cell_markers = {
        "T_Cell_Markers": [
            "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "IL2RA", "FOXP3"
        ],
        "B_Cell_Markers": [
            "MS4A1", "CD19", "CD79A", "CD79B", "IGHM", "IGHA1", "IGLC1"
        ],
        "Monocyte_Markers": [
            "CD14", "CD16", "CD68", "FCGR3A", "CSF1R", "CX3CR1"
        ],
        "NK_Cell_Markers": [
            "NCAM1", "KLRB1", "KLRD1", "GZMB", "PRF1", "NKG7"
        ],
        "Dendritic_Cell_Markers": [
            "ITGAX", "CD1C", "CLEC9A", "XCR1", "CADM1"
        ]
    }
    
    # Metabolic pathways
    metabolic_pathways = {
        "Glycolysis": [
            "HK1", "GPI", "PFKM", "ALDOA", "TPI1", "GAPDH", "PGK1", "PGAM1", "ENO1", "PKM"
        ],
        "Oxidative_Phosphorylation": [
            "NDUFA1", "NDUFA2", "NDUFB1", "SDHB", "UQCR1", "COX1", "ATP5A1", "ATP5B"
        ],
        "Fatty_Acid_Oxidation": [
            "CPT1A", "ACADM", "HADHA", "HADHB", "ECHS1", "ACAA2"
        ]
    }
    
    # Save gene sets
    import yaml
    
    with open(custom_dir / "cell_type_markers.yaml", 'w') as f:
        yaml.dump(cell_markers, f, default_flow_style=False)
    
    with open(custom_dir / "metabolic_pathways.yaml", 'w') as f:
        yaml.dump(metabolic_pathways, f, default_flow_style=False)
    
    print(f"Created example gene sets in {custom_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup data for sc_mechinterp tools")
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Data directory path")
    parser.add_argument("--species", choices=["human", "mouse"], default="human",
                       help="Species for GO annotations")
    parser.add_argument("--skip-go", action="store_true",
                       help="Skip GO data download")
    parser.add_argument("--example-sets", action="store_true", 
                       help="Create example gene sets")
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)
    
    print(f"Setting up data in {data_dir.absolute()}")
    
    # Setup GO data
    if not args.skip_go:
        success = setup_go_data(data_dir, args.species)
        if not success:
            print("Failed to setup GO data")
            sys.exit(1)
    
    # Create example gene sets
    if args.example_sets:
        create_example_gene_sets(data_dir)
    
    print("Data setup complete!")
    print(f"Data directory: {data_dir.absolute()}")
    print("\nTo use with scFeatureLens, update your config file:")
    print(f"  go_obo_path: {data_dir / 'go-basic.obo'}")
    print(f"  go_gaf_path: {data_dir / f'goa_{args.species}.gaf'}")


if __name__ == "__main__":
    main()
