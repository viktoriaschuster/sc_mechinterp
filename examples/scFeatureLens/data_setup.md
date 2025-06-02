# Data Directory

This directory contains data files used by the sc_mechinterp tools.

## Gene Ontology Files

For GO enrichment analysis, you need to download the following files:

### GO Basic Ontology
```bash
wget http://purl.obolibrary.org/obo/go/go-basic.obo -O go-basic.obo
```

### Human GO Annotations
```bash
wget ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz -O goa_human.gaf.gz
gunzip goa_human.gaf.gz
```

### Mouse GO Annotations (if needed)
```bash
wget ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/MGI/goa_mouse.gaf.gz -O goa_mouse.gaf.gz
gunzip goa_mouse.gaf.gz
```

## Directory Structure

```
data/
├── go-basic.obo           # GO ontology file
├── goa_human.gaf          # Human GO annotations
├── goa_mouse.gaf          # Mouse GO annotations (optional)
├── custom_gene_sets/      # Custom gene set files
│   ├── kegg_pathways.yaml
│   ├── hallmark_genes.yaml
│   └── cell_type_markers.yaml
└── example_data/          # Example datasets for testing
    ├── embeddings/
    ├── expression/
    └── gene_sets/
```

## Custom Gene Sets

You can create custom gene sets in YAML or JSON format:

```yaml
# Example: cell_type_markers.yaml
T_Cell_Markers:
  - CD3D
  - CD3E
  - CD3G
  - CD4
  - CD8A
  - CD8B

B_Cell_Markers:
  - MS4A1
  - CD19
  - CD79A
  - CD79B

Monocyte_Markers:
  - CD14
  - CD16
  - CD68
  - FCGR3A
```

## Usage

Most tools will automatically look for data files in this directory. You can override the default paths using configuration files or command-line arguments.
