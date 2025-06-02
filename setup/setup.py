#!/usr/bin/env python3
"""
Setup script for the sc_mechinterp collection of tools.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
README_PATH = Path(__file__).parent.parent / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Single-cell mechanistic interpretability tools"

# Read requirements
REQUIREMENTS_PATH = Path(__file__).parent / "requirements.txt"
if REQUIREMENTS_PATH.exists():
    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
        "anndata>=0.8.0",
        "scanpy>=1.8.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
        "goatools>=1.2.0",
        "psutil>=5.8.0"
    ]

setup(
    name="sc_mechinterp",
    version="0.1.0",
    author="Viktoria Schuster",
    author_email="vschuste@stanford.edu",
    description="A collection of tools for mechanistic interpretability analysis of single-cell RNA-seq data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sc_mechinterp",
    package_dir={"": str(Path(__file__).parent.parent)},
    packages=find_packages(where=str(Path(__file__).parent.parent)),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "scfeaturelens=scFeatureLens.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "scFeatureLens": ["config_example.yaml"],
    },
    zip_safe=False,
    keywords="single-cell, mechanistic interpretability, sparse autoencoders, gene expression",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/sc_mechinterp/issues",
        "Source": "https://github.com/yourusername/sc_mechinterp",
        "Documentation": "https://sc-mechinterp.readthedocs.io/",
    },
)
