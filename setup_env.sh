#!/bin/bash

# Quick setup script for sc_mechinterp development environment
# This script provides an automated way to set up the development environment

set -e  # Exit on any error

echo "🔬 Setting up sc_mechinterp development environment..."
echo "========================================================="

# Function to detect if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "✓ conda detected"
        return 0
    else
        echo "✗ conda not found"
        return 1
    fi
}

# Function to detect if poetry is available
check_poetry() {
    if command -v poetry &> /dev/null; then
        echo "✓ poetry detected"
        return 0
    else
        echo "✗ poetry not found"
        return 1
    fi
}

# Function to setup with conda
setup_conda() {
    echo "Setting up with conda..."
    
    # Source conda initialization
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source /opt/anaconda3/etc/profile.d/conda.sh
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        source ${HOME}/anaconda3/etc/profile.d/conda.sh
    elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        source ${HOME}/miniconda3/etc/profile.d/conda.sh
    else
        echo "❌ Could not find conda initialization script"
        echo "Please run 'conda init' first or activate conda manually"
        exit 1
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "sc_mechinterp"; then
        echo "⚠️  Environment 'sc_mechinterp' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n sc_mechinterp
        else
            echo "Using existing environment..."
            conda activate sc_mechinterp
            echo "Installing/updating package in development mode..."
            pip install -e .
            return 0
        fi
    fi
    
    # Create environment from file
    if [ -f "setup/environment.yml" ]; then
        echo "Creating environment from environment.yml..."
        conda env create -f setup/environment.yml
    else
        echo "Creating basic environment..."
        conda create -n sc_mechinterp python=3.10 -y
        conda activate sc_mechinterp
        
        # Install PyTorch
        echo "Installing PyTorch..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        
        # Install other dependencies
        pip install -r setup/requirements.txt
    fi
    
    echo "Activating environment..."
    conda activate sc_mechinterp
    
    echo "Installing package in development mode..."
    pip install -e .
}

# Function to setup with poetry
setup_poetry() {
    echo "Setting up with poetry..."
    
    # Install dependencies
    poetry install
    
    echo "Environment setup complete!"
    echo "To activate: poetry shell"
}

# Function to setup with venv
setup_venv() {
    echo "Setting up with venv..."
    
    ENV_NAME="sc_mechinterp_env"
    
    # Check if virtual environment already exists
    if [ -d "$ENV_NAME" ]; then
        echo "⚠️  Virtual environment '$ENV_NAME' already exists"
        read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$ENV_NAME"
        else
            echo "Using existing environment..."
            source "$ENV_NAME/bin/activate"
            pip install -e .
            return 0
        fi
    fi
    
    # Create virtual environment
    python -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio
    
    # Install dependencies
    pip install -r setup/requirements.txt
    
    # Install package in development mode
    pip install -e .
    
    echo "Environment setup complete!"
    echo "To activate: source $ENV_NAME/bin/activate"
}

# Function to verify installation
verify_installation() {
    echo ""
    echo "🧪 Verifying installation..."
    echo "=========================="
    
    # Source conda if available
    if command -v conda &> /dev/null; then
        if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
            source /opt/anaconda3/etc/profile.d/conda.sh
        elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
            source ${HOME}/anaconda3/etc/profile.d/conda.sh
        elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
            source ${HOME}/miniconda3/etc/profile.d/conda.sh
        fi
        
        # Try to activate conda environment
        if conda env list | grep -q "sc_mechinterp"; then
            conda activate sc_mechinterp
        fi
    fi
    
    # Test basic imports
    python -c "
import torch
import numpy
import pandas
import scipy
print('✓ Core packages imported successfully')
"
    
    # Test scFeatureLens import
    python -c "
from tools.scFeatureLens import SCFeatureLensPipeline
print('✓ scFeatureLens imported successfully')
"
    
    # Run basic tests if available
    if [ -f "tests/test_basic.py" ]; then
        echo "Running basic tests..."
        python tests/test_basic.py
    fi
    
    echo ""
    echo "🎉 Installation verified successfully!"
}

# Main setup logic
main() {
    echo "Choose your preferred environment manager:"
    echo "1) conda (recommended for data science)"
    echo "2) poetry (modern dependency management)"
    echo "3) venv (standard library)"
    echo "4) auto-detect"
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            if check_conda; then
                setup_conda
            else
                echo "❌ conda not available. Please install Miniconda or Anaconda first."
                exit 1
            fi
            ;;
        2)
            if check_poetry; then
                setup_poetry
            else
                echo "❌ poetry not available. Install with: curl -sSL https://install.python-poetry.org | python3 -"
                exit 1
            fi
            ;;
        3)
            setup_venv
            ;;
        4)
            echo "Auto-detecting available tools..."
            if check_conda; then
                setup_conda
            elif check_poetry; then
                setup_poetry
            else
                echo "Using venv as fallback..."
                setup_venv
            fi
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "📚 Next steps:"
    echo "============="
    echo "1. Run full validation: python validate_environment.py"
    echo "2. Try the basic example: python -m scFeatureLens.example --example basic"
    echo "3. Set up data files: python scripts/setup_data.py --example-sets"
    echo "4. Read the documentation:"
    echo "   - Quick start: cat QUICKSTART.md"
    echo "   - Detailed setup: cat ENVIRONMENT_SETUP.md"
    echo "   - Reproducibility: cat REPRODUCIBILITY_GUIDE.md"
    echo "   - Docker setup: cat DOCKER_GUIDE.md"
    echo "5. Start coding! 🚀"
    echo ""
    echo "🔧 Your isolated environment is ready!"
    echo "   Environment: $(which python)"
    echo "   Activate: conda activate sc_mechinterp  # (or appropriate command for your setup)"
}

# Run main function
main "$@"
