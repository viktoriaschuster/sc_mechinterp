#!/usr/bin/env python3
"""
Test basic functionality of scFeatureLens.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scFeatureLens.pipeline import SCFeatureLensPipeline, AnalysisConfig
    from scFeatureLens.sae import SparseAutoencoder
    print("✓ Successfully imported scFeatureLens modules")
except ImportError as e:
    print(f"✗ Failed to import scFeatureLens: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality without running full analysis."""
    print("\nTesting basic functionality...")
    
    # Test configuration
    try:
        config = AnalysisConfig(
            embeddings_path="dummy.pt",
            output_dir="test_output"
        )
        print("✓ AnalysisConfig creation works")
    except Exception as e:
        print(f"✗ AnalysisConfig creation failed: {e}")
        return False
    
    # Test SAE model
    try:
        sae = SparseAutoencoder(input_size=256, hidden_size=512)
        print("✓ SparseAutoencoder creation works")
    except Exception as e:
        print(f"✗ SparseAutoencoder creation failed: {e}")
        return False
    
    # Test pipeline initialization
    try:
        pipeline = SCFeatureLensPipeline(config)
        print("✓ SCFeatureLensPipeline initialization works")
    except Exception as e:
        print(f"✗ SCFeatureLensPipeline initialization failed: {e}")
        return False
    
    return True

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        "torch", "numpy", "pandas", "scipy", "sklearn", 
        "statsmodels", "anndata", "tqdm", "yaml"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "sklearn":
                import sklearn
            elif package == "yaml":
                import yaml
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running scFeatureLens basic tests...")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n✗ Import tests failed")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n✗ Basic functionality tests failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ All basic tests passed!")
    print("\nTo run a full example:")
    print("python -m scFeatureLens.example --example basic")

if __name__ == "__main__":
    main()
