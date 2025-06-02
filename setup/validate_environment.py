#!/usr/bin/env python3
"""
Environment validation script for scFeatureLens
Tests the isolated environment setup and verifies all components work correctly
"""

import sys
import os
import importlib
import subprocess
import platform
import tempfile
import shutil
from pathlib import Path


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")


def check_python_environment():
    """Check Python version and environment details"""
    print_subheader("Python Environment")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    
    if conda_env:
        print(f"✓ Conda environment: {conda_env}")
    elif in_venv:
        print(f"✓ Virtual environment: {sys.prefix}")
    else:
        print("⚠️  Warning: Not in a virtual environment")
        
    return sys.version_info >= (3, 8)


def check_core_packages():
    """Check if core packages are installed and working"""
    print_subheader("Core Package Imports")
    
    required_packages = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'torch', 
        'yaml', 'tqdm', 'psutil', 'anndata', 'goatools'
    ]
    
    optional_packages = ['scanpy']  # These are optional and may have compatibility issues
    
    results = {}
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
            results[package] = True
        except ImportError as e:
            print(f"✗ {package}: {e}")
            results[package] = False
            
    for package in optional_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
            results[package] = True
        except ImportError as e:
            print(f"⚠️  {package}: {e} (optional)")
            results[package] = False
    
    # Only check required packages for success
    required_success = all(results[pkg] for pkg in required_packages)
    return required_success


def check_scfeaturelens_import():
    """Check if scFeatureLens can be imported"""
    print_subheader("scFeatureLens Import Test")
    
    try:
        from tools.scFeatureLens import SCFeatureLensPipeline, SparseAutoencoder
        from tools.scFeatureLens.analysis_functions import differential_expression_analysis
        from tools.scFeatureLens.cli import main as cli_main
        print("✓ All scFeatureLens modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ scFeatureLens import failed: {e}")
        return False


def check_torch_functionality():
    """Check PyTorch functionality"""
    print_subheader("PyTorch Functionality")
    
    try:
        import torch
        
        # Check basic tensor operations
        x = torch.randn(10, 10)
        y = torch.mm(x, x.t())
        print(f"✓ Basic tensor operations work")
        
        # Check device availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (using CPU)")
            
        print(f"✓ PyTorch version: {torch.__version__}")
        return True
    except Exception as e:
        print(f"✗ PyTorch functionality test failed: {e}")
        return False


def run_basic_pipeline_test():
    """Run a basic pipeline test with synthetic data"""
    print_subheader("Basic Pipeline Test")
    
    try:
        import numpy as np
        import torch
        from tools.scFeatureLens import SCFeatureLensPipeline, AnalysisConfig
        
        # Create minimal synthetic data
        embeddings = torch.randn(50, 32)
        
        # Create basic config
        config = AnalysisConfig(
            embeddings_path="dummy_path",
            output_dir="test_output",
            sae_hidden_size=64,
            sae_l1_penalty=1e-3,
            sae_learning_rate=1e-3,
            sae_epochs=5,
            batch_size=16
        )
        
        # Initialize pipeline
        pipeline = SCFeatureLensPipeline(config)
        
        # Load embeddings into pipeline
        pipeline.embeddings = embeddings
        
        # Test SAE training (minimal)
        with tempfile.TemporaryDirectory() as temp_dir:
            sae_model = pipeline.train_sae()
            
        print("✓ Basic SAE training completed")
        print("✓ Pipeline initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic pipeline test failed: {e}")
        return False


def run_cli_test():
    """Test CLI functionality"""
    print_subheader("CLI Interface Test")
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, '-m', 'tools.scFeatureLens.cli', '--help'
        ], capture_output=True, text=True, timeout=30, check=False)
        
        if result.returncode == 0:
            print("✓ CLI help command works")
            return True
        else:
            print(f"✗ CLI help command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ CLI test timed out")
        return False
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False


def run_example_test():
    """Run the example script"""
    print_subheader("Example Script Test")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'tools.scFeatureLens.example', '--example', 'basic'
        ], capture_output=True, text=True, timeout=120, cwd=os.getcwd(), check=False)
        
        if result.returncode == 0:
            print("✓ Example script completed successfully")
            # Check if output files were created
            if os.path.exists('example_results'):
                print("✓ Example output directory created")
            return True
        else:
            print(f"✗ Example script failed:")
            print(f"  stdout: {result.stdout[-500:]}")  # Last 500 chars
            print(f"  stderr: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Example script timed out")
        return False
    except Exception as e:
        print(f"✗ Example script test failed: {e}")
        return False


def check_data_setup():
    """Check if data directories and files are properly set up"""
    print_subheader("Data Setup Check")
    
    expected_dirs = ['scripts', 'tests', 'setup', 'docs', 'examples', 'tools']
    expected_files = [
        'setup/requirements.txt', 'setup.py', 'setup/environment.yml',
        'setup_env.sh', 'README.md'
    ]
    
    missing_dirs = [d for d in expected_dirs if not os.path.exists(d)]
    missing_files = [f for f in expected_files if not os.path.exists(f)]
    
    if not missing_dirs and not missing_files:
        print("✓ All expected directories and files present")
        return True
    else:
        if missing_dirs:
            print(f"✗ Missing directories: {missing_dirs}")
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
        return False


def check_memory_usage():
    """Check memory usage and system resources"""
    print_subheader("System Resources")
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        print(f"Total memory: {memory.total / (1024**3):.2f} GB")
        print(f"Available memory: {memory.available / (1024**3):.2f} GB")
        print(f"Memory usage: {memory.percent:.1f}%")
        print(f"CPU cores: {cpu_count}")
        
        if memory.available / (1024**3) < 1.0:
            print("⚠️  Warning: Less than 1GB memory available")
            return False
        else:
            print("✓ Sufficient memory available")
            return True
            
    except Exception as e:
        print(f"✗ System resource check failed: {e}")
        return False


def cleanup_test_files():
    """Clean up any test files created during validation"""
    print_subheader("Cleanup")
    
    cleanup_paths = [
        'example_data',
        'example_results', 
        'test_output',
        '__pycache__'
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"✓ Cleaned up {path}")


def main():
    """Main validation function"""
    print_header("scFeatureLens Environment Validation")
    print("This script validates your isolated environment setup")
    print("and ensures all components are working correctly.")
    
    tests = [
        ("Python Environment", check_python_environment),
        ("Core Packages", check_core_packages), 
        ("scFeatureLens Import", check_scfeaturelens_import),
        ("PyTorch Functionality", check_torch_functionality),
        ("System Resources", check_memory_usage),
        ("Data Setup", check_data_setup),
        ("Basic Pipeline", run_basic_pipeline_test),
        ("CLI Interface", run_cli_test),
        ("Example Script", run_example_test),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your environment is ready for scFeatureLens.")
        print("\nNext steps:")
        print("1. Try running: python -m tools.scFeatureLens.example --example basic")
        print("2. Check out the documentation for usage examples")
        print("3. Start your analysis with your own data!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the correct environment")
        print("2. Run: pip install -e .")
        print("3. Check REPRODUCIBILITY_GUIDE.md for detailed setup instructions")
        return 1
    
    # Cleanup
    cleanup_test_files()
    return 0


if __name__ == "__main__":
    sys.exit(main())
