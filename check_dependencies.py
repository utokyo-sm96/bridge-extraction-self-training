#!/usr/bin/env python3
"""
Dependency Checker for Self-Training Framework
================================================

Checks all required and optional dependencies and reports their status.
Run this first to ensure your environment is properly configured.

Usage:
    python check_dependencies.py
"""

import sys
import platform
import subprocess


def check_python_version():
    """Check Python version."""
    print("\n" + "="*60)
    print("PYTHON VERSION")
    print("="*60)

    version = sys.version_info
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    if version.major == 3 and version.minor >= 9:
        print("  ✓ Python version OK (3.9+)")

        if version.minor >= 14:
            print("  ⚠ Python 3.14 detected - some packages may have compatibility issues")
            print("    Recommended: Python 3.10-3.12 for best compatibility")
        return True
    else:
        print(f"  ✗ Python 3.9+ required, found {version.major}.{version.minor}")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and importable."""
    import_name = import_name or package_name

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Import error: {e}"


def check_pytorch():
    """Check PyTorch installation and GPU availability."""
    print("\n" + "="*60)
    print("PYTORCH")
    print("="*60)

    success, result = check_package('torch')
    if not success:
        print(f"  ✗ PyTorch not installed: {result}")
        print("    Install: pip install torch")
        return False

    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")

    # Check compute device
    if torch.cuda.is_available():
        print(f"  ✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✓ Apple MPS (Metal) available")
    else:
        print("  ○ CPU only (training will be slower)")

    return True


def check_transformers():
    """Check transformers library."""
    print("\n" + "="*60)
    print("TRANSFORMERS")
    print("="*60)

    try:
        # Try importing just the version first
        import importlib.metadata
        version = importlib.metadata.version('transformers')
        print(f"  Package installed: transformers {version}")

        # Now try actual import
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
        return True

    except ImportError:
        print("  ✗ Transformers not installed")
        print("    Install: pip install transformers")
        return False

    except Exception as e:
        print(f"  ✗ Transformers import error: {e}")
        print()
        print("  This is likely a Python 3.14 compatibility issue.")
        print("  Workarounds:")
        print("    1. Try: pip install transformers==4.44.0")
        print("    2. Or use Python 3.10-3.12 for best compatibility")
        print("    3. The framework will use fallback mode without transformer training")
        return False


def check_required_packages():
    """Check required packages."""
    print("\n" + "="*60)
    print("REQUIRED PACKAGES")
    print("="*60)

    required = [
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
    ]

    all_ok = True
    for package, import_name in required:
        success, result = check_package(package, import_name)
        if success:
            print(f"  ✓ {package}: {result}")
        else:
            print(f"  ✗ {package}: {result}")
            all_ok = False

    return all_ok


def check_optional_packages():
    """Check optional packages."""
    print("\n" + "="*60)
    print("OPTIONAL PACKAGES")
    print("="*60)

    optional = [
        ('seaborn', 'seaborn', 'Enhanced visualizations'),
        ('python-docx', 'docx', 'DOCX file reading'),
        ('openai', 'openai', 'LLM baseline comparison'),
        ('tiktoken', 'tiktoken', 'Token counting'),
    ]

    for package, import_name, description in optional:
        success, result = check_package(package, import_name)
        if success:
            print(f"  ✓ {package}: {result} ({description})")
        else:
            print(f"  ○ {package}: not installed ({description})")


def check_data_directories():
    """Check if required data directories exist."""
    print("\n" + "="*60)
    print("DATA DIRECTORIES")
    print("="*60)

    from pathlib import Path

    directories = {
        'data/documents': 'Source DOCX files',
        'data/json': 'Extracted JSON files',
    }

    all_ok = True
    for dir_name, description in directories.items():
        path = Path(dir_name)
        if path.exists():
            # Count files
            docx_count = len(list(path.glob('*.docx')))
            txt_count = len(list(path.glob('*.txt')))
            json_count = len(list(path.glob('*.json')))

            file_counts = []
            if docx_count: file_counts.append(f"{docx_count} DOCX")
            if txt_count: file_counts.append(f"{txt_count} TXT")
            if json_count: file_counts.append(f"{json_count} JSON")

            if file_counts:
                print(f"  ✓ {dir_name}/: {', '.join(file_counts)}")
            else:
                print(f"  ○ {dir_name}/: exists but empty")
                all_ok = False
        else:
            print(f"  ✗ {dir_name}/: not found ({description})")
            all_ok = False

    return all_ok


def check_files():
    """Check if required files exist."""
    print("\n" + "="*60)
    print("FRAMEWORK FILES")
    print("="*60)

    from pathlib import Path

    required_files = [
        'self_training_framework.py',
        'evaluation_metrics.py',
        'baseline_comparison.py',
        'self_training_visualization.py',
        'run_self_training_pipeline.py',
    ]

    optional_files = [
        '01_extraction_prompt_v2.txt',
        '02_function_schema_v2.json',
        '.env',
    ]

    all_ok = True
    for filename in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} (REQUIRED)")
            all_ok = False

    for filename in optional_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ○ {filename} (optional)")

    return all_ok


def run_quick_import_test():
    """Try to import the main framework modules."""
    print("\n" + "="*60)
    print("FRAMEWORK IMPORT TEST")
    print("="*60)

    # Test basic imports without transformers
    try:
        print("  Testing core imports...")
        import numpy as np
        import json
        from pathlib import Path
        from collections import defaultdict
        from dataclasses import dataclass
        print("  ✓ Core Python modules OK")
    except Exception as e:
        print(f"  ✗ Core import failed: {e}")
        return False

    # Test ML imports
    try:
        print("  Testing ML imports...")
        import torch
        from sklearn.model_selection import KFold
        from scipy import stats
        print("  ✓ ML modules OK")
    except Exception as e:
        print(f"  ✗ ML import failed: {e}")
        return False

    # Test visualization
    try:
        print("  Testing visualization imports...")
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        print("  ✓ Visualization modules OK")
    except Exception as e:
        print(f"  ✗ Visualization import failed: {e}")
        return False

    # Test transformers (may fail on Python 3.14)
    try:
        print("  Testing transformers import...")
        import transformers
        from transformers import AutoTokenizer, AutoModel
        print("  ✓ Transformers OK")
        return True
    except Exception as e:
        print(f"  ⚠ Transformers failed: {e}")
        print("    The framework will run in fallback mode without transformer training")
        return True  # Still return True as we have a fallback


def print_summary(results):
    """Print overall summary."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_ok = all(results.values())

    if all_ok:
        print("  ✓ All checks passed! Ready to run the pipeline.")
        print()
        print("  Next steps:")
        print("    1. python run_self_training_pipeline.py --quick  # Quick test")
        print("    2. python run_self_training_pipeline.py          # Full run")
    else:
        print("  ⚠ Some checks failed. Please review above and fix issues.")

        if not results.get('python'):
            print("    - Install Python 3.9+")
        if not results.get('pytorch'):
            print("    - Install PyTorch: pip install torch")
        if not results.get('required'):
            print("    - Install requirements: pip install -r requirements.txt")
        if not results.get('data'):
            print("    - Ensure data directories exist with files")

    print()
    return all_ok


def main():
    """Run all dependency checks."""
    print("="*60)
    print("SELF-TRAINING FRAMEWORK - DEPENDENCY CHECK")
    print("="*60)

    results = {}

    results['python'] = check_python_version()
    results['pytorch'] = check_pytorch()
    results['transformers'] = check_transformers()
    results['required'] = check_required_packages()
    check_optional_packages()  # Don't block on optional
    results['data'] = check_data_directories()
    results['files'] = check_files()
    results['imports'] = run_quick_import_test()

    success = print_summary(results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
