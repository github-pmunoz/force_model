#!/usr/bin/env python3
"""
Verification script to check if the project setup is working correctly
"""

import sys
import importlib
import subprocess
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version >= (3, 6):
        print(f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        if version < (3, 7):
            print("⚠️  Python 3.6 detected - using compatibility mode")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is too old (requires 3.6+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name} ({version}) is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed")
        return False

def check_cython_compilation():
    """Check if Cython extensions are compiled"""
    cython_files = [
        'forces_cython.pyx',
        'forces_cython.c',
        'forces_cython.cp311-win_amd64.pyd',  # Windows
        'forces_cython.cpython-311-x86_64-linux-gnu.so',  # Linux
        'forces_cython.cpython-311-darwin.so'  # macOS
    ]
    
    has_source = os.path.exists('forces_cython.pyx')
    has_compiled = any(os.path.exists(f) for f in cython_files[2:])
    
    if has_source and has_compiled:
        print("✓ Cython extensions are compiled")
        return True
    elif has_source:
        print("⚠️  Cython source found but not compiled (performance will use Numba instead)")
        return False
    else:
        print("ℹ️  No Cython extensions found (using Numba optimization)")
        return False

def check_project_files():
    """Check if all required project files are present"""
    required_files = [
        'forces.py',
        'restoration_forces.py',
        'requirements.txt',
        'setup.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_core_functionality():
    """Test if core functionality works"""
    print("\n🧪 Testing core functionality...")
    
    try:
        # Test numpy and basic operations
        import numpy as np
        arr = np.array([1, 2, 3, 4, 5])
        print("✓ NumPy operations work")
        
        # Test matplotlib (non-interactive)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✓ Matplotlib works")
        
        # Test shapely
        from shapely.geometry import Polygon, LineString
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        line = LineString([(0, 0), (1, 1)])
        print("✓ Shapely geometry works")
        
        # Test numba
        from numba import jit
        @jit(nopython=True)
        def test_func(x):
            return x * 2
        result = test_func(5)
        print("✓ Numba JIT compilation works")
        
        # Test project imports
        import restoration_forces
        print("✓ Project modules import successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def main():
    """Main verification routine"""
    print("🔍 Verifying Force Simulation Project Setup")
    print("=" * 50)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📦 Checking dependencies...")
    required_packages = [
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('shapely', 'shapely'),
        ('numba', 'numba'),
        ('cython', 'cython'),
        ('setuptools', 'setuptools')
    ]
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    print("\n🔧 Checking compilation...")
    check_cython_compilation()  # This is optional, so don't fail if missing
    
    print("\n📁 Checking project files...")
    if not check_project_files():
        all_good = False
    
    # Test functionality
    if not test_core_functionality():
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! The project is ready to run.")
        print("\nTo start the simulation:")
        print("  python forces.py")
    else:
        print("❌ Some checks failed. Please review the issues above.")
        print("\nTo fix missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
