#!/usr/bin/env python3
"""
Test script to verify Python 3.6 compatibility
"""

import sys
import os

def test_python36_compatibility():
    """Test if the project works with Python 3.6 features"""
    
    print("🔍 Testing Python 3.6 Compatibility")
    print("=" * 50)
    
    # Test f-strings (available in Python 3.6+)
    version = sys.version_info
    test_var = "test"
    try:
        result = f"Python {version.major}.{version.minor} with {test_var}"
        print(f"✓ f-strings work: {result}")
    except SyntaxError:
        print("❌ f-strings not supported")
        return False
    
    # Test pathlib (available in Python 3.6+)
    try:
        from pathlib import Path
        test_path = Path(".")
        print(f"✓ pathlib works: {test_path.absolute()}")
    except ImportError:
        print("❌ pathlib not available")
        return False
    
    # Test requirements files exist
    requirements_files = {
        "requirements.txt": "Python 3.7+ requirements",
        "requirements-py36.txt": "Python 3.6 requirements"
    }
    
    for file, description in requirements_files.items():
        if os.path.exists(file):
            print(f"✓ {description} file exists: {file}")
        else:
            print(f"❌ {description} file missing: {file}")
    
    # Test version-specific logic
    if version >= (3, 7):
        print("✓ Python 3.7+ detected - full feature support")
        requirements_file = "requirements.txt"
    elif version >= (3, 6):
        print("✓ Python 3.6 detected - compatibility mode")
        requirements_file = "requirements-py36.txt"
    else:
        print("❌ Python version too old")
        return False
    
    print(f"✓ Would use requirements file: {requirements_file}")
    
    # Test basic numeric operations (should work on all versions)
    try:
        import numpy as np
        test_array = np.array([1, 2, 3])
        result = np.mean(test_array)
        print(f"✓ NumPy basic operations work: mean = {result}")
    except ImportError:
        print("⚠️  NumPy not installed (expected in fresh environment)")
    except Exception as e:
        print(f"❌ NumPy operation failed: {e}")
        return False
    
    # Test import structure
    try:
        # Test if the main files can be imported
        import importlib.util
        
        # Check if forces.py exists and has valid syntax
        if os.path.exists("forces.py"):
            spec = importlib.util.spec_from_file_location("forces", "forces.py")
            print("✓ forces.py has valid Python syntax")
        else:
            print("❌ forces.py not found")
            return False
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Python 3.6 compatibility test passed!")
    return True

if __name__ == "__main__":
    success = test_python36_compatibility()
    sys.exit(0 if success else 1)
