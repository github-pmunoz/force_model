#!/usr/bin/env python3
"""
Setup script for the force simulation project
This script will:
1. Install required dependencies
2. Compile Cython extensions
3. Verify the installation
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def main():
    """Main setup process"""
    print("🚀 Setting up Force Simulation Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {sys.version}")
    
    # Choose appropriate requirements file based on Python version
    if sys.version_info >= (3, 7):
        requirements_file = "requirements.txt"
        print("Using standard requirements for Python 3.7+")
    else:
        requirements_file = "requirements-py36.txt"
        print("Using Python 3.6 compatible requirements")
        if not os.path.exists(requirements_file):
            print("⚠️  Python 3.6 requirements file not found, using standard requirements")
            requirements_file = "requirements.txt"
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not run_command(f"pip install -r {requirements_file}", f"Installing Python packages from {requirements_file}"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Compile Cython extensions (optional but recommended for performance)
    print("\n🔧 Compiling Cython extensions...")
    if os.path.exists("forces_cython.pyx"):
        if run_command("python setup.py build_ext --inplace", "Compiling Cython extensions"):
            print("✓ Cython extensions compiled successfully")
        else:
            print("⚠️  Cython compilation failed (the project will still work with Numba)")
    else:
        print("⚠️  No Cython files found, skipping compilation")
    
    # Test the installation
    print("\n🧪 Testing installation...")
    test_code = """
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from numba import jit
import restoration_forces
print("✓ All imports successful")
"""
    
    try:
        exec(test_code)
        print("✓ Installation test passed")
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the simulation:")
    print("  python forces.py")
    print("\nTo run tests:")
    print("  python test_clamping.py")
    print("  python test_final_smoothing.py")
    print("  python test_safe_vertices.py")

if __name__ == "__main__":
    main()
