#!/usr/bin/env python3
"""
Deployment package creator for Force Simulation Project
This creates a clean deployment package with all necessary files
"""

import os
import shutil
import zipfile
from pathlib import Path

def create_deployment_package():
    """Create a deployment package with all necessary files"""
    
    # Files to include in deployment
    deployment_files = [
        # Core files (required)
        'forces.py',
        'restoration_forces.py',
        'requirements.txt',
        'requirements-py36.txt',
        'setup.py',
        'README.md',
        'PYTHON36_COMPATIBILITY.md',
        'install.py',
        'verify_setup.py',
        '.gitignore',
        
        # Optimization files (optional but recommended)
        'forces_cython.pyx',
        'forces_cpp.hpp',
        'enhanced_forces.py',
        'optimized_forces.py',
        
        # Test files (optional)
        'test_clamping.py',
        'test_final_smoothing.py',
        'test_safe_vertices.py',
        'run_forces_test.py',
    ]
    
    # Create deployment directory
    deploy_dir = Path('deployment_package')
    if deploy_dir.exists():
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir()
    
    # Copy files to deployment directory
    copied_files = []
    missing_files = []
    
    for file in deployment_files:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir / file)
            copied_files.append(file)
        else:
            missing_files.append(file)
    
    # Create a deployment info file
    info_content = f"""# Force Simulation Deployment Package

## Included Files ({len(copied_files)} files)
"""
    
    for file in sorted(copied_files):
        info_content += f"- {file}\n"
    
    if missing_files:
        info_content += f"\n## Missing Files ({len(missing_files)} files)\n"
        for file in sorted(missing_files):
            info_content += f"- {file} (optional)\n"
    
    info_content += f"""
## Installation Instructions

1. Extract all files to a directory
2. Run: python install.py
3. Or manually: pip install -r requirements.txt
4. Test: python verify_setup.py
5. Run: python forces.py

## Files Description

### Core Files (Required)
- **forces.py** - Main simulation engine
- **restoration_forces.py** - Restoration force calculations
- **requirements.txt** - Python dependencies (3.7+)
- **requirements-py36.txt** - Python 3.6 compatible dependencies
- **setup.py** - Cython compilation setup
- **README.md** - Complete documentation
- **PYTHON36_COMPATIBILITY.md** - Python 3.6 compatibility guide
- **install.py** - Automated installation script
- **verify_setup.py** - Setup verification script

### Optimization Files (Optional)
- **forces_cython.pyx** - Cython optimized functions
- **forces_cpp.hpp** - C++ optimization headers
- **enhanced_forces.py** - Enhanced simulation features
- **optimized_forces.py** - Various optimization approaches

### Test Files (Optional)
- **test_clamping.py** - Test endpoint clamping logic
- **test_final_smoothing.py** - Test final smoothing phase
- **test_safe_vertices.py** - Test safe vertex optimization
- **run_forces_test.py** - Run simulation without blocking

## System Requirements
- Python 3.6+ (see PYTHON36_COMPATIBILITY.md for 3.6 specifics)
- Windows/Linux/macOS
- 4GB RAM minimum
- Git (for version control)

## Python Version Support
- **Python 3.6**: Supported with limitations (see PYTHON36_COMPATIBILITY.md)
- **Python 3.7+**: Full support with optimal performance
- **Python 3.9+**: Recommended for best performance

## Installation Methods

### Automatic (Recommended)
```bash
python install.py
```

### Manual for Python 3.6
```bash
pip install -r requirements-py36.txt
python setup.py build_ext --inplace
```

### Manual for Python 3.7+
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Quick Start
```bash
python install.py
python forces.py
```
"""
    
    with open(deploy_dir / 'DEPLOYMENT_INFO.md', 'w') as f:
        f.write(info_content)
    
    # Create zip file
    zip_path = 'force_simulation_deployment.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, deploy_dir)
                zipf.write(file_path, arcname)
    
    print(f"✓ Deployment package created: {zip_path}")
    print(f"✓ Deployment directory created: {deploy_dir}")
    print(f"✓ Included {len(copied_files)} files")
    
    if missing_files:
        print(f"⚠️  Skipped {len(missing_files)} optional files")
    
    return deploy_dir, zip_path

if __name__ == "__main__":
    create_deployment_package()
