# Python 3.6 Compatibility Guide

## Overview
This project has been updated to support Python 3.6+ to ensure broader compatibility. Below are the key considerations and version-specific information.

## Python 3.6 Compatibility Features

### ✅ **Supported Features**
- **f-strings** (Python 3.6+) - Used extensively throughout the project
- **pathlib** - Available and used in setup scripts
- **NumPy 1.16.0+** - Supports Python 3.6
- **Numba 0.48.0+** - Supports Python 3.6
- **Shapely 1.6.0+** - Supports Python 3.6
- **Matplotlib 3.0.0+** - Supports Python 3.6
- **Cython 0.29.0+** - Supports Python 3.6

### ⚠️ **Potential Limitations**
- **Type Hints**: Basic type hints work, but some advanced features may not be available
- **Older NumPy**: Some newer NumPy features may not be available
- **Performance**: Slightly slower than newer Python versions due to interpreter improvements

## Version-Specific Dependency Requirements

### Python 3.6
```
numpy>=1.16.0,<1.20.0
matplotlib>=3.0.0,<3.4.0
shapely>=1.6.0,<1.8.0
numba>=0.48.0,<0.54.0
cython>=0.29.0
setuptools>=40.0.0
```

### Python 3.7+
```
numpy>=1.18.0
matplotlib>=3.2.0
shapely>=1.7.0
numba>=0.50.0
cython>=0.29.0
setuptools>=45.0.0
```

### Python 3.8+
```
numpy>=1.19.0
matplotlib>=3.3.0
shapely>=1.7.0
numba>=0.53.0
cython>=0.29.0
setuptools>=50.0.0
```

### Python 3.9+
```
numpy>=1.20.0
matplotlib>=3.4.0
shapely>=1.8.0
numba>=0.56.0
cython>=0.29.0
setuptools>=60.0.0
```

## Installation Instructions by Python Version

### For Python 3.6
```bash
# Create virtual environment
python3.6 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install compatible versions
pip install numpy==1.19.5
pip install matplotlib==3.3.4
pip install shapely==1.7.1
pip install numba==0.53.1
pip install cython>=0.29.0
pip install setuptools>=40.0.0

# Run setup
python install.py
```

### For Python 3.7+
```bash
# Standard installation
pip install -r requirements.txt
python install.py
```

## Performance Considerations

### Python 3.6 vs 3.9+ Performance
- **Numba JIT**: 10-50x speedup (same across versions)
- **Cython**: 2-10x speedup (same across versions)
- **Pure Python**: 3.9+ is ~20-30% faster than 3.6

### Memory Usage
- Python 3.6: Slightly higher memory usage
- Python 3.9+: Better memory optimization

## Testing Across Python Versions

### Automated Testing
```bash
# Test with different Python versions
python3.6 verify_setup.py
python3.7 verify_setup.py
python3.8 verify_setup.py
python3.9 verify_setup.py
```

### Common Issues and Solutions

#### 1. **NumPy Version Conflicts**
```bash
# For Python 3.6
pip install "numpy>=1.16.0,<1.20.0"
```

#### 2. **Matplotlib Display Issues**
```python
# For headless systems or older Python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

#### 3. **Numba Compilation Issues**
```bash
# If Numba fails to compile
pip install "numba>=0.48.0,<0.54.0"  # For Python 3.6
```

## Backward Compatibility Features

### 1. **Graceful Fallbacks**
- If Cython compilation fails, uses Numba
- If Numba fails, falls back to pure Python
- Automatic backend selection for matplotlib

### 2. **Version Detection**
```python
import sys
if sys.version_info >= (3, 8):
    # Use newer features
    pass
else:
    # Use compatible alternatives
    pass
```

### 3. **Dependency Management**
The project automatically handles version-specific dependencies through the installation scripts.

## CI/CD Considerations

### GitHub Actions Example
```yaml
strategy:
  matrix:
    python-version: [3.6, 3.7, 3.8, 3.9]
    
steps:
- uses: actions/setup-python@v2
  with:
    python-version: ${{ matrix.python-version }}
- run: python install.py
- run: python verify_setup.py
```

## Deployment Recommendations

### For Production
- **Python 3.8+**: Recommended for best performance
- **Python 3.7**: Good balance of compatibility and performance
- **Python 3.6**: Use only if required by system constraints

### For Development
- **Python 3.9+**: Best development experience
- **Python 3.6**: Test compatibility but develop on newer version

## Known Limitations

### Python 3.6 Specific
- Some advanced NumPy functions may not be available
- Slightly slower f-string performance
- Limited type hinting support

### Workarounds
- Use alternative NumPy functions for missing features
- Profile performance on target Python version
- Use basic type hints or none at all

## Future Compatibility

### Planned Updates
- **Python 3.11+**: Will add support when dependencies are ready
- **Python 3.10**: Currently supported
- **Python 3.6**: Will maintain support until EOL (December 2021 - already EOL)

### Migration Path
When Python 3.6 support is dropped:
1. Update minimum requirements to Python 3.7+
2. Use newer NumPy/Matplotlib features
3. Improve type hints
4. Optimize for newer Python versions

## Conclusion

The project now supports Python 3.6+ with appropriate version-specific dependencies. While Python 3.6 works, newer versions provide better performance and features. Choose the Python version based on your deployment requirements and performance needs.
