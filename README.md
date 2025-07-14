# Force Simulation Project

A sophisticated force-based polygon deformation simulation with restoration forces, optimization, and smoothing algorithms.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd forces
   ```

2. **Run the automated setup:**
   ```bash
   python install.py
   ```

   Or manually install:
   ```bash
   pip install -r requirements.txt
   python setup.py build_ext --inplace  # Optional: for Cython optimization
   ```

3. **Run the simulation:**
   ```bash
   python forces.py
   ```

## 📁 Project Structure

### Core Files (Required)
- `forces.py` - Main simulation engine
- `restoration_forces.py` - Restoration force calculations
- `requirements.txt` - Python dependencies
- `setup.py` - Cython compilation setup
- `install.py` - Automated installation script

### Optimization Files (Optional)
- `forces_cython.pyx` - Cython optimized functions
- `forces_cpp.hpp` - C++ optimization headers
- `enhanced_forces.py` - Enhanced simulation with additional features
- `optimized_forces.py` - Various optimization approaches

### Test Files (Optional)
- `test_clamping.py` - Test endpoint clamping logic
- `test_final_smoothing.py` - Test final smoothing phase
- `test_safe_vertices.py` - Test safe vertex optimization
- `run_forces_test.py` - Run main simulation without blocking

## 🔧 Dependencies

The project requires the following Python packages:

- **numpy** (≥1.21.0) - Numerical computations
- **matplotlib** (≥3.5.0) - Visualization
- **shapely** (≥1.8.0) - Geometric operations
- **numba** (≥0.56.0) - JIT compilation for performance
- **cython** (≥0.29.0) - Optional C extensions
- **setuptools** (≥60.0.0) - Build tools

## 🖥️ Platform Compatibility

### Windows
- Tested on Windows 10/11
- Use Command Prompt or PowerShell
- Visual Studio Build Tools may be required for Cython compilation

### Linux/macOS
- Should work out of the box
- GCC compiler required for Cython compilation

## 🚀 Performance Optimizations

The project includes multiple optimization layers:

1. **Numba JIT** - Automatic optimization (enabled by default)
2. **Cython Extensions** - Compiled C extensions (optional)
3. **C++ Backend** - Maximum performance (advanced users)

### Compiling Cython Extensions

```bash
# Windows
python setup.py build_ext --inplace

# Linux/macOS
python setup.py build_ext --inplace
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_clamping.py
python test_final_smoothing.py
python test_safe_vertices.py
```

## 📊 Usage Examples

### Basic Simulation
```python
from forces import main
main()  # Runs the full simulation with visualization
```

### Custom Parameters
```python
# Modify parameters in forces.py
distance_threshold = 0.15
max_iterations = 1500
restoration_strength = 0.5
```

## 🔍 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Cython compilation fails**: The project works without Cython using Numba
   ```bash
   # Skip Cython compilation, just run:
   python forces.py
   ```

3. **Matplotlib display issues**: For headless systems, use:
   ```python
   import matplotlib
   matplotlib.use('Agg')
   ```

### Performance Issues

- Ensure Numba is installed for JIT compilation
- Consider compiling Cython extensions for better performance
- Reduce `num_points` in `create_ellipse()` for faster computation

## 🤝 Contributing

The project is structured for easy extension:
- Add new force models in `restoration_forces.py`
- Implement new optimizations in separate files
- Add tests for new features

## 📈 Performance Benchmarks

- **Pure Python**: Baseline performance
- **Numba JIT**: 10-50x faster than pure Python
- **Cython**: 2-10x faster than Numba
- **C++ (planned)**: 10-100x faster than pure Python

## 🐛 Known Limitations

- Large polygons (>10,000 points) may be slow without C++ optimization
- Matplotlib visualization can be slow for many iterations
- Memory usage scales with polygon complexity

## 📝 License

[Add your license information here]

## 🔗 References

[Add any academic papers or references used]
