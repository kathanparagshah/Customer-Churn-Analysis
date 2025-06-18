# Packaging and Distribution Guide

This document provides a comprehensive guide to the packaging and distribution setup for the Bank Churn Analysis project.

## Overview

The project is now configured for distribution via PyPI with the following features:

- ✅ **Modern packaging** with `pyproject.toml` and `setup.py`
- ✅ **Automated CI/CD** with GitHub Actions
- ✅ **Multi-target publishing** (TestPyPI and PyPI)
- ✅ **Command-line tools** for easy usage
- ✅ **Comprehensive testing** before release
- ✅ **Automated GitHub releases** with artifacts

## Package Configuration

### Files Added/Modified

1. **`pyproject.toml`** - Modern Python packaging configuration
2. **`setup.py`** - Traditional packaging (maintained for compatibility)
3. **`MANIFEST.in`** - Controls which files are included in distributions
4. **`.github/workflows/ci.yml`** - Enhanced CI/CD pipeline
5. **`scripts/release.sh`** - Automated release script
6. **Documentation** - Installation, release, and packaging guides

### Package Details

- **Name**: `bank-churn-analysis`
- **Current Version**: `0.2.0`
- **Python Support**: 3.8, 3.9, 3.10, 3.11
- **License**: MIT
- **Repository**: https://github.com/kathanparagshah/Customer-Churn-Analysis

## Installation Methods

### 1. From PyPI (Production)
```bash
pip install bank-churn-analysis
```

### 2. From TestPyPI (Testing)
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis
```

### 3. From Source (Development)
```bash
git clone https://github.com/kathanparagshah/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
pip install -e .[dev]
```

## Command Line Tools

After installation, these tools are available:

```bash
churn-predict    # Train churn prediction models
churn-segment    # Perform customer segmentation  
churn-explain    # Generate model explanations
```

## Release Process

### Automated Release (Recommended)

Use the provided release script:

```bash
# Test release (publishes to TestPyPI)
./scripts/release.sh 0.2.1 test

# Production release (publishes to PyPI)
./scripts/release.sh 0.2.1 prod
```

### Manual Release

1. **Update version** in `pyproject.toml` and `setup.py`
2. **Commit changes**: `git commit -am "Bump version to X.Y.Z"`
3. **Create tag**:
   - Test: `git tag X.Y.Z-test`
   - Production: `git tag vX.Y.Z`
4. **Push**: `git push origin main && git push origin <tag>`

### GitHub Actions Workflow

The CI/CD pipeline automatically:

1. **Runs tests** on Python 3.8-3.11
2. **Builds packages** (wheel and source distribution)
3. **Publishes to TestPyPI** on any tag
4. **Publishes to PyPI** on `v*` tags only
5. **Creates GitHub releases** with built artifacts

## Required Secrets

Configure these in your GitHub repository settings:

- `PYPI_API_TOKEN` - PyPI API token for production releases
- `TEST_PYPI_API_TOKEN` - TestPyPI API token for test releases
- `KAGGLE_USERNAME` - Kaggle username (for testing)
- `KAGGLE_KEY` - Kaggle API key (for testing)

## Testing the Package

### Local Testing
```bash
# Build the package
python3 -m build

# Install locally
pip install dist/bank_churn_analysis-*.whl

# Test command line tools
churn-predict --help
```

### TestPyPI Testing
```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis

# Test functionality
python -c "import src.data.load_data; print('Success!')"
```

## Package Structure

The package includes:

```
bank-churn-analysis/
├── src/
│   ├── data/           # Data loading and processing
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── tests/          # Test suite
│   └── visualization/  # Plotting utilities
├── deployment/         # API and Docker files
├── notebooks/          # Jupyter notebooks
└── scripts/           # Utility scripts
```

## Dependencies

### Core Dependencies
- pandas, numpy, scikit-learn
- xgboost, lightgbm
- shap, lime (explainability)
- fastapi, uvicorn (API)
- matplotlib, seaborn, plotly (visualization)

### Optional Dependencies
- `[dev]` - Development tools (pytest, black, flake8)
- `[docs]` - Documentation tools (sphinx)

## Monitoring and Maintenance

### After Release
1. **Monitor downloads** on PyPI
2. **Check for issues** in GitHub Issues
3. **Update documentation** as needed
4. **Plan next release** based on feedback

### Version Management
- Follow [Semantic Versioning](https://semver.org/)
- Use pre-release versions for testing
- Maintain CHANGELOG.md

## Troubleshooting

### Common Issues

1. **Build failures**: Check `pyproject.toml` syntax
2. **Import errors**: Verify package structure and `__init__.py` files
3. **Missing files**: Update `MANIFEST.in`
4. **CI/CD failures**: Check GitHub Actions logs

### Getting Help

- Check [GitHub Issues](https://github.com/kathanparagshah/Customer-Churn-Analysis/issues)
- Review [PyPI documentation](https://packaging.python.org/)
- Consult [GitHub Actions docs](https://docs.github.com/en/actions)

## Future Enhancements

- [ ] Add conda-forge distribution
- [ ] Implement semantic release automation
- [ ] Add performance benchmarks to CI
- [ ] Create Docker images for releases
- [ ] Add integration with package registries