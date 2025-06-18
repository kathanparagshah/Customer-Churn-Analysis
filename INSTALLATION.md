# Installation Guide

## Installing from PyPI

Once published, you can install the bank-churn-analysis package directly from PyPI:

```bash
# Install the latest stable version
pip install bank-churn-analysis

# Install with development dependencies
pip install bank-churn-analysis[dev]

# Install with documentation dependencies
pip install bank-churn-analysis[docs]

# Install with all optional dependencies
pip install bank-churn-analysis[dev,docs]
```

## Installing from TestPyPI

To install pre-release versions from TestPyPI:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis
```

## Installing from Source

For development or to get the latest features:

```bash
# Clone the repository
git clone https://github.com/kathanparagshah/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

## Command Line Tools

After installation, you'll have access to these command-line tools:

```bash
# Train churn prediction models
churn-predict

# Perform customer segmentation
churn-segment

# Generate model explanations
churn-explain
```

## Kaggle API Setup

To download data from Kaggle, you need to set up your Kaggle API credentials:

1. Create a Kaggle account at https://www.kaggle.com
2. Go to your account settings and create a new API token
3. Download the `kaggle.json` file
4. Place it in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

Alternatively, set environment variables:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

## Docker Installation

You can also run the application using Docker:

```bash
# Build the Docker image
docker build -t bank-churn-analysis .

# Run with docker-compose
docker-compose up
```

## Verification

To verify your installation:

```python
import src.data.load_data
import src.models.train_churn
import src.models.segment
import src.models.explain

print("Bank Churn Analysis package installed successfully!")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct environment and all dependencies are installed
2. **Kaggle API errors**: Verify your Kaggle credentials are properly configured
3. **Memory issues**: Some models require significant RAM; consider using a machine with at least 8GB RAM
4. **Permission errors**: On Unix systems, you might need to use `sudo` or check file permissions

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/kathanparagshah/Customer-Churn-Analysis/issues)
2. Review the documentation in the repository
3. Create a new issue with detailed error information

## Development Setup

For contributors:

```bash
# Clone and setup development environment
git clone https://github.com/kathanparagshah/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev,docs]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
flake8 src/
black src/
```