# Release Process

This document outlines the release process for the Bank Churn Analysis package.

## Release Types

### Development Releases (TestPyPI)
- **Trigger**: Any git tag (e.g., `0.2.1-alpha`, `0.3.0-beta`, `test-release`)
- **Destination**: TestPyPI
- **GitHub Release**: Created as pre-release

### Production Releases (PyPI)
- **Trigger**: Git tags starting with `v` (e.g., `v0.2.0`, `v1.0.0`, `v2.1.3`)
- **Destination**: Both TestPyPI and PyPI
- **GitHub Release**: Created as stable release

## Creating a Release

### 1. Prepare the Release

```bash
# Update version in pyproject.toml and setup.py
# Update CHANGELOG.md with new features and fixes
# Commit changes
git add .
git commit -m "Prepare release v0.2.1"
git push origin main
```

### 2. Create and Push Tag

#### For Development/Test Release:
```bash
# Create a test tag
git tag 0.2.1-alpha
git push origin 0.2.1-alpha
```

#### For Production Release:
```bash
# Create a production tag (must start with 'v')
git tag v0.2.1
git push origin v0.2.1
```

### 3. Automated Process

Once you push a tag, GitHub Actions will automatically:

1. **Run Tests**: Execute the full test suite
2. **Build Package**: Create wheel and source distributions
3. **Publish to TestPyPI**: For any tag
4. **Publish to PyPI**: Only for `v*` tags
5. **Create GitHub Release**: With built artifacts attached

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., `1.0.0`)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Pre-release Versions
- **Alpha**: `1.0.0-alpha.1`
- **Beta**: `1.0.0-beta.1`
- **Release Candidate**: `1.0.0-rc.1`

## Required Secrets

Ensure these secrets are configured in your GitHub repository:

### PyPI Secrets
1. **PYPI_API_TOKEN**: Your PyPI API token
   - Go to https://pypi.org/manage/account/token/
   - Create a new token with upload permissions
   - Add to GitHub Secrets

2. **TEST_PYPI_API_TOKEN**: Your TestPyPI API token
   - Go to https://test.pypi.org/manage/account/token/
   - Create a new token with upload permissions
   - Add to GitHub Secrets

### Kaggle Secrets (for testing)
3. **KAGGLE_USERNAME**: Your Kaggle username
4. **KAGGLE_KEY**: Your Kaggle API key

## Manual Release (if needed)

If you need to release manually:

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Post-Release Checklist

1. **Verify Installation**: Test installing from PyPI/TestPyPI
2. **Update Documentation**: Ensure installation instructions are current
3. **Announce Release**: Update README, social media, etc.
4. **Plan Next Release**: Update project roadmap

## Testing Releases

### Testing from TestPyPI
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ bank-churn-analysis

# Test basic functionality
python -c "import src.data.load_data; print('Success!')"
```

### Testing from PyPI
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from PyPI
pip install bank-churn-analysis

# Test command line tools
churn-predict --help
churn-segment --help
churn-explain --help
```

## Rollback Process

If a release has critical issues:

1. **Immediate**: Remove the problematic release from PyPI (if possible)
2. **Create Hotfix**: Fix the issue in a new patch version
3. **Release Hotfix**: Follow normal release process
4. **Communicate**: Notify users about the issue and fix

## Release Notes Template

```markdown
## [v0.2.1] - 2024-01-15

### Added
- New feature descriptions

### Changed
- Modified functionality descriptions

### Fixed
- Bug fix descriptions

### Removed
- Deprecated feature removals

### Security
- Security-related changes
```

## Monitoring

After release, monitor:

- **Download Statistics**: PyPI download counts
- **Issue Reports**: GitHub issues related to the release
- **User Feedback**: Community responses
- **CI/CD Status**: Ensure automated processes are working