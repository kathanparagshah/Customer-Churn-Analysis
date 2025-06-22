# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.6] - 2024-12-19

### Changed
- Replaced deprecated `pandas-profiling` with modern `ydata-profiling>=4.0.0,<5.0.0`
- Loosened `joblib` version pin from `==1.1.1` to `>=1.1.0,<1.4.0` for better compatibility
- Downgraded `pre-commit` to `3.5.5` for Python 3.8 compatibility
- Updated CI workflow to use minimal test dependencies for faster builds

## [0.2.0] - 2024-12-19

### Added
- Comprehensive unit test suite with 95%+ coverage
- Robust error handling and logging throughout the pipeline
- Support for configurable project root directory
- Enhanced data validation and encoding fallback mechanisms
- Detailed API documentation and usage examples
- Docker deployment configuration
- Continuous integration setup

### Fixed
- **Critical**: Fixed F1 score key mismatch in model evaluation metrics
  - Changed from `f1_score` to `f1` to match scikit-learn's classification_report output
  - Affects: `src/models/train_churn.py`, `src/tests/test_models.py`

- **Critical**: Resolved project root path handling issues
  - Added dynamic project root detection and configuration
  - Fixed file path resolution across different execution contexts
  - Affects: All modules requiring file I/O operations

- **Data Pipeline**: Fixed missing `processed_dir` attribute in DataLoader
  - Added proper initialization of processed data directory
  - Ensures consistent data processing workflow
  - Affects: `src/data/load_data.py`

- **Segmentation**: Fixed TypeError in customer segmentation analysis
  - Resolved DataFrame indexing issues in `analyze_segments` method
  - Added proper error handling for edge cases
  - Affects: `src/models/segment.py`

- **Testing**: Fixed missing 'characteristics' key in segmentation output
  - Added required output structure for segment analysis
  - Ensures API compatibility and test consistency
  - Affects: `src/models/segment.py`, `src/tests/test_models.py`

- **Data Loading**: Fixed file path handling in encoding tests
  - Corrected temporary file path resolution in unit tests
  - Improved test isolation and reliability
  - Affects: `src/tests/test_data_pipeline.py`

### Enhanced
- **Age Binning**: Improved age group categorization logic
  - More intuitive age ranges: Young (18-30), Middle-aged (31-50), Senior (51+)
  - Better handling of edge cases and missing values

- **Balance Handling**: Enhanced zero-balance customer processing
  - Improved detection and categorization of inactive accounts
  - Better feature engineering for balance-related metrics

- **Outlier Detection**: Refined outlier identification algorithms
  - More robust statistical methods for anomaly detection
  - Configurable thresholds for different data distributions

- **Encoding Fallback**: Improved CSV file encoding detection
  - Automatic fallback to alternative encodings (latin-1, cp1252)
  - Better error messages and logging for encoding issues

- **Documentation**: Comprehensive README and API documentation
  - Detailed installation and setup instructions
  - Complete usage examples and best practices
  - Testing guidelines and CI/CD information

### Technical Improvements
- Enhanced logging configuration with structured output
- Improved error handling with descriptive messages
- Better separation of concerns in model classes
- More robust data validation and preprocessing
- Optimized memory usage in large dataset processing

### Testing
- Added comprehensive unit tests for all major components
- Implemented integration tests for end-to-end workflows
- Added API endpoint testing with FastAPI TestClient
- Configured automated testing with pytest and coverage reporting
- Set up continuous integration for multiple Python versions

### Dependencies
- Updated core dependencies to latest stable versions
- Added development dependencies for testing and documentation
- Improved dependency management with version pinning
- Added optional dependencies for enhanced functionality

---

## [0.1.0] - 2024-12-01

### Added
- Initial project structure and core functionality
- Data loading and preprocessing pipeline
- Customer churn prediction model
- Customer segmentation analysis
- Basic exploratory data analysis notebooks
- FastAPI web service for model deployment
- Docker containerization support

### Features
- Machine learning pipeline for churn prediction
- K-means clustering for customer segmentation
- Feature engineering and selection
- Model interpretability with SHAP values
- RESTful API for model inference
- Jupyter notebooks for analysis and visualization

---

## Release Notes

### Version 0.2.0 Highlights

This release focuses on **stability, testing, and production readiness**. Key improvements include:

1. **Comprehensive Testing**: Added 67 unit tests covering all major components
2. **Robust Error Handling**: Fixed critical bugs and improved error recovery
3. **Enhanced Documentation**: Complete setup and usage instructions
4. **Production Ready**: Docker deployment and CI/CD configuration
5. **API Stability**: Consistent output formats and error responses

### Breaking Changes

- **Model Metrics**: F1 score key changed from `f1_score` to `f1` in evaluation results
- **Project Structure**: Project root must be properly configured for file operations
- **API Responses**: Segmentation analysis now includes 'characteristics' field

### Migration Guide

To upgrade from v0.1.0 to v0.2.0:

1. Update your code to use the new F1 score key:
   ```python
   # Old
   f1 = metrics['f1_score']
   
   # New
   f1 = metrics['f1']
   ```

2. Ensure project root is properly configured:
   ```python
   from src.utils.config import setup_project_root
   setup_project_root()
   ```

3. Update API clients to handle new segmentation response format:
   ```python
   # Response now includes 'characteristics' field
   response = segmentation_api.analyze_segments(data)
   characteristics = response['characteristics']
   ```

### Contributors

- Core development and testing improvements
- Documentation enhancements
- Bug fixes and stability improvements

### Acknowledgments

Thanks to all contributors who helped make this release more robust and production-ready!