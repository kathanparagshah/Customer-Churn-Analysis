# Enhanced Kaggle Download Functionality

This guide explains the enhanced `download_from_kaggle()` method in the DataLoader class, which provides robust data acquisition from Kaggle with retry logic, credential management, and error handling.

## Features

### 🔐 Flexible Credential Management
The method supports multiple credential sources in order of priority:

1. **Project Root `kaggle.json`** - Reads from `Customer Churn Analysis/kaggle.json`
2. **Environment Variables** - Uses `KAGGLE_USERNAME` and `KAGGLE_KEY`
3. **Default Kaggle Location** - Falls back to `~/.kaggle/kaggle.json`

### 🔄 Retry Logic with Exponential Backoff
- Up to 3 retry attempts (configurable)
- Exponential backoff: 1s, 2s, 4s between retries
- Graceful handling of temporary network issues

### 📦 Automatic ZIP Handling
- Detects downloaded ZIP archives
- Automatically extracts contents
- Cleans up ZIP files after extraction

### 🛡️ Robust Error Handling
- Returns expected file path even on failure (for pipeline continuity)
- Comprehensive logging of all operations
- Handles missing credentials gracefully

## Usage Examples

### Basic Usage
```python
from src.data.load_data import DataLoader

loader = DataLoader()

# Download with default parameters
file_path = loader.download_from_kaggle()
print(f"File downloaded to: {file_path}")
print(f"File exists: {file_path.exists()}")
```

### Custom Dataset and Parameters
```python
# Download a different dataset with custom retry count
file_path = loader.download_from_kaggle(
    dataset_name="your-username/your-dataset",
    filename="your_file.csv",
    max_retries=5
)
```

### Integration with Data Pipeline
```python
# The method integrates seamlessly with the existing pipeline
success = loader.run_full_pipeline(download_data=True)
if success:
    df = loader.load_csv_data()
    print(f"Loaded {len(df)} rows of data")
```

## Method Signature

```python
def download_from_kaggle(
    self, 
    dataset_name: str = "mashlyn/customer-churn-modeling", 
    filename: str = "Churn_Modelling.csv", 
    max_retries: int = 3
) -> Path:
    """
    Download dataset from Kaggle using the API with retry logic and credential handling.
    
    Args:
        dataset_name: Kaggle dataset identifier (default: mashlyn/customer-churn-modeling)
        filename: Expected filename to download (default: Churn_Modelling.csv)
        max_retries: Maximum number of retry attempts (default: 3)
        
    Returns:
        Path: Path to the downloaded file (even if download fails, returns expected path)
    """
```

## Credential Setup

### Option 1: Project Root kaggle.json
Place your `kaggle.json` file in the project root:
```json
{
    "username": "your-kaggle-username",
    "key": "your-api-key"
}
```

### Option 2: Environment Variables
```bash
export KAGGLE_USERNAME="your-kaggle-username"
export KAGGLE_KEY="your-api-key"
```

### Option 3: Default Kaggle Location
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## CI/CD Integration

The enhanced functionality includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that demonstrates how to configure Kaggle credentials in CI environments:

### GitHub Secrets Setup
1. Go to your repository Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API key

### Workflow Features
- Automatically sets up Kaggle credentials from secrets or project files
- Runs tests with proper credential configuration
- Includes integration tests for download functionality
- Supports multiple Python versions (3.8-3.11)

## Error Handling

The method handles various error scenarios gracefully:

### Missing Credentials
```
2025-06-17 16:39:20,953 - data.load_data - ERROR - No Kaggle credentials found. Please provide credentials via:
2025-06-17 16:39:20,953 - data.load_data - ERROR - 1. kaggle.json in project root
2025-06-17 16:39:20,953 - data.load_data - ERROR - 2. Environment variables KAGGLE_USERNAME and KAGGLE_KEY
2025-06-17 16:39:20,953 - data.load_data - ERROR - 3. ~/.kaggle/kaggle.json
```

### Network/API Failures
```
2025-06-17 16:39:41,425 - data.load_data - WARNING - Download attempt 1 failed: Permission 'datasets.get' was denied
2025-06-17 16:39:41,425 - data.load_data - INFO - Retrying in 1 seconds...
```

### Graceful Degradation
Even when downloads fail, the method returns the expected file path, allowing the rest of the pipeline to continue with existing data or alternative data sources.

## Testing

Use the provided test scripts to verify functionality:

```bash
# Comprehensive test
python test_kaggle_download.py

# Simple download test
python simple_download_test.py
```

## Dependencies

Ensure the following packages are installed:
```
kaggle==1.5.16
pandas>=1.3.0
numpy>=1.21.0
```

These are already included in `requirements.txt`.

## Troubleshooting

### Common Issues

1. **Permission Denied**: Verify your Kaggle API key has the correct permissions
2. **Dataset Not Found**: Check the dataset name format (username/dataset-name)
3. **Network Issues**: The retry logic will handle temporary network problems
4. **File Not Found After Download**: Check if the dataset contains the expected filename

### Debug Mode
Enable debug logging to see detailed operation logs:
```python
import logging
logging.getLogger('data.load_data').setLevel(logging.DEBUG)
```

## Implementation Details

### Helper Methods

- `_setup_kaggle_credentials()`: Handles credential configuration from multiple sources
- `_extract_and_cleanup_zip()`: Manages ZIP file extraction and cleanup

### Return Value
The method always returns a `Path` object pointing to the expected file location, ensuring pipeline continuity even when downloads fail.

### Integration Points
The enhanced method integrates with:
- `run_full_pipeline()`: Updated to handle the new return type
- `load_csv_data()`: Works seamlessly with downloaded files
- Existing validation and processing methods

## Future Enhancements

Potential improvements for future versions:
- Support for multiple file downloads
- Progress bars for large downloads
- Checksum verification
- Resume interrupted downloads
- Custom download directories per dataset