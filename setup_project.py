#!/usr/bin/env python3
"""
Bank Churn Analysis Project Setup Script

Automated setup script that initializes the complete project environment,
installs dependencies, creates necessary directories, downloads data,
and validates the setup.

Usage:
    python setup_project.py                    # Full setup
    python setup_project.py --skip-data        # Skip data download
    python setup_project.py --dev              # Development setup
    python setup_project.py --check            # Check existing setup

Author: Bank Churn Analysis Team
Date: 2024
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import time

# ANSI color codes for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


class ProjectSetup:
    """Main class for setting up the bank churn analysis project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.setup_log = []
        self.errors = []
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp and level."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        
        # Print with colors
        color = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "WARNING": Colors.YELLOW,
            "ERROR": Colors.RED
        }.get(level, Colors.WHITE)
        
        print(f"{color}{log_entry}{Colors.NC}")
        
        if level == "ERROR":
            self.errors.append(message)
    
    def run_command(self, command: List[str], description: str, 
                   check: bool = True, cwd: Optional[Path] = None) -> bool:
        """Run a shell command and log the result."""
        try:
            self.log(f"Running: {description}")
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.returncode == 0:
                self.log(f"✅ {description} completed successfully", "SUCCESS")
                return True
            else:
                self.log(f"❌ {description} failed: {result.stderr}", "ERROR")
                return False
                
        except subprocess.CalledProcessError as e:
            self.log(f"❌ {description} failed: {e.stderr}", "ERROR")
            return False
        except Exception as e:
            self.log(f"❌ {description} failed: {str(e)}", "ERROR")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        self.log("Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            self.log(f"❌ Python 3.8+ required, found {version.major}.{version.minor}", "ERROR")
            return False
        
        self.log(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible", "SUCCESS")
        return True
    
    def check_system_dependencies(self) -> bool:
        """Check if required system dependencies are available."""
        self.log("Checking system dependencies...")
        
        required_commands = ['git', 'curl']
        optional_commands = ['docker', 'docker-compose', 'make']
        
        all_good = True
        
        for cmd in required_commands:
            if shutil.which(cmd) is None:
                self.log(f"❌ Required command '{cmd}' not found", "ERROR")
                all_good = False
            else:
                self.log(f"✅ Found {cmd}", "SUCCESS")
        
        for cmd in optional_commands:
            if shutil.which(cmd) is None:
                self.log(f"⚠️  Optional command '{cmd}' not found", "WARNING")
            else:
                self.log(f"✅ Found {cmd}", "SUCCESS")
        
        return all_good
    
    def create_directory_structure(self) -> bool:
        """Create the project directory structure."""
        self.log("Creating project directory structure...")
        
        directories = [
            'data/raw',
            'data/interim', 
            'data/processed',
            'notebooks',
            'src/data',
            'src/features',
            'src/models',
            'src/tests',
            'src/visualization',
            'models',
            'reports/figures',
            'reports/test_results',
            'deployment',
            'deployment/monitoring',
            '.github/workflows'
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log(f"Created directory: {directory}")
            
            self.log("✅ Directory structure created successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to create directory structure: {str(e)}", "ERROR")
            return False
    
    def install_python_dependencies(self, dev: bool = False) -> bool:
        """Install Python dependencies."""
        self.log("Installing Python dependencies...")
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.log("❌ requirements.txt not found", "ERROR")
            return False
        
        # Install requirements
        success = self.run_command(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            "Installing project dependencies"
        )
        
        if not success:
            return False
        
        # Install development dependencies if requested
        if dev:
            dev_packages = [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'pytest-html>=3.1.0',
                'pytest-xdist>=3.0.0',
                'black>=22.0.0',
                'flake8>=5.0.0',
                'mypy>=1.0.0',
                'isort>=5.10.0',
                'bandit>=1.7.0',
                'safety>=2.0.0',
                'jupyter>=1.0.0',
                'jupyterlab>=3.0.0'
            ]
            
            success = self.run_command(
                [sys.executable, '-m', 'pip', 'install'] + dev_packages,
                "Installing development dependencies"
            )
        
        return success
    
    def setup_git_repository(self) -> bool:
        """Initialize git repository if not already initialized."""
        self.log("Setting up git repository...")
        
        git_dir = self.project_root / '.git'
        if git_dir.exists():
            self.log("✅ Git repository already initialized", "SUCCESS")
            return True
        
        # Initialize git repository
        success = self.run_command(
            ['git', 'init'],
            "Initializing git repository"
        )
        
        if not success:
            return False
        
        # Create .gitignore if it doesn't exist
        gitignore_file = self.project_root / '.gitignore'
        if not gitignore_file.exists():
            gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
PIPFILE.lock

# Virtual environments
venv/
env/
.venv/
.env/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/
.tox/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Data files
data/raw/*.csv
data/raw/*.parquet
data/interim/*.parquet
data/processed/*.parquet

# Model files
models/*.pkl
models/*.joblib
models/*.h5

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Environment variables
.env
.env.local
.env.*.local

# Reports
reports/test_results/
reports/figures/*.png
reports/figures/*.jpg
reports/figures/*.pdf

# Temporary files
tmp/
temp/
*.tmp
*.temp
""".strip()
            
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            
            self.log("Created .gitignore file")
        
        return True
    
    def download_sample_data(self) -> bool:
        """Download or create sample data for testing."""
        self.log("Setting up sample data...")
        
        # Create sample data if Kaggle data is not available
        sample_data_script = self.project_root / 'src' / 'data' / 'create_sample_data.py'
        
        sample_script_content = '''
#!/usr/bin/env python3
"""
Create sample data for testing the bank churn analysis pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data(n_samples=10000, output_path=None):
    """Create realistic sample bank customer data."""
    np.random.seed(42)
    
    # Generate sample data
    data = pd.DataFrame({
        'RowNumber': range(1, n_samples + 1),
        'CustomerId': range(15634602, 15634602 + n_samples),
        'Surname': [f'Customer_{i}' for i in range(n_samples)],
        'CreditScore': np.random.randint(350, 850, n_samples),
        'Geography': np.random.choice(['France', 'Spain', 'Germany'], n_samples, p=[0.5, 0.25, 0.25]),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 92, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 250000, n_samples),
        'NumOfProducts': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.4, 0.08, 0.02]),
        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'EstimatedSalary': np.random.uniform(11.58, 199992.48, n_samples),
    })
    
    # Create realistic churn patterns
    churn_prob = 0.2  # Base churn probability
    
    # Adjust churn probability based on features
    prob_adjustments = (
        (data['Age'] > 60) * 0.1 +  # Older customers more likely to churn
        (data['NumOfProducts'] == 1) * 0.15 +  # Single product customers
        (data['IsActiveMember'] == 0) * 0.2 +  # Inactive members
        (data['Balance'] == 0) * 0.1 +  # Zero balance
        (data['CreditScore'] < 500) * 0.1  # Low credit score
    )
    
    final_churn_prob = np.clip(churn_prob + prob_adjustments, 0, 0.8)
    data['Exited'] = np.random.binomial(1, final_churn_prob)
    
    # Add some missing values to make it realistic
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data.loc[missing_indices[:len(missing_indices)//2], 'CreditScore'] = np.nan
    data.loc[missing_indices[len(missing_indices)//2:], 'EstimatedSalary'] = np.nan
    
    if output_path:
        data.to_csv(output_path, index=False)
        print(f"Sample data saved to {output_path}")
    
    return data

if __name__ == '__main__':
    # Create sample data
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / 'data' / 'raw' / 'Churn_Modelling.csv'
    
    # Create directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate and save sample data
    sample_data = create_sample_data(output_path=output_file)
    
    print(f"Created sample dataset with {len(sample_data)} records")
    print(f"Churn rate: {sample_data['Exited'].mean():.2%}")
    print(f"Missing values: {sample_data.isnull().sum().sum()}")
'''
        
        # Write the sample data creation script
        with open(sample_data_script, 'w') as f:
            f.write(sample_script_content)
        
        # Run the script to create sample data
        success = self.run_command(
            [sys.executable, str(sample_data_script)],
            "Creating sample data"
        )
        
        return success
    
    def create_data_dictionary(self) -> bool:
        """Create a data dictionary file."""
        self.log("Creating data dictionary...")
        
        data_dict_content = """
# Bank Customer Churn Dataset - Data Dictionary

## Overview
This dataset contains information about bank customers and whether they churned (left the bank) or not.

## Features

| Column Name | Data Type | Description | Valid Range/Values | Missing Values |
|-------------|-----------|-------------|-------------------|----------------|
| RowNumber | Integer | Row index | 1 to N | No |
| CustomerId | Integer | Unique customer identifier | Positive integers | No |
| Surname | String | Customer surname | Text | No |
| CreditScore | Integer | Customer credit score | 350-850 | Possible |
| Geography | String | Customer country | France, Spain, Germany | No |
| Gender | String | Customer gender | Male, Female | No |
| Age | Integer | Customer age in years | 18-92 | No |
| Tenure | Integer | Number of years as bank customer | 0-10 | No |
| Balance | Float | Account balance | 0.0-250000.0 | No |
| NumOfProducts | Integer | Number of bank products used | 1-4 | No |
| HasCrCard | Integer | Has credit card (binary) | 0, 1 | No |
| IsActiveMember | Integer | Is active member (binary) | 0, 1 | No |
| EstimatedSalary | Float | Estimated annual salary | 11.58-199992.48 | Possible |
| Exited | Integer | Customer churned (target variable) | 0, 1 | No |

## Target Variable
- **Exited**: 1 if customer left the bank, 0 if customer stayed

## Data Quality Notes
- CreditScore may have missing values (~1% of records)
- EstimatedSalary may have missing values (~1% of records)
- All other fields should be complete

## Privacy and Compliance
- Customer names (Surname) are anonymized
- CustomerIds are anonymized identifiers
- No personally identifiable information (PII) is included
- Data complies with GDPR and banking privacy regulations

## Usage Guidelines
- Use for churn prediction modeling
- Suitable for classification algorithms
- Consider feature engineering for better model performance
- Handle missing values appropriately before modeling

## Data Source
- Synthetic dataset based on realistic banking scenarios
- Created for educational and research purposes
- Not actual customer data
""".strip()
        
        data_dict_file = self.project_root / 'data' / 'data_dictionary.md'
        
        try:
            with open(data_dict_file, 'w') as f:
                f.write(data_dict_content)
            
            self.log("✅ Data dictionary created successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to create data dictionary: {str(e)}", "ERROR")
            return False
    
    def validate_setup(self) -> bool:
        """Validate that the setup was successful."""
        self.log("Validating project setup...")
        
        validation_checks = [
            ("README.md exists", (self.project_root / 'README.md').exists()),
            ("requirements.txt exists", (self.project_root / 'requirements.txt').exists()),
            ("Source directory exists", (self.project_root / 'src').exists()),
            ("Data directory exists", (self.project_root / 'data').exists()),
            ("Models directory exists", (self.project_root / 'models').exists()),
            ("Reports directory exists", (self.project_root / 'reports').exists()),
            ("Deployment directory exists", (self.project_root / 'deployment').exists()),
            ("Test configuration exists", (self.project_root / 'pytest.ini').exists()),
            ("Test runner exists", (self.project_root / 'run_tests.py').exists()),
            ("Makefile exists", (self.project_root / 'Makefile').exists()),
        ]
        
        all_passed = True
        
        for check_name, check_result in validation_checks:
            if check_result:
                self.log(f"✅ {check_name}", "SUCCESS")
            else:
                self.log(f"❌ {check_name}", "ERROR")
                all_passed = False
        
        # Test import of key modules
        try:
            import pandas
            import numpy
            import sklearn
            self.log("✅ Core data science libraries importable", "SUCCESS")
        except ImportError as e:
            self.log(f"❌ Failed to import core libraries: {str(e)}", "ERROR")
            all_passed = False
        
        return all_passed
    
    def run_initial_tests(self) -> bool:
        """Run initial tests to verify setup."""
        self.log("Running initial tests...")
        
        # Check if test runner exists
        test_runner = self.project_root / 'run_tests.py'
        if not test_runner.exists():
            self.log("❌ Test runner not found", "ERROR")
            return False
        
        # Run dependency check
        success = self.run_command(
            [sys.executable, 'run_tests.py', '--check-deps'],
            "Checking test dependencies"
        )
        
        return success
    
    def generate_setup_report(self) -> None:
        """Generate a setup report."""
        report_file = self.project_root / 'setup_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("Bank Churn Analysis Project Setup Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Setup completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Setup Log:\n")
            f.write("-" * 20 + "\n")
            for log_entry in self.setup_log:
                f.write(f"{log_entry}\n")
            
            if self.errors:
                f.write("\nErrors Encountered:\n")
                f.write("-" * 20 + "\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
            
            f.write("\nNext Steps:\n")
            f.write("-" * 20 + "\n")
            f.write("1. Review the setup report\n")
            f.write("2. Run 'make test' to verify everything works\n")
            f.write("3. Run 'make data-pipeline' to process data\n")
            f.write("4. Run 'make train-models' to train models\n")
            f.write("5. Run 'make deploy' to deploy the API\n")
        
        self.log(f"Setup report generated: {report_file}")
    
    def print_summary(self) -> None:
        """Print setup summary."""
        print("\n" + "=" * 60)
        print(f"{Colors.BOLD}{Colors.BLUE}🚀 BANK CHURN ANALYSIS PROJECT SETUP COMPLETE{Colors.NC}")
        print("=" * 60)
        
        if not self.errors:
            print(f"{Colors.GREEN}✅ Setup completed successfully!{Colors.NC}")
            print("\n📋 Next steps:")
            print(f"  1. {Colors.CYAN}make test{Colors.NC} - Run tests to verify setup")
            print(f"  2. {Colors.CYAN}make data-pipeline{Colors.NC} - Process the data")
            print(f"  3. {Colors.CYAN}make train-models{Colors.NC} - Train ML models")
            print(f"  4. {Colors.CYAN}make deploy{Colors.NC} - Deploy the API")
            print(f"\n📚 Documentation:")
            print(f"  - README.md - Project overview")
            print(f"  - data/data_dictionary.md - Data documentation")
            print(f"  - reports/project_summary.md - Detailed project report")
        else:
            print(f"{Colors.RED}❌ Setup completed with {len(self.errors)} error(s){Colors.NC}")
            print("\n🔧 Please fix the following issues:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the project setup."""
    parser = argparse.ArgumentParser(
        description="Setup script for Bank Churn Analysis project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--skip-data', action='store_true', help='Skip data download')
    parser.add_argument('--dev', action='store_true', help='Install development dependencies')
    parser.add_argument('--check', action='store_true', help='Check existing setup')
    parser.add_argument('--no-git', action='store_true', help='Skip git repository setup')
    parser.add_argument('--no-tests', action='store_true', help='Skip initial tests')
    
    args = parser.parse_args()
    
    # Initialize setup
    project_root = Path(__file__).parent
    setup = ProjectSetup(project_root)
    
    print(f"{Colors.BOLD}{Colors.BLUE}🏦 Bank Churn Analysis Project Setup{Colors.NC}")
    print("=" * 40)
    
    try:
        # Check mode
        if args.check:
            setup.log("Running setup validation check...")
            if setup.validate_setup():
                setup.log("✅ Project setup is valid", "SUCCESS")
            else:
                setup.log("❌ Project setup has issues", "ERROR")
            return
        
        # Step 1: Check system requirements
        if not setup.check_python_version():
            sys.exit(1)
        
        if not setup.check_system_dependencies():
            setup.log("⚠️  Some system dependencies are missing, but continuing...", "WARNING")
        
        # Step 2: Create directory structure
        if not setup.create_directory_structure():
            sys.exit(1)
        
        # Step 3: Install Python dependencies
        if not setup.install_python_dependencies(dev=args.dev):
            sys.exit(1)
        
        # Step 4: Setup git repository
        if not args.no_git:
            setup.setup_git_repository()
        
        # Step 5: Setup data
        if not args.skip_data:
            setup.download_sample_data()
            setup.create_data_dictionary()
        
        # Step 6: Validate setup
        if not setup.validate_setup():
            setup.log("⚠️  Setup validation failed, but continuing...", "WARNING")
        
        # Step 7: Run initial tests
        if not args.no_tests:
            setup.run_initial_tests()
        
        # Generate report and summary
        setup.generate_setup_report()
        setup.print_summary()
        
    except KeyboardInterrupt:
        setup.log("\n⚠️  Setup interrupted by user", "WARNING")
        sys.exit(130)
    except Exception as e:
        setup.log(f"❌ Unexpected error during setup: {str(e)}", "ERROR")
        sys.exit(1)


if __name__ == '__main__':
    main()