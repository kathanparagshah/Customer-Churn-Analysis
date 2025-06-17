#!/usr/bin/env python3
"""
Setup script for Bank Customer Churn Analysis project.

This package provides a comprehensive machine learning solution for predicting
customer churn in banking, featuring automated data pipelines, model explainability,
and deployment-ready APIs.

Author: Bank Churn Analysis Team
Date: 2024
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="bank-churn-analysis",
    version="0.2.0",
    author="Bank Churn Analysis Team",
    author_email="team@bankchurn.com",
    description="Comprehensive ML solution for predicting customer churn in banking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kathanparagshah/Customer-Churn-Analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "churn-predict=src.models.train_churn:main",
            "churn-segment=src.models.segment:main",
            "churn-explain=src.models.explain:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning churn-prediction banking customer-analytics",
    project_urls={
        "Bug Reports": "https://github.com/kathanparagshah/Customer-Churn-Analysis/issues",
        "Source": "https://github.com/kathanparagshah/Customer-Churn-Analysis",
        "Documentation": "https://github.com/kathanparagshah/Customer-Churn-Analysis/blob/main/README.md",
    },
)