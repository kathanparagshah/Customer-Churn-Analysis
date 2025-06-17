# Bank Customer Churn Dataset - Data Dictionary

## Dataset Overview

**Source**: Kaggle - Bank Customer Churn Dataset  
**Original Size**: ~10,000 records  
**Target Variable**: `Exited` (1 = Churned, 0 = Retained)  
**Data Type**: Tabular, structured data  
**Update Frequency**: Static dataset for analysis  

## Schema Definition

| Column Name | Data Type | Description | Valid Range/Values | Missing Values Policy |
|-------------|-----------|-------------|-------------------|----------------------|
| `RowNumber` | Integer | Sequential row identifier | 1 to N | Not allowed - Drop if missing |
| `CustomerId` | Integer | Unique customer identifier | Positive integers | Not allowed - Drop if missing |
| `Surname` | String | Customer last name | Text, 2-50 characters | **PII - Remove after processing** |
| `CreditScore` | Integer | Credit score | 350-850 | Impute with median by geography |
| `Geography` | String | Customer country | France, Spain, Germany | Impute with mode |
| `Gender` | String | Customer gender | Male, Female | Impute with mode |
| `Age` | Integer | Customer age in years | 18-95 | Impute with median |
| `Tenure` | Integer | Years as bank customer | 0-10 | Impute with median |
| `Balance` | Float | Account balance | >= 0 | Impute with 0 (assume closed account) |
| `NumOfProducts` | Integer | Number of bank products | 1-4 | Impute with mode |
| `HasCrCard` | Integer | Has credit card flag | 0, 1 | Impute with mode |
| `IsActiveMember` | Integer | Active member flag | 0, 1 | Impute with 0 (conservative) |
| `EstimatedSalary` | Float | Estimated annual salary | > 0 | Impute with median by geography |
| `Exited` | Integer | **TARGET**: Customer churned | 0, 1 | Not allowed - Drop if missing |

## Data Quality Rules

### Validation Checks

1. **Schema Validation**
   - All columns must be present
   - Data types must match specification
   - No duplicate CustomerIds allowed

2. **Range Validation**
   - CreditScore: 350 ≤ value ≤ 850
   - Age: 18 ≤ value ≤ 95
   - Tenure: 0 ≤ value ≤ 10
   - Balance: value ≥ 0
   - NumOfProducts: 1 ≤ value ≤ 4

3. **Categorical Validation**
   - Geography: Must be in [France, Spain, Germany]
   - Gender: Must be in [Male, Female]
   - Binary flags: Must be in [0, 1]

### Data Quality Thresholds

- **Missing Data**: < 5% per column acceptable
- **Outliers**: Values beyond 3 standard deviations flagged for review
- **Duplicates**: 0% tolerance for duplicate CustomerIds
- **Data Freshness**: Not applicable (static dataset)

## Privacy & Compliance

### PII (Personally Identifiable Information)

| Column | PII Level | Handling Policy |
|--------|-----------|----------------|
| `CustomerId` | Pseudo-identifier | Hash or anonymize for production |
| `Surname` | Direct PII | **REMOVE** after initial processing |
| `CreditScore` | Sensitive | Encrypt in production systems |
| `EstimatedSalary` | Sensitive | Encrypt in production systems |

### Data Governance

1. **Access Control**
   - Raw data: Data science team only
   - Processed data: Analytics team access
   - Models: Production deployment team

2. **Retention Policy**
   - Raw data: 2 years
   - Processed data: 5 years
   - Model artifacts: Indefinite (versioned)

3. **Audit Trail**
   - Log all data access and transformations
   - Version control for all processing scripts
   - Document model lineage and dependencies

## Feature Engineering Guidelines

### Derived Features

1. **Tenure-Balance Ratio**: `Balance / (Tenure + 1)`
2. **Age Groups**: Binned age categories
3. **Salary-to-Balance Ratio**: `EstimatedSalary / (Balance + 1)`
4. **Product Density**: Products per tenure year
5. **Credit Score Bins**: Low, Medium, High categories

### Encoding Strategies

- **Geography**: One-hot encoding (3 categories)
- **Gender**: Binary encoding (0/1)
- **Ordinal Features**: Label encoding where appropriate
- **Numeric Features**: StandardScaler for model training

## Data Lineage

```
Kaggle API → Raw CSV → Schema Validation → Missing Value Imputation → 
Feature Engineering → Scaling/Encoding → Train/Test Split → Model Training
```

## Quality Monitoring

### Automated Checks

1. **Daily Validation** (if data updates)
   - Schema compliance
   - Missing value rates
   - Outlier detection

2. **Weekly Reports**
   - Data quality dashboard
   - Distribution drift analysis
   - Feature correlation changes

3. **Monthly Reviews**
   - Data governance compliance
   - PII handling audit
   - Model performance vs. data quality

## Contact Information

**Data Steward**: Data Science Team  
**Privacy Officer**: Compliance Team  
**Technical Contact**: ML Engineering Team  

---

*Last Updated*: Project Initialization  
*Version*: 1.0  
*Next Review*: Monthly