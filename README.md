# Credit Risk Customer Segmentation using K-Means

## Overview
This project applies unsupervised machine learning techniques to segment borrowers based on their credit behaviour using Lending Club loan data.

The objective is to identify distinct borrower profiles that can support risk-based decision-making, portfolio monitoring, and targeted credit strategies.

---

## Dataset
- Source: Lending Club Loan Dataset
- Observations: ~2.2 million loans
- Features used:
  - Debt-to-Income Ratio (DTI)
  - Credit Utilisation
  - Delinquencies (last 2 years)
  - Credit Inquiries (credit hunger)
  - Employment Length
  - Average FICO Score

---

## Methodology

### 1. Data Cleaning
- Handled missing values using median imputation and filtering
- Converted categorical variables:
  - `emp_length` → numeric
  - `loan_status` → simplified into:
    - Good
    - Bad
    - Delinquent
    - Ongoing
- Treated invalid values (e.g., DTI > 100, utilisation > 100)

---

### 2. Feature Engineering
- Created:
  - `avg_fico` = mean of FICO range
- Renamed:
  - `inq_last_6mths` → credit_hunger
  - `revol_util` → credit_utilisation

---

### 3. Handling Skewness
Highly skewed variables were log-transformed:
- `credit_hunger`
- `delinq_2yrs`

---

### 4. Scaling
- StandardScaler applied before clustering

---

### 5. Clustering
- Algorithm: MiniBatch K-Means
- Optimal clusters determined using:
  - Elbow Method
  - Silhouette Score

**Optimal k = 4**

---

## Results

### Cluster Summary

| Cluster | Description |
|--------|------------|
| 0 | Credit-stressed borrowers with high activity and delinquency |
| 1 | Young / thin credit profile with high leverage |
| 2 | Experienced but highly leveraged borrowers |
| 3 | Prime borrowers with strong credit profiles |

---

### Key Insights

- High FICO borrowers exhibit significantly lower default rates
- Credit utilisation and delinquency are strong drivers of risk
- Credit inquiries ("credit hunger") indicate elevated risk behaviour
- Borrower risk is driven by **combined behaviour**, not a single metric

---

### Validation

Cluster-wise loan performance:

| Cluster | Bad Rate |
|--------|--------|
| 0 | 16% |
| 1 | 13% |
| 2 | 12% |
| 3 | 6% |

---

## Visualisations
- Distribution plots
- Boxplots (outlier detection)
- Correlation heatmap
- Pairplots
- Elbow curve

---

## Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## Conclusion
The analysis demonstrates that borrower segmentation using clustering can uncover meaningful patterns in credit behaviour. These segments can support more granular risk assessment and targeted lending strategies.

---

## Future Improvements
- Apply PCA for dimensionality reduction
- Compare with hierarchical clustering
- Build supervised models within each cluster
- Incorporate additional behavioural features

---

## Author
Salini Sankarankutty Menon
MSc Banking & Finance, King’s College London
