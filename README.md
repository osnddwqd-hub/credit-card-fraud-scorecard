# Credit Card Fraud Scorecard

An interpretable transaction-level fraud-risk project built with Weight of
Evidence (WOE), Information Value (IV), logistic regression and a 0–1000 risk
score. The final output assigns each transaction a fraud probability, risk
score, risk tier and recommended action.

## Project Highlights

- Processes 284,807 transactions with a fraud rate of about 0.17%.
- Uses training-only quantile binning and Laplace-smoothed WOE.
- Ranks features by IV and fits an interpretable logistic-regression model.
- Converts log odds to a business-friendly score where a lower score means
  higher fraud risk.
- Produces a deployable scoring bundle and transaction decision list.

Results from the completed analysis:

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9807** |
| KS statistic | **0.9364** |
| Score range observed | **94–865** |
| Median score | **≈699** |
| Transactions below 500 | **≈5.02%** |
| Fraud captured below 500 | **≈96.43%** |

## Business Problem

Fraud detection is highly imbalanced: fraudulent transactions represent only
about 0.17% of the data. Accuracy alone is therefore misleading. This project
focuses on ranking risk and translating model output into operational actions.

| Score | Risk level | Action |
|---:|---|---|
| 0–300 | High Risk | Block Transaction |
| 301–500 | Medium Risk | Manual Review |
| 501–1000 | Low Risk | Approve |

The thresholds are derived from score-band fraud rates and are demonstration
rules rather than production banking policy.

## Dataset

The project uses Kaggle's ULB Credit Card Fraud Detection dataset.

| Item | Description |
|---|---|
| Rows | 284,807 transactions |
| Columns | 31 |
| Features | `Time`, `V1`–`V28`, `Amount` |
| Target | `Class`: 0 normal, 1 fraud |
| Fraud cases | Approximately 492 |

`V1`–`V28` are PCA-transformed anonymous variables. The raw dataset is excluded
from GitHub; see [`data/README.md`](data/README.md).

## Methodology

### 1. Exploratory analysis

The first notebook checks data quality, class imbalance, amount distribution
and time patterns. Because of extreme imbalance, evaluation emphasizes ROC-AUC,
KS and fraud capture rather than overall accuracy.

### 2. WOE and IV

Continuous variables are divided into quantile bins using the training sample.
For each bin:

```text
WOE = ln(distribution of normal transactions / distribution of fraud transactions)
```

Laplace smoothing prevents infinite WOE values when a bin contains no fraud.
Information Value summarizes the separation contributed by each feature.

The selected variables in the completed analysis were:

```text
V14, V12, V3, V4, V11, V10, V17, V2, V16, V27
```

### 3. Logistic scorecard

A class-weighted logistic-regression model is trained on WOE-transformed
features. The hold-out evaluation produced:

- ROC-AUC: 0.9807
- KS: 0.9364

### 4. Score scaling

The project uses:

| Parameter | Value |
|---|---:|
| Base Score | 400 |
| PDO | 35 |
| Base good-to-bad odds | 100 |
| Score limits | 0–1000 |

Higher scores represent lower estimated fraud probability. The fitted model,
WOE definitions and score parameters are stored together in
`model/scorecard_bundle.pkl` after Notebook 03 is run.

## Score-band Findings

| Score band | Transactions | Fraud | Fraud rate |
|---:|---:|---:|---:|
| 0–200 | 60 | 47 | 78.33% |
| 201–300 | 29 | 3 | 10.34% |
| 301–400 | 136 | 2 | 1.47% |
| 401–500 | 936 | 2 | 0.21% |
| 501–600 | 3,223 | 1 | 0.03% |

The sharp decline in fraud rate as score increases shows that the score
successfully ranks transactions by risk. In the completed experiment, reviewing
or blocking scores below 500 covered about 5.02% of transactions while capturing
about 96.43% of fraud cases in the evaluated sample.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── score_band_analysis.csv
│   └── risk_strategy_summary.csv
├── figures/
├── model/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Scorecard_Model.ipynb
│   └── 04_Risk_Scoring.ipynb
└── src/
    └── scorecard.py
```

## How to Run

```bash
git clone https://github.com/osnddwqd-hub/credit-card-fraud-scorecard.git
cd credit-card-fraud-scorecard
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Download `creditcard.csv` as described in `data/README.md`, then run the
notebooks in order:

```text
01_EDA.ipynb
    ↓
02_Feature_Engineering.ipynb
    ↓
03_Scorecard_Model.ipynb
    ↓
04_Risk_Scoring.ipynb
```

Notebook 03 creates the model bundle. Notebook 04 demonstrates scoring without
requiring a separate new CSV by sampling demonstration transactions from the
original dataset.

## Outputs

After all notebooks are run, the project produces:

- fraud-distribution and evaluation figures;
- IV feature ranking;
- ROC and KS curves;
- score distribution and score-band analysis;
- `model/scorecard_bundle.pkl`;
- `data/scored_transactions_demo.csv`;
- `data/risk_strategy_summary.csv`.

## Limitations

- The dataset contains PCA-anonymized variables, limiting business-level feature
  interpretation.
- Random stratified validation does not reproduce chronological concept drift.
- Score thresholds require cost-sensitive validation before production use.
- The project does not model investigator capacity, transaction value lost or
  customer friction caused by false positives.
- Reported metrics should be reproduced in the target environment before being
  used in a formal report.

## Future Improvements

- add precision–recall AUC and cost-based threshold optimization;
- compare the scorecard with tree-based models;
- use temporal validation and stability monitoring;
- add probability calibration;
- expose the scorer through a small batch pipeline or API;
- monitor population stability index after deployment.

## Author

Mathematics master's student building practical projects in data analysis,
statistical modeling and business decision support.

