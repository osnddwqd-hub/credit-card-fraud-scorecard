# Dataset

This project uses the **Credit Card Fraud Detection** dataset published on
Kaggle by the Machine Learning Group of ULB.

- Kaggle dataset: `mlg-ulb/creditcardfraud`
- File name: `creditcard.csv`
- Shape: 284,807 transactions × 31 columns
- Target: `Class` (`0` normal, `1` fraud)
- Fraud cases: approximately 492 (0.17%)

The raw dataset is not committed to GitHub. Place it at:

```text
data/raw/creditcard.csv
```

With KaggleHub installed, it can also be downloaded in Python:

```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print(path)
```

Copy `creditcard.csv` from the returned folder to `data/raw/` before running
the notebooks. The generated model and processed outputs are reproducible from
the raw file.

