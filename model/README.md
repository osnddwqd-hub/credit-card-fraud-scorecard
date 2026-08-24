# Model artifact

Run `notebooks/03_Scorecard_Model.ipynb` to generate:

```text
model/scorecard_bundle.pkl
```

The bundle stores the training-time WOE bin boundaries, WOE mappings, selected
features, fitted logistic-regression model and score-scaling parameters. The
binary file is regenerated rather than committed so the complete training
process remains auditable.

