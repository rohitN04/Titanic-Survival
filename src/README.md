# src/ — Titanic Survival Prediction Module

This directory contains the modular Python source code for the Titanic Survival Prediction project. Each file handles a specific responsibility in the ML pipeline, ensuring clean separation of concerns and reusability.

## Module Overview

| File | Responsibility |
|------|---------------|
| `data_preprocessing.py` | Data loading (Kaggle), EDA, missing value analysis & visualization |
| `feature_engineering.py` | Row-local feature creation (FamilySize, IsAlone, Title, FarePerPerson, FareGroup, AgeGroup, TicketFreq) |
| `model_training.py` | sklearn Pipeline construction (imputation + scaling + encoding), GridSearchCV training, cross-validation |
| `evaluate.py` | Evaluation visualizations — confusion matrices, ROC curves, overfitting analysis, feature importance, key findings |
| `predict.py` | Standalone inference script — loads saved model and predicts on new data |
| `main.py` | End-to-end pipeline orchestrator — runs the full workflow from data loading to final predictions |
| `__init__.py` | Package exports for notebook/external imports |

## How to Run

### Full Pipeline
```bash
cd /path/to/project
python -m src.main
```

This will:
1. Load the Titanic dataset from Kaggle
2. Perform exploratory data analysis with visualizations
3. Split data (80/20 stratified) **before** any preprocessing
4. Apply row-local feature engineering to train, validation, and test sets independently
5. Train 3 models (Logistic Regression, Decision Tree, Random Forest) using sklearn Pipelines
6. Tune hyperparameters with GridSearchCV (5-fold CV)
7. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC, and 10-fold CV
8. Generate all evaluation plots → saved to `src/plots/`
9. Predict on the truly unseen Kaggle test set → saved to `src/predictions/`
10. Save the best model to `src/model.pkl`

### Inference Only
```bash
python -m src.predict --model src/model.pkl --input test.csv --output submission.csv
```

### Import as Package (e.g., from notebook)
```python
from src import load_data, engineer_features, build_pipeline, train_and_evaluate
from src import plot_confusion_matrices, plot_roc_curves, plot_overfitting_analysis
```

## Output Structure

```
src/
├── plots/                        # Generated visualizations
│   ├── eda_plots.png             # 2×2 EDA grid (survival, sex, age, fare)
│   ├── missing_data_analysis.png # Missing data bar chart
│   ├── correlation_heatmap.png   # Feature correlation heatmap
│   ├── confusion_matrices.png    # Side-by-side confusion matrices
│   ├── roc_curves_comparison.png # ROC curves for all models
│   ├── overfitting_analysis.png  # Train vs Test vs CV accuracy
│   ├── feature_importance.png    # LR coefficients + RF Gini importance
│   └── key_findings.png          # Survival by sex, class, title
├── predictions/
│   └── titanic_submission.csv    # 418 predictions on unseen test set
└── model.pkl                     # Saved best sklearn Pipeline
```

## Data Leakage Prevention

All statistical preprocessing (imputation, scaling, encoding) is encapsulated inside **sklearn Pipelines** that are fitted **only on training data**. Feature engineering uses only row-local operations (each row's values are self-contained), making it safe to apply independently to each split.

## Dependencies

See `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
kagglehub>=0.2.0
joblib>=1.2.0
```
