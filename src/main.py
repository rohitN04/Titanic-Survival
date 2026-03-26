"""
Main pipeline for the Titanic Survival Prediction project.

Orchestrates the full ML workflow:
    1. Load & explore data
    2. Train/test split (before any preprocessing)
    3. Row-local feature engineering (after split)
    4. Train models via sklearn Pipelines (no data leakage)
    5. Evaluate with multiple metrics and visualizations
    6. Generate final predictions on truly unseen Kaggle test set
    7. Save best model to disk

Usage:
    python -m src.main
"""

import sys
import os
import warnings
import shutil
import glob

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import set_config

# Local module imports (works whether run as `python -m src.main` or directly)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_preprocessing import load_data, explore_data_overview, analyze_missing_data, plot_eda, plot_correlation_heatmap
from feature_engineering import engineer_features
from model_training import build_pipeline, train_and_evaluate, NUMERIC_COLS, CATEGORICAL_COLS
from evaluate import (
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison,
    plot_overfitting_analysis, plot_dual_feature_importance,
    plot_key_findings
)

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def main():
    print("=" * 60)
    print("  Titanic Survival Prediction — ML Pipeline")
    print("=" * 60)

    # --- Setup output directories ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_plots_dir = os.path.join(base_dir, "plots")
    output_pred_dir = os.path.join(base_dir, "predictions")
    os.makedirs(output_plots_dir, exist_ok=True)
    os.makedirs(output_pred_dir, exist_ok=True)
    print(f"\n📁 Output directories: plots/ and predictions/ inside src/\n")

    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print("\nLoading data from Kaggle...")
    df, test_df = load_data()
    print(f"   Training set shape: {df.shape}")
    print(f"   Test set shape:     {test_df.shape}\n")

    # ================================================================
    # 2. DATA EXPLORATION & ANALYSIS
    # ================================================================
    print("\nExploring data...")
    explore_data_overview(df)
    print()
    missing_report = analyze_missing_data(df)
    print(missing_report)
    print()
    plot_eda(df)

    # ================================================================
    # 3. TRAIN/TEST SPLIT — BEFORE any preprocessing (prevents leakage)
    # ================================================================
    print("\nSplitting data into 80/20 train/validation sets (stratified)...")
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Survived']
    )
    print(f"   Train size: {train_df.shape[0]}  |  Validation size: {val_df.shape[0]}")

    # ================================================================
    # 4. FEATURE ENGINEERING — applied AFTER split (row-local only)
    # ================================================================
    print("\nApplying row-local feature engineering...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)
    print("   Features added: FamilySize, IsAlone, Title, FarePerPerson, FareGroup, AgeGroup, TicketFreq")

    # Correlation heatmap (on training data only to avoid leakage)
    plot_correlation_heatmap(train_df)

    # ================================================================
    # 5. PREPARE FEATURE MATRICES
    # ================================================================
    features = list(NUMERIC_COLS) + list(CATEGORICAL_COLS)
    X_train = train_df[features]
    y_train = train_df['Survived']
    X_val = val_df[features]
    y_val = val_df['Survived']

    # ================================================================
    # 6. MODEL TRAINING & HYPERPARAMETER TUNING
    # ================================================================
    print("\nTraining models with GridSearchCV...")
    models_and_params = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'classifier__C': [0.01, 0.1, 1, 10]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'classifier__max_depth': [3, 5, 7, 10, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 5, 10],
                'classifier__min_samples_split': [2, 5]
            }
        }
    }

    results, best_models = train_and_evaluate(
        models_and_params, X_train, X_val, y_train, y_val,
        NUMERIC_COLS, CATEGORICAL_COLS
    )

    # ================================================================
    # 7. EVALUATION VISUALIZATIONS
    # ================================================================
    print("\nGenerating evaluation visualizations...")

    # 7.1 Confusion Matrices
    plot_confusion_matrices(results, y_val)

    # 7.2 ROC Curves
    plot_roc_curves(results, y_val)

    # 7.3 Model Comparison Table
    comparison = plot_model_comparison(results)

    # 7.4 Overfitting Analysis
    plot_overfitting_analysis(results)

    # 7.5 Feature Importance (LR + RF side-by-side)
    plot_dual_feature_importance(best_models, NUMERIC_COLS, CATEGORICAL_COLS)

    # 7.6 Key Findings
    plot_key_findings(train_df)

    # ================================================================
    # 8. FINAL PREDICTIONS ON TRULY UNSEEN DATA
    # ================================================================
    print("\nGenerating final predictions on unseen Kaggle test set...")

    # Select best model dynamically
    best_model_name = max(results, key=lambda k: results[k]['Test Accuracy'])
    final_model = best_models[best_model_name]

    print(f"   Best Model:           {best_model_name}")
    print(f"   Validation Accuracy:  {results[best_model_name]['Test Accuracy']:.4f}")
    print(f"   ROC-AUC:              {results[best_model_name]['ROC-AUC']:.4f}")
    print(f"   F1-Score:             {results[best_model_name]['F1-Score']:.4f}")

    # Generate predictions
    test_features = test_df[features]
    final_predictions = final_model.predict(test_features)

    # Create submission CSV
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': final_predictions
    })
    output_csv = 'titanic_submission.csv'
    submission.to_csv(output_csv, index=False)
    print(f"\n   Submission file: {output_csv} ({len(submission)} predictions)")
    print(f"   Predicted survival rate: {final_predictions.mean()*100:.1f}%")
    print(submission.head(10).to_string())

    # ================================================================
    # 9. SAVE MODEL TO DISK
    # ================================================================
    model_path = os.path.join(base_dir, 'model.pkl')
    joblib.dump(final_model, model_path)
    print(f"\nBest model saved to: {model_path}")

    # ================================================================
    # 10. ORGANIZE OUTPUTS INTO FOLDERS
    # ================================================================
    print("\nOrganizing outputs into type folders...")

    # Move plots (PNGs generated in CWD) to src/plots/
    cwd = os.getcwd()
    moved_plots = 0
    for file in glob.glob(os.path.join(cwd, "*.png")):
        filename = os.path.basename(file)
        dest = os.path.join(output_plots_dir, filename)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(file, dest)
        moved_plots += 1

    # Move submission CSV to src/predictions/
    csv_path = os.path.join(cwd, output_csv)
    dest_csv = os.path.join(output_pred_dir, output_csv)
    if os.path.exists(csv_path):
        if os.path.exists(dest_csv):
            os.remove(dest_csv)
        shutil.move(csv_path, dest_csv)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("  ✅ Pipeline Complete!")
    print("=" * 60)
    print(f"   → {moved_plots} plots saved to: {os.path.abspath(output_plots_dir)}/")
    print(f"   → Predictions saved to: {os.path.abspath(dest_csv)}")
    print(f"   → Model saved to: {os.path.abspath(model_path)}")
    print(f"   → Best Model: {best_model_name} "
          f"(Acc={results[best_model_name]['Test Accuracy']:.4f}, "
          f"AUC={results[best_model_name]['ROC-AUC']:.4f})")


if __name__ == "__main__":
    main()
