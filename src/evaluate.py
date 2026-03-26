"""
Evaluation and visualization module for the Titanic Survival Prediction project.

Provides functions to create confusion matrices, ROC curves,
model comparison tables, overfitting analysis, feature importance charts,
and key-findings summary plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


def plot_confusion_matrices(results, y_test):
    """Plot confusion matrices for all models side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

    if n == 1:
        axes = [axes]

    for idx, (name, metrics) in enumerate(results.items()):
        cm = confusion_matrix(y_test, metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        axes[idx].set_title(f'{name}\nAcc: {metrics["Test Accuracy"]:.4f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models on the same axes."""
    plt.figure(figsize=(8, 6))
    colors = ['darkorange', 'green', 'blue', 'red', 'purple']

    for idx, (name, metrics) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        auc_score = metrics['ROC-AUC']
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                 label=f'{name} (AUC = {auc_score:.3f})')

    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results):
    """Print a comparison table of all model metrics."""
    comparison = pd.DataFrame({
        name: {
            'Train Accuracy': f'{m["Train Accuracy"]:.4f}',
            'Test Accuracy': f'{m["Test Accuracy"]:.4f}',
            'Overfit Gap': f'{m["Train Accuracy"] - m["Test Accuracy"]:.4f}',
            'Precision': f'{m["Precision"]:.4f}',
            'Recall': f'{m["Recall"]:.4f}',
            'F1-Score': f'{m["F1-Score"]:.4f}',
            'ROC-AUC': f'{m["ROC-AUC"]:.4f}',
            '10-Fold CV': f'{m["CV Mean"]:.4f} ± {m["CV Std"]:.4f}'
        } for name, m in results.items()
    })
    print('\n=== Model Comparison Summary ===')
    print(comparison.to_string())
    return comparison


def plot_overfitting_analysis(results):
    """
    Create a grouped bar chart comparing Train / Test / CV accuracy
    to check for overfitting across all models.

    Parameters
    ----------
    results : dict
        Per-model metrics dictionary from train_and_evaluate().
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(results.keys())
    train_accs = [results[m]['Train Accuracy'] for m in models]
    test_accs = [results[m]['Test Accuracy'] for m in models]
    cv_means = [results[m]['CV Mean'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, train_accs, width, label='Train Accuracy', color='#3498db')
    ax.bar(x, test_accs, width, label='Test Accuracy', color='#2ecc71')
    ax.bar(x + width, cv_means, width, label='10-Fold CV Mean', color='#e74c3c')

    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Test vs CV Accuracy (Overfitting Check)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0.7, 1.0)

    for i, (tr, te, cv) in enumerate(zip(train_accs, test_accs, cv_means)):
        ax.text(i - width, tr + 0.005, f'{tr:.3f}', ha='center', fontsize=9)
        ax.text(i, te + 0.005, f'{te:.3f}', ha='center', fontsize=9)
        ax.text(i + width, cv + 0.005, f'{cv:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def get_feature_names(pipeline, numeric_cols, categorical_cols):
    """Extract feature names from a fitted pipeline's preprocessor."""
    preprocessor = pipeline.named_steps['preprocessor']
    cat_features = list(
        preprocessor.named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(categorical_cols)
    )
    return list(numeric_cols) + cat_features


def plot_dual_feature_importance(best_models, numeric_cols, categorical_cols):
    """
    Plot side-by-side feature importance for Logistic Regression
    (|coefficient| values) and Random Forest (Gini importance).

    Parameters
    ----------
    best_models : dict
        Mapping of model name → fitted Pipeline.
    numeric_cols : list[str]
    categorical_cols : list[str]
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    feature_names = get_feature_names(
        best_models['Logistic Regression'], numeric_cols, categorical_cols
    )

    # --- Logistic Regression ---
    lr_pipeline = best_models['Logistic Regression']
    lr_model = lr_pipeline.named_steps['classifier']
    importances_lr = np.abs(lr_model.coef_[0])
    feat_imp_lr = pd.DataFrame({'Feature': feature_names, 'Importance': importances_lr})
    feat_imp_lr = feat_imp_lr.sort_values('Importance', ascending=True)

    axes[0].barh(feat_imp_lr['Feature'], feat_imp_lr['Importance'], color='steelblue')
    axes[0].set_xlabel('|Coefficient|')
    axes[0].set_title('Feature Importance\n(Logistic Regression)',
                      fontsize=13, fontweight='bold')

    # --- Random Forest ---
    rf_pipeline = best_models['Random Forest']
    rf_model = rf_pipeline.named_steps['classifier']
    importances_rf = rf_model.feature_importances_
    feat_imp_rf = pd.DataFrame({'Feature': feature_names, 'Importance': importances_rf})
    feat_imp_rf = feat_imp_rf.sort_values('Importance', ascending=True)

    axes[1].barh(feat_imp_rf['Feature'], feat_imp_rf['Importance'], color='forestgreen')
    axes[1].set_xlabel('Gini Importance')
    axes[1].set_title('Feature Importance\n(Random Forest)',
                      fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(pipeline, feature_names, model_name='Model'):
    """Plot horizontal bar chart of feature importances (single model)."""
    model = pipeline.named_steps['classifier']

    if hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        xlabel = '|Coefficient|'
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        xlabel = 'Gini Importance'
    else:
        print(f'{model_name} does not support feature importance.')
        return

    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue')
    plt.xlabel(xlabel)
    plt.title(f'Feature Importance ({model_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_key_findings(df):
    """
    Create a 1×3 subplot showing key survival patterns:
        1. Survival by Sex
        2. Survival by Passenger Class
        3. Survival by Title

    Parameters
    ----------
    df : pd.DataFrame
        Titanic training dataframe (with 'Title' column engineered).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Key Findings: Survival Distribution by Category',
                 fontsize=16, fontweight='bold')

    # Survival by Sex
    sns.countplot(data=df, x='Sex', hue='Survived', palette='pastel', ax=axes[0])
    axes[0].set_title('Survival by Sex', fontsize=12)

    # Survival by Passenger Class
    sns.countplot(data=df, x='Pclass', hue='Survived', palette='pastel', ax=axes[1])
    axes[1].set_title('Survival by Passenger Class', fontsize=12)

    # Survival by Title
    sns.countplot(data=df, x='Title', hue='Survived', palette='pastel', ax=axes[2])
    axes[2].set_title('Survival by Title', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('key_findings.png', dpi=150, bbox_inches='tight')
    plt.show()
