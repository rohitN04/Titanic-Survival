"""
Titanic Survival Prediction source package.

Exposes key functions for data processing, feature engineering,
model training, evaluation, and visualization.
"""

from .data_preprocessing import (
    load_data, explore_data_overview, analyze_missing_data,
    plot_eda, plot_correlation_heatmap
)
from .feature_engineering import engineer_features
from .model_training import build_pipeline, train_and_evaluate, NUMERIC_COLS, CATEGORICAL_COLS
from .evaluate import (
    plot_confusion_matrices, plot_roc_curves, plot_model_comparison,
    plot_overfitting_analysis, plot_dual_feature_importance,
    plot_feature_importance, plot_key_findings, get_feature_names
)
from .predict import predict

__all__ = [
    # data preprocessing
    'load_data',
    'explore_data_overview',
    'analyze_missing_data',
    'plot_eda',
    'plot_correlation_heatmap',
    # feature engineering
    'engineer_features',
    # model training
    'build_pipeline',
    'train_and_evaluate',
    'NUMERIC_COLS',
    'CATEGORICAL_COLS',
    # evaluation & visualization
    'plot_confusion_matrices',
    'plot_roc_curves',
    'plot_model_comparison',
    'plot_overfitting_analysis',
    'plot_dual_feature_importance',
    'plot_feature_importance',
    'plot_key_findings',
    'get_feature_names',
    # inference
    'predict',
]
