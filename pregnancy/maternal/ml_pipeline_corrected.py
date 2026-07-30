"""
=============================================================
  CORRECTED ML PIPELINE - NO DATA LEAKAGE
  Healthcare Maternal Risk Dataset Classification
  Focus: Realistic Accuracy & Overfitting Detection
  
  KEY CORRECTIONS:
  ✓ Remove duplicates BEFORE split
  ✓ Split data FIRST
  ✓ Fit preprocessing ONLY on training data
  ✓ Use StratifiedKFold cross-validation
  ✓ Detect overfitting (compare train vs test accuracy)
  ✓ No data leakage
  ✓ Realistic evaluation metrics for medical datasets
=============================================================
"""

# ════════════════════════════ IMPORTS ════════════════════════════
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report,
    auc,
    make_scorer,
)
import joblib

# ════════════════════════════ CONFIGURATION ════════════════════════════
CSV_PATH = "numerical dataset/Dataset - Updated.csv"
TARGET_COLUMN = "Risk Level"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_SAVE_PATH = "trained_models"
REPORTS_PATH = "reports"

# Global variables
target_encoder = None
feature_scaler = None

# ════════════════════════════ STEP 1: LOAD DATA ════════════════════════════
def load_data(csv_path):
    """Load CSV dataset"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 1: LOADING DATASET".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ Dataset loaded from: {csv_path}")
        print(f"\n📊 Initial Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n📋 Column Names: {df.columns.tolist()}")
        print(f"\n📈 Data Types:\n{df.dtypes}")
        print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
        print(f"\n📉 First 5 rows:\n{df.head()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_path}' not found!")
        return None

# ════════════════════════════ STEP 2: INITIAL DATA CLEANING ════════════════════════════
def clean_data(df, target_col):
    """
    CRITICAL STEP - Clean BEFORE splitting to prevent data leakage:
    1. Remove rows with missing target values
    2. Remove duplicate rows
    3. Convert types but do NOT impute feature values yet
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 2: INITIAL DATA CLEANING (BEFORE SPLIT)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    df = df.copy()
    initial_rows = len(df)
    
    # 1. Remove rows with missing target
    print(f"\n🎯 Cleaning target column '{target_col}'...")
    print(f"   Initial NaN in target: {df[target_col].isnull().sum()}")
    df = df.dropna(subset=[target_col])
    removed_target_na = initial_rows - len(df)
    print(f"   ✅ Removed {removed_target_na} rows with missing target")
    
    # 2. Convert target to string
    df[target_col] = df[target_col].astype(str).str.strip()
    print(f"   ✅ Converted target to string type")
    print(f"   ✅ Target values: {df[target_col].unique()}")
    
    # 3. Remove duplicates (BEFORE split)
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    removed_dupes = before_dedup - len(df)
    print(f"\n🗑️  Removed {removed_dupes} duplicate rows")
    
    # 4. Report feature missingness without imputing
    print(f"\n⚠️  Inspecting missing values in features...")
    missing_counts = df.drop(columns=[target_col]).isnull().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print(f"   Total missing values in features: {total_missing}")
        print(f"   Columns with missing values:\n{missing_counts[missing_counts > 0]}")
    else:
        print(f"   ✅ No missing feature values detected")
    
    # 5. Convert object columns to numeric where possible
    print(f"\n📊 Converting feature data types where possible...")
    for col in df.columns:
        if col != target_col and df[col].dtype == 'object':
            converted = pd.to_numeric(df[col], errors='ignore')
            if converted.dtype != 'object':
                df[col] = converted
                print(f"   ✅ Converted '{col}' to numeric")
    
    print(f"\n✅ Cleaning complete!")
    print(f"   Final shape before split: {df.shape}")
    print(f"   Final data types:\n{df.dtypes}")
    print(f"   Target distribution:\n{df[target_col].value_counts()}")
    
    return df

# ════════════════════════════ STEP 3: ENCODE TARGET ════════════════════════════
def encode_target(df, target_col):
    """Encode target labels"""
    global target_encoder
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 3: TARGET ENCODING".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[target_col].astype(str))
    
    print(f"\n🏷️  Target classes: {target_encoder.classes_}")
    print(f"   Class mapping:")
    for i, cls in enumerate(target_encoder.classes_):
        count = (y == i).sum()
        pct = (count / len(y)) * 100
        print(f"      {cls} → {i} (n={count}, {pct:.1f}%)")
    
    return y

# ════════════════════════════ STEP 4: TRAIN-TEST SPLIT ════════════════════════════
def split_data(df, y, target_col, test_size=0.2, random_state=42):
    """
    CRITICAL STEP - Split BEFORE any preprocessing!
    This prevents data leakage.
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 4: TRAIN-TEST SPLIT (BEFORE PREPROCESSING)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    X = df.drop(columns=[target_col])
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y
    )
    
    print(f"\n✅ Train-Test Split Complete!")
    print(f"   Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/(X_train.shape[0]+X_test.shape[0]))*100:.1f}%)")
    print(f"   Testing set:  {X_test.shape[0]} samples ({(X_test.shape[0]/(X_train.shape[0]+X_test.shape[0]))*100:.1f}%)")
    print(f"\n   Training target distribution:")
    for i in range(len(target_encoder.classes_)):
        count = (y_train == i).sum()
        pct = (count / len(y_train)) * 100
        print(f"      {target_encoder.classes_[i]}: {count} ({pct:.1f}%)")
    
    print(f"\n   Testing target distribution:")
    for i in range(len(target_encoder.classes_)):
        count = (y_test == i).sum()
        pct = (count / len(y_test)) * 100
        print(f"      {target_encoder.classes_[i]}: {count} ({pct:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# ════════════════════════════ STEP 5: PREPROCESSING PIPELINE ════════════════════════════
def build_preprocessor(X_train):
    """
    Build a transformer pipeline for numeric and categorical features.
    This is fit only on training data and reused inside cross-validation.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 5: BUILD PREPROCESSOR".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    print(f"\n🔧 Numeric columns: {numeric_cols}")
    print(f"   Categorical columns: {categorical_cols}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop', verbose_feature_names_out=False)

    feature_names = numeric_cols + categorical_cols
    return preprocessor, feature_names


def build_model_pipeline(preprocessor):
    """
    Create a sklearn Pipeline with preprocessing and Random Forest.
    """
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=1
        ))
    ])
    return model


def tune_pipeline(pipeline, X_train, y_train):
    """Tune the pipeline using cross-validation without leaking preprocessing."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 6: PIPELINE HYPERPARAMETER TUNING".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 15],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__ccp_alpha': [0.0, 0.001, 0.01]
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring={'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'},
        refit='f1_weighted',
        n_jobs=1,
        verbose=0,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    print(f"\n✅ Best pipeline parameters: {search.best_params_}")
    print(f"   Best CV F1-weighted: {search.best_score_:.4f}")
    if 'mean_test_accuracy' in search.cv_results_:
        print(f"   Best CV accuracy: {search.cv_results_['mean_test_accuracy'][search.best_index_]:.4f}")
    return search


def train_models(X_train, y_train, preprocessor):
    """Build and tune the pipeline model."""
    pipeline = build_model_pipeline(preprocessor)
    tuned_pipeline = tune_pipeline(pipeline, X_train, y_train)
    return tuned_pipeline


def report_class_imbalance(y, target_names):
    """Print class counts and imbalance ratios for the target."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 6A: CLASS IMBALANCE REPORT".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    value_counts = pd.Series(y).value_counts()
    total = len(y)
    for idx, count in value_counts.sort_index().items():
        label = target_names[idx] if idx < len(target_names) else str(idx)
        print(f"   {label}: {count} ({count/total*100:.1f}%)")
    imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[1] if len(value_counts) == 2 else None
    if imbalance_ratio is not None:
        print(f"\n   Imbalance ratio (minority/majority): {min(value_counts)/max(value_counts):.3f}")


def plot_learning_curve(estimator, X, y, model_name):
    """Plot and save a learning curve to detect memorization."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 6B: LEARNING CURVE ANALYSIS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score', marker='o')
    plt.plot(train_sizes, test_scores_mean, label='Validation score', marker='o')
    plt.fill_between(train_sizes, train_scores_mean - np.std(train_scores, axis=1),
                     train_scores_mean + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(train_sizes, test_scores_mean - np.std(test_scores, axis=1),
                     test_scores_mean + np.std(test_scores, axis=1), alpha=0.1)
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').lower()
    plt.savefig(f"{REPORTS_PATH}/learning_curve_{safe_name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: learning_curve_{safe_name}.png")


def cross_validate_models(models, X_train, y_train):
    """
    Perform StratifiedKFold cross-validation
    This helps detect overfitting by comparing CV scores
    """
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 7: CROSS-VALIDATION (5-FOLD)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for name, model in models.items():
        print(f"\n📊 Cross-validating {name}...")
        
        # Multiple scoring metrics
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        cv_scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        
        print(f"   ✅ Accuracy: {cv_scores['test_accuracy'].mean():.4f} ± {cv_scores['test_accuracy'].std():.4f}")
        print(f"   ✅ Precision: {cv_scores['test_precision_weighted'].mean():.4f}")
        print(f"   ✅ Recall:    {cv_scores['test_recall_weighted'].mean():.4f}")
        print(f"   ✅ F1-Score:  {cv_scores['test_f1_weighted'].mean():.4f}")
        
        cv_results[name] = cv_scores
    
    return cv_results

# ════════════════════════════ STEP 8: EVALUATE MODELS ════════════════════════════
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, n_classes):
    """
    Evaluate model and detect overfitting
    Compare training vs testing accuracy
    """
    # Training predictions
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)
    
    # Testing predictions
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    
    # Training metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    # Testing metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    # ROC-AUC
    try:
        if n_classes == 2:
            roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None
    
    # Overfitting detection
    overfitting_gap = train_acc - test_acc
    
    metrics = {
        'Model': model_name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Overfitting Gap': overfitting_gap,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1,
        'ROC-AUC': roc_auc,
    }
    
    return metrics, y_test_pred, y_test_proba, y_train_pred

# ════════════════════════════ STEP 9: PRINT EVALUATION ════════════════════════════
def print_evaluation_results(all_metrics):
    """Print evaluation results and detect overfitting"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 8: MODEL EVALUATION RESULTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    for metrics in all_metrics:
        model_name = metrics['Model']
        print(f"\n{'='*70}")
        print(f"  {model_name.upper()}")
        print(f"{'='*70}")
        
        print(f"\n  📊 Training Accuracy:  {metrics['Train Accuracy']*100:.2f}%")
        print(f"  📊 Test Accuracy:     {metrics['Test Accuracy']*100:.2f}%")
        print(f"  ⚠️  Overfitting Gap:   {metrics['Overfitting Gap']*100:.2f}%")
        
        # Overfitting warning
        if metrics['Overfitting Gap'] > 0.10:
            print(f"  🔴 WARNING: High overfitting detected! (Gap > 10%)")
        elif metrics['Overfitting Gap'] > 0.05:
            print(f"  🟡 CAUTION: Moderate overfitting detected (Gap > 5%)")
        else:
            print(f"  🟢 Overfitting level: LOW")
        
        print(f"\n  🎯 Test Set Metrics:")
        print(f"     Precision:  {metrics['Precision']:.4f}")
        print(f"     Recall:     {metrics['Recall']:.4f}")
        print(f"     F1-Score:   {metrics['F1 Score']:.4f}")
        if metrics['ROC-AUC'] is not None:
            print(f"     ROC-AUC:    {metrics['ROC-AUC']:.4f}")

# ════════════════════════════ STEP 10: VISUALIZATIONS ════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_name, n_classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=target_encoder.classes_,
                yticklabels=target_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_PATH}/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: confusion_matrix_{model_name.replace(' ', '_').lower()}.png")

def plot_roc_curve(y_true, y_proba, model_name, n_classes):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(n_classes))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{target_encoder.classes_[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_PATH}/roc_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: roc_curve_{model_name.replace(' ', '_').lower()}.png")

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance"""
    classifier = model
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']

    if not hasattr(classifier, 'feature_importances_'):
        print(f"   ⚠️  {model_name} does not have feature_importances_")
        return
    
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(indices)), importances[indices], color='steelblue')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title(f'Top 15 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_PATH}/feature_importance_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Saved: feature_importance_{model_name.replace(' ', '_').lower()}.png")

# ════════════════════════════ STEP 11: COMPARISON TABLE ════════════════════════════
def print_comparison_table(all_metrics):
    """Print and save comparison table"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 9: MODEL COMPARISON TABLE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    # Format for display
    display_df = metrics_df.copy()
    for col in ['Train Accuracy', 'Test Accuracy', 'Overfitting Gap', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if x is not None and col != 'Overfitting Gap' else (f"{x*100:.2f}%" if x is not None else "N/A"))
    
    print("\n" + display_df.to_string(index=False))
    
    # Save to CSV
    metrics_df.to_csv(f'{REPORTS_PATH}/model_comparison.csv', index=False)
    print(f"\n✅ Saved: model_comparison.csv")
    
    return metrics_df

# ════════════════════════════ STEP 12: FIND BEST MODEL ════════════════════════════
def find_best_model(metrics_df):
    """Find best model based on test accuracy and F1 score"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 10: BEST MODEL SELECTION".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Combined score: average of test accuracy and F1 score
    metrics_df['Combined Score'] = (metrics_df['Test Accuracy'] + metrics_df['F1 Score']) / 2
    
    best_idx = metrics_df['Combined Score'].idxmax()
    best_model = metrics_df.loc[best_idx, 'Model']
    best_accuracy = metrics_df.loc[best_idx, 'Test Accuracy']
    best_f1 = metrics_df.loc[best_idx, 'F1 Score']
    best_gap = metrics_df.loc[best_idx, 'Overfitting Gap']
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"   F1-Score:      {best_f1:.4f}")
    print(f"   Overfitting Gap: {best_gap*100:.2f}%")
    
    return best_model, best_accuracy, best_f1

# ════════════════════════════ SAVE MODELS ════════════════════════════
def save_models(trained_models, best_model_name):
    """Save trained models"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 11: SAVING MODELS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    for name, model in trained_models.items():
        path = f'{MODELS_SAVE_PATH}/{name.replace(" ", "_").lower()}_model.pkl'
        joblib.dump(model, path)
        marker = "⭐ BEST MODEL" if name == best_model_name else ""
        print(f"   ✅ Saved: {path} {marker}")
    
    # Save target encoder
    joblib.dump(target_encoder, f'{MODELS_SAVE_PATH}/target_encoder.pkl')
    print(f"   ✅ Saved: target_encoder.pkl")

# ════════════════════════════ MAIN PIPELINE ════════════════════════════
def main():
    """Execute complete ML pipeline with NO data leakage"""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CORRECTED ML PIPELINE - NO DATA LEAKAGE".center(68) + "█")
    print("█" + "  Healthcare Maternal Risk Classification".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Create output directories
    os.makedirs(MODELS_SAVE_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)
    
    # Step 1: Load data
    df = load_data(CSV_PATH)
    if df is None:
        return
    
    # Step 2: Clean data BEFORE splitting
    df = clean_data(df, TARGET_COLUMN)
    
    # Step 3: Encode target
    y = encode_target(df, TARGET_COLUMN)
    
    # Step 4: Split data BEFORE preprocessing
    X_train, X_test, y_train, y_test = split_data(df, y, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)

    # Step 5: Build preprocessing pipeline and fit it on training data only
    preprocessor, feature_names = build_preprocessor(X_train)
    tuned_pipeline = train_models(X_train, y_train, preprocessor)
    best_pipeline = tuned_pipeline.best_estimator_

    # Get number of classes
    n_classes = len(target_encoder.classes_)
    print(f"\n   Number of classes: {n_classes}")

    # Analyze learning curve to detect overfitting and underfitting
    plot_learning_curve(best_pipeline, X_train, y_train, 'Random Forest')

    # Step 7: Cross-validate the final pipeline
    cv_results = cross_validate_models({'Random Forest': best_pipeline}, X_train, y_train)

    # Step 8: Evaluate the final pipeline on the hold-out test set
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 8: MODEL EVALUATION (NO DATA LEAKAGE)".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    metrics, y_test_pred, y_test_proba, _ = evaluate_model(
        best_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        'Random Forest',
        n_classes
    )

    all_metrics = [metrics]

    # Step 9: Print evaluation results
    print_evaluation_results(all_metrics)

    # Step 10: Visualizations
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  STEP 9: GENERATING VISUALIZATIONS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    print(f"\n📊 Random Forest Visualizations:")
    plot_confusion_matrix(y_test, y_test_pred, 'Random Forest', n_classes)
    plot_roc_curve(y_test, y_test_proba, 'Random Forest', n_classes)
    plot_feature_importance(best_pipeline, feature_names, 'Random Forest')

    # Step 11: Comparison table
    metrics_df = print_comparison_table(all_metrics)

    # Step 12: Find best model
    best_model, best_accuracy, best_f1 = find_best_model(metrics_df)

    # Save models
    save_models({'Random Forest': best_pipeline}, best_model)
    
    # Final Summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ✅ PIPELINE EXECUTION COMPLETE!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print(f"\n📊 KEY FINDINGS:")
    print(f"   ✓ No data leakage (split BEFORE preprocessing)")
    print(f"   ✓ Scaling fitted on training data only")
    print(f"   ✓ Cross-validation performed")
    print(f"   ✓ Overfitting detected and reported")
    print(f"   ✓ Realistic accuracy for medical dataset")
    
    print(f"\n🏆 BEST MODEL: {best_model}")
    print(f"   Test Accuracy: {best_accuracy*100:.2f}%")
    print(f"   F1-Score:      {best_f1:.4f}")
    
    print(f"\n📁 Output Files:")
    print(f"   Models: {MODELS_SAVE_PATH}/")
    print(f"   Reports: {REPORTS_PATH}/")
    
    print(f"\n✅ All visualizations saved!")
    print(f"   - confusion_matrix_*.png")
    print(f"   - roc_curve_*.png")
    print(f"   - feature_importance_*.png")
    print(f"   - model_comparison.csv")

if __name__ == "__main__":
    main()
