#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning Prediction of Longitudinal Data Based on LPA Classification (Enhanced Version)
Adapted for Python 3.6+ / Local Environment / New scikit-learn
Core Improvement: Combine t1 and t2 into vectors, add temporal features
LPA Classes: 2 classes (Low Burnout / High Burnout)
"""
import matplotlib
matplotlib.use('Agg')  # 切换到Agg后端，完全避免tkinter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import warnings
import pickle
import sys
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures

# ==================== Path Configuration ====================
DATA_PATH = r"D:\临时文件\文档\文献\DATA_LPA.xlsx"
OUTPUT_BASE_DIR = r"D:\临时文件\文档\文献\厌学分析 数据图表"  # 你的指定路径
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Compatibility for GaussianMixture
try:
    from sklearn.mixture import GaussianMixture
    GMM_MODE = "new"
    print("✅ Using new version of GaussianMixture")
except ImportError:
    print("❌ GaussianMixture not found, please install scikit-learn")
    exit()

# Filter warnings
warnings.filterwarnings('ignore')

# Set plot style (no Chinese font needed for English labels)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'

print("=" * 80)
print("Machine Learning Prediction of Longitudinal Data Based on LPA Classification")
print(f"Data Source: {DATA_PATH}")
print(f"Output Directory: {OUTPUT_BASE_DIR}")
print("LPA Classes: 2 classes (Low Burnout / High Burnout)")
print("=" * 80)
print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version.split()[0]}")
print(f"scikit-learn GMM Mode: {GMM_MODE}")
print("=" * 80)

# ==================== 1. Data Loading ====================
print("\n【Step 1】Data Loading")
print("-" * 50)

try:
    df_raw = pd.read_excel(DATA_PATH, sheet_name="无空缺值数据")
    print(f"✅ Raw data loaded successfully: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    print(f"⚠️ Please check file path: {DATA_PATH}")
    exit()

print("\nFirst 3 rows of data:")
print(df_raw.head(3))

# ==================== 2. LPA Classification (2 Classes) ====================
print("\n【Step 2】LPA Classification Based on t3 Burnout Data (2 Classes)")
print("-" * 50)

# Check target column (BURNOUT_t3)
if 'BURNOUT_t3' not in df_raw.columns:
    burnout_t3_cols = [col for col in df_raw.columns if 'BURNOUT' in col.upper() and '_t3' in col]
    if len(burnout_t3_cols) > 0:
        target_col = burnout_t3_cols[0]
        print(f"Using alternative variable: {target_col}")
    else:
        print("❌ No t3 burnout variables found")
        exit()
else:
    target_col = 'BURNOUT_t3'

# Extract t3 burnout scores
X_cluster = df_raw[[target_col]].values
print(f"Clustering data dimension: {X_cluster.shape[0]} samples × 1 time point (t3)")

# Fixed to 2 classes
n_classes = 2
gmm = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=42)
gmm.fit(X_cluster)
labels = gmm.predict_proba(X_cluster).argmax(axis=1)

# Calculate means for class naming
means = [X_cluster[labels == i].mean() for i in range(n_classes)]
sorted_idx = np.argsort(means)
label_mapping = {sorted_idx[i]: i for i in range(n_classes)}
labels_sorted = np.array([label_mapping[label] for label in labels])

# Class names (English, no extra words)
class_names = ['Low Burnout Group', 'High Burnout Group']
df_raw['lpa_class'] = labels_sorted
df_raw['lpa_label'] = df_raw['lpa_class'].map({i: name for i, name in enumerate(class_names)})

print("\n📊 LPA Classification Results (Based on t3, 2 Classes):")
class_counts = df_raw['lpa_class'].value_counts().sort_index()
for i in range(n_classes):
    count = class_counts[i] if i in class_counts.index else 0
    pct = count / len(df_raw) * 100
    mean_val = X_cluster[df_raw['lpa_class'] == i].mean()
    print(f"\n【{class_names[i]}】:")
    print(f"  Sample Count: {count} ({pct:.2f}%)")
    print(f"  t3 Mean Score: {mean_val:.2f}")

# Visualization (separate plots)
# Plot 1: t3 Distribution
plt.figure(figsize=(8, 6))
colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
for i in range(n_classes):
    mask = df_raw['lpa_class'] == i
    plt.hist(X_cluster[mask], bins=20, alpha=0.6, color=colors[i], label=class_names[i])
plt.xlabel('t3 Burnout Score')
plt.ylabel('Frequency')
plt.title('t3 Burnout Score Distribution (2 Classes)')
plt.legend()
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '1_t3_distribution_2class.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ t3 Distribution plot saved: {image_path}")

# Plot 2: Class Mean Comparison
plt.figure(figsize=(8, 6))
means_by_class = [X_cluster[df_raw['lpa_class'] == i].mean() for i in range(n_classes)]
plt.bar(class_names, means_by_class, color=colors, alpha=0.7)
plt.ylabel('Mean Score')
plt.title('t3 Mean Score by Class (2 Classes)')
for i, v in enumerate(means_by_class):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '2_class_mean_comparison_2class.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Class Mean Comparison plot saved: {image_path}")

# ==================== 3. Vector Feature Construction ====================
print("\n【Step 3】Vector Feature Construction (Combine t1 and t2 into vectors)")
print("-" * 50)
print("Core Improvement: Combine t1 and t2 observations into vectors, preserve temporal structure")

# Find valid variables with t1 and t2 measurements
valid_vars = []
for col in df_raw.columns:
    if '_t1' in col:
        base = col.replace('_t1', '')
        if f"{base}_t2" in df_raw.columns:
            valid_vars.append(base)

# Exclude burnout variables
feature_vars = [v for v in valid_vars if 'BURNOUT' not in v.upper()]
print(f"\nVariables available for prediction: {len(feature_vars)}")
print(f"Variable list: {sorted(feature_vars)[:10]}...")

# Construct vector features
feature_df = pd.DataFrame()
feature_df['id'] = df_raw['S_ID_t1'] if 'S_ID_t1' in df_raw.columns else range(len(df_raw))

for var in sorted(feature_vars):
    t1 = f"{var}_t1"
    t2 = f"{var}_t2"

    if t1 in df_raw.columns and t2 in df_raw.columns:
        # 1. Original vector [t1, t2]
        feature_df[f'{var}_t1'] = df_raw[t1]
        feature_df[f'{var}_t2'] = df_raw[t2]

        # 2. Vector norm (Euclidean distance)
        feature_df[f'{var}_norm'] = np.sqrt(df_raw[t1] ** 2 + df_raw[t2] ** 2)

        # 3. Vector angle (relative to origin)
        feature_df[f'{var}_angle'] = np.arctan2(df_raw[t2], df_raw[t1] + 1e-10)

        # 4. Change features
        feature_df[f'{var}_change'] = df_raw[t2] - df_raw[t1]
        feature_df[f'{var}_change_rate'] = np.where(
            np.abs(df_raw[t1]) > 1e-10,
            (df_raw[t2] - df_raw[t1]) / np.abs(df_raw[t1]),
            0
        )

        # 5. Statistical features
        feature_df[f'{var}_mean'] = df_raw[[t1, t2]].mean(axis=1)
        feature_df[f'{var}_std'] = df_raw[[t1, t2]].std(axis=1).fillna(0)
        feature_df[f'{var}_max'] = df_raw[[t1, t2]].max(axis=1)
        feature_df[f'{var}_min'] = df_raw[[t1, t2]].min(axis=1)
        feature_df[f'{var}_range'] = feature_df[f'{var}_max'] - feature_df[f'{var}_min']

        # 6. Trend features
        feature_df[f'{var}_increasing'] = (df_raw[t2] > df_raw[t1]).astype(int)
        feature_df[f'{var}_stable'] = (np.abs(df_raw[t2] - df_raw[t1]) < 0.1).astype(int)

        # 7. Interaction features (with time)
        feature_df[f'{var}_t1_sq'] = df_raw[t1] ** 2
        feature_df[f'{var}_t2_sq'] = df_raw[t2] ** 2
        feature_df[f'{var}_t1_t2'] = df_raw[t1] * df_raw[t2]

print(f"\n📊 Feature Construction Complete:")
print(f"  - Total Features: {feature_df.shape[1] - 3}")
print(f"  - Sample Size: {len(feature_df)}")

# Handle missing values
missing_counts = feature_df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\n⚠️ Missing values found, imputing with median")
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        feature_df[col].fillna(feature_df[col].median(), inplace=True)

# ==================== 4. Feature Engineering Optimization ====================
print("\n【Step 4】Feature Engineering Optimization")
print("-" * 50)

# Define feature columns
feature_cols = [col for col in feature_df.columns
                if col not in ['id', 'lpa_class', 'lpa_label']]

X = feature_df[feature_cols].values
y = df_raw['lpa_class'].values

print(f"Original Feature Matrix: {X.shape}")

# 1. Add polynomial features (2nd order interaction)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_poly = poly.fit_transform(X)
print(f"After adding interaction terms: {X_poly.shape}")

# 2. Feature selection
selector = SelectKBest(f_classif, k=min(100, X_poly.shape[1]))
X_selected = selector.fit_transform(X_poly, y)
print(f"After feature selection: {X_selected.shape}")

# Final feature matrix
X_final = X_selected
selected_indices = selector.get_support(indices=True)

# Get feature names (compatible with new/old sklearn)
if hasattr(poly, 'get_feature_names_out'):
    selected_features = [poly.get_feature_names_out(feature_cols)[i] for i in selected_indices]
else:
    selected_features = [poly.get_feature_names(feature_cols)[i] for i in selected_indices]
print(f"\nFinal number of features used: {X_final.shape[1]}")

# ==================== Add Variable Correlation Heatmap ====================
print("\n【Step 4.1】Variable Correlation Analysis")
print("-" * 50)

# Select top 20 features for correlation analysis
top_20_features = selected_features[:20]
top_20_indices = [i for i, feat in enumerate(selected_features) if feat in top_20_features]
X_corr = X_final[:, top_20_indices]

# Create correlation matrix
corr_matrix = pd.DataFrame(X_corr, columns=top_20_features).corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap (Top 20 Features)')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '3_feature_correlation_heatmap.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Feature Correlation Heatmap saved: {image_path}")

# ==================== 5. Data Splitting ====================
print("\n【Step 5】Data Preparation (Balance classes with class_weight)")
print("-" * 50)

# Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining Set: {X_train.shape[0]} samples")
print(f"Test Set: {X_test.shape[0]} samples")

# Check class distribution
print("\nClass Distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for i, count in zip(unique, counts):
    print(f"  {class_names[i]}: {count} ({count / len(y_train) * 100:.2f}%)")

# Use original data (balance with class_weight)
X_train_resampled, y_train_resampled = X_train, y_train

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Data Preprocessing Complete")

# ==================== 6. Define Models ====================
print("\n【Step 6】Define Models")
print("-" * 50)

# Base models
base_models = {
    'Logistic Regression': LogisticRegression(
        solver='lbfgs',
        max_iter=2000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=3,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=5,
        random_state=42,
        objective='binary',
        verbose=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ),
    'MLP Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
}

print(f"Total models defined: {len(base_models)}")
for name in base_models.keys():
    print(f"  - {name}")

# ==================== 7. Model Training and Evaluation ====================
print("\n【Step 7】Model Training and Evaluation")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_model = None
best_f1 = 0
best_model_name = ""

# Store ROC curve data for all models
roc_data = {}

for name, model in base_models.items():
    print(f"\n{'=' * 60}")
    print(f"📊 Training Model: {name}")
    print(f"{'=' * 60}")

    try:
        # Cross validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled,
                                    cv=cv, scoring='f1_macro')

        # Train model
        model.fit(X_train_scaled, y_train_resampled)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        y_pred_train = model.predict(X_train_scaled)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision_macro = precision_score(y_test, y_pred, average='macro')
        test_recall_macro = recall_score(y_test, y_pred, average='macro')
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
        train_f1_macro = f1_score(y_train_resampled, y_pred_train, average='macro')

        # Per-class F1
        f1_per_class = f1_score(y_test, y_pred, average=None)

        # Save results
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'test_precision_macro': test_precision_macro,
            'test_recall_macro': test_recall_macro,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
            'train_f1_macro': train_f1_macro,
            'f1_per_class': f1_per_class,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }

        # Save ROC data
        if y_pred_proba is not None and n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

        print(f"\n📈 5-Fold Cross Validation:")
        print(f"  F1(macro): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        print(f"\n📈 Test Set Results:")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  Precision(macro): {test_precision_macro:.4f}")
        print(f"  Recall(macro): {test_recall_macro:.4f}")
        print(f"  F1(macro): {test_f1_macro:.4f}")
        print(f"  F1(weighted): {test_f1_weighted:.4f}")

        print(f"\n📊 Model Diagnosis:")
        print(f"  Training F1: {train_f1_macro:.4f}")
        print(f"  Test F1: {test_f1_macro:.4f}")
        print(f"  Overfitting Gap: {train_f1_macro - test_f1_macro:.4f}")

        # Per-class F1
        print(f"\n📊 Per-class F1 Scores:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  {class_names[i]}: {f1:.4f}")

        # Update best model
        if test_f1_macro > best_f1:
            best_f1 = test_f1_macro
            best_model = model
            best_model_name = name

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        results[name] = {'error': str(e)}

# ==================== 8. Ensemble Model ====================
print("\n【Step 8】Ensemble Model Creation")
print("-" * 50)

# Collect good models for ensemble
good_models = []
good_model_names = []
for name, res in results.items():
    if 'error' not in res and res['test_f1_macro'] > 0.5:
        good_models.append((name, res['model']))
        good_model_names.append(name)

if len(good_models) >= 2:
    print(f"Using {len(good_models)} models for ensemble: {good_model_names}")

    # Hard voting
    estimators = [(name, model) for name, model in good_models]
    voting_clf_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_clf_hard.fit(X_train_scaled, y_train_resampled)
    y_pred_voting_hard = voting_clf_hard.predict(X_test_scaled)
    f1_voting_hard = f1_score(y_test, y_pred_voting_hard, average='macro')

    # Soft voting (requires predict_proba)
    estimators_proba = [(name, model) for name, model in good_models
                        if hasattr(model, 'predict_proba')]
    if len(estimators_proba) >= 2:
        voting_clf_soft = VotingClassifier(estimators=estimators_proba, voting='soft')
        voting_clf_soft.fit(X_train_scaled, y_train_resampled)
        y_pred_voting_soft = voting_clf_soft.predict(X_test_scaled)
        f1_voting_soft = f1_score(y_test, y_pred_voting_soft, average='macro')

        # Save ensemble ROC data
        y_pred_proba_soft = voting_clf_soft.predict_proba(X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_soft[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba_soft[:, 1])
        roc_data['Soft Voting Ensemble'] = {'fpr': fpr, 'tpr': tpr, 'auc': auc}

        print(f"\n📊 Ensemble Model Results:")
        print(f"  Hard Voting F1: {f1_voting_hard:.4f}")
        print(f"  Soft Voting F1: {f1_voting_soft:.4f}")

        # Update best model
        if f1_voting_soft > best_f1:
            best_f1 = f1_voting_soft
            best_model = voting_clf_soft
            best_model_name = "Soft Voting Ensemble"
        elif f1_voting_hard > best_f1:
            best_f1 = f1_voting_hard
            best_model = voting_clf_hard
            best_model_name = "Hard Voting Ensemble"

# ==================== 9. Results Summary ====================
print("\n【Step 9】Results Summary")
print("-" * 50)

# Create results dataframe
results_summary = []
for name, res in results.items():
    if 'error' not in res:
        results_summary.append({
            'Model': name,
            'CV_F1': res['cv_mean'],
            'Test_Accuracy': res['test_accuracy'],
            'Precision': res['test_precision_macro'],
            'Recall': res['test_recall_macro'],
            'F1_Score': res['test_f1_macro'],
            'F1_Weighted': res['test_f1_weighted'],
            'Overfitting_Gap': res['train_f1_macro'] - res['test_f1_macro']
        })

results_df = pd.DataFrame(results_summary)
results_df = results_df.sort_values('F1_Score', ascending=False)

print("\n📊 Model Performance Comparison (Sorted by F1 Score):")
print("=" * 100)
print(results_df.round(4).to_string(index=False))
print("=" * 100)

# Random guess baseline
random_baseline = 1 / n_classes
print(f"\nRandom Guess Baseline: {random_baseline:.3f}")
print(f"Best Model Improvement Over Random: {(best_f1 - random_baseline) / random_baseline * 100:.1f}%")

# ==================== 10. Learning Curve (Separate Plot) ====================
print("\n【Step 10】Learning Curve Analysis")
print("-" * 50)

if best_model is not None:
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            best_model, X_train_scaled, y_train_resampled,
            cv=5, scoring='f1_macro', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training F1')
        plt.plot(train_sizes, test_mean, 'o-', color='red', label='Validation F1')
        plt.xlabel('Number of Training Samples')
        plt.ylabel('F1 Score')
        plt.title(f'{best_model_name} - Learning Curve (2 Classes)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        image_path = os.path.join(OUTPUT_BASE_DIR, '4_learning_curve_2class.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Learning Curve saved: {image_path}")
    except Exception as e:
        print(f"⚠️ Learning Curve generation failed: {e}")

# ==================== 11. Visualization (Separate Plots, All Models ROC) ====================
print("\n【Step 11】Generate Visualization Plots (2 Classes)")
print("-" * 50)

# Plot 1: Model Performance Comparison (Separate Plot)
plt.figure(figsize=(12, 8))
x = np.arange(len(results_df))
width = 0.15
metrics = ['CV_F1', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
    offset = (i - 2) * width
    plt.bar(x + offset, results_df[metric], width,
            label=metric, color=color, alpha=0.8)

plt.axhline(y=random_baseline, color='gray', linestyle='--', label='Random Guess')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Performance Comparison (2 Classes)')
plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.legend(loc='lower right', ncol=2)
plt.ylim([0, 1])
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '5_model_performance_comparison.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Model Performance Comparison plot saved: {image_path}")

# Plot 2: Confusion Matrix (Separate Plot)
plt.figure(figsize=(8, 6))
if best_model_name in results or best_model_name in ['Hard Voting Ensemble', 'Soft Voting Ensemble']:
    if best_model_name in results:
        cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    else:
        y_pred_best = best_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred_best)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{best_model_name} - Confusion Matrix (2 Classes)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    image_path = os.path.join(OUTPUT_BASE_DIR, '6_confusion_matrix.png')
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion Matrix plot saved: {image_path}")

# Plot 3: Per-class F1 (Separate Plot)
plt.figure(figsize=(8, 6))
if best_model_name in results and 'f1_per_class' in results[best_model_name]:
    f1_per_class = results[best_model_name]['f1_per_class']
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    plt.bar(class_names, f1_per_class, color=colors[:len(class_names)])
    plt.ylim([0, 1])
    plt.ylabel('F1 Score')
    plt.title(f'{best_model_name} - F1 Score by Class (2 Classes)')
    for i, v in enumerate(f1_per_class):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    image_path = os.path.join(OUTPUT_BASE_DIR, '7_per_class_f1.png')
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Per-class F1 plot saved: {image_path}")

# Plot 4: Overfitting Diagnosis (Separate Plot)
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Overfitting_Gap'], color='coral')
plt.axhline(y=0.1, color='red', linestyle='--', label='Warning Threshold (0.1)')
plt.xlabel('Model')
plt.ylabel('Overfitting Gap (Train F1 - Test F1)')
plt.title('Overfitting Diagnosis')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '8_overfitting_diagnosis.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Overfitting Diagnosis plot saved: {image_path}")

# Plot 5: All Models ROC Curve (Combined Plot)
plt.figure(figsize=(10, 8))
colors_roc = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
for i, (name, data) in enumerate(roc_data.items()):
    plt.plot(data['fpr'], data['tpr'], color=colors_roc[i],
             label=f'{name} (AUC = {data["auc"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for All Models (2 Classes)')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '9_all_models_roc.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Combined ROC Curve plot saved: {image_path}")

# Plot 6: Results Summary (Separate Plot)
plt.figure(figsize=(8, 6))
summary_text = f"Best Model: {best_model_name}\n"
summary_text += f"F1 Score: {best_f1:.4f}\n"
summary_text += f"Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}\n"
summary_text += f"Improvement Over Random: {(best_f1 - random_baseline) / random_baseline * 100:.1f}%\n"
summary_text += f"Number of Features: {X_final.shape[1]}\n"
summary_text += f"Training Samples: {X_train_scaled.shape[0]}\n"
summary_text += f"LPA Classes: 2"

plt.text(0.5, 0.5, summary_text,
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue"))
plt.axis('off')
plt.title('Analysis Summary (2 Classes)')
plt.tight_layout()
image_path = os.path.join(OUTPUT_BASE_DIR, '10_results_summary.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Results Summary plot saved: {image_path}")

# ==================== 12. Feature Importance Analysis (Separate Plot) ====================
print("\n【Step 12】Feature Importance Analysis")
print("-" * 50)

# Analyze tree-based model feature importance
importance_dict = {}
for name, res in results.items():
    if 'error' not in res and hasattr(res['model'], 'feature_importances_'):
        importances = res['model'].feature_importances_
        if len(importances) == len(selected_features):
            importance_dict[name] = pd.Series(importances, index=selected_features)
        else:
            print(f"⚠️ Feature importance dimension mismatch for {name}")

if importance_dict:
    try:
        importance_df = pd.DataFrame(importance_dict)
        importance_df['Mean_Importance'] = importance_df.mean(axis=1)
        importance_df = importance_df.sort_values('Mean_Importance', ascending=False)

        print("\n📊 Feature Importance Ranking (Top 20):")
        print(importance_df[['Mean_Importance']].head(20).round(4))

        # Visualization (Separate Plot)
        plt.figure(figsize=(12, 10))
        top_features = importance_df.head(20)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features['Mean_Importance'].values)
        plt.yticks(y_pos, top_features.index, fontsize=9)
        plt.xlabel('Mean Importance')
        plt.title('Feature Importance Ranking (Top 20) - 2 Classes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        image_path = os.path.join(OUTPUT_BASE_DIR, '11_feature_importance_2class.png')
        plt.savefig(image_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature Importance plot saved: {image_path}")

        # Save
        csv_path = os.path.join(OUTPUT_BASE_DIR, 'feature_importance_2class.csv')
        importance_df.to_csv(csv_path, encoding='utf-8-sig')
        print(f"✅ Feature Importance saved: {csv_path}")
    except Exception as e:
        print(f"⚠️ Feature Importance analysis failed: {e}")

# ==================== 13. Save Results ====================
print("\n【Step 13】Save Results (2 Classes)")
print("-" * 50)

# Save results summary
try:
    csv_path = os.path.join(OUTPUT_BASE_DIR, 'final_results_2class.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Results summary saved: {csv_path}")
except Exception as e:
    print(f"⚠️ Results summary save failed: {e}")

# Save best model
if best_model is not None:
    try:
        pkl_path = os.path.join(OUTPUT_BASE_DIR, 'best_model_2class.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✅ Best model saved: {pkl_path}")
    except Exception as e:
        print(f"⚠️ Best model save failed: {e}")

# Save preprocessing objects
try:
    # Scaler
    pkl_path = os.path.join(OUTPUT_BASE_DIR, 'scaler_2class.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {pkl_path}")

    # Polynomial features
    pkl_path = os.path.join(OUTPUT_BASE_DIR, 'poly_2class.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(poly, f)
    print(f"✅ Polynomial features saved: {pkl_path}")

    # Feature selector
    pkl_path = os.path.join(OUTPUT_BASE_DIR, 'selector_2class.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(selector, f)
    print(f"✅ Feature selector saved: {pkl_path}")

    # Feature names
    pkl_path = os.path.join(OUTPUT_BASE_DIR, 'feature_names_2class.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(selected_features, f)
    print(f"✅ Feature names saved: {pkl_path}")
except Exception as e:
    print(f"⚠️ Preprocessing objects save failed: {e}")

# Save predictions
try:
    if best_model_name in results and 'error' not in results[best_model_name]:
        y_pred_best = results[best_model_name]['y_pred']
    elif best_model_name in ['Hard Voting Ensemble', 'Soft Voting Ensemble']:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = None

    if y_pred_best is not None:
        csv_path = os.path.join(OUTPUT_BASE_DIR, 'predictions_2class.csv')
        prediction_df = pd.DataFrame({
            'True_Class': [class_names[i] for i in y_test],
            'Predicted_Class': [class_names[i] for i in y_pred_best],
            'True_Label': y_test,
            'Predicted_Label': y_pred_best
        })
        prediction_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Predictions saved: {csv_path}")
except Exception as e:
    print(f"⚠️ Predictions save failed: {e}")

# ==================== 14. Generate Report (English) ====================
print("\n【Step 14】Generate Complete Report (2 Classes)")
print("-" * 50)

try:
    txt_path = os.path.join(OUTPUT_BASE_DIR, 'analysis_report_2class.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Machine Learning Prediction of Longitudinal Data Based on LPA Classification\n")
        f.write(f"Data Source: {DATA_PATH}\n")
        f.write(f"Output Directory: {OUTPUT_BASE_DIR}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Report Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python Version: {sys.version.split()[0]}\n")
        f.write(f"scikit-learn GMM Mode: {GMM_MODE}\n")
        f.write(f"LPA Classes: 2 (Low Burnout / High Burnout)\n\n")

        f.write("1. Data Overview\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total Samples: {len(feature_df)}\n")
        f.write(f"Original Features: {len(feature_cols)}\n")
        f.write(f"Engineered Features: {X_final.shape[1]}\n")
        f.write(f"Training Set: {X_train_scaled.shape[0]} samples\n")
        f.write(f"Test Set: {X_test.shape[0]} samples\n\n")

        f.write("2. LPA Classification Results (Based on t3, 2 Classes)\n")
        f.write("-" * 50 + "\n")
        for i in range(n_classes):
            count = (df_raw['lpa_class'] == i).sum()
            pct = count / len(df_raw) * 100
            f.write(f"{class_names[i]}: {count} samples ({pct:.2f}%)\n")
        f.write("\n")

        f.write("3. Model Performance Comparison\n")
        f.write("-" * 50 + "\n")
        f.write(results_df.round(4).to_string())
        f.write("\n\n")

        f.write("4. Best Model\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}\n")
        f.write(f"Improvement Over Random Guess: {(best_f1 - random_baseline) / random_baseline * 100:.1f}%\n\n")

        f.write("5. Improvement Measures Summary\n")
        f.write("-" * 50 + "\n")
        f.write("1. Vector Features: Combine t1 and t2 into vectors, preserve temporal structure\n")
        f.write("2. Feature Engineering: Add vector norm, angle, interaction terms\n")
        f.write("3. Class Balance: Use class_weight instead of SMOTE (Python 3 compatible)\n")
        f.write("4. Model Ensemble: Voting ensemble of multiple models\n")
        f.write("5. Feature Selection: SelectKBest for optimal features\n")
        f.write("6. Polynomial Features: Add feature interaction terms\n")
        f.write("7. Parameter Optimization: Fix XGBoost/LightGBM binary classification parameters\n")
        f.write("8. LPA Optimization: Fixed to 2-class classification\n\n")

        f.write("6. Conclusions and Recommendations\n")
        f.write("-" * 50 + "\n")
        if best_f1 > 0.7:
            perf_level = "Excellent"
        elif best_f1 > 0.6:
            perf_level = "Good"
        elif best_f1 > 0.5:
            perf_level = "Moderate"
        elif best_f1 > random_baseline:
            perf_level = "Weakly Effective"
        else:
            perf_level = "Ineffective"

        f.write(f"1. Prediction Performance: {perf_level}\n")

        if best_f1 > 0.6:
            f.write("2. Model can be used for practical prediction, but professional judgment is recommended\n")
            f.write("3. Focus on top-ranked features in feature importance\n")
        elif best_f1 > 0.5:
            f.write("2. Model has certain predictive ability, use with caution\n")
            f.write("3. Recommend collecting more longitudinal data, add time points\n")
        else:
            f.write("2. Current model has limited predictive ability\n")
            f.write("3. Recommended improvements:\n")
            f.write("   - Collect more relevant predictor variables\n")
            f.write("   - Add more measurement time points (e.g., t0, t1, t2, t3)\n")
            f.write("   - Use deep learning models (e.g., LSTM) for time series\n")
            f.write("   - Consider non-linear and interaction effects\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✅ Analysis report saved: {txt_path}")
except Exception as e:
    print(f"⚠️ Report generation failed: {e}")

print("\n" + "=" * 80)
print("Analysis Complete! (2 Classes Version)")
print("=" * 80)

print(f"\n📊 Final Results Summary:")
print(f"  - Best Model: {best_model_name}")
print(f"  - F1 Score: {best_f1:.4f}")
print(f"  - Accuracy: {results_df.iloc[0]['Test_Accuracy']:.4f}")
print(f"  - Improvement Over Random: {(best_f1 - random_baseline) / random_baseline * 100:.1f}%")
print(f"  - LPA Classes: 2 (Low Burnout / High Burnout)")

print(f"\n📁 All output files saved to: {OUTPUT_BASE_DIR}")
# ==================== 15. SHAP Analysis (Logistic Regression, English) ====================
print("\n【Step 15】SHAP Analysis (Logistic Regression Model)")
print("-" * 50)

try:
    import shap
    print(f"✅ SHAP library version: {shap.__version__}")

    # Select Logistic Regression model
    best_model_shap = base_models['Logistic Regression']
    print(f"✅ Selected Logistic Regression model for SHAP analysis")

    # Initialize SHAP LinearExplainer
    explainer = shap.LinearExplainer(
        best_model_shap,
        X_train_scaled[:100],
        feature_perturbation="interventional"
    )
    print(f"✅ SHAP LinearExplainer initialized successfully")

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_scaled)
    print(f"✅ SHAP values calculated, shape: {shap_values.shape}")

    # Prepare feature names
    feature_names = selected_features
    assert len(feature_names) == shap_values.shape[1], \
        f"Feature name count ({len(feature_names)}) does not match SHAP value dimension ({shap_values.shape[1]})!"

    # 1. Mean Absolute SHAP Values (Bar Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Mean Absolute SHAP Values (Logistic Regression)", fontsize=14)
    plt.tight_layout()
    shap_bar_path = os.path.join(OUTPUT_BASE_DIR, 'shap_bar_plot_logreg.png')
    plt.savefig(shap_bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Mean Absolute SHAP Value Plot saved: {shap_bar_path}")

    # 2. Global SHAP Summary Plot (Beeswarm Plot)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, show=False)
    plt.title("Global SHAP Summary Plot (Logistic Regression)", fontsize=14)
    plt.tight_layout()
    shap_summary_path = os.path.join(OUTPUT_BASE_DIR, 'shap_summary_plot_logreg.png')
    plt.savefig(shap_summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Global SHAP Summary Plot saved: {shap_summary_path}")

    # 3. SHAP Dependence Plots (Top 10 Features)
    shap_sum = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_sum
    }).sort_values('shap_value', ascending=False)

    top_10_features = shap_importance['feature'].head(10).tolist()
    print(f"\n📊 Top 10 most influential features: {top_10_features}")

    for i, feature in enumerate(top_10_features):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X_test_scaled,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence Plot: {feature} (Logistic Regression)", fontsize=14)
        plt.tight_layout()
        dep_path = os.path.join(OUTPUT_BASE_DIR, f'shap_dep_{i+1}_{feature[:20]}.png')
        plt.savefig(dep_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP Dependence Plot {i+1}/10 saved: {dep_path}")

    # 4. Save SHAP results
    shap_importance_path = os.path.join(OUTPUT_BASE_DIR, 'shap_feature_importance_logreg.csv')
    shap_importance.to_csv(shap_importance_path, index=False, encoding='utf-8-sig')
    print(f"✅ SHAP feature importance saved: {shap_importance_path}")

    print(f"\n✅ SHAP analysis for Logistic Regression completed, all plots saved to: {OUTPUT_BASE_DIR}")

except ModuleNotFoundError:
    print("❌ SHAP library not installed, please run 'pip install shap -i https://pypi.tuna.tsinghua.edu.cn/simple'")
except AssertionError as e:
    print(f"❌ Feature name configuration error: {e}")
except Exception as e:
    print(f"❌ SHAP analysis failed: {e}")

print("\n" + "=" * 80)
print("SHAP Analysis Module for Logistic Regression Completed!")
print("=" * 80)