"""
Chapter 4: In-Depth Model Evaluation, Error Analysis, Feature Importance & Kaggle Leakage Audit
Author: Hoang Xuan Phi Long (Team Leader - Group 01)
Course: Machine Learning - Ho Chi Minh City University of Industry and Trade (HUIT)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Ensure root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)
import ckd_preprocessing

SEED = 42
np.random.seed(SEED)


def setup_tufte_style():
    """Configure Matplotlib to follow Edward Tufte's high data-ink ratio principles."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (11, 5.5)
    plt.rcParams['font.size'] = 10.5
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.25
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['grid.linestyle'] = '--'


def load_all_splits():
    """Load raw train/test splits, labels, and split_manifest for patient traceability."""
    proc_dir = os.path.join(BASE_DIR, 'data', 'processed')
    X_train_raw = pd.read_csv(os.path.join(proc_dir, 'X_train_raw.csv'))
    y_train = pd.read_csv(os.path.join(proc_dir, 'y_train.csv'))['classification'].astype(int)
    X_test_raw = pd.read_csv(os.path.join(proc_dir, 'X_test_raw.csv'))
    y_test = pd.read_csv(os.path.join(proc_dir, 'y_test.csv'))['classification'].astype(int)
    manifest = pd.read_csv(os.path.join(proc_dir, 'split_manifest.csv'))
    return X_train_raw, y_train, X_test_raw, y_test, manifest


def bootstrap_ci(y_true, y_pred, y_prob, n_boot=1000, seed=SEED):
    """Compute 95% bootstrap confidence intervals for key clinical metrics using fast vectorized NumPy."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    idx = rng.randint(0, n, size=(n_boot, n))
    yt = y_true_arr[idx]
    yp = y_pred_arr[idx]

    tp = np.sum((yt == 1) & (yp == 1), axis=1).astype(float)
    tn = np.sum((yt == 0) & (yp == 0), axis=1).astype(float)
    fp = np.sum((yt == 0) & (yp == 1), axis=1).astype(float)
    fn = np.sum((yt == 1) & (yp == 0), axis=1).astype(float)

    recalls = np.divide(tp, tp + fn, out=np.ones_like(tp), where=(tp + fn) > 0)
    specs = np.divide(tn, tn + fp, out=np.ones_like(tn), where=(tn + fp) > 0)
    accs = (tp + tn) / n
    f1s = np.divide(2 * tp, 2 * tp + fp + fn, out=np.ones_like(tp), where=(2 * tp + fp + fn) > 0)

    aucs = []
    for b in range(min(n_boot, 300)):
        yt_b = yt[b]
        if 0 < np.sum(yt_b) < n:
            aucs.append(roc_auc_score(yt_b, y_prob_arr[idx[b]]))

    def ci_bounds(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        'recall_ci': ci_bounds(recalls),
        'spec_ci': ci_bounds(specs),
        'acc_ci': ci_bounds(accs),
        'f1_ci': ci_bounds(f1s),
        'auc_ci': ci_bounds(aucs),
    }


def build_and_fit_models(X_train, y_train):
    """Fit all 6 baseline pipelines and 3 Chapter-3 GridSearchCV optimal tuned pipelines on X_train."""
    baselines = {
        'Logistic Regression (Baseline)': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        'K-Nearest Neighbors (Baseline)': KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
        'SVM (Baseline)': SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED),
        'Decision Tree (Baseline)': DecisionTreeClassifier(criterion='gini', random_state=SEED),
        'Random Forest (Baseline)': RandomForestClassifier(n_estimators=100, criterion='gini', random_state=SEED),
        'XGBoost (Baseline)': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='logloss', random_state=SEED, n_jobs=1),
    }

    fitted_baselines = {}
    for name, clf in baselines.items():
        pipe = Pipeline([
            ('preprocessor', ckd_preprocessing.make_preprocessor()),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_baselines[name] = pipe

    # Optimal hyperparameters obtained from Chapter 3 GridSearchCV (scoring='recall', 5-Fold Stratified CV on 280 Train samples)
    tuned_specs = {
        'XGBoost (Tuned)': XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3, subsample=1.0,
            eval_metric='logloss', random_state=SEED, n_jobs=1
        ),
        'Random Forest (Tuned)': RandomForestClassifier(
            n_estimators=50, max_depth=None, min_samples_split=2, criterion='gini',
            random_state=SEED, n_jobs=1
        ),
        'SVM (Tuned)': SVC(
            C=1.0, gamma='scale', kernel='linear', probability=True, random_state=SEED
        )
    }

    fitted_tuned = {}
    cv_scores = {
        'XGBoost (Tuned)': 0.9771428571428572,
        'Random Forest (Tuned)': 0.9885714285714287,
        'SVM (Tuned)': 0.9885714285714287
    }
    best_params = {
        'XGBoost (Tuned)': {'classifier__n_estimators': 100, 'classifier__learning_rate': 0.05, 'classifier__max_depth': 3, 'classifier__subsample': 1.0},
        'Random Forest (Tuned)': {'classifier__n_estimators': 50, 'classifier__max_depth': None, 'classifier__min_samples_split': 2, 'classifier__criterion': 'gini'},
        'SVM (Tuned)': {'classifier__C': 1.0, 'classifier__gamma': 'scale', 'classifier__kernel': 'linear'},
    }
    for name, clf in tuned_specs.items():
        pipe = Pipeline([
            ('preprocessor', ckd_preprocessing.make_preprocessor()),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_tuned[name] = pipe

    return fitted_baselines, fitted_tuned, cv_scores, best_params


def evaluate_comprehensive_test_set(fitted_baselines, fitted_tuned, X_test, y_test, tables_dir):
    """Section 4.1: Comprehensive evaluation on independent test set (n=80) with 95% Bootstrap CI."""
    all_models = {}
    for k, v in fitted_tuned.items():
        all_models[k] = ('Tuned', v)
    for k, v in fitted_baselines.items():
        all_models[k] = ('Baseline', v)

    rows = []
    predictions = {}
    probabilities = {}

    for name, (group, model) in all_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        predictions[name] = y_pred
        probabilities[name] = y_prob

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_pred, beta=2.0, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ll = log_loss(y_test, y_prob)

        ci = bootstrap_ci(y_test, y_pred, y_prob)

        rows.append({
            'Model': name,
            'Stage': group,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'Accuracy': acc,
            'Accuracy_95CI': f"[{ci['acc_ci'][0]:.4f}, {ci['acc_ci'][1]:.4f}]",
            'Precision': prec,
            'Recall': rec,
            'Recall_95CI': f"[{ci['recall_ci'][0]:.4f}, {ci['recall_ci'][1]:.4f}]",
            'Specificity': spec,
            'Specificity_95CI': f"[{ci['spec_ci'][0]:.4f}, {ci['spec_ci'][1]:.4f}]",
            'F1_Score': f1,
            'F1_95CI': f"[{ci['f1_ci'][0]:.4f}, {ci['f1_ci'][1]:.4f}]",
            'F2_Score': f2,
            'ROC_AUC': roc_auc,
            'ROC_AUC_95CI': f"[{ci['auc_ci'][0]:.4f}, {ci['auc_ci'][1]:.4f}]",
            'PR_AUC': pr_auc,
            'Brier_Score': brier,
            'Log_Loss': ll
        })

    df_test = pd.DataFrame(rows)
    df_test.to_csv(os.path.join(tables_dir, 'ch4_test_comprehensive_metrics.csv'), index=False)
    return df_test, predictions, probabilities


def evaluate_repeated_cv_stability(fitted_tuned, fitted_baselines, X_train, y_train, tables_dir):
    """Evaluate 5x5 Repeated Stratified K-Fold CV (25 folds) to measure statistical stability."""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    fold_data = []
    for tr_idx, va_idx in rskf.split(X_train, y_train):
        prep = ckd_preprocessing.make_preprocessor()
        X_tr = prep.fit_transform(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        X_va = prep.transform(X_train.iloc[va_idx])
        fold_data.append((X_tr, y_train.iloc[tr_idx].to_numpy(), X_va, y_train.iloc[va_idx].to_numpy()))

    models_to_check = {
        'XGBoost (Tuned)': fitted_tuned['XGBoost (Tuned)'],
        'Random Forest (Tuned)': fitted_tuned['Random Forest (Tuned)'],
        'SVM (Tuned)': fitted_tuned['SVM (Tuned)'],
        'Logistic Regression (Baseline)': fitted_baselines['Logistic Regression (Baseline)'],
        'Decision Tree (Baseline)': fitted_baselines['Decision Tree (Baseline)'],
        'K-Nearest Neighbors (Baseline)': fitted_baselines['K-Nearest Neighbors (Baseline)'],
    }

    rows = []
    for name, pipe in models_to_check.items():
        recs, aucs, f1s, accs = [], [], [], []
        base_clf = pipe.named_steps['classifier']
        for X_tr, y_tr, X_va, y_va in fold_data:
            clf = clone(base_clf)
            clf.fit(X_tr, y_tr)
            yp = clf.predict(X_va)
            ypr = clf.predict_proba(X_va)[:, 1]
            recs.append(recall_score(y_va, yp))
            aucs.append(roc_auc_score(y_va, ypr))
            f1s.append(f1_score(y_va, yp))
            accs.append(accuracy_score(y_va, yp))
        rows.append({
            'Model': name,
            'Folds': 25,
            'Recall_Mean': float(np.mean(recs)),
            'Recall_Std': float(np.std(recs)),
            'Recall_Min': float(np.min(recs)),
            'ROC_AUC_Mean': float(np.mean(aucs)),
            'ROC_AUC_Std': float(np.std(aucs)),
            'F1_Mean': float(np.mean(f1s)),
            'F1_Std': float(np.std(f1s)),
            'Accuracy_Mean': float(np.mean(accs)),
            'Accuracy_Std': float(np.std(accs)),
        })

    df_stab = pd.DataFrame(rows)
    df_stab.to_csv(os.path.join(tables_dir, 'ch4_repeated_cv_stability.csv'), index=False)
    return df_stab


def analyze_thresholds(probabilities, y_test, tables_dir, figures_dir):
    """Section 4.3.2: Classification Threshold Tuning across tau in [0.05, 0.95]."""
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    tuned_names = ['XGBoost (Tuned)', 'Random Forest (Tuned)', 'SVM (Tuned)']

    rows = []
    for name in tuned_names:
        y_prob = probabilities[name]
        for tau in thresholds:
            y_pred_tau = (y_prob >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_tau, labels=[0, 1]).ravel()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = f1_score(y_test, y_pred_tau, zero_division=0)
            f2 = fbeta_score(y_test, y_pred_tau, beta=2.0, zero_division=0)
            youden_j = rec + spec - 1.0
            rows.append({
                'Model': name,
                'Threshold': tau,
                'TP': int(tp),
                'TN': int(tn),
                'FP': int(fp),
                'FN': int(fn),
                'Recall': rec,
                'Specificity': spec,
                'Precision': prec,
                'F1_Score': f1,
                'F2_Score': f2,
                'Youden_J': youden_j
            })

    df_thresh = pd.DataFrame(rows)
    df_thresh.to_csv(os.path.join(tables_dir, 'ch4_threshold_tuning.csv'), index=False)

    # Fine-grained plot across tau in [0.05, 0.95]
    tau_grid = np.linspace(0.05, 0.95, 91)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for idx, name in enumerate(tuned_names):
        ax = axes[idx]
        y_prob = probabilities[name]
        recs, specs, precs, f2s = [], [], [], []
        for t in tau_grid:
            yp = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yp, labels=[0, 1]).ravel()
            recs.append(tp / (tp + fn))
            specs.append(tn / (tn + fp))
            precs.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
            f2s.append(fbeta_score(y_test, yp, beta=2.0, zero_division=0))

        ax.plot(tau_grid, recs, label='Recall (Sensitivity)', color='#d62728', lw=2.2)
        ax.plot(tau_grid, specs, label='Specificity', color='#1f77b4', lw=2.0, linestyle='--')
        ax.plot(tau_grid, f2s, label='F2-Score (Clinical)', color='#2ca02c', lw=1.8, linestyle='-.')
        ax.axvline(0.50, color='#666666', linestyle=':', lw=1.2, label='Default (0.50)')
        ax.axvline(0.35, color='#ff7f0e', linestyle='--', lw=1.2, label='Screening (0.35)')
        ax.set_title(name, fontsize=11, pad=8)
        ax.set_xlabel('Decision Threshold (tau)')
        if idx == 0:
            ax.set_ylabel('Metric Score')
        ax.set_ylim(0.75, 1.03)
        sns.despine(ax=ax, top=True, right=True)
        if idx == 2:
            ax.legend(loc='lower left', frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_threshold_tuning_curves.png'), dpi=300)
    plt.close()

    return df_thresh


def perform_error_analysis(
    fitted_baselines, fitted_tuned, predictions, probabilities,
    X_train_raw, y_train, X_test_raw, y_test, manifest, tables_dir, figures_dir
):
    """Section 4.4: Deep patient-level Error & Borderline Case Analysis on Test & CV OOF."""
    test_manifest = manifest[manifest['split'] == 'test'].sort_values('position').reset_index(drop=True)
    train_manifest = manifest[manifest['split'] == 'train'].sort_values('position').reset_index(drop=True)

    # Identify any Test sample misclassified by ANY of the 9 models OR with borderline probability (0.10 <= p <= 0.90)
    all_model_names = list(fitted_tuned.keys()) + list(fitted_baselines.keys())
    error_records = []

    for pos in range(len(X_test_raw)):
        true_y = int(y_test.iloc[pos])
        row_raw = X_test_raw.iloc[pos]
        n_missing = int(row_raw.isna().sum())
        missing_cols = [c for c in ckd_preprocessing.FEATURES if pd.isna(row_raw[c])]

        misclassified_by = []
        for mname in all_model_names:
            if int(predictions[mname][pos]) != true_y:
                misclassified_by.append(mname)

        p_xgb = float(probabilities['XGBoost (Tuned)'][pos])
        p_rf = float(probabilities['Random Forest (Tuned)'][pos])
        p_svm = float(probabilities['SVM (Tuned)'][pos])
        p_lr = float(probabilities['Logistic Regression (Baseline)'][pos])
        p_dt = float(probabilities['Decision Tree (Baseline)'][pos])
        p_knn = float(probabilities['K-Nearest Neighbors (Baseline)'][pos])

        is_borderline = any(0.05 <= p <= 0.95 for p in [p_xgb, p_rf, p_svm])

        if len(misclassified_by) > 0 or is_borderline:
            error_records.append({
                'Split': 'Test',
                'Test_Position': pos,
                'Original_ID': int(test_manifest.loc[pos, 'id']),
                'Source_Row': int(test_manifest.loc[pos, 'source_row']),
                'True_Label': true_y,
                'True_Class': 'CKD (1)' if true_y == 1 else 'Not CKD (0)',
                'Missing_Count_24': n_missing,
                'Missing_Percent': round(n_missing / 24.0 * 100, 1),
                'Missing_Features': ', '.join(missing_cols) if missing_cols else 'None',
                'Prob_XGBoost_Tuned': round(p_xgb, 4),
                'Prob_RF_Tuned': round(p_rf, 4),
                'Prob_SVM_Tuned': round(p_svm, 4),
                'Prob_LogReg_Base': round(p_lr, 4),
                'Prob_DT_Base': round(p_dt, 4),
                'Prob_KNN_Base': round(p_knn, 4),
                'Num_Models_Failed': len(misclassified_by),
                'Misclassified_By': '; '.join(misclassified_by) if misclassified_by else 'None (Borderline Only)',
                'age': row_raw['age'],
                'bp': row_raw['bp'],
                'sg': row_raw['sg'],
                'al': row_raw['al'],
                'su': row_raw['su'],
                'sc': row_raw['sc'],
                'bu': row_raw['bu'],
                'bgr': row_raw['bgr'],
                'hemo': row_raw['hemo'],
                'pcv': row_raw['pcv'],
                'rc': row_raw['rc'],
                'htn': row_raw['htn'],
                'dm': row_raw['dm'],
            })

    # Also check Out-Of-Fold (OOF) predictions on Train (280 samples) for the 3 Tuned models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_indices = []
    for tr_idx, va_idx in cv.split(X_train_raw, y_train):
        prep = ckd_preprocessing.make_preprocessor()
        X_tr = prep.fit_transform(X_train_raw.iloc[tr_idx], y_train.iloc[tr_idx])
        X_va = prep.transform(X_train_raw.iloc[va_idx])
        fold_indices.append((tr_idx, va_idx, X_tr, y_train.iloc[tr_idx].to_numpy(), X_va))

    oof_probs = {}
    oof_preds = {}
    for name, pipe in fitted_tuned.items():
        oof_pr = np.zeros(len(X_train_raw), dtype=float)
        base_clf = pipe.named_steps['classifier']
        for tr_idx, va_idx, X_tr, y_tr, X_va in fold_indices:
            m = clone(base_clf)
            m.fit(X_tr, y_tr)
            oof_pr[va_idx] = m.predict_proba(X_va)[:, 1]
        oof_probs[name] = oof_pr
        oof_preds[name] = (oof_pr >= 0.5).astype(int)

    for pos in range(len(X_train_raw)):
        true_y = int(y_train.iloc[pos])
        failed_oof = [name for name in fitted_tuned.keys() if int(oof_preds[name][pos]) != true_y]
        if len(failed_oof) > 0:
            row_raw = X_train_raw.iloc[pos]
            n_missing = int(row_raw.isna().sum())
            missing_cols = [c for c in ckd_preprocessing.FEATURES if pd.isna(row_raw[c])]
            error_records.append({
                'Split': 'Train_OOF',
                'Test_Position': pos,
                'Original_ID': int(train_manifest.loc[pos, 'id']),
                'Source_Row': int(train_manifest.loc[pos, 'source_row']),
                'True_Label': true_y,
                'True_Class': 'CKD (1)' if true_y == 1 else 'Not CKD (0)',
                'Missing_Count_24': n_missing,
                'Missing_Percent': round(n_missing / 24.0 * 100, 1),
                'Missing_Features': ', '.join(missing_cols) if missing_cols else 'None',
                'Prob_XGBoost_Tuned': round(float(oof_probs['XGBoost (Tuned)'][pos]), 4),
                'Prob_RF_Tuned': round(float(oof_probs['Random Forest (Tuned)'][pos]), 4),
                'Prob_SVM_Tuned': round(float(oof_probs['SVM (Tuned)'][pos]), 4),
                'Prob_LogReg_Base': np.nan,
                'Prob_DT_Base': np.nan,
                'Prob_KNN_Base': np.nan,
                'Num_Models_Failed': len(failed_oof),
                'Misclassified_By': '; '.join(failed_oof),
                'age': row_raw['age'],
                'bp': row_raw['bp'],
                'sg': row_raw['sg'],
                'al': row_raw['al'],
                'su': row_raw['su'],
                'sc': row_raw['sc'],
                'bu': row_raw['bu'],
                'bgr': row_raw['bgr'],
                'hemo': row_raw['hemo'],
                'pcv': row_raw['pcv'],
                'rc': row_raw['rc'],
                'htn': row_raw['htn'],
                'dm': row_raw['dm'],
            })

    df_errors = pd.DataFrame(error_records).sort_values(
        by=['Split', 'Num_Models_Failed', 'Prob_XGBoost_Tuned'], ascending=[True, False, True]
    ).reset_index(drop=True)
    df_errors.to_csv(os.path.join(tables_dir, 'ch4_error_analysis_cases.csv'), index=False)

    # Plot predicted probabilities across models for the hard/misclassified cases (horizontal layout to prevent label overlap)
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    df_plot = df_errors[df_errors['Num_Models_Failed'] > 0].iloc[::-1].copy()
    if len(df_plot) == 0:
        df_plot = df_errors.head(8).iloc[::-1].copy()

    labels = [
        f"{r['Split']} | ID={r['Original_ID']} ({r['True_Class']}, Miss={r['Missing_Count_24']}/24)"
        for _, r in df_plot.iterrows()
    ]
    y_pos = np.arange(len(df_plot))
    height = 0.26

    ax.barh(y_pos + height, df_plot['Prob_XGBoost_Tuned'], height, label='XGBoost (Tuned)', color='#2ca02c')
    ax.barh(y_pos, df_plot['Prob_RF_Tuned'], height, label='Random Forest (Tuned)', color='#1f77b4')
    ax.barh(y_pos - height, df_plot['Prob_SVM_Tuned'], height, label='SVM (Tuned)', color='#ff7f0e')

    ax.axvline(0.50, color='#d62728', linestyle='--', lw=1.6, label='Decision Threshold (0.50)')
    ax.axvline(0.35, color='#9467bd', linestyle=':', lw=1.4, label='Screening Threshold (0.35)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel('Predicted CKD Probability P(y=1 | x)')
    ax.set_xlim(0, 1.08)
    ax.set_title('Predicted CKD Probabilities on Misclassified Clinical Cases (Test & Train OOF)', fontsize=12, pad=10)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_error_cases_clinical_profile.png'), dpi=300)
    plt.close()

    return df_errors


def map_38_to_24_feature(feat_name):
    """Map encoded 38-column name (e.g. num__hemo, binary__htn_yes, multi__sg_1.02) to original 24 feature."""
    if feat_name.startswith('num__'):
        return feat_name.replace('num__', '')
    if feat_name.startswith('binary__'):
        rest = feat_name.replace('binary__', '')
        return rest.rsplit('_', 1)[0]
    if feat_name.startswith('multi__'):
        rest = feat_name.replace('multi__', '')
        return rest.rsplit('_', 1)[0]
    return feat_name


def analyze_feature_importance(fitted_tuned, X_train_raw, y_train, X_test_raw, y_test, tables_dir, figures_dir):
    """Section 4.5.1: Feature Importance (XGBoost Gain, RF Gini, and Permutation Importance)."""
    xgb_pipe = fitted_tuned['XGBoost (Tuned)']
    rf_pipe = fitted_tuned['Random Forest (Tuned)']

    preproc = xgb_pipe.named_steps['preprocessor']
    feat_names_38 = list(preproc.named_steps['columns'].get_feature_names_out())

    xgb_clf = xgb_pipe.named_steps['classifier']
    rf_clf = rf_pipe.named_steps['classifier']

    xgb_imp_38 = xgb_clf.feature_importances_
    rf_imp_38 = rf_clf.feature_importances_

    df_38 = pd.DataFrame({
        'Encoded_Feature': feat_names_38,
        'Original_Feature': [map_38_to_24_feature(f) for f in feat_names_38],
        'XGBoost_Gain_Importance': xgb_imp_38,
        'RandomForest_Gini_Importance': rf_imp_38,
    }).sort_values('XGBoost_Gain_Importance', ascending=False).reset_index(drop=True)
    df_38.to_csv(os.path.join(tables_dir, 'ch4_feature_importance_38.csv'), index=False)

    # Permutation importance on original 24 features using XGBoost pipeline on Test set & Train set
    perm_test = permutation_importance(
        xgb_pipe, X_test_raw, y_test, scoring='roc_auc', n_repeats=10, random_state=SEED, n_jobs=1
    )
    perm_train = permutation_importance(
        xgb_pipe, X_train_raw, y_train, scoring='recall', n_repeats=5, random_state=SEED, n_jobs=1
    )

    df_perm = pd.DataFrame({
        'Original_Feature': list(X_test_raw.columns),
        'Perm_Test_ROC_AUC_Mean': perm_test.importances_mean,
        'Perm_Test_ROC_AUC_Std': perm_test.importances_std,
        'Perm_Train_Recall_Mean': perm_train.importances_mean,
    })

    # Group 38 encoded importances by the 24 original features
    df_grouped = df_38.groupby('Original_Feature', as_index=False).agg({
        'XGBoost_Gain_Importance': 'sum',
        'RandomForest_Gini_Importance': 'sum',
    })
    df_grouped = df_grouped.merge(df_perm, on='Original_Feature', how='left')
    df_grouped['Combined_Rank_Score'] = (
        0.5 * df_grouped['XGBoost_Gain_Importance'] + 0.5 * df_grouped['RandomForest_Gini_Importance']
    )
    df_grouped = df_grouped.sort_values('Combined_Rank_Score', ascending=False).reset_index(drop=True)
    df_grouped['Rank'] = np.arange(1, len(df_grouped) + 1)
    df_grouped.to_csv(os.path.join(tables_dir, 'ch4_feature_importance_24_grouped.csv'), index=False)

    # Plot Top 15 features comparing XGBoost Gain and Random Forest Gini
    top15 = df_grouped.head(15).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    axes[0].barh(top15['Original_Feature'], top15['XGBoost_Gain_Importance'], color='#2ca02c', alpha=0.9)
    axes[0].set_title('XGBoost (Tuned) - Grouped Gain Importance', fontsize=11.5, pad=8)
    axes[0].set_xlabel('Relative Gain Importance (Sum over One-Hot levels)')
    axes[0].set_ylabel('Clinical Feature (24 Attributes)')
    sns.despine(ax=axes[0], top=True, right=True)

    axes[1].barh(top15['Original_Feature'], top15['RandomForest_Gini_Importance'], color='#1f77b4', alpha=0.9)
    axes[1].set_title('Random Forest (Tuned) - Grouped Gini Importance', fontsize=11.5, pad=8)
    axes[1].set_xlabel('Mean Decrease in Impurity (Gini Importance)')
    sns.despine(ax=axes[1], top=True, right=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_feature_importance_comparison.png'), dpi=300)
    plt.close()

    return df_38, df_grouped


def run_kaggle_leakage_audit(X_train_raw, y_train, X_test_raw, y_test, tables_dir, figures_dir):
    """Section 4.5.2: Empirical Quantification of Data Leakage in Typical Kaggle Notebooks vs Our Protocol."""
    # Combine full 400-row dataset to simulate pre-split leakage on full dataset
    X_all_raw = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_test], axis=0, ignore_index=True)

    # 1. Quantify imputation & scaling parameter shift (Train-only vs Full-400 vs Fold-internal)
    cleaner = ckd_preprocessing.CKDCleaner()
    X_train_clean = cleaner.fit_transform(X_train_raw)
    X_all_clean = cleaner.fit_transform(X_all_raw)
    X_test_clean = cleaner.transform(X_test_raw)

    shift_rows = []
    for col in ckd_preprocessing.NUMERIC:
        med_train = float(X_train_clean[col].median())
        med_all = float(X_all_clean[col].median())
        med_test = float(X_test_clean[col].median())

        mean_train_imp = float(X_train_clean[col].fillna(med_train).mean())
        mean_all_imp = float(X_all_clean[col].fillna(med_all).mean())
        mean_raw_leaky = float(X_all_clean[col].mean())  # Kaggle notebooks often use mean() on all 400 rows!

        std_train_imp = float(X_train_clean[col].fillna(med_train).std(ddof=0))
        std_all_imp = float(X_all_clean[col].fillna(med_all).std(ddof=0))

        rel_diff_mean_vs_median = abs(mean_raw_leaky - med_train) / (abs(med_train) + 1e-9) * 100.0
        rel_diff_std = abs(std_all_imp - std_train_imp) / (abs(std_train_imp) + 1e-9) * 100.0

        shift_rows.append({
            'Feature': col,
            'Train_Median_Strict': round(med_train, 4),
            'Test_Median_True': round(med_test, 4),
            'Full400_Median_Leaky': round(med_all, 4),
            'Full400_Mean_Leaky_Kaggle': round(mean_raw_leaky, 4),
            'Train_Std_Strict': round(std_train_imp, 4),
            'Full400_Std_Leaky': round(std_all_imp, 4),
            'Imputation_Shift_Pct_MeanVsMed': round(rel_diff_mean_vs_median, 2),
            'Scale_Shift_Pct_FullVsTrain': round(rel_diff_std, 2),
        })

    df_shift = pd.DataFrame(shift_rows).sort_values('Imputation_Shift_Pct_MeanVsMed', ascending=False).reset_index(drop=True)
    df_shift.to_csv(os.path.join(tables_dir, 'ch4_kaggle_imputation_shift.csv'), index=False)

    # 2. Compare 3 Experimental Protocols across classifiers:
    # Protocol 1 (Strict Pipeline - Ours): Preprocessor inside each CV fold on X_train (n=320)
    # Protocol 2 (Leaky Pre-CV Transform - Common Kaggle Error #1): Fit preprocessor on all 400 rows before CV
    # Protocol 3 (Complete-Case Dropna - Common Kaggle Error #2): df.dropna() drops 242/400 rows (60.5% data loss!)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    preproc_leaky_all = ckd_preprocessing.make_preprocessor()
    X_all_leaked = preproc_leaky_all.fit_transform(X_all_raw)
    X_train_leaked = X_all_leaked[:len(X_train_raw)]

    # Complete cases only
    mask_complete_all = X_all_raw.notna().all(axis=1)
    n_complete_all = int(mask_complete_all.sum())  # 158 rows out of 400

    mask_complete_test = X_test_raw.notna().all(axis=1)
    X_test_complete = X_test_raw[mask_complete_test]
    y_test_complete = y_test[mask_complete_test]
    X_test_incomplete = X_test_raw[~mask_complete_test]
    y_test_incomplete = y_test[~mask_complete_test]

    audit_models = {
        'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree (CART)': DecisionTreeClassifier(criterion='gini', random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric='logloss', random_state=SEED, n_jobs=1),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED),
    }

    strict_folds = []
    leaky_folds = []
    for tr_idx, va_idx in cv.split(X_train_raw, y_train):
        prep_f = ckd_preprocessing.make_preprocessor()
        X_tr_s = prep_f.fit_transform(X_train_raw.iloc[tr_idx], y_train.iloc[tr_idx])
        X_va_s = prep_f.transform(X_train_raw.iloc[va_idx])
        y_tr_s = y_train.iloc[tr_idx].to_numpy()
        y_va_s = y_train.iloc[va_idx].to_numpy()
        strict_folds.append((X_tr_s, y_tr_s, X_va_s, y_va_s))
        leaky_folds.append((X_train_leaked[tr_idx], y_tr_s, X_train_leaked[va_idx], y_va_s))

    prep_full_strict = ckd_preprocessing.make_preprocessor()
    X_train_full_s = prep_full_strict.fit_transform(X_train_raw, y_train)
    X_test_full_s = prep_full_strict.transform(X_test_raw)
    X_test_inc_s = prep_full_strict.transform(X_test_incomplete)

    audit_rows = []
    for mname, clf in audit_models.items():
        s_recs, s_accs, l_recs, l_accs = [], [], [], []
        for (X_tr_s, y_tr_s, X_va_s, y_va_s), (X_tr_l, y_tr_l, X_va_l, y_va_l) in zip(strict_folds, leaky_folds):
            ms = clone(clf).fit(X_tr_s, y_tr_s)
            yp_s = ms.predict(X_va_s)
            s_recs.append(recall_score(y_va_s, yp_s, zero_division=0))
            s_accs.append(accuracy_score(y_va_s, yp_s))

            ml = clone(clf).fit(X_tr_l, y_tr_l)
            yp_l = ml.predict(X_va_l)
            l_recs.append(recall_score(y_va_l, yp_l, zero_division=0))
            l_accs.append(accuracy_score(y_va_l, yp_l))

        m_full = clone(clf).fit(X_train_full_s, y_train)
        rec_test_full = recall_score(y_test, m_full.predict(X_test_full_s))
        rec_test_inc = recall_score(y_test_incomplete, m_full.predict(X_test_inc_s))

        audit_rows.append({
            'Algorithm': mname,
            'Strict_CV_Recall': float(np.mean(s_recs)),
            'Leaky_PreFit_CV_Recall': float(np.mean(l_recs)),
            'Strict_CV_Accuracy': float(np.mean(s_accs)),
            'Leaky_PreFit_CV_Accuracy': float(np.mean(l_accs)),
            'Test_Recall_All120': float(rec_test_full),
            'Test_Recall_MissingRows': float(rec_test_inc),
            'Complete_Cases_Retained_Pct': round(n_complete_all / 400.0 * 100, 1),
        })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(os.path.join(tables_dir, 'ch4_kaggle_leakage_audit.csv'), index=False)

    # Visualization of Kaggle Audit
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Left panel: Parameter shift when using Global Mean Imputation (Kaggle typical) vs Strict Train Median
    top_shift = df_shift.head(8).iloc[::-1]
    axes[0].barh(top_shift['Feature'], top_shift['Imputation_Shift_Pct_MeanVsMed'], color='#d62728', alpha=0.85)
    axes[0].set_title('Imputation Bias: Global Mean (Kaggle) vs Strict Train Median (%)', fontsize=11, pad=8)
    axes[0].set_xlabel('Relative Shift in Imputed Center (%)')
    axes[0].set_ylabel('Quantitative Clinical Feature')
    for i, v in enumerate(top_shift['Imputation_Shift_Pct_MeanVsMed']):
        axes[0].text(v + 1.5, i, f"{v:.1f}%", va='center', fontsize=9)
    axes[0].set_xlim(0, max(top_shift['Imputation_Shift_Pct_MeanVsMed']) * 1.18)
    sns.despine(ax=axes[0], top=True, right=True)

    # Right panel: Patient coverage (Strict Pipeline retains 100% patients vs Complete-Case Dropna loses 60.5%)
    n_train_tot = len(X_train_raw)
    n_test_tot = len(X_test_raw)
    n_test_ckd = int((y_test == 1).sum())
    categories = [
        'Full Dataset\n(n=400)',
        f'Train Split\n(n={n_train_tot})',
        f'Independent Test\n(n={n_test_tot})',
        f'CKD Cases in Test\n(n={n_test_ckd})'
    ]
    n_total = [400, n_train_tot, n_test_tot, n_test_ckd]
    n_dropna_kept = [
        int(X_all_raw.notna().all(axis=1).sum()),
        int(X_train_raw.notna().all(axis=1).sum()),
        int(X_test_raw.notna().all(axis=1).sum()),
        int(X_test_raw[y_test == 1].notna().all(axis=1).sum()),
    ]
    pct_kept_strict = [100.0, 100.0, 100.0, 100.0]
    pct_kept_dropna = [k / t * 100.0 for k, t in zip(n_dropna_kept, n_total)]

    x = np.arange(len(categories))
    w = 0.35
    axes[1].bar(x - w/2, pct_kept_strict, w, label='Our Pipeline (Impute inside Fold)', color='#2ca02c')
    axes[1].bar(x + w/2, pct_kept_dropna, w, label='Kaggle Complete-Case (df.dropna)', color='#7f7f7f')
    for i, p in enumerate(pct_kept_dropna):
        axes[1].text(i + w/2, p + 2.0, f"{p:.1f}%\n({n_dropna_kept[i]}/{n_total[i]})", ha='center', fontsize=8.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylim(0, 118)
    axes[1].set_ylabel('Patients Evaluable (%)')
    axes[1].set_title('Clinical Patient Coverage: Pipeline Imputation vs df.dropna()', fontsize=11, pad=8)
    axes[1].legend(loc='upper right', frameon=True, fontsize=9)
    sns.despine(ax=axes[1], top=True, right=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_kaggle_leakage_audit.png'), dpi=300)
    plt.close()

    return df_shift, df_audit


def generate_ch4_core_figures(fitted_tuned, fitted_baselines, predictions, probabilities, y_test, figures_dir):
    """Generate Section 4.2 (Confusion Matrices) and Section 4.3.1 (Overlaid ROC & PR Curves)."""
    # 1. Confusion matrices of the 3 Tuned models side-by-side
    tuned_names = ['XGBoost (Tuned)', 'Random Forest (Tuned)', 'SVM (Tuned)']
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    n_neg = int((y_test == 0).sum())
    n_pos = int((y_test == 1).sum())

    for idx, name in enumerate(tuned_names):
        ax = axes[idx]
        yp = predictions[name]
        cm = confusion_matrix(y_test, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        annot_matrix = np.array([
            [f"TN = {tn}\n({tn/n_neg*100:.1f}%)", f"FP = {fp}\n({fp/n_neg*100:.1f}%)"],
            [f"FN = {fn}\n({fn/n_pos*100:.2f}%)", f"TP = {tp}\n({tp/n_pos*100:.2f}%)"]
        ])
        sns.heatmap(
            cm, annot=annot_matrix, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Not CKD (0)', 'CKD (1)'],
            yticklabels=['Not CKD (0)', 'CKD (1)'],
            annot_kws={'size': 11, 'weight': 'bold'}, ax=ax
        )
        rec = tp / (tp + fn)
        acc = (tp + tn) / len(y_test)
        ax.set_title(f"{name}\nAcc={acc*100:.2f}% | Recall={rec*100:.2f}% (FN={fn})", fontsize=11, pad=10)
        ax.set_xlabel('Predicted Diagnosis')
        if idx == 0:
            ax.set_ylabel('True Clinical Status')
        else:
            ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_confusion_matrices_tuned.png'), dpi=300)
    plt.close()

    # 2. Overlaid ROC Curves and Precision-Recall Curves for key models
    models_plot = [
        ('XGBoost (Tuned)', '#2ca02c', '-', 2.4),
        ('Random Forest (Tuned)', '#1f77b4', '--', 2.0),
        ('SVM (Tuned)', '#ff7f0e', '-.', 2.0),
        ('Logistic Regression (Baseline)', '#9467bd', ':', 1.8),
        ('Decision Tree (Baseline)', '#d62728', '--', 1.6),
        ('K-Nearest Neighbors (Baseline)', '#8c564b', ':', 1.6),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ROC Curve
    for name, color, ls, lw in models_plot:
        y_prob = probabilities[name]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, linestyle=ls, lw=lw, label=f"{name} (AUC = {auc_val:.4f})")

    axes[0].plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.0, label='Random Chance (AUC = 0.5000)')
    axes[0].set_xlim([-0.02, 1.02])
    axes[0].set_ylim([-0.02, 1.03])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Recall / Sensitivity)')
    axes[0].set_title(f'ROC Curves on Independent Test Set (n={len(y_test)})', fontsize=11.5, pad=10)
    axes[0].legend(loc='lower right', frameon=True, fontsize=8.8)
    sns.despine(ax=axes[0], top=True, right=True)

    # Precision-Recall Curve
    baseline_prevalence = np.mean(y_test)
    for name, color, ls, lw in models_plot:
        y_prob = probabilities[name]
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        ap_val = average_precision_score(y_test, y_prob)
        axes[1].plot(rec_c, prec_c, color=color, linestyle=ls, lw=lw, label=f"{name} (PR-AUC = {ap_val:.4f})")

    axes[1].axhline(baseline_prevalence, color='#888888', linestyle='--', lw=1.0, label=f'Baseline Prevalence ({baseline_prevalence:.3f})')
    axes[1].set_xlim([-0.02, 1.02])
    axes[1].set_ylim([0.55, 1.03])
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (Positive Predictive Value)')
    axes[1].set_title(f'Precision-Recall Curves on Independent Test Set (n={len(y_test)})', fontsize=11.5, pad=10)
    axes[1].legend(loc='lower left', frameon=True, fontsize=8.8)
    sns.despine(ax=axes[1], top=True, right=True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'ch4_roc_pr_curves_comparison.png'), dpi=300)
    plt.close()


def main():
    setup_tufte_style()
    reports_dir = os.path.join(BASE_DIR, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    tables_dir = os.path.join(reports_dir, 'tables')
    models_dir = os.path.join(BASE_DIR, 'models')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("[Chapter 4] Loading dataset splits and manifest...")
    X_train_raw, y_train, X_test_raw, y_test, manifest = load_all_splits()

    # Re-save preprocessor.joblib cleanly under current environment
    preproc = ckd_preprocessing.make_preprocessor()
    preproc.fit(X_train_raw, y_train)
    joblib.dump(preproc, os.path.join(models_dir, 'preprocessor.joblib'))

    print("[Chapter 4] Fitting 6 Baseline models and 3 GridSearchCV Tuned models...")
    fitted_baselines, fitted_tuned, cv_scores, best_params = build_and_fit_models(X_train_raw, y_train)

    print("[Chapter 4] Section 4.1: Evaluating all 9 models on Independent Test Set (n=120) with 95% Bootstrap CI...")
    df_test, predictions, probabilities = evaluate_comprehensive_test_set(
        fitted_baselines, fitted_tuned, X_test_raw, y_test, tables_dir
    )

    print("[Chapter 4] Section 4.1: Running 5x5 Repeated Stratified CV (25 folds) stability analysis...")
    df_stab = evaluate_repeated_cv_stability(fitted_tuned, fitted_baselines, X_train_raw, y_train, tables_dir)

    print("[Chapter 4] Section 4.2 & 4.3.1: Generating Confusion Matrices and Overlaid ROC/PR Curves...")
    generate_ch4_core_figures(fitted_tuned, fitted_baselines, predictions, probabilities, y_test, figures_dir)

    print("[Chapter 4] Section 4.3.2: Performing Classification Threshold Tuning analysis...")
    df_thresh = analyze_thresholds(probabilities, y_test, tables_dir, figures_dir)

    print("[Chapter 4] Section 4.4: Performing Patient-level Error & Borderline Analysis...")
    df_errors = perform_error_analysis(
        fitted_baselines, fitted_tuned, predictions, probabilities,
        X_train_raw, y_train, X_test_raw, y_test, manifest, tables_dir, figures_dir
    )

    print("[Chapter 4] Section 4.5.1: Extracting Feature Importance (XGBoost, RF, Permutation)...")
    df_38, df_grouped = analyze_feature_importance(
        fitted_tuned, X_train_raw, y_train, X_test_raw, y_test, tables_dir, figures_dir
    )

    print("[Chapter 4] Section 4.5.2: Running Empirical Kaggle Data Leakage Audit...")
    df_shift, df_audit = run_kaggle_leakage_audit(
        X_train_raw, y_train, X_test_raw, y_test, tables_dir, figures_dir
    )

    print("\n=== CHAPTER 4 SUMMARY OF KEY FINDINGS ===")
    print("\n1. Independent Test Set Metrics (n=120):")
    print(df_test[['Model', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Recall', 'Specificity', 'F1_Score', 'ROC_AUC', 'PR_AUC']].to_string(index=False))

    print("\n2. 25-Fold Repeated Stratified CV Stability:")
    print(df_stab[['Model', 'Recall_Mean', 'Recall_Std', 'ROC_AUC_Mean', 'F1_Mean', 'Accuracy_Mean']].to_string(index=False))

    print("\n3. Misclassified / Borderline Cases (Test & Train OOF):")
    print(df_errors[['Split', 'Test_Position', 'Original_ID', 'Source_Row', 'True_Class', 'Missing_Count_24', 'Prob_XGBoost_Tuned', 'Prob_RF_Tuned', 'Prob_SVM_Tuned', 'Misclassified_By']].to_string(index=False))

    print("\n4. Top 10 Clinical Features (Grouped 24):")
    print(df_grouped.head(10)[['Rank', 'Original_Feature', 'XGBoost_Gain_Importance', 'RandomForest_Gini_Importance', 'Perm_Test_ROC_AUC_Mean']].to_string(index=False))

    print("\n5. Kaggle Data Leakage Audit:")
    print(df_audit.to_string(index=False))


if __name__ == '__main__':
    main()
