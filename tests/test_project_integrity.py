import os
import sys
import json
import unittest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ckd_preprocessing


class TestProjectIntegrity(unittest.TestCase):
    """End-to-end verification across Chapter 2 (EDA/Preprocess), Chapter 3 (Training), and Chapter 4 (Evaluation)."""

    def test_feature_contract_and_splits(self):
        """Verify feature_contract.json and data/processed splits match exact specifications."""
        contract_path = os.path.join(BASE_DIR, 'feature_contract.json')
        self.assertTrue(os.path.exists(contract_path))
        with open(contract_path, 'r', encoding='utf-8') as f:
            contract = json.load(f)

        self.assertEqual(contract['input_features'], ckd_preprocessing.FEATURES)
        self.assertEqual(contract['input_count'], 24)
        self.assertEqual(contract['output_count'], 38)

        proc_dir = os.path.join(BASE_DIR, 'data', 'processed')
        X_train_raw = pd.read_csv(os.path.join(proc_dir, 'X_train_raw.csv'))
        X_test_raw = pd.read_csv(os.path.join(proc_dir, 'X_test_raw.csv'))
        y_train = pd.read_csv(os.path.join(proc_dir, 'y_train.csv'))['classification']
        y_test = pd.read_csv(os.path.join(proc_dir, 'y_test.csv'))['classification']

        self.assertEqual(X_train_raw.shape, (320, 24))
        self.assertEqual(X_test_raw.shape, (80, 24))
        self.assertEqual(int(y_train.sum()), 200)
        self.assertEqual(int(y_test.sum()), 50)

    def test_preprocessor_artifact(self):
        """Verify models/preprocessor.joblib transforms X_test_raw to a finite (80, 38) matrix."""
        prep_path = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')
        self.assertTrue(os.path.exists(prep_path))
        preprocessor = joblib.load(prep_path)

        X_test_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_test_raw.csv'))
        X_test_proc = preprocessor.transform(X_test_raw)
        self.assertEqual(X_test_proc.shape, (80, 38))
        self.assertTrue(np.isfinite(X_test_proc).all())

    def test_best_model_on_independent_test_set(self):
        """Verify models/best_model.joblib achieves 100% Recall (0 False Negatives) on independent Test set."""
        model_path = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
        pipeline = joblib.load(model_path)

        X_test_raw = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'X_test_raw.csv'))
        y_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'y_test.csv'))['classification'].astype(int)

        y_pred = pipeline.predict(X_test_raw)
        y_prob = pipeline.predict_proba(X_test_raw)[:, 1]

        self.assertEqual(recall_score(y_test, y_pred), 1.0)
        self.assertEqual(accuracy_score(y_test, y_pred), 1.0)
        self.assertEqual(roc_auc_score(y_test, y_prob), 1.0)

    def test_all_chapter_deliverables_present(self):
        """Verify all required notebooks, reports, tables, and figures for Chapters 2, 3, and 4 exist."""
        required_files = [
            'notebook/NoteBook_EDA_HuynhCongTao.ipynb',
            'notebook/training_model.ipynb',
            'notebook/evaluation_chapter4.ipynb',
            'reports/CHUONG_2_HuynhCongTao.md',
            'reports/CHUONG_3_ToHongVu.md',
            'reports/CHUONG_4_HoangXuanPhiLong.md',
            'reports/experiment_summary.md',
            'reports/slide_ready_notes.md',
            'reports/tables/ch4_test_comprehensive_metrics.csv',
            'reports/tables/ch4_repeated_cv_stability.csv',
            'reports/tables/ch4_threshold_tuning.csv',
            'reports/tables/ch4_error_analysis_cases.csv',
            'reports/tables/ch4_feature_importance_24_grouped.csv',
            'reports/tables/ch4_kaggle_leakage_audit.csv',
            'reports/figures/ch4_confusion_matrices_tuned.png',
            'reports/figures/ch4_roc_pr_curves_comparison.png',
            'reports/figures/ch4_threshold_tuning_curves.png',
            'reports/figures/ch4_error_cases_clinical_profile.png',
            'reports/figures/ch4_feature_importance_comparison.png',
            'reports/figures/ch4_kaggle_leakage_audit.png',
        ]
        for rel_path in required_files:
            full_path = os.path.join(BASE_DIR, rel_path)
            self.assertTrue(os.path.exists(full_path), f"Missing deliverable: {rel_path}")
            self.assertGreater(os.path.getsize(full_path), 0, f"Empty deliverable: {rel_path}")


if __name__ == '__main__':
    unittest.main()

