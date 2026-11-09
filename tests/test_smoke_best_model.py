import os
import sys
import unittest
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Ensure project root is on sys.path so ckd_preprocessing can be resolved
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ckd_preprocessing

class TestSmokeBestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib'))
        cls.assertTrue(os.path.exists(cls.model_path), f"Tep checkpoint khong ton tai tai: {cls.model_path}")
        cls.pipeline = joblib.load(cls.model_path)

    def test_pipeline_structure(self):
        """Kiem tra cau truc Pipeline chua ca bo tien xu ly va bo phan lop."""
        self.assertIsInstance(self.pipeline, Pipeline)
        self.assertTrue(hasattr(self.pipeline, 'predict'))
        self.assertTrue(hasattr(self.pipeline, 'predict_proba'))
        # Kiem tra buoc tien xu ly va phan lop
        self.assertIn('preprocessor', self.pipeline.named_steps)
        self.assertIn('classifier', self.pipeline.named_steps)

    def test_inference_ckd_case(self):
        """Kiem tra suy luan tren ca benh nhan CKD nguy co cao."""
        sample_ckd = pd.DataFrame([{
            'age': 65.0, 'bp': 90.0, 'sg': '1.010', 'al': '3', 'su': '1',
            'rbc': 'abnormal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'present',
            'bgr': 190.0, 'bu': 75.0, 'sc': 4.5, 'sod': 130.0, 'pot': 5.2,
            'hemo': 8.5, 'pcv': 28.0, 'wc': 9500.0, 'rc': 3.2,
            'htn': 'yes', 'dm': 'yes', 'cad': 'yes', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
        }])
        pred = self.pipeline.predict(sample_ckd)[0]
        probs = self.pipeline.predict_proba(sample_ckd)[0]
        
        self.assertEqual(pred, 1, "Ca nguy co cao phai duoc du doan la CKD (nhan 1)")
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        self.assertGreater(probs[1], 0.70)

    def test_inference_healthy_case(self):
        """Kiem tra suy luan tren ca benh nhan khoe manh."""
        sample_healthy = pd.DataFrame([{
            'age': 30.0, 'bp': 70.0, 'sg': '1.025', 'al': '0', 'su': '0',
            'rbc': 'normal', 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': 90.0, 'bu': 22.0, 'sc': 0.8, 'sod': 142.0, 'pot': 4.1,
            'hemo': 15.5, 'pcv': 45.0, 'wc': 6500.0, 'rc': 5.2,
            'htn': 'no', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }])
        pred = self.pipeline.predict(sample_healthy)[0]
        probs = self.pipeline.predict_proba(sample_healthy)[0]
        
        self.assertEqual(pred, 0, "Ca khoe manh phai duoc du doan la Not CKD (nhan 0)")
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)
        self.assertGreater(probs[0], 0.70)

    def test_resilience_to_missing_values(self):
        """Kiem tra kha nang chong chiu khi du lieu khuyet thieu (NaN) nho SimpleImputer."""
        sample_nan = pd.DataFrame([{
            'age': 48.0, 'bp': 80.0, 'sg': '1.020', 'al': np.nan, 'su': '0',
            'rbc': np.nan, 'pc': 'normal', 'pcc': 'notpresent', 'ba': 'notpresent',
            'bgr': np.nan, 'bu': 36.0, 'sc': np.nan, 'sod': 138.0, 'pot': np.nan,
            'hemo': 11.2, 'pcv': np.nan, 'wc': 7200.0, 'rc': np.nan,
            'htn': 'yes', 'dm': 'no', 'cad': 'no', 'appet': 'good', 'pe': 'no', 'ane': 'no'
        }])
        pred = self.pipeline.predict(sample_nan)[0]
        probs = self.pipeline.predict_proba(sample_nan)[0]
        self.assertIn(pred, [0, 1])
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)

    def test_rejection_of_schema_violation(self):
        """Kiem tra tinh hop le: tu choi dau vao sai ten cot hoac thieu cot."""
        invalid_sample = pd.DataFrame([{
            'age': 50.0, 'bp': 80.0, 'wrong_column': 123
        }])
        with self.assertRaises(ValueError):
            self.pipeline.predict(invalid_sample)

    def test_latency_performance(self):
        """Kiem tra hieu nang suy luan: do tre trung binh phai duoi 50ms."""
        sample = pd.DataFrame([{
            'age': 55.0, 'bp': 80.0, 'sg': '1.015', 'al': '2', 'su': '0',
            'rbc': 'normal', 'pc': 'abnormal', 'pcc': 'present', 'ba': 'notpresent',
            'bgr': 140.0, 'bu': 58.0, 'sc': 3.4, 'sod': 135.0, 'pot': 4.9,
            'hemo': 9.8, 'pcv': 32.0, 'wc': 7800.0, 'rc': 3.9,
            'htn': 'yes', 'dm': 'yes', 'cad': 'no', 'appet': 'poor', 'pe': 'yes', 'ane': 'yes'
        }])
        # Warmup
        self.pipeline.predict(sample)
        
        start = time.time()
        iterations = 50
        for _ in range(iterations):
            self.pipeline.predict(sample)
        avg_latency = (time.time() - start) / iterations
        self.assertLess(avg_latency, 0.05, f"Do tre qua cao: {avg_latency*1000:.2f}ms")

if __name__ == '__main__':
    unittest.main()
