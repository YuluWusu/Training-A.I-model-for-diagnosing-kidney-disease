# Slide-Ready Presentation Notes: Chapter 3 Model Training & Evaluation
*Concise, verifiable bullet points for defense presentation slides*

---

### Slide 1: Experimental Design & Leakage Prevention
- **Dataset Scale**: 400 clinical patient records; 24 attributes (11 quantitative numerical, 13 categorical).
- **Split Scheme**: 70/30 Stratified Partition (280 Train / 120 Independent Test, seed=42).
- **Leakage Prevention**: All imputation (median/mode) and scaling (StandardScaler) fitted strictly within each 5-Fold cross-validation split via scikit-learn Pipeline.
- **Evaluation Criteria**: Prioritizing **Recall (Sensitivity)** to minimize deadly False Negatives (unscreened CKD cases).

---

### Slide 2: 6-Model Baseline Cross-Validation Benchmark
- **Logistic Regression (L2)**: Fast linear baseline; Mean Recall = 0.9886, Mean ROC-AUC = 0.9997.
- **K-Nearest Neighbors (k=5)**: Instance-based; Mean Recall = 0.8971.
- **Support Vector Machine (RBF)**: Maximum margin hyperplane; Mean Recall = 0.9771, Mean ROC-AUC = 1.0000.
- **Decision Tree (CART)**: Rule-based splitting; Mean Recall = 0.9429.
- **Random Forest (100 trees)**: Bagging ensemble; Mean Recall = 0.9829, Mean ROC-AUC = 0.9992.
- **XGBoost (Boosting)**: Sequential gradient boosting; Mean Recall = 0.9714, Mean ROC-AUC = 0.9962.
*Note for Presenter: Classical algorithms use boosting rounds, trees, or iterations. They do NOT use neural network epochs or Adam.*

---

### Slide 3: Hyperparameter Tuning via GridSearchCV
- **Tuned Architectures**: Random Forest, XGBoost, and Support Vector Machine.
- **Optimization Target**: `scoring='recall'` under 5-Fold Stratified Cross-Validation.
- **Optimal Hyperparameters**:
  - XGBoost: `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `subsample=1.0`.
  - Random Forest: `n_estimators=50`, `max_depth=None`, `min_samples_split=2`, `criterion='gini'`.
  - SVM: `C=1.0`, `gamma='scale'`, `kernel='linear'`.

---

### Slide 4: Independent Test Verification & Model Selection
- **Selected Architecture**: **XGBoost**
- **Test Performance (n=120)**:
  - Accuracy: **99.2%** (119/120 correct)
  - Recall (Sensitivity): **98.7%** (74/75 CKD cases detected)
  - False Negatives: **1 patients missed**
  - Specificity: **100.0%** (45/45 healthy cases verified)
  - ROC-AUC: **0.9997**
- **Deployment Readiness**: Preprocessor + Classifier saved as unified `models/best_model.joblib`. Sub-50ms inference latency.
