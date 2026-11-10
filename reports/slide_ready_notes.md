# Slide-Ready Presentation Notes: Chapter 3 Model Training & Evaluation
*Concise, verifiable bullet points for defense presentation slides*

---

### Slide 1: Experimental Design & Leakage Prevention
- **Dataset Scale**: 400 clinical patient records; 24 attributes (11 quantitative numerical, 13 categorical).
- **Split Scheme**: 80/20 Stratified Partition (320 Train / 80 Independent Test, seed=42).
- **Leakage Prevention**: All imputation (median/mode) and scaling (StandardScaler) fitted strictly within each 5-Fold cross-validation split via scikit-learn Pipeline.
- **Evaluation Criteria**: Prioritizing **Recall (Sensitivity)** to eliminate deadly False Negatives (unscreened CKD cases).

---

### Slide 2: 6-Model Baseline Cross-Validation Benchmark
- **Logistic Regression (L2)**: Fast linear baseline; Mean Recall = 0.9950, Mean ROC-AUC = 0.9998.
- **K-Nearest Neighbors (k=5)**: Instance-based; Mean Recall = 0.9250.
- **Support Vector Machine (RBF)**: Maximum margin hyperplane; Mean Recall = 1.0000, Mean ROC-AUC = 1.0000.
- **Decision Tree (CART)**: Rule-based splitting; Mean Recall = 0.9650.
- **Random Forest (100 trees)**: Bagging ensemble; Mean Recall = 0.9900, Mean ROC-AUC = 0.9992.
- **XGBoost (Boosting)**: Sequential gradient boosting; Mean Recall = 0.9900, Mean ROC-AUC = 0.9979.
*Note for Presenter: Classical algorithms use boosting rounds, trees, or iterations. They do NOT use neural network epochs or Adam.*

---

### Slide 3: Hyperparameter Tuning via GridSearchCV
- **Tuned Architectures**: Random Forest, XGBoost, and Support Vector Machine.
- **Optimization Target**: `scoring='recall'` under 5-Fold Stratified Cross-Validation.
- **Optimal Hyperparameters**:
  - XGBoost: `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`, `subsample=1.0`.
  - Random Forest: `n_estimators=50`, `max_depth=None`, `min_samples_split=2`, `criterion='gini'`.
  - SVM: `C=1.0`, `gamma='scale'`, `kernel='rbf'`.

---

### Slide 4: Independent Test Verification & Model Selection
- **Selected Architecture**: **XGBoost**
- **Test Performance (n=80)**:
  - Accuracy: **100.0%** (80/80 correct)
  - Recall (Sensitivity): **100.0%** (50/50 CKD cases detected)
  - False Negatives: **0 patients missed**
  - Specificity: **100.0%** (30/30 healthy cases verified)
  - ROC-AUC: **1.0000**
- **Deployment Readiness**: Preprocessor + Classifier saved as unified `models/best_model.joblib`. Sub-50ms inference latency.
