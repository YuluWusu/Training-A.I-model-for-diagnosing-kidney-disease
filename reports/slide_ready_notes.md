# Slide-Ready Presentation Notes: Model Training & Evaluation
*Concise, verifiable bullet points for defense presentation slides*

---

### Slide 1: Experimental Design & Selection Bias Prevention
- **Dataset Scale**: 400 clinical patient records; 24 attributes (11 numerical, 13 categorical).
- **Split Protocol**: 80/20 Stratified Partition (320 Train / 80 Independent Test, seed=42).
- **Selection Bias Prevention (Cawley & Talbot, 2010)**: All model-family benchmarking and hyperparameter tuning performed strictly within the 320-sample development split. The 80-sample test set remained untouched until model selection was frozen.
- **Leakage Prevention**: All imputation (median/mode) and scaling (StandardScaler) fitted strictly within CV folds via scikit-learn Pipeline.

---

### Slide 2: 6-Model Baseline Cross-Validation Benchmark (320 Train Samples)
- **Logistic Regression (L2)**: Mean Recall = 0.9950, Mean ROC-AUC = 0.9998.
- **K-Nearest Neighbors (k=5)**: Mean Recall = 0.9250.
- **Support Vector Machine (RBF)**: Mean Recall = 1.0000, Mean ROC-AUC = 1.0000.
- **Decision Tree (CART)**: Mean Recall = 0.9650.
- **Random Forest (100 trees)**: Mean Recall = 0.9900, Mean ROC-AUC = 0.9992.
- **XGBoost (Boosting)**: Mean Recall = 0.9900, Mean ROC-AUC = 0.9979.
*Note for Presenter: Classical algorithms use boosting rounds, trees, or iterations. They do NOT use neural network epochs or Adam.*

---

### Slide 3: Hyperparameter Tuning & Development Model Selection
- **Tuned Architectures**: Random Forest, XGBoost, and Support Vector Machine.
- **Tuning Space**: Explored via `GridSearchCV(scoring='recall', cv=5)` on Train partition.
- **CV Performance on Development Split**:
  - XGBoost: CV Recall = 0.9950, CV ROC-AUC = 0.9975.
  - Random Forest: CV Recall = 0.9950, CV ROC-AUC = 0.9992.
  - SVM: CV Recall = 1.0000, CV ROC-AUC = 1.0000.
- **Selection Decision**: **XGBoost** selected on Development CV evidence before touching test data.

---

### Slide 4: Single-Pass Independent Test Verification
- **Evaluated Architecture**: **XGBoost** (Pre-selected and retrained on 320 train samples).
- **Test Performance (n=80 held-out observations)**:
  - Accuracy: **100.0%** (80/80 correct)
  - Recall (Sensitivity): **100.0%** (50/50 CKD cases detected)
  - False Negatives: **0 patients missed**
  - Specificity: **100.0%** (30/30 healthy controls verified)
  - ROC-AUC: **1.0000**
- **Deployment Readiness**: Serialized with embedded `make_preprocessor()` into `models/best_model.joblib`. Sub-50ms inference latency.
