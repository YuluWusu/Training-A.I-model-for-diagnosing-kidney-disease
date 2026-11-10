# CHƯƠNG 3: HUẤN LUYỆN, TINH CHỈNH VÀ ĐÓNG GÓI MÔ HÌNH

**Người thực hiện: Tô Hồng Vũ — Nhóm 01**

Chương này trình bày quy trình thiết lập thực nghiệm học máy có giám sát cho bài toán chẩn đoán bệnh thận mạn tính (CKD), bao gồm: xây dựng phương thức kiểm chứng chéo phân tầng 5-Fold (5-Fold Stratified Cross-Validation) tích hợp trực tiếp bộ tiền xử lý `make_preprocessor()` nhằm ngăn chặn rò rỉ dữ liệu, huấn luyện và đánh giá sơ bộ 6 thuật toán cơ sở (Baseline Models), tinh chỉnh siêu tham số bằng `GridSearchCV` với hàm mục tiêu ưu tiên độ nhạy lâm sàng (`scoring='recall'`), và đóng gói toàn bộ quy trình thành file nhị phân `models/best_model.joblib` kèm bộ kiểm thử tự động (Smoke Test) để bàn giao cho Chương 4 và Chương 5.

---

## 3.1. Thiết lập môi trường thực nghiệm và Phương pháp kiểm chứng chéo

### 3.1.1. Đầu vào thực nghiệm và Hợp đồng dữ liệu (Feature Ordering Contract)

Đầu vào của Chương 3 được tiếp nhận từ bộ bàn giao chuẩn hóa của Chương 2 (`data/processed/`):
- **Tập huấn luyện cú pháp sạch chưa điền khuyết (`X_train_raw.csv`, `y_train.csv`):** Gồm $N_{\text{train}} = 320$ bệnh nhân (200 ca `ckd = 1`, 120 ca `notckd = 0`), giữ nguyên 24 thuộc tính đầu vào theo đúng thứ tự trong `feature_contract.json`:
  `age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane`.
- **Tập kiểm thử độc lập (`X_test_raw.csv`, `y_test.csv`):** Gồm $N_{\text{test}} = 80$ bệnh nhân (50 ca `ckd = 1`, 30 ca `notckd = 0`), được cô lập hoàn toàn trong suốt quá trình đánh giá Baseline và tìm kiếm siêu tham số.

### 3.1.2. Cơ chế chống rò rỉ dữ liệu trong 5-Fold Stratified Cross-Validation

Thay vì sử dụng ma trận `X_train.csv` (đã được fit `SimpleImputer` và `StandardScaler` trên toàn bộ 320 mẫu trước khi chia fold), nhóm sử dụng `X_train_raw.csv` và đóng gói hàm `ckd_preprocessing.make_preprocessor()` vào bên trong đối tượng `sklearn.pipeline.Pipeline` cùng với bộ phân lớp:

```python
pipeline = Pipeline([
    ('preprocessor', ckd_preprocessing.make_preprocessor()),
    ('classifier', clf)
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Với kiến trúc này, tại mỗi vòng lặp thứ $k \in \{1, 2, 3, 4, 5\}$ của `StratifiedKFold`:
1. Tập huấn luyện nội bộ của fold ($4/5 \times 320 = 256$ mẫu, gồm 160 ca CKD và 96 ca Not CKD) được dùng để học `median` (cho 11 biến số), `mode` (cho 13 biến phân loại) và tham số chuẩn hóa Z-score ($\mu, \sigma$), sau đó biến đổi thành ma trận $256 \times 38$ để huấn luyện bộ phân lớp.
2. Tập thẩm định của fold ($1/5 \times 320 = 64$ mẫu, gồm 40 ca CKD và 24 ca Not CKD) chỉ gọi phương thức `.transform()` dựa trên thống kê đã học từ 256 mẫu huấn luyện của chính fold đó.
3. Hạt giống ngẫu nhiên được cố định tại `SEED = 42` xuyên suốt bộ chia fold và các thuật toán có yếu tố ngẫu nhiên để đảm bảo khả năng tái lập 100%.

---

## 3.2. Huấn luyện thực nghiệm và Đánh giá sơ bộ 6 mô hình Baseline

### 3.2.1. Cấu hình 6 thuật toán cơ sở (Baseline Classifiers)

Nhóm lựa chọn 6 thuật toán đại diện cho 4 trường phái học máy có giám sát đã trình bày tại Chương 1:
1. **Logistic Regression (`penalty='l2', C=1.0, max_iter=1000, random_state=42`):** Mô hình tuyến tính chuẩn với điều chuẩn Ridge ($L_2$) tối ưu hóa hàm mất mát Binary Cross-Entropy.
2. **K-Nearest Neighbors (`n_neighbors=5, metric='minkowski', p=2`):** Thuật toán dựa trên khoảng cách Euclidean với $k=5$ láng giềng gần nhất.
3. **Support Vector Machine (`kernel='rbf', C=1.0, probability=True, random_state=42`):** Máy học vector hỗ trợ cực đại hóa lề mềm với hàm nhân bán kính (RBF Kernel).
4. **Decision Tree — CART (`criterion='gini', random_state=42`):** Cây quyết định đơn lẻ phân chia nút theo độ giảm tạp chất Gini.
5. **Random Forest (`n_estimators=100, criterion='gini', random_state=42`):** Mô hình tổ hợp song song (Bagging) gồm 100 cây quyết định độc lập.
6. **XGBoost (`n_estimators=100, learning_rate=0.1, max_depth=3, eval_metric='logloss', random_state=42`):** Mô hình tổ hợp tuần tự tăng cường độ dốc (Gradient Boosted Decision Trees) có điều chuẩn $L_1/L_2$.

### 3.2.2. Kết quả đối sánh 6 mô hình Baseline qua 5-Fold Stratified CV

Kết quả trung bình (Mean) và độ lệch chuẩn (Std) qua 5 folds trên tập Train ($n = 320$) được trích xuất từ `reports/baseline_cv_metrics.csv` và minh họa tại Hình `reports/figures/baseline_comparison.png`:

![So sánh độ nhạy lâm sàng (Recall) và ROC-AUC trung bình 5-Fold CV của 6 mô hình Baseline](figures/baseline_comparison.png)

| Hạng | Thuật toán Baseline | Accuracy (Mean ± Std) | Precision (Mean) | Recall / Sensitivity (Mean) | F1-Score (Mean) | ROC-AUC (Mean) | Thời gian chạy 5 Folds (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** | **Support Vector Machine (RBF)** | 1,0000 ± 0,0000 | 1,0000 | **1,0000** | 1,0000 | 1,0000 | 0,18s |
| **2** | **Logistic Regression (L2)** | 0,9906 ± 0,0077 | 0,9902 | **0,9950** | 0,9925 | 0,9998 | 0,22s |
| **3** | **Random Forest (100 trees)** | 0,9875 ± 0,0117 | 0,9901 | **0,9900** | 0,9900 | 0,9992 | 0,85s |
| **4** | **XGBoost (Boosting)** | 0,9812 ± 0,0182 | 0,9805 | **0,9900** | 0,9851 | 0,9979 | 0,38s |
| **5** | **Decision Tree (CART)** | 0,9625 ± 0,0159 | 0,9756 | **0,9650** | 0,9700 | 0,9617 | 0,12s |
| **6** | **K-Nearest Neighbors (k=5)** | 0,9500 ± 0,0182 | 0,9947 | **0,9250** | 0,9584 | 0,9929 | 0,14s |

**Nhận xét sơ bộ từ thực nghiệm Baseline:**
- Ba kiến trúc phi tuyến mạnh (**SVM**, **Random Forest**, **XGBoost**) cùng với **Logistic Regression** đều đạt độ nhạy trung bình $\text{Recall} \ge 0,9900$ và $\text{ROC-AUC} \ge 0,9979$.
- **Decision Tree đơn lẻ** ($\text{Recall} = 0,9650$, $\text{ROC-AUC} = 0,9617$) có phương sai cao hơn và dễ bị khớp quá mức (overfitting) vào các ngưỡng cắt cục bộ của từng fold.
- **K-Nearest Neighbors** có độ nhạy thấp nhất ($\text{Recall} = 0,9250$, bỏ sót trung bình 7,5% số ca CKD), chứng tỏ phép đo khoảng cách Euclidean trong không gian 38 chiều nhạy cảm mạnh với các bản ghi bị điền khuyết nhiều thuộc tính.
- Từ kết quả này, nhóm chọn 3 đại diện mạnh nhất thuộc 3 họ thuật toán khác nhau (**Random Forest**, **XGBoost**, **Support Vector Machine**) để đưa vào giai đoạn tinh chỉnh siêu tham số chuyên sâu.

---

## 3.3. Tinh chỉnh siêu tham số mô hình (Hyperparameter Tuning via GridSearchCV)

### 3.3.1. Không gian tìm kiếm tham số và Tiêu chí tối ưu hóa

Nhóm sử dụng `GridSearchCV` kết hợp `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` với hàm mục tiêu **`scoring='recall'`** nhằm ưu tiên các cấu hình siêu tham số giảm thiểu tối đa số ca Âm tính giả (False Negative):

| Mô hình tinh chỉnh | Lưới siêu tham số khảo sát (`param_grid`) | Số tổ hợp | Tổng số lần fit (× 5 Folds) | Bộ siêu tham số tối ưu (`best_params_`) |
|---|---|---:|---:|---|
| **XGBoost** | `n_estimators`: `[50, 100, 150]`<br>`learning_rate`: `[0.01, 0.05, 0.1]`<br>`max_depth`: `[2, 3, 4]`<br>`subsample`: `[0.8, 1.0]` | 54 | 270 | `n_estimators = 100`<br>`learning_rate = 0.05`<br>`max_depth = 3`<br>`subsample = 1.0` |
| **Random Forest** | `n_estimators`: `[50, 100, 150]`<br>`max_depth`: `[None, 5, 8]`<br>`min_samples_split`: `[2, 4]`<br>`criterion`: `['gini', 'entropy']` | 36 | 180 | `n_estimators = 50`<br>`max_depth = None`<br>`min_samples_split = 2`<br>`criterion = 'gini'` |
| **Support Vector Machine** | `C`: `[0.1, 1.0, 10.0]`<br>`gamma`: `['scale', 'auto', 0.01, 0.1]`<br>`kernel`: `['rbf', 'linear']` | 24 | 120 | `C = 1.0`<br>`gamma = 'scale'`<br>`kernel = 'rbf'` |

### 3.3.2. So sánh kết quả mô hình trước và sau khi tinh chỉnh

Bảng đối chiếu hiệu năng trước và sau khi tinh chỉnh bằng `GridSearchCV` (trích xuất từ `reports/baseline_cv_metrics.csv` và `reports/tuned_test_metrics.csv`):

| Mô hình | CV Recall (Trước Tuning) | CV Recall (Sau Tuning) | Mức cải thiện CV Recall | Test Accuracy ($n=80$) | Test Recall ($n=80$) | Test Precision ($n=80$) | Test F1-Score ($n=80$) | Test ROC-AUC ($n=80$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost** | 0,9900 | **0,9950** | **+0,0050** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| **Random Forest** | 0,9900 | **0,9950** | **+0,0050** | 0,9875 | 0,9800 | 1,0000 | 0,9899 | 1,0000 |
| **Support Vector Machine** | 1,0000 | **1,0000** | 0,0000 | 0,9875 | 0,9800 | 1,0000 | 0,9899 | 1,0000 |

Việc giảm tốc độ học (`learning_rate`) của **XGBoost** từ `0.1` (Baseline) xuống `0.05` kết hợp chiều sâu cây `max_depth = 3` và `n_estimators = 100` giúp quá trình cập nhật trọng số theo đạo hàm bậc hai mượt hơn, nâng độ nhạy 5-Fold CV từ `0,9900` lên `0,9950` và đạt độ nhạy tuyệt đối `1,0000` (50/50 ca CKD) trên tập kiểm thử độc lập. Tương tự, **Random Forest** với `n_estimators = 50` cũng nâng CV Recall từ `0,9900` lên `0,9950`.

---

## 3.4. Đóng gói và Lưu trữ mô hình hoàn chỉnh (Model Serialization & Smoke Test)

### 3.4.1. Hợp nhất Preprocessing Pipeline và Mô hình phân lớp tối ưu

Sau khi hoàn tất giai đoạn lựa chọn siêu tham số, toàn bộ Pipeline thống nhất của mô hình tối ưu (`XGBoost (Tuned)`) được huấn luyện lại (`.fit()`) trên toàn bộ $320$ mẫu của tập `X_train_raw.csv` và xuất ra định dạng nhị phân tại `models/best_model.joblib`:

```python
best_pipeline = Pipeline([
    ('preprocessor', ckd_preprocessing.make_preprocessor()),
    ('classifier', XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=1.0, eval_metric='logloss', random_state=42
    ))
])
best_pipeline.fit(X_train_raw, y_train)
joblib.dump(best_pipeline, 'models/best_model.joblib')
```

Cấu trúc đóng gói này đảm bảo nguyên tắc **"Single Artifact Deployment"**: ứng dụng Web ở Chương 5 chỉ cần nạp duy nhất một file `models/best_model.joblib` và truyền vào `pandas.DataFrame` gồm 24 cột thô; mọi bước làm sạch chuỗi, kiểm tra miền giá trị, điền khuyết, mã hóa One-Hot 38 chiều, chuẩn hóa Z-score và dự báo xác suất đều diễn ra tự động bên trong Pipeline.

### 3.4.2. Kiểm thử tải lại mô hình tự động (Automated Smoke Test Suite)

Để đảm bảo mô hình đóng gói hoạt động độc lập, chính xác và an toàn trước khi bàn giao cho Hoàng Xuân Phi Long (Chương 4) và Nguyễn Hoài Phú (Chương 5), bộ kiểm thử tự động `tests/test_smoke_best_model.py` gồm 6 bài kiểm tra (`unittest` / `pytest`) đã được thiết lập và vượt qua 100%:
1. `test_pipeline_structure`: Xác nhận đối tượng nạp từ `models/best_model.joblib` là `sklearn.pipeline.Pipeline` có đủ hai bước `'preprocessor'` và `'classifier'`, hỗ trợ cả `.predict()` và `.predict_proba()`.
2. `test_inference_ckd_case`: Kiểm tra suy luận trên hồ sơ bệnh nhân CKD điển hình trả về nhãn `1` với xác suất $P(y=1|x) > 0,70$.
3. `test_inference_healthy_case`: Kiểm tra suy luận trên hồ sơ người khỏe mạnh trả về nhãn `0` với xác suất $P(y=0|x) > 0,70$.
4. `test_resilience_to_missing_values`: Kiểm tra khả năng tự động xử lý bản ghi bị khuyết nhiều thuộc tính (`np.nan`) mà không phát sinh lỗi runtime.
5. `test_rejection_of_schema_violation`: Xác nhận `CKDCleaner` lập tức từ chối và ném ngoại lệ `ValueError` khi đầu vào thiếu cột, thừa cột hoặc sai tên thuộc tính.
6. `test_latency_performance`: Xác nhận thời gian suy luận trung bình mỗi bệnh nhân đạt dưới $50\text{ ms}$ (thực tế đạt $< 8\text{ ms}$), sẵn sàng phục vụ thời gian thực qua RESTful API.

---

## Tài liệu tham khảo

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
3. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

