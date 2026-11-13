# Training A.I. Model for Diagnosing Chronic Kidney Disease (CKD)

**Đồ Án Học Phần: Học Máy (Machine Learning)**  
**Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) — Khoa Công Nghệ Thông Tin**

---

## 1. Giới thiệu Đề tài & Phân công Nhóm 01

Dự án xây dựng quy trình Học máy có giám sát (Supervised Machine Learning Pipeline) chuẩn mực, chống rò rỉ dữ liệu (Data Leakage-Free) để chẩn đoán sớm **Bệnh Thận Mạn Tính (Chronic Kidney Disease — CKD)** trên bộ dữ liệu lâm sàng UCI (400 bệnh nhân, 24 thuộc tính sinh hóa máu, nước tiểu và bệnh sử). Mục tiêu lâm sàng cốt lõi là **tối đa hóa Độ nhạy (Recall / Sensitivity)** nhằm giảm thiểu tối đa nguy cơ bỏ sót bệnh nhân mắc bệnh (False Negatives).

| STT | Thành viên | Vai trò & Phụ trách | Sản phẩm bàn giao chính trong Repository |
|:---:|---|---|---|
| 1 | **Hoàng Xuân Phi Long** *(Nhóm trưởng)* | Điều phối tổng thể, thực hiện **Chương 4** (Đánh giá chuyên sâu trên tập Test độc lập 120 mẫu, phân tích lỗi cấp độ bệnh nhân, giải thích Feature Importance, thực nghiệm đối sánh Data Leakage trên Kaggle) | `src/evaluate_chapter4.py`, `notebook/evaluation_chapter4.ipynb`, `reports/CHUONG_4_HoangXuanPhiLong.md`, `reports/tables/ch4_*.csv`, `reports/figures/ch4_*.png` |
| 2 | **Huỳnh Công Tạo** | Thực hiện **Chương 2** (Khảo sát dữ liệu lâm sàng EDA, trực quan hóa phân phối & tương quan, thiết kế `CKDCleaner` và `make_preprocessor` chống rò rỉ dữ liệu trên cấu hình chia 70/30) | `ckd_preprocessing.py`, `feature_contract.json`, `models/preprocessor.joblib`, `data/processed/`, `notebook/NoteBook_EDA_HuynhCongTao.ipynb`, `reports/CHUONG_2_HuynhCongTao.md`, `reports/CHUONG_2_updates_70_30.md` |
| 3 | **Tô Hồng Vũ** | Thực hiện **Chương 3** (Huấn luyện 6 mô hình Baseline với 5-Fold Stratified CV trên 280 mẫu Train, tinh chỉnh siêu tham số `GridSearchCV`, đóng gói `best_model.joblib` và Smoke Test) | `src/export_reports.py`, `notebook/training_model.ipynb`, `models/best_model.joblib`, `reports/CHUONG_3_ToHongVu.md`, `tests/test_smoke_best_model.py` |
| 4 | **Nguyễn Hoài Phú** | Thực hiện **Chương 5** (Xây dựng ứng dụng Web chẩn đoán tương tác tích hợp `models/best_model.joblib` và kiểm thử lâm sàng) | Tích hợp trực tiếp `models/best_model.joblib`, `ckd_preprocessing.py` và `feature_contract.json` |

---

## 2. Kiến trúc Pipeline Chống Rò rỉ Dữ liệu (Leakage-Free Protocol)

1. **Chia tập phân tầng độc lập (70/30 Stratified Split, `test_size=0.30`, `seed=42`):**
   - **Tập Huấn luyện (`X_train_raw.csv`, `y_train.csv`):** 280 bản ghi (175 ca `ckd=1`, 105 ca `notckd=0`).
   - **Tập Kiểm thử độc lập (`X_test_raw.csv`, `y_test.csv`):** 120 bản ghi (75 ca `ckd=1`, 45 ca `notckd=0`), được cô lập hoàn toàn và chỉ dùng để kiểm định ở Chương 3 & Chương 4.
2. **Tiền xử lý nhúng trong từng Fold CV (`ckd_preprocessing.py`):**
   - `CKDCleaner`: Chuẩn hóa chuỗi, xử lý ký tự ẩn (`\t`, `?`), kiểm tra nghiêm ngặt hợp đồng 24 thuộc tính đầu vào (`feature_contract.json`) và miền giá trị hợp lệ.
   - `ColumnTransformer`:
     - **11 biến định lượng (`NUMERIC`):** Điền khuyết bằng trung vị (`SimpleImputer(strategy='median')`) $\to$ Chuẩn hóa Z-score (`StandardScaler`).
     - **10 biến nhị phân (`BINARY`):** Điền khuyết bằng yếu vị (`SimpleImputer(strategy='most_frequent')`) $\to$ `OneHotEncoder(drop='if_binary')`.
     - **3 biến thứ bậc nhiều mức (`MULTI`: `sg`, `al`, `su`):** Điền khuyết bằng `most_frequent` $\to$ `OneHotEncoder` bảo toàn các mức $\to$ Ma trận đầu ra **38 đặc trưng** (`280 × 38` trên Train và `120 × 38` trên Test).

---

## 3. Kết quả Thực nghiệm Tổng hợp (Chương 3 & Chương 4 — Tỷ lệ 70/30)

### 3.1. Kết quả 5-Fold Stratified Cross-Validation trên tập Train ($n = 280$)

| Thuật toán (Baseline) | Accuracy (Mean ± Std) | Precision (Mean) | Recall / Sensitivity (Mean) | F1-Score (Mean) | ROC-AUC (Mean) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (L2)** | 0,9893 ± 0,0087 | 0,9944 | **0,9886** | 0,9914 | 0,9997 |
| **Random Forest (100 trees)** | 0,9786 ± 0,0208 | 0,9828 | **0,9829** | 0,9828 | 0,9992 |
| **Support Vector Machine (RBF)** | 0,9857 ± 0,0175 | 1,0000 | **0,9771** | 0,9882 | 1,0000 |
| **XGBoost (Boosting)** | 0,9679 ± 0,0134 | 0,9776 | **0,9714** | 0,9742 | 0,9962 |
| **Decision Tree (CART)** | 0,9536 ± 0,0290 | 0,9821 | **0,9429** | 0,9617 | 0,9571 |
| **K-Nearest Neighbors (k=5)** | 0,9357 ± 0,0143 | 1,0000 | **0,8971** | 0,9456 | 0,9839 |

### 3.2. Kết quả Kiểm định trên Tập Test Độc lập ($n = 120$: 75 CKD / 45 Not CKD, kèm 95% Bootstrap CI)

| Mô hình | TP | TN | FP | FN | Test Accuracy (95% CI) | Test Recall (95% CI) | Specificity | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Tuned — Best Model)** | **74** | **45** | **0** | **1** | **0,9917** `[0,9750; 1,0000]` | **0,9867** `[0,9558; 1,0000]` | **1,0000** | **0,9933** | **0,9997** | **0,9998** | **0,0088** |
| **Random Forest (Tuned)** | 74 | 44 | 1 | 1 | 0,9833 `[0,9583; 1,0000]` | 0,9867 `[0,9558; 1,0000]` | 0,9778 | 0,9867 | 0,9997 | 0,9998 | 0,0107 |
| **SVM (Tuned)** | 73 | 45 | 0 | 2 | 0,9833 `[0,9583; 1,0000]` | 0,9733 `[0,9315; 1,0000]` | 1,0000 | 0,9865 | 0,9994 | 0,9997 | 0,0148 |

---

## 4. Cấu trúc Thư mục Dự án

```text
Training-A.I-model-for-diagnosing-kidney-disease/
├── ckd_preprocessing.py                # Module CKDCleaner & make_preprocessor() (bắt buộc đi kèm joblib)
├── feature_contract.json               # Hợp đồng 24 thuộc tính đầu vào, 38 thuộc tính đầu ra & miền giá trị
├── verification.json                   # Kết quả tự kiểm định tính toàn vẹn dữ liệu Chương 2
├── requirements.txt                    # Danh sách thư viện và phiên bản chuẩn
├── data/
│   ├── raw/
│   │   └── kidney_disease.csv          # Tập dữ liệu gốc UCI CKD (400 dòng)
│   └── processed/                      # Dữ liệu phân tách 70/30 (Train=280, Test=120) & split_manifest.csv
├── models/
│   ├── preprocessor.joblib             # Pipeline tiền xử lý đã fit trên 280 mẫu Train
│   └── best_model.joblib               # Pipeline hoàn chỉnh (Preprocessor + Tuned XGBoost) dùng cho Web App
├── notebook/
│   ├── NoteBook_EDA_HuynhCongTao.ipynb # Notebook Chương 2: EDA & Tiền xử lý dữ liệu (70/30)
│   ├── training_model.ipynb            # Notebook Chương 3: Huấn luyện 6 Baseline & GridSearchCV (70/30)
│   └── evaluation_chapter4.ipynb       # Notebook Chương 4: Đánh giá độc lập, Phân tích lỗi & Kaggle Audit (70/30)
├── src/
│   ├── export_reports.py               # Script tự động chạy thực nghiệm Chương 3 & xuất best_model.joblib
│   └── evaluate_chapter4.py            # Script tự động chạy toàn bộ phân tích chuyên sâu Chương 4
├── reports/
│   ├── CHUONG_2_HuynhCongTao.md        # Báo cáo chi tiết Chương 2 (70/30)
│   ├── CHUONG_2_updates_70_30.md       # Bảng đối chiếu nhanh các con số Chương 2 (80/20 -> 70/30) để sửa Word
│   ├── CHUONG_3_ToHongVu.md            # Báo cáo chi tiết Chương 3 (70/30)
│   ├── CHUONG_4_HoangXuanPhiLong.md    # Báo cáo chi tiết Chương 4 (70/30)
│   ├── experiment_summary.md           # Tóm tắt thực nghiệm tiếng Anh
│   ├── slide_ready_notes.md            # Ghi chú chuẩn bị Slide bảo vệ đồ án
│   ├── figures/                        # Toàn bộ biểu đồ EDA (01-06), Chương 3 và Chương 4 (ch4_*.png)
│   └── tables/                         # Toàn bộ bảng thống kê CSV của Chương 2 và Chương 4 (ch4_*.csv)
└── tests/
    ├── test_smoke_best_model.py        # Smoke test kiểm tra nạp mô hình, suy luận, xử lý NaN & độ trễ (<50ms)
    └── test_project_integrity.py       # Kiểm thử tích hợp toàn dự án (Chương 2 -> Chương 3 -> Chương 4)
```

---

## 5. Hướng dẫn Cài đặt & Chạy Tái lập Thực nghiệm

```bash
# 1. Cài đặt thư viện
python -m pip install -r requirements.txt

# 2. Chạy pipeline huấn luyện & xuất mô hình (Chương 3)
python src/export_reports.py

# 3. Chạy pipeline đánh giá chuyên sâu, phân tích lỗi & đối sánh Kaggle (Chương 4)
python src/evaluate_chapter4.py

# 4. Chạy toàn bộ bộ kiểm thử tự động (10/10 tests)
python -m pytest -v
```

---

## 6. Hướng dẫn Tích hợp cho Chương 5 (Ứng dụng Web — Nguyễn Hoài Phú)

File `models/best_model.joblib` đã đóng gói **cả bộ tiền xử lý (`CKDCleaner` + `ColumnTransformer`) và mô hình `XGBClassifier`**. Backend API chỉ cần truyền trực tiếp `pandas.DataFrame` có đúng 24 cột thô theo `feature_contract.json`:

```python
import joblib
import pandas as pd
import ckd_preprocessing  # Bắt buộc import để nạp lớp CKDCleaner

model = joblib.load("models/best_model.joblib")

patient_df = pd.DataFrame([{
    "age": 55.0, "bp": 80.0, "sg": "1.015", "al": "2", "su": "0",
    "rbc": "normal", "pc": "abnormal", "pcc": "present", "ba": "notpresent",
    "bgr": 140.0, "bu": 58.0, "sc": 3.4, "sod": 135.0, "pot": 4.9,
    "hemo": 9.8, "pcv": 32.0, "wc": 7800.0, "rc": 3.9,
    "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes"
}])

pred_label = int(model.predict(patient_df)[0])          # 1: CKD, 0: Not CKD
ckd_prob = float(model.predict_proba(patient_df)[0, 1]) # Xác suất mắc CKD (0.0 -> 1.0)
```
