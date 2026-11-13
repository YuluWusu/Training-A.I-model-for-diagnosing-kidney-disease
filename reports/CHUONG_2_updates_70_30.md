# BẢNG ĐỐI CHIẾU CẬP NHẬT SỐ LIỆU CHƯƠNG 2 (CHUYỂN TỪ 80/20 SANG 70/30)

Tài liệu này liệt kê **chính xác từng con số thay đổi ở Chương 2** khi chuyển tỷ lệ chia tập dữ liệu từ **80/20 (`test_size=0.20`, 320 Train / 80 Test)** sang **70/30 (`test_size=0.30`, 280 Train / 120 Test, giữ nguyên `stratify=y` và `random_state=42`)** để bạn sửa nhanh trong file Word báo cáo.

> **Lưu ý quan trọng:** Bạn cũng có thể copy trực tiếp từ file [CHUONG_2_HuynhCongTao.md](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/CHUONG_2_HuynhCongTao.md) vì toàn bộ file đó đã được cập nhật sẵn theo tỷ lệ 70/30 mới. Toàn bộ 6 biểu đồ (`01_class_split.png` đến `06_categorical_train.png`) trong `reports/figures/` cũng đã được vẽ lại trên tập Train 280 mẫu / Test 120 mẫu.

---

## 1. Những mục KHÔNG THAY ĐỔI ở Chương 2 (Giữ nguyên 100%)
Các mục kiểm kê chất lượng trên toàn bộ 400 mẫu ban đầu **không hề thay đổi**:
- **Mục 2.1 (Tổng quan & Từ điển thuộc tính):** 400 bản ghi, 24 thuộc tính dự báo (11 biến định lượng, 13 biến định loại gồm 10 biến nhị phân và 3 biến nhiều mức `sg`, `al`, `su`), mã `SHA-256 = 835c57ec100db2d3751165fae615fec425caf444cca23bab6a853d4c08bf5556`.
- **Mục 2.2.1 (Phân phối nhãn toàn cục):** 250 CKD (`62,5%`) và 150 Not CKD (`37,5%`), tỷ lệ `250/150 ≈ 1,67`.
- **Mục 2.2.2 (Bảng kiểm kê khuyết thiếu toàn cục 400 mẫu):** Tổng cộng `1.012 / 9.600` ô thiếu (`10,54%`), `158/400` dòng đủ 24 thuộc tính, xóa dòng thiếu (`dropna`) mất `242/400` dòng (`60,5%`). Top 3 cột thiếu nhiều nhất toàn cục vẫn là `rbc` (152 ô, `38,00%`), `rc` (131 ô, `32,75%`), `wc` (106 ô, `26,50%`).
- **Mục 2.2.3 (Lỗi định dạng cú pháp):** Bảng `format_audit.csv` không đổi.
- **Mục 2.3.4 (Mã hóa One-Hot):** `24 đầu vào → 38 đầu ra = 11 + 10 + 5 + 6 + 6` không đổi.
- **Mục 2.3.2 (Mode của 13 biến phân loại trên Train - `learned_modes.csv`):** Cả 13 biến phân loại đều giữ nguyên `mode_train` cũ (`rbc=normal`, `pc=normal`, `pcc=notpresent`, `ba=notpresent`, `htn=no`, `dm=no`, `cad=no`, `appet=good`, `pe=no`, `ane=no`, `sg=1.02`, `al=0`, `su=0`).

---

## 2. Các đoạn văn cần sửa nhanh trong Word Chương 2 (Find & Replace)

### A. Mục 2.2.4 — Phân phối biến số và ngoại lai (trên tập Train)
- **Câu cũ (80/20):**
  > Mẫu số của tỷ lệ ngoại lai là số giá trị không thiếu của biến, không phải mặc định 320.
  > ...
  > Trong lần chạy hiện tại, sc có 42/304 giá trị quan sát bị đánh dấu (13,82%), bgr có 33/282 (11,70%), bp có 31/309 (10,03%). Các giá trị lớn nhất trên Train lần lượt gồm sc = 76, bgr = 490 và bu = 322; sod có giá trị nhỏ nhất 4,5.
- **Câu mới (70/30) — Thay bằng:**
  > Mẫu số của tỷ lệ ngoại lai là số giá trị không thiếu của biến, không phải mặc định **280**.
  > ...
  > Trong lần chạy hiện tại (trên tập Train **280 mẫu**), **bgr có 33/248 giá trị quan sát bị đánh dấu (13,31%), sc có 32/265 (12,08%), bp có 28/272 (10,29%), bu có 26/262 (9,92%)**. Các giá trị lớn nhất trên Train lần lượt gồm sc = 76, bgr = 490 và bu = 322; sod có giá trị nhỏ nhất 4,5.

### B. Mục 2.2.5 — Tương quan Pearson & Spearman (trên tập Train)
- **Câu cũ (80/20):**
  > Kết quả Pearson trên Train cho thấy hemo–pcv đạt 0,897 với 260 cặp quan sát, pcv–rc đạt 0,815 với 212 cặp, hemo–rc đạt 0,814 với 211 cặp. Cặp sc–sod có Pearson −0,812 với 242 cặp nhưng Spearman chỉ khoảng −0,54.
- **Câu mới (70/30) — Thay bằng:**
  > Kết quả Pearson trên Train (**280 mẫu**) cho thấy **hemo–pcv đạt 0,898 với 231 cặp quan sát, pcv–rc đạt 0,817 với 191 cặp, hemo–rc đạt 0,811 với 190 cặp**. Cặp **sc–sod có Pearson −0,815 với 211 cặp** nhưng Spearman chỉ khoảng **−0,532 (xấp xỉ −0,53)**. Ngoài ra, cặp **bu–hemo đạt Pearson −0,601 với 232 cặp** (Spearman −0,587) và **bu–rc đạt Pearson −0,596 với 184 cặp**.

### C. Mục 2.3.6 — Chia tập phân tầng (Bảng Train/Test)
- **Câu & Bảng cũ (80/20):**
  > Dữ liệu được chia Train 80%, Test 20%, random_state=42 và stratify=y.
  > - Train: 120 notckd | 200 ckd | 320 Tổng
  > - Test: 30 notckd | 50 ckd | 80 Tổng
  > - "...toàn bộ 320 dòng trước CV."
- **Câu & Bảng mới (70/30) — Thay bằng:**
  > Dữ liệu được chia **Train 70%, Test 30%**, `random_state=42` và `stratify=y`. Kết quả với file hiện tại:
  >
  > | Tập | notckd (0) | ckd (1) | Tổng | Tỷ lệ |
  > |---|---:|---:|---:|---:|
  > | **Train** | **105** | **175** | **280** | **70,0%** |
  > | **Test** | **45** | **75** | **120** | **30,0%** |
  > | **Tổng** | **150** | **250** | **400** | **100,0%** |
  >
  > - "...không dùng ma trận đã được fit tiền xử lý trên toàn bộ **280 dòng** trước CV. Trong từng fold của 5-Fold Stratified CV, Pipeline được fit lại chỉ trên **224 mẫu huấn luyện** của fold và kiểm định trên **56 mẫu validation** (gồm đúng **35 CKD và 21 Not CKD** mỗi fold)."

### D. Mục 2.4.3 & Kết luận chương — Kích thước ma trận và các file bàn giao
- `preprocessor.joblib`: Pipeline fit trên **280 dòng Train** (thay vì 320 dòng).
- `data/processed/X_train.csv`: Kích thước **`280 × 38`**, không NaN/Inf (thay vì `320 × 38`).
- `data/processed/X_test.csv`: Kích thước **`120 × 38`**, không NaN/Inf (thay vì `80 × 38`).
- `data/processed/X_train_raw.csv`: Kích thước **`280 × 24`**, sạch cú pháp, còn NaN (thay vì `320 × 24`).
- `data/processed/X_test_raw.csv`: Kích thước **`120 × 24`**, sạch cú pháp, còn NaN (thay vì `80 × 24`).
- **Kết luận chương:** *"Dữ liệu được chia phân tầng thành **280 dòng Train và 120 dòng Test**."*

---

## 3. Bảng chi tiết số lượng ngoại lai IQR trên tập Train 280 mẫu (`reports/tables/outliers_iqr_train.csv`)

| Biến số (`feature`) | Q1 (Train 280) | Q3 (Train 280) | Ngưỡng dưới (`lower`) | Ngưỡng trên (`upper`) | Số ngoại lai (`flagged` mới 280) | Số quan sát (`observed` mới 280) | Tỷ lệ ngoại lai mới (`%`) | *(Đối chiếu cũ 320: `flagged / observed (%))`* | Min (Train) | Max (Train) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`bgr`** | 100,00 | 150,75 | 23,875 | 226,875 | **33** | **248** | **13,31%** | *(33 / 282 = 11,70%)* | 22,0 | 490,0 |
| **`sc`** | 0,90 | 3,20 | −2,550 | 6,650 | **32** | **265** | **12,08%** | *(42 / 304 = 13,82%)* | 0,4 | 76,0 |
| **`bp`** | 70,00 | 80,00 | 55,000 | 95,000 | **28** | **272** | **10,29%** | *(31 / 309 = 10,03%)* | 50,0 | 180,0 |
| **`bu`** | 28,00 | 69,50 | −34,250 | 131,750 | **26** | **262** | **9,92%** | *(29 / 301 = 9,63%)* | 1,5 | 322,0 |
| **`sod`** | 135,00 | 142,00 | 124,500 | 152,500 | **11** | **214** | **5,14%** | *(12 / 245 = 4,90%)* | 4,5 | 150,0 |
| **`wc`** | 6300,00 | 9800,00 | 1050,000 | 15050,000 | **7** | **211** | **3,32%** | *(8 / 236 = 3,39%)* | 2200,0 | 26400,0 |
| **`age`** | 43,00 | 64,00 | 11,500 | 95,500 | **5** | **274** | **1,82%** | *(8 / 312 = 2,56%)* | 5,0 | 90,0 |
| **`pot`** | 3,80 | 4,90 | 2,150 | 6,550 | **2** | **213** | **0,94%** | *(2 / 244 = 0,82%)* | 2,5 | 7,6 |
| **`pcv`** | 32,00 | 45,00 | 12,500 | 64,500 | **1** | **233** | **0,43%** | *(1 / 262 = 0,38%)* | 9,0 | 54,0 |
| **`hemo`** | 10,30 | 15,00 | 3,250 | 22,050 | **1** | **245** | **0,41%** | *(1 / 277 = 0,36%)* | 3,1 | 17,8 |
| **`rc`** | 3,90 | 5,50 | 1,500 | 7,900 | **0** | **193** | **0,00%** | *(0 / 215 = 0,00%)* | 2,1 | 6,5 |

---

## 4. Bảng tham số học được trên tập Train 280 mẫu (`reports/tables/learned_numeric_parameters.csv`)

| Biến số (`feature`) | `median_train` mới (280 mẫu) | `mean_after_imputation` mới (280 mẫu) | `scale_after_imputation` mới ($\sigma_{\text{train}}$, 280 mẫu) | *(Đối chiếu cũ 320 mẫu: Median / Mean / Scale)* |
|---|---:|---:|---:|---|
| **`age`** | **54,0** | **51,8429** | **16,2618** | *(54,0 / 51,1594 / 16,9201)* |
| **`bp`** | **80,0** | **76,8571** | **14,2721** | *(80,0 / 76,8750 / 13,9278)* |
| **`bgr`** | **120,0** | **141,7143** | **71,1172** | *(120,5 / 142,1375 / 71,5264)* |
| **`bu`** | **44,0** | **57,8379** | **48,4332** | *(42,0 / 56,5288 / 48,1953)* |
| **`sc`** | **1,3** | **3,1950** | **6,1886** | *(1,2 / 3,0509 / 5,9016)* |
| **`sod`** | **138,0** | **137,1696** | **10,1246** | *(138,0 / 137,2672 / 9,7053)* |
| **`pot`** | **4,3** | **4,3304** | **0,6644** | *(4,3 / 4,3400 / 0,6593)* |
| **`hemo`** | **12,6** | **12,4936** | **2,7677** | *(12,6 / 12,5297 / 2,7609)* |
| **`pcv`** | **40,0** | **38,8536** | **8,2564** | *(40,0 / 38,9156 / 8,1376)* |
| **`wc`** | **8000,0** | **8314,6429** | **2710,9368** | *(8100,0 / 8324,0625 / 2646,7933)* |
| **`rc`** | **4,8** | **4,7361** | **0,8530** | *(4,8 / 4,7438 / 0,8356)* |

---

## 5. Bảng hệ số tương quan mạnh nhất & Số cặp quan sát trên Train 280 mẫu

*(Trích xuất từ `correlation_pearson_train.csv`, `correlation_spearman_train.csv` và `correlation_pair_counts_train.csv`)*

| Hạng | Cặp biến số | Pearson $r$ mới (280 mẫu) | Spearman $\rho$ mới (280 mẫu) | Số cặp quan sát mới ($n_{\text{pair}} / 280$) | *(Đối chiếu cũ 320 mẫu: Pearson $r$ / $n_{\text{pair}}$)* |
|:---:|---|---:|---:|---:|---|
| 1 | **`hemo` – `pcv`** | **+0,8984** *(≈ 0,898)* | **+0,8654** | **231** | *(+0,897 / 260 cặp)* |
| 2 | **`pcv` – `rc`** | **+0,8171** *(≈ 0,817)* | **+0,7816** | **191** | *(+0,815 / 212 cặp)* |
| 3 | **`sc` – `sod`** | **−0,8150** *(≈ −0,815)* | **−0,5316** *(≈ −0,53)* | **211** | *(−0,812 / 242 cặp)* |
| 4 | **`hemo` – `rc`** | **+0,8114** *(≈ 0,811)* | **+0,7686** | **190** | *(+0,814 / 211 cặp)* |
| 5 | **`bu` – `hemo`** | **−0,6011** *(≈ −0,601)* | **−0,5867** | **232** | *(−0,606 / 263 cặp)* |
| 6 | **`bu` – `rc`** | **−0,5961** *(≈ −0,596)* | **−0,5841** | **184** | *(−0,612 / 205 cặp)* |
| 7 | **`bu` – `pcv`** | **−0,5955** *(≈ −0,596)* | **−0,5669** | **222** | *(−0,601 / 250 cặp)* |
| 8 | **`bu` – `sc`** | **+0,5149** *(≈ +0,515)* | **+0,7038** | **262** | *(+0,520 / 298 cặp)* |

---

## 6. Danh sách 6 biểu đồ Chương 2 cần thay thế trong Word

Bạn vào thư mục `reports/figures/` và copy đè 6 hình sau vào Chương 2 trong file Word:
1. [01_class_split.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/01_class_split.png) — Kiểm kê toàn bộ dữ liệu (400 mẫu) và Phân tầng Train (280 mẫu: 105 notckd, 175 ckd) / Test (120 mẫu: 45 notckd, 75 ckd).
2. [02_missing_train.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/02_missing_train.png) — Tỷ lệ và vị trí khuyết thiếu trên tập Train 280 mẫu.
3. [03_histograms_train.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/03_histograms_train.png) — Histogram phân phối 11 biến định lượng trên tập Train 280 mẫu.
4. [04_boxplots_train.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/04_boxplots_train.png) — Boxplot 11 biến định lượng theo nhãn trên tập Train 280 mẫu.
5. [05_correlations_train.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/05_correlations_train.png) — Ma trận tương quan Pearson & Spearman trên tập Train 280 mẫu.
6. [06_categorical_train.png](file:///d:/2%20Personal%20Activities/Study%20Folder/University/N%C4%83m%20Ba/H%E1%BB%8Dc%20m%C3%A1y/%C4%90%E1%BB%93%20%C3%81n%20H%E1%BB%8Dc%20M%C3%A1y/Training-A.I-model-for-diagnosing-kidney-disease-main/reports/figures/06_categorical_train.png) — Biểu đồ cột chồng tỷ lệ lớp trong 13 biến phân loại trên tập Train 280 mẫu.

