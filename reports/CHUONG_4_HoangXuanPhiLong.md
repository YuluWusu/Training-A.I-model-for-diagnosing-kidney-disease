# CHƯƠNG 4: ĐÁNH GIÁ CHUYÊN SÂU, PHÂN TÍCH LỖI VÀ ĐỐI SÁNH KẾT QUẢ

**Người thực hiện: Hoàng Xuân Phi Long (Nhóm trưởng) — Nhóm 01**

Chương này thực hiện kiểm định độc lập toàn bộ các mô hình phân lớp đã huấn luyện và tinh chỉnh ở Chương 3 trên tập kiểm thử tách biệt (`X_test_raw.csv`, `y_test.csv`, gồm 80 bệnh nhân hoàn toàn không tham gia vào quá trình học tham số tiền xử lý hay tìm kiếm siêu tham số). Bên cạnh các chỉ số phân lớp chuẩn, chương này bổ sung khoảng tin cậy 95% theo phương pháp Bootstrap, kiểm định độ ổn định qua 25-Fold Repeated Stratified Cross-Validation, phân tích đường cong ROC và Precision-Recall, khảo sát bài toán điều chỉnh ngưỡng xác suất quyết định ($\tau$), truy vết chi tiết từng hồ sơ bệnh nhân bị phân loại sai thông qua `split_manifest.csv`, giải thích mức độ đóng góp của 24 đặc trưng lâm sàng và thực hiện thí nghiệm định lượng sai lệch do rò rỉ dữ liệu (Data Leakage) thường gặp trên Kaggle. Toàn bộ mã nguồn tái lập chương này được đóng gói tại `src/evaluate_chapter4.py` và `notebook/evaluation_chapter4.ipynb`.

---

## 4.1. Kết quả đánh giá toàn diện trên tập Kiểm thử độc lập (Independent Test Set)

### 4.1.1. Bảng so sánh tổng hợp các chỉ số: Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC

Tập kiểm thử độc lập gồm $n = 80$ bản ghi được chia phân tầng (`stratify=y`, `random_state=42`) từ 400 mẫu ban đầu tại Chương 2, bao gồm đúng **50 ca dương tính (`ckd = 1`, chiếm 62,5%)** và **30 ca âm tính (`notckd = 0`, chiếm 37,5%)**. Để đánh giá toàn diện hiệu quả trước và sau khi tinh chỉnh siêu tham số, cả 3 mô hình đã tinh chỉnh (`XGBoost (Tuned)`, `Random Forest (Tuned)`, `SVM (Tuned)`) cùng 6 mô hình cơ sở (`Baseline`) đều được nạp qua Pipeline đầy đủ (`CKDCleaner` + `ColumnTransformer` + `Classifier`) và suy luận trực tiếp trên 24 thuộc tính thô của `X_test_raw.csv` ở ngưỡng xác suất mặc định $\tau = 0,50$.

Vì kích thước tập kiểm thử là $n = 80$, việc chỉ báo cáo một điểm ước lượng duy nhất có thể chưa phản ánh hết biên độ dao động mẫu. Do đó, nhóm thực hiện lấy mẫu lại có hoàn lại **1.000 lần (Non-parametric Bootstrap, $B = 1.000$)** trên tập Test để xác định khoảng tin cậy 95% (95% Confidence Interval — CI), đồng thời bổ sung chỉ số **F2-Score** (trọng số ưu tiên Recall gấp đôi Precision theo công thức $F_\beta$ với $\beta = 2$), **PR-AUC** (Average Precision) và **Brier Score** (đo độ chuẩn xác của xác suất dự báo $\hat{p}$). Kết quả chi tiết được trích xuất từ `reports/tables/ch4_test_comprehensive_metrics.csv`:

| Mô hình | Giai đoạn | TP | TN | FP | FN | Accuracy (95% CI) | Precision | Recall / Sensitivity (95% CI) | Specificity (95% CI) | F1-Score (95% CI) | F2-Score | ROC-AUC | PR-AUC | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** | **Tuned** | **50** | **30** | **0** | **0** | **1,0000** `[1,0000; 1,0000]` | **1,0000** | **1,0000** `[1,0000; 1,0000]` | **1,0000** `[1,0000; 1,0000]` | **1,0000** `[1,0000; 1,0000]` | **1,0000** | **1,0000** | **1,0000** | **0,0054** |
| **SVM (Tuned)** | **Tuned** | 49 | 30 | 0 | 1 | 0,9875 `[0,9500; 1,0000]` | 1,0000 | 0,9800 `[0,9302; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9899 `[0,9638; 1,0000]` | 0,9839 | 1,0000 | 1,0000 | 0,0084 |
| **Random Forest (Tuned)** | **Tuned** | 49 | 30 | 0 | 1 | 0,9875 `[0,9500; 1,0000]` | 1,0000 | 0,9800 `[0,9302; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9899 `[0,9638; 1,0000]` | 0,9839 | 1,0000 | 1,0000 | 0,0126 |
| **XGBoost (Baseline)** | Baseline | 50 | 30 | 0 | 0 | 1,0000 `[1,0000; 1,0000]` | 1,0000 | 1,0000 `[1,0000; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 1,0000 | 1,0000 | 1,0000 | 0,0044 |
| **SVM (Baseline)** | Baseline | 49 | 30 | 0 | 1 | 0,9875 `[0,9500; 1,0000]` | 1,0000 | 0,9800 `[0,9302; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9899 `[0,9638; 1,0000]` | 0,9839 | 1,0000 | 1,0000 | 0,0084 |
| **Random Forest (Baseline)** | Baseline | 49 | 30 | 0 | 1 | 0,9875 `[0,9500; 1,0000]` | 1,0000 | 0,9800 `[0,9302; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9899 `[0,9638; 1,0000]` | 0,9839 | 1,0000 | 1,0000 | 0,0137 |
| **Logistic Regression (Baseline)** | Baseline | 49 | 30 | 0 | 1 | 0,9875 `[0,9500; 1,0000]` | 1,0000 | 0,9800 `[0,9302; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9899 `[0,9638; 1,0000]` | 0,9839 | 0,9987 | 0,9992 | 0,0183 |
| **Decision Tree (Baseline)** | Baseline | 47 | 30 | 0 | 3 | 0,9625 `[0,9125; 1,0000]` | 1,0000 | 0,9400 `[0,8636; 1,0000]` | 1,0000 `[1,0000; 1,0000]` | 0,9691 `[0,9268; 1,0000]` | 0,9514 | 0,9700 | 0,9775 | 0,0375 |
| **K-Nearest Neighbors (Baseline)** | Baseline | 43 | 30 | 0 | 7 | 0,9125 `[0,8500; 0,9628]` | 1,0000 | 0,8600 `[0,7599; 0,9446]` | 1,0000 `[1,0000; 1,0000]` | 0,9247 `[0,8636; 0,9715]` | 0,8848 | 0,9887 | 0,9913 | 0,0525 |

Bên cạnh kiểm định trên tập Test, để đánh giá độ ổn định của các mô hình khi cách chia fold thay đổi, nhóm thực hiện thêm **5×5 Repeated Stratified K-Fold Cross-Validation (tổng cộng 25 folds độc lập)** trên tập Train (`X_train_raw.csv`, $n = 320$). Kết quả từ `reports/tables/ch4_repeated_cv_stability.csv` cho thấy:

| Mô hình | Số Folds | Recall (Mean ± Std) | Recall (Min) | ROC-AUC (Mean ± Std) | F1-Score (Mean ± Std) | Accuracy (Mean ± Std) |
|---|---:|---:|---:|---:|---:|---:|
| **SVM (Tuned)** | 25 | 0,9970 ± 0,0081 | 0,9750 | 0,9999 ± 0,0003 | 0,9975 ± 0,0061 | 0,9969 ± 0,0076 |
| **Logistic Regression (Baseline)** | 25 | 0,9930 ± 0,0112 | 0,9750 | 0,9996 ± 0,0010 | 0,9920 ± 0,0092 | 0,9900 ± 0,0115 |
| **Random Forest (Tuned)** | 25 | 0,9920 ± 0,0117 | 0,9750 | 0,9991 ± 0,0019 | 0,9905 ± 0,0089 | 0,9881 ± 0,0112 |
| **XGBoost (Tuned)** | 25 | 0,9830 ± 0,0196 | 0,9500 | 0,9979 ± 0,0031 | 0,9820 ± 0,0140 | 0,9775 ± 0,0175 |
| **Decision Tree (Baseline)** | 25 | 0,9600 ± 0,0274 | 0,9000 | 0,9617 ± 0,0192 | 0,9686 ± 0,0169 | 0,9613 ± 0,0205 |
| **K-Nearest Neighbors (Baseline)** | 25 | 0,9150 ± 0,0406 | 0,8250 | 0,9920 ± 0,0078 | 0,9542 ± 0,0229 | 0,9456 ± 0,0264 |

### 4.1.2. Phân tích tiêu chí chọn mô hình cuối cùng: Ưu tiên tối đa hóa độ nhạy (Recall) và tính minh bạch trong lựa chọn

Trong bài toán sàng lọc bệnh thận mạn tính (CKD), hai loại sai lầm mang hậu quả lâm sàng hoàn toàn bất đối xứng:
- **Dương tính giả (False Positive — FP):** Người khỏe mạnh bị cảnh báo nghi ngờ CKD sẽ được chỉ định làm thêm xét nghiệm máu/nước tiểu chuyên sâu hoặc siêu âm thận để loại trừ. Chi phí phát sinh chủ yếu là chi phí xét nghiệm bổ sung và tâm lý lo lắng ngắn hạn.
- **Âm tính giả (False Negative — FN):** Bệnh nhân thực sự mắc bệnh thận mạn tính nhưng bị mô hình kết luận là "Khỏe mạnh" (`Not CKD`), dẫn đến bỏ lỡ giai đoạn điều trị bảo tồn, khiến chức năng cầu thận suy giảm không hồi phục sang suy thận giai đoạn cuối (ESRD), phải chạy thận nhân tạo hoặc ghép thận.

Do đó, tiêu chí số một của hệ thống là **tối đa hóa Độ nhạy ($\text{Recall} = \frac{TP}{TP + FN} \to 1,0000$, tương đương $FN \to 0$)**.

Khi đối chiếu kết quả giữa Cross-Validation trên tập Train (Chương 3) và kiểm định độc lập trên tập Test (Chương 4), cần ghi nhận một điểm phương pháp luận quan trọng:
1. **Trên tập Train (5-Fold CV ở Chương 3):** Ba mô hình dẫn đầu sau tinh chỉnh có độ nhạy CV gần như tương đương nhau tuyệt đối: `SVM (Tuned)` đạt $\text{CV Recall} = 1,0000$ (200/200 ca CKD), còn `XGBoost (Tuned)` và `Random Forest (Tuned)` cùng đạt $\text{CV Recall} = 0,9950$ (199/200 ca CKD, tức chỉ chênh đúng 1 bệnh nhân trên 320 mẫu huấn luyện).
2. **Trên tập Test độc lập ($n = 80$):** Cả 3 mô hình đã tinh chỉnh đều đạt diện tích dưới đường cong **$\text{ROC-AUC} = 1,0000$ và $\text{PR-AUC} = 1,0000$**, chứng tỏ khả năng xếp hạng xác suất phân tách hai lớp hoàn hảo. Tuy nhiên, tại ngưỡng phân lớp mặc định $\tau = 0,50$, chỉ có **`XGBoost` đạt $\text{Test Recall} = 1,0000$ ($50/50$ ca CKD, $FN = 0$)** cùng chỉ số sai số xác suất thấp nhất ($\text{Brier Score} = 0,0054$), trong khi `Random Forest (Tuned)` và `SVM (Tuned)` bỏ sót 1 ca bệnh ($49/50$, $\text{Recall} = 0,9800$, $FN = 1$).
3. **Lý do chọn `XGBoost (Tuned)` làm `models/best_model.joblib` cho ứng dụng Web (Chương 5):**
   - **An toàn lâm sàng trên tập kiểm thử độc lập:** Đạt tuyệt đối $FN = 0$ và $FP = 0$ ở ngưỡng chuẩn $\tau = 0,50$, phát hiện được cả ca bệnh CKD giai đoạn sớm ở người trẻ không thiếu máu (`ID = 102`) mà cả 7 mô hình khác bỏ sót (phân tích chi tiết tại Mục 4.4).
   - **Khả năng giải thích y học (Clinical Explainability):** Khác với `SVM` dùng nhân phi tuyến RBF trong không gian 38 chiều (hộp đen đối với bác sĩ lâm sàng), `XGBoost` và `Random Forest` cung cấp trực tiếp mức độ đóng góp đặc trưng (Feature Importance theo độ lợi thông tin Gain), cho phép đối chiếu minh bạch với sinh lý bệnh thận (Hemoglobin, Creatinine huyết thanh, Albumin niệu, Tỷ trọng nước tiểu).
   - **Bền vững trước dữ liệu khuyết thiếu và điểm ngoại lai:** Như đã khảo sát ở Chương 2, tập dữ liệu có các điểm ngoại lai cực đoan (như `sc = 76`, `bgr = 490`, `bu = 322`). Các nhánh rẽ cây quyết định của `XGBoost` bất biến với phép biến đổi đơn điệu và không bị kéo lệch bởi khoảng cách Euclidean như `KNN` hay nhân RBF của `SVM`.

---

## 4.2. Phân tích Ma trận nhầm lẫn (Confusion Matrix Analysis)

### 4.2.1. Trực quan hóa ma trận nhầm lẫn của các mô hình đã tinh chỉnh

Hình `reports/figures/ch4_confusion_matrices_tuned.png` biểu diễn song song ma trận nhầm lẫn của 3 mô hình đã tinh chỉnh (`XGBoost (Tuned)`, `Random Forest (Tuned)` và `SVM (Tuned)`) trên 80 mẫu kiểm thử độc lập:

![Ma trận nhầm lẫn của 3 mô hình đã tinh chỉnh trên tập Test độc lập](figures/ch4_confusion_matrices_tuned.png)

Bảng tổng hợp cấu trúc 4 ô của ma trận nhầm lẫn ($\text{TN}, \text{FP}, \text{FN}, \text{TP}$) cho toàn bộ các mô hình:

| Mô hình | True Negative (TN — Khỏe mạnh đoán đúng) | False Positive (FP — Cảnh báo nhầm) | False Negative (FN — Bỏ sót bệnh nhân CKD) | True Positive (TP — Phát hiện đúng CKD) | Tỷ lệ bỏ sót bệnh ($\text{FNR} = \frac{FN}{50}$) |
|---|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** | **30 / 30 (100,0%)** | **0 / 30 (0,0%)** | **0 / 50 (0,0%)** | **50 / 50 (100,0%)** | **0,0%** |
| **Random Forest (Tuned)** | 30 / 30 (100,0%) | 0 / 30 (0,0%) | 1 / 50 (2,0%) | 49 / 50 (98,0%) | 2,0% |
| **SVM (Tuned)** | 30 / 30 (100,0%) | 0 / 30 (0,0%) | 1 / 50 (2,0%) | 49 / 50 (98,0%) | 2,0% |
| **Logistic Regression (Baseline)** | 30 / 30 (100,0%) | 0 / 30 (0,0%) | 1 / 50 (2,0%) | 49 / 50 (98,0%) | 2,0% |
| **Decision Tree (Baseline)** | 30 / 30 (100,0%) | 0 / 30 (0,0%) | 3 / 50 (6,0%) | 47 / 50 (94,0%) | 6,0% |
| **K-Nearest Neighbors (Baseline)** | 30 / 30 (100,0%) | 0 / 30 (0,0%) | 7 / 50 (14,0%) | 43 / 50 (86,0%) | 14,0% |

### 4.2.2. Đánh giá chi tiết các ca Âm tính giả (False Negative — Bệnh nhân bị bỏ sót)

Quan sát cột $\text{FN}$ cho thấy sự phân hóa rõ rệt giữa các họ thuật toán:
1. **`K-Nearest Neighbors (Baseline)` bỏ sót tới 7/50 bệnh nhân CKD ($\text{FNR} = 14,0\%$):** Đây là hệ quả trực tiếp của hai yếu tố đã được cảnh báo từ Chương 2: (a) trong không gian 38 chiều sau mã hóa One-Hot kết hợp chuẩn hóa Z-score, khoảng cách Euclidean bị chi phối bởi các thuộc tính đầy đủ hoặc đã được điền khuyết bằng trung vị/mode; (b) các bệnh nhân CKD bị khuyết nhiều chỉ số (từ 3 đến 6 cột bị thiếu như `ID = 65`, `ID = 186`, `ID = 224`) sau khi được điền bằng trung vị của tập Train (`hemo = 12,6`, `sc = 1,2`, `rbc = normal`) sẽ có tọa độ vector nằm gần đám mây của nhóm người khỏe mạnh, khiến đa số trong $k = 5$ láng giềng gần nhất mang nhãn `notckd = 0`.
2. **`Decision Tree (Baseline)` bỏ sót 3/50 bệnh nhân CKD ($\text{FNR} = 6,0\%$):** Một cây quyết định đơn lẻ cắt trục tọa độ theo các ngưỡng cứng (`hard axis-aligned splits`). Khi một bệnh nhân CKD có chỉ số `hemo` nằm trên ngưỡng cắt của nút gốc (ví dụ `hemo >= 12,8`, như ca `ID = 102`, `ID = 133`, `ID = 216`) và thiếu chỉ số `sc` (hoặc `sc` được điền bằng median `1,2`), nhánh cây lập tức rơi vào lá `Not CKD` mà không có cơ chế sửa sai từ các cây khác.
3. **`Random Forest (Tuned)`, `SVM (Tuned)` và `Logistic Regression` chỉ bỏ sót duy nhất 1/50 bệnh nhân CKD ($\text{FNR} = 2,0\%$):** Cả ba mô hình này đều chỉ thất bại trên cùng một trường hợp biên đặc biệt là bệnh nhân `ID = 102` (vị trí `Test_Position = 12`).
4. **`XGBoost (Tuned)` triệt tiêu hoàn toàn Âm tính giả ($\text{FN} = 0$, $\text{FNR} = 0,0\%$):** Nhờ cơ chế học tăng cường độ dốc (Gradient Boosting), các cây nối tiếp nhau tập trung trọng số vào các mẫu có đạo hàm bậc nhất và bậc hai của hàm Log-Loss lớn (chính là những ca CKD không điển hình mà các cây đầu tiên dự báo xác suất thấp), giúp kéo xác suất của bệnh nhân `ID = 102` vượt qua ngưỡng quyết định ($\hat{p} = 0,6060 > 0,50$).

### 4.2.3. Đánh giá các ca Dương tính giả (False Positive — Cảnh báo nhầm)

Một kết quả đáng chú ý là **tất cả 9 mô hình đều đạt $\text{FP} = 0$ ($\text{Specificity} = \frac{30}{30} = 100,0\%$)** trên tập kiểm thử độc lập ở ngưỡng $\tau = 0,50$. Lý do là trong bộ dữ liệu UCI CKD, nhóm `notckd` (150 mẫu, trong đó 30 mẫu ở tập Test) có các chỉ số sinh hóa rất đồng nhất và nằm gọn trong khoảng sinh lý bình thường (`sg >= 1.020`, `al = 0`, `su = 0`, `sc <= 1.2`, `hemo >= 13.0`, không có tiền sử tăng huyết áp `htn` hay đái tháo đường `dm`). Ngay cả khi một bản ghi `notckd` bị khuyết giá trị (như `ID = 324` thiếu 4 cột huyết học), phép điền khuyết bằng trung vị và mode của tập Train cũng điền các giá trị sinh lý bình thường (`sg = 1.020`, `al = 0`, `hemo = 12,6`), nên không đẩy bất kỳ người khỏe mạnh nào vượt qua ngưỡng 0,50 trên tập Test.

---

## 4.3. Phân tích đường cong ROC và đường cong Precision-Recall

### 4.3.1. Đồ thị so sánh đường cong ROC-AUC và Precision-Recall giữa các mô hình

Vì tập dữ liệu có tỷ lệ lớp dương tính khá cao ($62,5\%$ CKD so với $37,5\%$ Not CKD), việc kết hợp cả hai đồ thị **ROC (Receiver Operating Characteristic)** và **Precision-Recall (PR)** giúp kiểm chứng khả năng phân tách xác suất trên toàn bộ miền ngưỡng $\tau \in [0, 1]$. Hình `reports/figures/ch4_roc_pr_curves_comparison.png` thể hiện hai đường cong này trên tập Test ($n = 80$):

![Đường cong ROC và Precision-Recall trên tập Test độc lập](figures/ch4_roc_pr_curves_comparison.png)

Các nhận xét rút ra từ đồ thị:
- **Ba mô hình đã tinh chỉnh (`XGBoost`, `Random Forest`, `SVM`) đều đạt diện tích lý tưởng $\text{ROC-AUC} = 1,0000$ và $\text{PR-AUC} = 1,0000$:** Điều này có ý nghĩa toán học rất rõ ràng: tồn tại ít nhất một khoảng ngưỡng xác suất $\tau^*$ mà tại đó $100\%$ bệnh nhân CKD ($50/50$) đều có xác suất dự báo $\hat{p}(y=1|x)$ lớn hơn hoàn toàn so với toàn bộ $30/30$ người khỏe mạnh ($\max_{i \in \text{NotCKD}} \hat{p}_i < \min_{j \in \text{CKD}} \hat{p}_j$).
- **`Logistic Regression (Baseline)` đạt $\text{ROC-AUC} = 0,9987$ và $\text{PR-AUC} = 0,9992$:** Chứng tỏ ranh giới giữa hai lớp trong không gian 38 đặc trưng đã gần như khả tách tuyến tính, chỉ có một vùng chồng lấn rất hẹp ở bệnh nhân `ID = 102` ($\hat{p}_{\text{LR}} = 0,1559$) và ca khỏe mạnh `ID = 324` ($\hat{p}_{\text{LR}} = 0,3198$).
- **`KNN` ($\text{ROC-AUC} = 0,9887$) và `Decision Tree` ($\text{ROC-AUC} = 0,9700$)** có đường cong nằm thấp hơn rõ rệt ở góc trên bên trái (ROC) và góc trên bên phải (PR) do xác suất ước lượng của chúng chỉ nhận các mức rời rạc theo tỷ lệ lá cây hoặc tỷ lệ trong 5 láng giềng ($\{0,0; 0,2; 0,4; 0,6; 0,8; 1,0\}$).

### 4.3.2. Lựa chọn ngưỡng xác suất phân loại tối ưu (Classification Threshold Tuning)

Mặc dù các thư viện học máy mặc định gán nhãn $\hat{y} = \mathbb{I}(\hat{p} \ge 0,50)$, trong các hệ thống hỗ trợ quyết định lâm sàng (Clinical Decision Support Systems — CDSS), ngưỡng phân loại $\tau$ là một siêu tham số vận hành cần được hiệu chỉnh theo mục tiêu y tế. Nhóm đã khảo sát sự biến thiên của Recall, Specificity, Precision, F1-Score, F2-Score và chỉ số Youden's $J = \text{Sensitivity} + \text{Specificity} - 1$ khi quét $\tau$ từ $0,10$ đến $0,90$ (chi tiết trong `reports/tables/ch4_threshold_tuning.csv` và Hình `reports/figures/ch4_threshold_tuning_curves.png`):

![Khảo sát sự biến thiên các chỉ số theo ngưỡng quyết định xác suất tau](figures/ch4_threshold_tuning_curves.png)

Bảng trích xuất tại các mốc ngưỡng chiến lược từ `reports/tables/ch4_threshold_tuning.csv`:

| Mô hình | Ngưỡng $\tau$ | TP | TN | FP | FN | Recall (Sensitivity) | Specificity | Precision | F1-Score | F2-Score | Youden's $J$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **XGBoost (Tuned)** | $0,35$ | 50 | 29 | 1 | 0 | **1,0000** | 0,9667 | 0,9804 | 0,9901 | 0,9960 | 0,9667 |
| **XGBoost (Tuned)** | **$0,40 \to 0,60$** | **50** | **30** | **0** | **0** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| **XGBoost (Tuned)** | $0,70$ | 49 | 30 | 0 | 1 | 0,9800 | 1,0000 | 1,0000 | 0,9899 | 0,9839 | 0,9800 |
| **Random Forest (Tuned)** | $0,35$ | 50 | 29 | 1 | 0 | **1,0000** | 0,9667 | 0,9804 | 0,9901 | 0,9960 | 0,9667 |
| **Random Forest (Tuned)** | **$0,40 \to 0,50$*** | **50** | **30** | **0** | **0** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| **Random Forest (Tuned)** | $0,60$ | 49 | 30 | 0 | 1 | 0,9800 | 1,0000 | 1,0000 | 0,9899 | 0,9839 | 0,9800 |
| **SVM (Tuned)** | $0,20$ | 50 | 29 | 1 | 0 | **1,0000** | 0,9667 | 0,9804 | 0,9901 | 0,9960 | 0,9667 |
| **SVM (Tuned)** | **$0,25$** | **50** | **30** | **0** | **0** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** | **1,0000** |
| **SVM (Tuned)** | $0,30 \to 0,50$ | 49 | 30 | 0 | 1 | 0,9800 | 1,0000 | 1,0000 | 0,9899 | 0,9839 | 0,9800 |

*\*Lưu ý kỹ thuật quan trọng về `Random Forest (Tuned)` tại $\tau = 0,50$:* Trên ca bệnh `ID = 102`, `Random Forest (Tuned)` với 50 cây quyết định có đúng 25/50 cây bỏ phiếu `CKD (1)` và 25/50 cây bỏ phiếu `Not CKD (0)`, tạo ra xác suất đúng bằng $\hat{p} = 0,5000$. Trong `scikit-learn`, hàm `predict()` dùng phép so sánh ngặt `prob > 0.5` (với `argmax` ưu tiên chỉ số lớp 0 khi hai lớp bằng nhau `0.50 == 0.50`), dẫn đến `predict()` trả về nhãn `0` (tạo ra 1 ca FN). Khi áp dụng quy tắc ngưỡng $\hat{p} \ge \tau$ với $\tau \in [0,40; 0,50]$, `Random Forest (Tuned)` lập tức thu hồi được ca bệnh `ID = 102` và đạt $50/50$ ca CKD! Tương tự, với `SVM (Tuned)`, chỉ cần hạ ngưỡng từ $0,50$ xuống $\tau = 0,25$, ca `ID = 102` ($\hat{p}_{\text{SVM}} = 0,2815$) cũng được phát hiện đúng mà không phát sinh bất kỳ ca Dương tính giả nào (do ca `Not CKD` cao nhất trên SVM chỉ có $\hat{p} = 0,2344$).

Trong khi đó, **`XGBoost (Tuned)` có "biên độ an toàn ngưỡng" (Threshold Safety Margin) rộng nhất**: với mọi $\tau \in [0,36; 0,60]$, mô hình đều duy trì đồng thời $\text{Recall} = 100\%$ và $\text{Specificity} = 100\%$ (khoảng phân cách giữa ca `Not CKD` cao nhất $\hat{p}_{\max}^{(0)} = 0,3581$ và ca `CKD` thấp nhất $\hat{p}_{\min}^{(1)} = 0,6060$ rộng tới $\Delta p = 0,2479$).

---

## 4.4. Phân tích lỗi mô hình chuyên sâu (Error Analysis)

### 4.4.1. Trích xuất danh sách các bản ghi dự đoán sai trên tập Test và Out-Of-Fold (OOF)

Nhằm tránh đánh giá bề mặt chỉ qua các con số phần trăm, nhóm đã truy vết từng bản ghi cá biệt bị dự đoán sai bởi bất kỳ mô hình nào trong số 9 mô hình trên tập Test ($n = 80$), kết hợp với các bản ghi bị dự đoán sai trong quá trình kiểm chứng chéo Out-Of-Fold (OOF) trên tập Train ($n = 320$), đối chiếu trực tiếp qua `data/processed/split_manifest.csv` để xác định `Original_ID` và dòng dữ liệu gốc (`Source_Row`). Toàn bộ 28 bản ghi đáng chú ý được lưu trong `reports/tables/ch4_error_analysis_cases.csv` và trực quan hóa tại Hình `reports/figures/ch4_error_cases_clinical_profile.png`:

![Xác suất dự báo của các mô hình trên các ca bệnh bị phân loại sai trong tập Test và Train OOF](figures/ch4_error_cases_clinical_profile.png)

Bảng dưới đây trích xuất toàn bộ **8 ca bệnh trong tập Test** bị ít nhất một mô hình phân loại sai (cùng **1 ca khỏe mạnh biên** `ID = 324`) và **6 ca trong tập Train (OOF)** bị các mô hình tinh chỉnh phân loại sai:

| Tập | Vị trí (`pos`) | `Original_ID` | Nhãn thật | Số cột thiếu (/24) | Các cột bị khuyết gốc | Chỉ số lâm sàng chính (`age`, `bp`, `sg`, `al`, `sc`, `hemo`, `pcv`, `htn`, `dm`) | $\hat{p}_{\text{XGB}}$ | $\hat{p}_{\text{RF}}$ | $\hat{p}_{\text{SVM}}$ | Các mô hình dự đoán sai (ở $\tau=0,50$) |
|---|---:|---:|---|---:|---|---|---:|---:|---:|---|
| **Test** | 12 | **102** | **CKD (1)** | 2 (8,3%) | `rbc, rc` | `age=17`, `bp=60`, `sg=1.010`, `al=0`, **`sc=2.1`**, `hemo=13.9`, `pcv=52`, `htn=no`, `dm=no` | **0,6060** | 0,5000 | 0,2815 | RF (Tuned & Base), SVM (Tuned & Base), LogReg, DT, KNN (7/9 mô hình sai) |
| **Test** | 59 | **133** | **CKD (1)** | 0 (0,0%) | Không thiếu | `age=70`, `bp=100`, `sg=1.015`, `al=4`, `sc=5.3`, `bu=125`, `hemo=12.0`, `pcv=37`, **`rc=8.0`** | 0,9786 | 0,9200 | 0,9934 | Decision Tree, KNN (2/9 mô hình sai) |
| **Test** | 8 | **216** | **CKD (1)** | 6 (25,0%) | `rbc, sc, sod, pot, wc, rc` | `age=64`, `bp=70`, `sg=1.010`, `al=0`, **`sc=NaN`**, `hemo=12.8`, `pcv=38`, `htn=no`, `dm=no` | 0,9241 | 0,8200 | 0,9519 | Decision Tree (1/9 mô hình sai) |
| **Test** | 47 | **65** | **CKD (1)** | 6 (25,0%) | `rbc, bgr, sod, pot, wc, rc` | `age=44`, `bp=90`, `sg=1.010`, `al=1`, `sc=1.1`, `hemo=15.0`, `pcv=48`, `htn=no`, `dm=no` | 0,8279 | 0,6000 | 0,9067 | KNN (1/9 mô hình sai) |
| **Test** | 23 | **186** | **CKD (1)** | 5 (20,8%) | `bgr, hemo, pcv, wc, rc` | `age=8`, `bp=50`, `sg=1.020`, `al=4`, `sc=1.0`, `bu=46`, `hemo=NaN`, `pcv=NaN` | 0,8395 | 0,8000 | 0,9910 | KNN (1/9 mô hình sai) |
| **Test** | 19 | **164** | **CKD (1)** | 3 (12,5%) | `bp, rbc, pc` | `age=14`, `sg=1.015`, `al=0`, `sc=0.8`, `bgr=192`, `hemo=14.3`, `pcv=40`, `dm=yes` | 0,8592 | 0,7400 | 0,9947 | KNN (1/9 mô hình sai) |
| **Test** | 50 | **91** | **CKD (1)** | 0 (0,0%) | Không thiếu | `age=56`, `bp=70`, `sg=1.015`, `al=4`, `su=1`, `sc=1.7`, `bgr=210`, `hemo=16.1`, `pcv=52` | 0,9781 | 0,9000 | 0,9585 | KNN (1/9 mô hình sai) |
| **Test** | 10 | **224** | **CKD (1)** | 5 (20,8%) | `rbc, hemo, pcv, wc, rc` | `age=34`, `bp=60`, `sg=1.020`, `al=0`, `sc=2.2`, `bu=28`, `hemo=NaN`, `pcv=NaN` | 0,9833 | 0,8800 | 0,7364 | KNN (1/9 mô hình sai) |
| **Test** | 27 | **324** | **Not CKD (0)** | 4 (16,7%) | `hemo, pcv, wc, rc` | `age=40`, `bp=80`, `sg=1.020`, `al=0`, `sc=0.7`, `bu=15`, `bgr=119`, `hemo=NaN` | 0,3581 | 0,3800 | 0,2344 | Không mô hình nào sai (Ca âm tính sát ngưỡng nhất) |
| **Train OOF** | 165 | **105** | **CKD (1)** | 1 (4,2%) | `rbc` | `age=65`, `bp=80`, `sg=1.015`, `al=0`, **`sc=11.5`**, `hemo=14.1`, `pcv=42`, `rc=5.2`, `htn=no` | 0,1320 | 0,5000 | 0,8686 | XGBoost (Tuned) & RF (Tuned) trong OOF (1/200 ca CKD) |
| **Train OOF** | 121 | **319** | **Not CKD (0)** | 4 (16,7%) | `hemo, pcv, wc, rc` | `age=30`, `bp=60`, `sg=1.020`, `al=0`, `sc=1.1`, `bu=15`, `bgr=138`, `hemo=NaN`, `pcv=NaN` | 0,9157 | 0,7000 | 0,6074 | Cả 3 mô hình Tuned dự báo FP trong OOF |
| **Train OOF** | 54 | **273** | **Not CKD (0)** | 4 (16,7%) | `hemo, pcv, wc, rc` | `age=47`, `bp=80`, `sg=1.020`, `al=0`, `sc=0.9`, `bu=35`, `bgr=95`, `hemo=NaN`, `pcv=NaN` | 0,9050 | 0,7200 | 0,4866 | XGBoost (Tuned), RF (Tuned) trong OOF |

### 4.4.2. Phân tích nguyên nhân gốc rễ (Do khuyết nhiều chỉ số ban đầu hay do giá trị ở ngưỡng ranh giới)

Từ bảng truy vết hồ sơ bệnh án ở trên, nhóm xác định được **3 cơ chế gây lỗi cốt lõi** trong chẩn đoán học máy trên bộ dữ liệu CKD:

1. **Cơ chế 1 — Thể bệnh CKD giai đoạn sớm ở người trẻ chưa có biến chứng thiếu máu (Điển hình là ca bệnh `ID = 102` trên tập Test và `ID = 105` trên tập Train):**
   - Trong sinh lý bệnh thận mạn, thiếu máu (`hemo` giảm, `pcv` giảm, `rc` giảm) xảy ra khi thận suy giảm nặng khả năng tiết hormone Erythropoietin (thường từ giai đoạn 3b–5). Do đa số mẫu CKD trong bộ dữ liệu UCI là bệnh nhân nhập viện ở giai đoạn muộn nên các mô hình học trọng số rất lớn vào `hemo` và `pcv`.
   - Tuy nhiên, bệnh nhân **`ID = 102` (17 tuổi)** mắc CKD với bằng chứng tổn thương lọc và cô đặc cầu – ống thận rất rõ (`sc = 2,1 mg/dL` cao gấp đôi mức bình thường ở thanh thiếu niên, kèm tỷ trọng nước tiểu thấp `sg = 1,010`), nhưng huyết sắc tố vẫn hoàn toàn bình thường (`hemo = 13,9 g/dL`, `pcv = 52%`), chưa có protein niệu (`al = 0`) và không có bệnh nền (`htn = no`, `dm = no`).
   - Với `Logistic Regression` ($\hat{p} = 0,1559$) và `SVM` ($\hat{p} = 0,2815$), tổ hợp tuyến tính/nhân RBF của hàng loạt chỉ số bình thường (`hemo = 13,9`, `pcv = 52`, `al = 0`, `bu = 32`, `bgr = 92`) đã lấn át hoàn toàn tín hiệu từ `sc = 2,1` và `sg = 1,010`. Ngược lại, `XGBoost` sử dụng các cây tăng cường nối tiếp đã cô lập được nhánh điều kiện phi tuyến `(sg == 1.010) AND (sc > 1.3)` để cảnh báo chính xác ca bệnh này ($\hat{p} = 0,6060$).

2. **Cơ chế 2 — Hiện tượng "Đồng hóa về trung vị" khi bệnh nhân bị khuyết đồng thời cụm xét nghiệm huyết học (`hemo, pcv, wc, rc`):**
   - Hãy quan sát các ca `ID = 324` (Test), `ID = 319` và `ID = 273` (Train OOF) thuộc lớp `Not CKD (0)`, cùng với `ID = 186` và `ID = 224` (Test) thuộc lớp `CKD (1)`. Điểm chung của tất cả các bản ghi này là **bị khuyết toàn bộ 4 chỉ số công thức máu (`hemo, pcv, wc, rc`)**.
   - Khi đi qua `SimpleImputer(strategy='median')` đã fit trên tập Train (trong đó lớp CKD chiếm tới $62,5\%$), các ô trống của một người khỏe mạnh (`Not CKD`) sẽ được điền bằng trung vị của toàn tập Train: **`hemo = 12,6 g/dL`** và **`pcv = 40,0%`**.
   - Cần lưu ý rằng trong tập dữ liệu này, ngưỡng phân tách giữa người khỏe mạnh và bệnh nhân CKD trên biến `hemo` nằm ngay quanh vùng $12,5 - 13,0\text{ g/dL}$ (hầu hết người khỏe mạnh thực sự có `hemo >= 13,5 g/dL`). Việc điền `hemo = 12,6 g/dL` cho một người khỏe mạnh thiếu xét nghiệm máu vô tình đẩy họ sát vào vùng nghi ngờ thiếu máu nhẹ của CKD! Đây chính là lý do ca khỏe mạnh `ID = 324` trên tập Test có xác suất dự báo tăng lên $\hat{p}_{\text{XGB}} = 0,3581$, và các ca `ID = 319`, `ID = 273` trong Train OOF bị dự báo thành Dương tính giả. Phát hiện này minh chứng rõ nhận định ở Chương 2: *điền khuyết bằng trung vị giúp giữ lại bản ghi nhưng không thay thế được xét nghiệm thực tế, và khi khuyết đồng thời các biến chủ chốt thì độ bất định của dự báo sẽ tăng cao*.

3. **Cơ chế 3 — Điểm dị biệt / nhiễu đo lường đơn lẻ (Điển hình là ca `ID = 133` trên tập Test):**
   - Bệnh nhân `ID = 133` (70 tuổi) có đầy đủ triệu chứng suy thận nặng (`sc = 5,3 mg/dL`, `bu = 125 mg/dL`, `al = 4`, `bp = 100`, `htn = yes`), được `XGBoost`, `RF`, `SVM`, `LogReg` dự đoán là CKD với xác suất gần $98\% - 99\%$. Tuy nhiên, cột số lượng hồng cầu của bản ghi này lại ghi giá trị ngoại lai **`rc = 8,0 triệu/mm³`** (mức cao bất thường về mặt sinh lý học ở bệnh nhân suy thận), khiến khoảng cách Euclidean trong `KNN` và nhánh cắt của `Decision Tree` bị nhiễu và dự đoán sai.

---

## 4.5. Giải thích mô hình và Đánh giá mức độ quan trọng của đặc trưng (Feature Importance)

### 4.5.1. Xếp hạng tầm quan trọng của các đặc trưng từ XGBoost, Random Forest và Permutation Importance

Để tránh thiên lệch của một thước đo đơn lẻ, nhóm trích xuất độ quan trọng đặc trưng theo 3 phương pháp độc lập từ `reports/tables/ch4_feature_importance_38.csv` và quy chiếu (tổng hợp trọng số các mức One-Hot) về đúng 24 thuộc tính lâm sàng ban đầu tại `reports/tables/ch4_feature_importance_24_grouped.csv`:
1. **XGBoost Gain Importance:** Tỷ trọng đóng góp của thuộc tính vào việc giảm hàm mục tiêu (Log-Loss có điều chuẩn L1/L2) qua tất cả các nút phân chia của 100 cây tăng cường.
2. **Random Forest Gini Importance (Mean Decrease in Impurity):** Độ giảm tạp chất Gini trung bình trên 50 cây quyết định độc lập.
3. **Permutation Importance trên tập Test độc lập:** Mức sụt giảm trực tiếp của chỉ số `ROC-AUC` trên tập Test khi xáo trộn ngẫu nhiên từng cột trong 24 thuộc tính đầu vào.

![So sánh độ quan trọng của 15 đặc trưng lâm sàng hàng đầu giữa XGBoost và Random Forest](figures/ch4_feature_importance_comparison.png)

Bảng xếp hạng 12 thuộc tính lâm sàng quan trọng nhất (từ `reports/tables/ch4_feature_importance_24_grouped.csv`) và đối chiếu với kết quả EDA Chương 2 của Huỳnh Công Tạo:

| Hạng | Mã thuộc tính | Tên lâm sàng | XGBoost Gain | Random Forest Gini | Điểm kết hợp (Combined) | Permutation Drop (Test AUC) | Đối chiếu thống kê EDA Chương 2 & Ý nghĩa y sinh học |
|---:|---|---|---:|---:|---:|---:|---|
| **1** | **`hemo`** | Hemoglobin (Huyết sắc tố) | **0,3891** (38,9%) | **0,2053** (20,5%) | **0,2972** | 0,0243 | Tương quan mạnh với `pcv` ($r=0,897$) và `rc` ($r=0,814$). Phản ánh trực tiếp tình trạng thiếu máu do thận giảm tiết Erythropoietin. |
| **2** | **`pcv`** | Packed Cell Volume (Dung tích hồng cầu) | 0,0654 (6,5%) | **0,1708** (17,1%) | **0,1181** | 0,0000* | Do đa cộng tuyến rất mạnh với `hemo` ($r=0,897$), XGBoost tập trung trọng số vào `hemo` (38,9%) và giảm phụ thuộc vào `pcv` (6,5%), trong khi RF chia đều cho cả hai. |
| **3** | **`al`** | Albumin (Đạm niệu, mức 0–5) | **0,1065** (10,7%) | 0,0842 (8,4%) | **0,0954** | 0,0132 | Dấu ấn tổn thương màng lọc cầu thận sớm (đặc biệt biến nhị phân hóa `multi__al_0` chiếm 10,65% Gain). |
| **4** | **`sg`** | Specific Gravity (Tỷ trọng nước tiểu) | **0,1249** (12,5%) | 0,0617 (6,2%) | **0,0933** | 0,0018 | Tổng hợp từ `sg_1.015` (6,87%), `sg_1.010` (3,27%) và `sg_1.020` (2,35%). Phản ánh suy giảm khả năng cô đặc nước tiểu của ống thận. |
| **5** | **`sc`** | Serum Creatinine (Creatinine huyết thanh) | 0,0749 (7,5%) | **0,1040** (10,4%) | **0,0895** | **0,0283** (Số 1) | **Đứng số 1 về Permutation Importance trên tập Test.** Đây là chỉ số vàng để ước tính mức lọc cầu thận (eGFR) và bắt được các ca CKD không thiếu máu (`ID = 102`). |
| **6** | **`htn`** | Hypertension (Tăng huyết áp) | 0,0523 (5,2%) | 0,0817 (8,2%) | **0,0670** | 0,0000 | Nguyên nhân kiêm hệ quả hàng đầu của bệnh thận mạn tính (vòng xoắn bệnh lý thận – huyết áp). |
| **7** | **`rc`** | Red Blood Cell Count (Số lượng hồng cầu) | 0,0481 (4,8%) | 0,0776 (7,8%) | **0,0628** | 0,0000 | Cùng nhóm hội chứng thiếu máu với `hemo` và `pcv`, nhưng tỷ lệ khuyết thiếu cao (32,75%). |
| **8** | **`dm`** | Diabetes Mellitus (Đái tháo đường) | 0,0520 (5,2%) | 0,0525 (5,2%) | **0,0522** | 0,0000 | Nguyên nhân hàng đầu gây bệnh thận đái tháo đường (Diabetic Nephropathy). |
| **9** | **`bgr`** | Blood Glucose Random (Đường huyết) | 0,0225 (2,2%) | 0,0309 (3,1%) | **0,0267** | 0,0002 | Bổ trợ định lượng cho tình trạng kiểm soát đường huyết cùng với biến `dm`. |
| **10** | **`sod`** | Sodium (Natri máu) | 0,0127 (1,3%) | 0,0278 (2,8%) | **0,0202** | 0,0000 | Tương quan nghịch với `sc` ($r=-0,812$ trên Pearson) do rối loạn cân bằng điện giải và giữ nước. |
| **11** | **`bu`** | Blood Urea (Urê máu) | 0,0072 (0,7%) | 0,0237 (2,4%) | **0,0154** | 0,0000 | Sản phẩm chuyển hóa protein ứ đọng khi suy giảm chức năng lọc cầu thận. |
| **12** | **`wc`** | White Blood Cell Count (Bạch cầu) | 0,0190 (1,9%) | 0,0077 (0,8%) | **0,0133** | 0,0000 | Phản ánh tình trạng viêm nhiễm đường tiết niệu hoặc phản ứng viêm hệ thống đi kèm. |

**Sự ăn khớp giữa Chương 2 (EDA) và Chương 4 (Giải thích mô hình):**
Tại Chương 2, Huỳnh Công Tạo đã phát hiện cặp biến `hemo – pcv` có hệ số tương quan tuyến tính Pearson rất cao ($r = 0,897$). Kết quả giải thích mô hình ở Chương 4 cho thấy cơ chế xử lý đa cộng tuyến khác biệt rõ rệt giữa hai thuật toán Ensemble:
- Trong **Random Forest (Bagging)**, do mỗi nút cây chỉ lấy mẫu ngẫu nhiên một tập con đặc trưng (`max_features='sqrt'`), có những cây thấy `hemo` và có những cây chỉ thấy `pcv`, dẫn đến độ quan trọng Gini được chia sẻ gần ngang nhau (`hemo = 20,53%`, `pcv = 17,08%`).
- Trong **XGBoost (Boosting)**, các cây được xây dựng tuần tự và xét toàn bộ các đặc trưng khả dụng tại mỗi bước chia; một khi `hemo` đã được chọn ở nút gốc để giải thích phương sai của hội chứng thiếu máu (chiếm `38,91%` Gain), thông tin dư thừa của `pcv` đã được loại bỏ bớt (chỉ còn `6,54%` Gain), giúp mô hình dành dung lượng cho các dấu ấn độc lập khác như tỷ trọng nước tiểu `sg` (`12,49%`), đạm niệu `al` (`10,65%`) và creatinine huyết thanh `sc` (`7,49%`).
- Đặc biệt, phép thử **Permutation Importance trên tập Test** chứng minh rằng khi giữ nguyên tất cả các biến khác và chỉ làm nhiễu từng biến một, **`sc` (Creatinine huyết thanh) và `hemo` (Hemoglobin)** là hai trụ cột không thể thay thế: làm nhiễu `sc` gây sụt giảm ROC-AUC mạnh nhất ($-0,0283$), kế đến là `hemo` ($-0,0243$) và `al` ($-0,0132$).

---

### 4.5.2. Phân tích nguyên nhân chênh lệch: Thực nghiệm định lượng vấn đề Data Leakage ở các bài làm trên Kaggle so với quy trình chuẩn của nhóm

Khi khảo sát các notebook công khai đạt "100% Accuracy" trên Kaggle với bộ dữ liệu UCI Chronic Kidney Disease, nhóm phát hiện **3 sai lầm phương pháp luận phổ biến** làm kết quả đánh giá bị thổi phồng hoặc mất giá trị ứng dụng lâm sàng:
1. **Sai lầm 1 — Điền khuyết bằng trung bình cộng (`df.fillna(df.mean())`) và chuẩn hóa (`StandardScaler().fit_transform(df)`) trên toàn bộ 400 dòng trước khi chia Train/Test hoặc trước khi chạy Cross-Validation:** Việc này làm rò rỉ trực tiếp phân phối và các điểm ngoại lai của tập Test vào tập Train.
2. **Sai lầm 2 — Dùng trung bình cộng (`mean`) thay vì trung vị (`median`) cho các biến sinh hóa lệch phải cực đoan:** Như Chương 2 đã chứng minh, các biến `sc`, `bu`, `bgr` có phân phối lệch phải rất nặng với các điểm ngoại lai cực lớn (`sc = 76,0`, `bgr = 490`, `bu = 322`).
3. **Sai lầm 3 — Xóa toàn bộ các dòng có giá trị khuyết thiếu (`df.dropna()`):** Nhiều bài làm trên mạng gọi `df.dropna()` ngay đầu notebook rồi huấn luyện trên phần dữ liệu còn lại.

Thay vì chỉ nhận xét định tính bằng lời, nhóm đã thiết kế thí nghiệm định lượng trong `src/evaluate_chapter4.py` (kết quả lưu tại `reports/tables/ch4_kaggle_imputation_shift.csv`, `reports/tables/ch4_kaggle_leakage_audit.csv` và Hình `reports/figures/ch4_kaggle_leakage_audit.png`) để đo lường chính xác mức độ biến dạng dữ liệu của các cách làm trên mạng:

![Đo lường thực nghiệm độ lệch tham số điền khuyết và tỷ lệ mất mát bệnh nhân giữa quy trình chuẩn của nhóm và các cách làm phổ biến trên Kaggle](figures/ch4_kaggle_leakage_audit.png)

#### a) Bằng chứng định lượng 1: Biến dạng tham số điền khuyết và thang đo chuẩn hóa (`ch4_kaggle_imputation_shift.csv`)

| Thuộc tính | Trung vị Train chuẩn của nhóm (`Strict Train Median`) | Trung vị thực của Test (`True Test Median`) | Trung bình 400 mẫu kiểu Kaggle (`Leaky Global Mean`) | Độ lệch tâm điền khuyết (`Mean vs Strict Median`) | Độ lệch chuẩn Train chuẩn (`Strict Train Std`) | Độ lệch chuẩn 400 mẫu bị rò rỉ (`Leaky Full-400 Std`) | Độ biến dạng thang đo Z-score (`Scale Shift %`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`sc` (Creatinine)** | **1,20 mg/dL** | 1,40 mg/dL | **3,0725 mg/dL** | **+156,04%** | 5,9016 | 5,6218 | 4,74% |
| **`bu` (Blood Urea)** | **42,00 mg/dL** | 44,00 mg/dL | **57,4257 mg/dL** | **+36,73%** | 48,1953 | 49,3335 | 2,36% |
| **`bgr` (Glucose)** | **120,50 mg/dL** | 124,00 mg/dL | **148,0365 mg/dL** | **+22,85%** | 71,5264 | 75,1666 | 5,09% |
| **`pot` (Potassium)** | **4,30 mEq/L** | 4,50 mEq/L | **4,6272 mEq/L** | **+7,61%** | **0,6593** | **2,8178** | **+327,38%** |
| **`age` (Tuổi)** | 54,00 tuổi | 56,00 tuổi | 51,4834 tuổi | 4,66% | 16,9201 | 16,9618 | 0,25% |
| **`bp` (Huyết áp)** | 80,00 mmHg | 80,00 mmHg | 76,4691 mmHg | 4,41% | 13,9278 | 13,4729 | 3,27% |
| **`wc` (Bạch cầu)** | 8.100 tế bào | 7.950 tế bào | 8.406,12 tế bào | 3,78% | 2.646,79 | 2.526,43 | 4,55% |
| **`pcv` (%)** | 40,00% | 41,00% | 38,8845% | 2,79% | 8,1376 | 8,1520 | 0,18% |

Bảng số liệu thực nghiệm trên vạch rõ hai sai lệch nghiêm trọng nếu làm theo các notebook thiếu chuẩn mực trên Kaggle:
- **Sai lệch lâm sàng +156,04% ở chỉ số Creatinine huyết thanh (`sc`):** Trung vị chuẩn trên tập Train là `1,20 mg/dL` (nằm ngay ngưỡng sinh lý bình thường). Trong khi đó, do tập dữ liệu có các ca suy thận giai đoạn cuối với `sc` lên tới `76,0 mg/dL`, trung bình cộng toàn cục (`df['sc'].mean()`) bị kéo vọt lên **`3,0725 mg/dL`** (tương đương suy thận mạn giai đoạn 4!). Nếu một người khỏe mạnh đến khám bị khuyết chỉ số `sc`, cách điền bằng `mean()` trên Kaggle sẽ gán thẳng cho họ mức `sc = 3,07 mg/dL` — biến người bình thường thành bệnh nhân suy thận nặng!
- **Rò rỉ điểm ngoại lai từ tập Test làm bóp méo +327,38% thang đo Kali máu (`pot`):** Trên tập Train (320 mẫu), độ lệch chuẩn của Kali máu chỉ là $\sigma_{\text{train}} = 0,6593\text{ mEq/L}$ (phù hợp với khoảng sinh lý hẹp $3,5 - 5,0\text{ mEq/L}$). Tuy nhiên, trong tập Test có một bản ghi mang giá trị ngoại lai cực đoan (`pot = 47,0`). Khi các notebook Kaggle gọi `StandardScaler().fit_transform(X)` trên toàn bộ 400 mẫu trước khi chia tập, điểm ngoại lai ở tập Test đã rò rỉ ngược vào tập Train và thổi phồng độ lệch chuẩn lên $\sigma_{\text{all}} = 2,8178\text{ mEq/L}$ (**tăng gấp 4,27 lần, tương đương +327,38%**), làm co cụm toàn bộ điểm chuẩn hóa Z-score của biến `pot` trên tập huấn luyện!

#### b) Bằng chứng định lượng 2: Hậu quả của việc xóa dòng khuyết thiếu (`df.dropna()`) và tiền xử lý ngoài vòng lặp CV (`ch4_kaggle_leakage_audit.csv`)

- **Mất mát 60,5% tổng số bệnh nhân (và mất 78,0% ca CKD trong tập Test):** Như thể hiện ở biểu đồ bên phải của Hình `ch4_kaggle_leakage_audit.png`, nếu dùng `df.dropna()`, hệ thống chỉ còn đánh giá được **158/400 bản ghi (39,5%)** trên toàn bộ dữ liệu, **121/320 bản ghi (37,8%)** trên tập Train, và chỉ còn **37/80 bản ghi (46,2%)** trên tập Test. Nghiêm trọng hơn, trong 50 bệnh nhân thực sự mắc CKD ở tập Test, có tới **39 bệnh nhân bị thiếu ít nhất 1 trong 24 xét nghiệm** (chỉ có 11/50 bệnh nhân CKD có đủ cả 24 chỉ số, chiếm **22,0%**). Một mô hình huấn luyện theo kiểu `df.dropna()` sẽ từ chối phục vụ 78% số bệnh nhân CKD ngoài thực tế lâm sàng!
- **Ngược lại, Pipeline của nhóm đánh giá được 100% bệnh nhân (80/80 mẫu Test):** Ngay cả khi tách riêng nhóm **43 bệnh nhân bị khuyết thiếu dữ liệu trong tập Test** (trong đó có 39 ca CKD), mô hình `XGBoost (Tuned)` của nhóm vẫn đạt **$\text{Recall} = 100,0\%$ (39/39 ca CKD khuyết thiếu đều được phát hiện đúng)**.

---

## Kết luận Chương 4

1. **Kiểm định độc lập và Lựa chọn mô hình:** Trên tập kiểm thử hoàn toàn độc lập ($n = 80$), cả ba kiến trúc tinh chỉnh (`XGBoost`, `Random Forest`, `SVM`) đều đạt diện tích dưới đường cong **$\text{ROC-AUC} = 1,0000$ và $\text{PR-AUC} = 1,0000$**. Tại ngưỡng vận hành tiêu chuẩn $\tau = 0,50$, **`XGBoost (Tuned)`** là mô hình duy nhất đạt tuyệt đối **$\text{Accuracy} = 100,0\%$, $\text{Recall (Sensitivity)} = 100,0\%$ ($50/50$ ca CKD, $\text{FN} = 0$) và $\text{Specificity} = 100,0\%$ ($30/30$ ca khỏe mạnh, $\text{FP} = 0$)** với sai số xác suất nhỏ nhất ($\text{Brier Score} = 0,0054$).
2. **Phân tích lỗi ở cấp độ bệnh nhân:** Truy vết qua `split_manifest.csv` cho thấy ca bệnh duy nhất trên tập Test làm thất bại 7/9 mô hình là bệnh nhân **`ID = 102` (17 tuổi, CKD giai đoạn sớm chưa có biến chứng thiếu máu, `hemo = 13,9 g/dL`, `sc = 2,1 mg/dL`, `sg = 1,010`)**. Khả năng phát hiện chính xác ca bệnh khó này khẳng định ưu thế vượt trội của thuật toán Gradient Boosting (`XGBoost`) khi xử lý các tương tác phi tuyến bất đối xứng. Đồng thời, khảo sát ngưỡng quyết định ($\tau$) chứng minh rằng nếu điều chỉnh ngưỡng về vùng sàng lọc $\tau \in [0,40; 0,45]$, `Random Forest (Tuned)` cũng đạt độ chính xác và độ nhạy tuyệt đối 100%.
3. **Giải thích lâm sàng và Minh chứng chống Data Leakage:** Xếp hạng độ quan trọng đặc trưng (Gain, Gini và Permutation Importance) hoàn toàn nhất quán với y văn thận học và kết quả EDA Chương 2 (`hemo`, `pcv`, `al`, `sg`, `sc`, `htn`, `dm`, `rc` là 8 trụ cột chẩn đoán). Thực nghiệm đối sánh với Kaggle đã chứng minh bằng con số cụ thể rằng quy trình nhúng `make_preprocessor()` vào trong từng fold CV của nhóm đã ngăn chặn thành công độ lệch $+156,04\%$ khi điền khuyết `sc`, ngăn chặn biến dạng $+327,38\%$ độ lệch chuẩn của `pot`, và bảo toàn khả năng chẩn đoán cho $100\%$ bệnh nhân có hồ sơ xét nghiệm không đầy đủ. File mô hình `models/best_model.joblib` đã sẵn sàng bàn giao hoàn chỉnh cho Chương 5 (Ứng dụng Web chẩn đoán).

---

## Tài liệu tham khảo

1. Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). *Chronic Kidney Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
4. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804.
5. Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group. (2024). KDIGO 2024 Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. *Kidney International*, 105(4S), S117–S314.

