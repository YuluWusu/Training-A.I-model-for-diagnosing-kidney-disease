# CHƯƠNG 2 KHÁM PHÁ TRỰC QUAN HÓA VÀ TIỀN XỬ LÝ DỮ LIỆU

**Người thực hiện: Huỳnh Công Tạo — MSSV 2001240431 — Nhóm 01**

Chương này khảo sát bộ dữ liệu bệnh thận mạn tính, xác định các vấn đề chất lượng và xây dựng quy trình chuyển đổi dữ liệu phục vụ huấn luyện mô hình. Sản phẩm gồm notebook EDA, đối tượng tiền xử lý đã fit trên tập huấn luyện, các tập dữ liệu bàn giao và hợp đồng thứ tự đặc trưng. Mọi số liệu mô tả trong chương được đối chiếu với file kidney_disease.csv đi kèm; chương này chưa thực hiện đánh giá chất lượng mô hình phân lớp.

## 2.1 Tổng quan bộ dữ liệu

### 2.1.1 Nguồn gốc và quy mô

Bộ dữ liệu Chronic Kidney Disease được công bố tại UCI Machine Learning Repository, DOI 10.24432/C5G020. Tài liệu nguồn mô tả 400 bản ghi với 24 thuộc tính dự báo và một nhãn phân lớp. File CSV đang sử dụng có 400 dòng, 26 cột gồm 24 thuộc tính, cột id và cột classification. Cột id là mã truy vết, không đưa vào đầu vào mô hình.

Notebook chạy được với bản dữ liệu được gắn vào Kaggle. Metadata trong CSV không xác định được đường dẫn trang Kaggle hoặc người tải lên, nên không gán một đường dẫn Kaggle chưa được xác minh. Mã SHA-256, tên file và phiên bản thư viện được lưu trong feature_contract.json để xác định đúng dữ liệu của mỗi lần thực nghiệm.

Bài toán là phân lớp nhị phân: ckd được mã hóa thành 1, notckd thành 0. Giá trị 1 là lớp dương trong các chỉ số Recall, F1 và ROC-AUC ở các chương sau. Nhãn không được sử dụng để điền khuyết đặc trưng, lựa chọn giá trị median hoặc chuẩn hóa thang đo.

### 2.1.2 Từ điển thuộc tính

Theo cách tổ chức của đề cương, 24 thuộc tính được chia thành 11 biến định lượng và 13 biến định loại. Trong 13 biến định loại, 10 biến có hai mức; sg, al, su có nhiều mức. Mặc dù ba biến này hiển thị dưới dạng số, quy trình xử lý chúng như biến phân loại theo đặc tả dữ liệu, tránh áp đặt khoảng cách tuyến tính giữa các mức.

| Nhóm | Thuộc tính | Ý nghĩa |
|---|---|---|
| Định lượng | age, bp | Tuổi và huyết áp |
| Định lượng | bgr, bu, sc | Đường huyết ngẫu nhiên, urê và creatinin |
| Định lượng | sod, pot | Natri và kali |
| Định lượng | hemo, pcv, wc, rc | Hemoglobin, thể tích khối hồng cầu, số lượng bạch cầu và hồng cầu |
| Nhiều mức | sg, al, su | Tỷ trọng, albumin và đường trong nước tiểu |
| Nhị phân | rbc, pc | Hồng cầu nước tiểu và tế bào mủ |
| Nhị phân | pcc, ba | Cụm tế bào mủ và vi khuẩn |
| Nhị phân | htn, dm, cad | Tăng huyết áp, đái tháo đường, bệnh động mạch vành |
| Nhị phân | appet, pe, ane | Thèm ăn, phù chân, thiếu máu |

Đơn vị và các mức cụ thể của từng biến được trình bày trong bảng data_dictionary của notebook và reports/tables/data_dictionary.csv. Giá trị khuyết thiếu không được coi là một chẩn đoán âm tính. Chẳng hạn rbc bị thiếu không đồng nghĩa rbc = normal.

## 2.2 Phân tích khám phá và trực quan hóa dữ liệu

### 2.2.1 Phân phối biến mục tiêu

Trước làm sạch, nhãn được ghi thành ba chuỗi khác nhau: 248 giá trị ckd, 2 giá trị ckd kèm tab và 150 giá trị notckd. Sau chuẩn hóa khoảng trắng, dữ liệu có đúng hai lớp:

| Lớp | Mã | Số dòng | Tỷ lệ |
|---|---:|---:|---:|
| Không CKD theo nhãn dữ liệu | 0 | 150 | 37,5% |
| CKD | 1 | 250 | 62,5% |
| Tổng | | 400 | 100% |

Tỷ lệ lớp lớn so với lớp nhỏ là 250/150 ≈ 1,67. Nếu chỉ dự đoán tất cả là CKD, Accuracy trên toàn bộ dữ liệu sẽ bằng 62,5%; đây là phép tính tham chiếu từ phân phối nhãn, không phải kết quả mô hình đã huấn luyện. Vì vậy, các chương sau cần đánh giá thêm Recall, F1, ROC-AUC và ma trận nhầm lẫn.

Chương 2 chưa áp dụng lấy mẫu lại. Nếu nhóm thử SMOTE hoặc các phương pháp cân bằng lớp, thao tác đó phải thực hiện trong phần huấn luyện của từng fold, không thực hiện trên toàn bộ dữ liệu trước khi chia tập.

Hình 2.1 trong notebook (01_class_split.png) thể hiện phân phối nhãn và việc duy trì tỷ lệ hai lớp sau chia Train–Test.

### 2.2.2 Khuyết thiếu

Việc chỉ đếm ô trống ban đầu sẽ bỏ sót các token dấu hỏi có tab. Sau chuẩn hóa, 24 thuộc tính có tổng cộng 1.012 ô thiếu trên 9.600 ô, tương ứng khoảng 10,54%. Có 158 dòng đủ toàn bộ 24 thuộc tính. Xóa tất cả dòng có ít nhất một giá trị thiếu sẽ loại 242/400 dòng, tức 60,5% dữ liệu. Do đó, nhóm sử dụng phương pháp điền khuyết thay vì xóa toàn bộ các dòng này.

| Thuộc tính | Số thiếu | Tỷ lệ | Thuộc tính | Số thiếu | Tỷ lệ |
|---|---:|---:|---|---:|---:|
| age | 9 | 2,25% | sod | 87 | 21,75% |
| bp | 12 | 3,00% | pot | 88 | 22,00% |
| sg | 47 | 11,75% | hemo | 52 | 13,00% |
| al | 46 | 11,50% | pcv | 71 | 17,75% |
| su | 49 | 12,25% | wc | 106 | 26,50% |
| rbc | 152 | 38,00% | rc | 131 | 32,75% |
| pc | 65 | 16,25% | htn | 2 | 0,50% |
| pcc | 4 | 1,00% | dm | 2 | 0,50% |
| ba | 4 | 1,00% | cad | 2 | 0,50% |
| bgr | 44 | 11,00% | appet | 1 | 0,25% |
| bu | 19 | 4,75% | pe | 1 | 0,25% |
| sc | 17 | 4,25% | ane | 1 | 0,25% |

rbc, rc và wc là các cột thiếu nhiều nhất. Điền khuyết giúp giữ số mẫu nhưng không phục hồi được thông tin chưa quan sát. Đặc biệt, điền mode hoặc median nhiều lần có thể làm tăng tần suất của giá trị phổ biến và làm giảm độ phân tán.

Bảng trên là kiểm kê chất lượng dữ liệu toàn bộ. Hình 2.2 (02_missing_train.png) được vẽ riêng trên Train: biểu đồ cột thể hiện tỷ lệ thiếu, còn ma trận màu thể hiện vị trí thiếu theo từng bản ghi. Từ biểu đồ này không thể tự kết luận cơ chế thiếu là MCAR, MAR hoặc MNAR; cần thông tin về quá trình thu thập để đánh giá các giả định đó.

### 2.2.3 Lỗi định dạng và bất thường

Các lỗi thực tế bao gồm ckd kèm tab ở nhãn; yes hoặc no có khoảng trắng/tab ở dm và cad; số trong pcv, wc có tab ở đầu; dấu hỏi có tab ở pcv, wc, rc. Notebook dùng repr để hiển thị ký tự ẩn, kèm số lần xuất hiện trong format_audit.csv.

Quy trình chuẩn hóa gồm loại khoảng trắng đầu/cuối, chuyển chữ thường, nhận diện các token khuyết thiếu và ép kiểu số có kiểm tra. Chuỗi 5, 5.0 trong rc đều trở thành số thực 5.0. Với sg, al, su, dữ liệu được quy về biểu diễn chuỗi thống nhất, chẳng hạn 1.020 thành 1.02, 0.0 thành 0, trước khi mã hóa.

Nếu xuất hiện một chuỗi không chuyển được sang số, một giá trị vô hạn hoặc một mức phân loại ngoài miền cho phép, chương trình dừng và báo lỗi. Cách này giúp phát hiện sự thay đổi của nguồn dữ liệu thay vì biến lỗi thành giá trị thiếu một cách âm thầm. Các giá trị số âm bị từ chối theo miền áp dụng của các biến; chương trình không tự suy diễn ngưỡng lâm sàng cho các giá trị lớn.

Notebook cũng kiểm tra id duy nhất và các bản ghi trùng toàn bộ đặc trưng cùng nhãn. Nếu có trùng, quy trình yêu cầu xem xét loại trùng hoặc chia theo nhóm trước khi tiếp tục, tránh để cùng một bản ghi xuất hiện ở cả Train và Test. id duy nhất vẫn không chứng minh chắc chắn mỗi dòng là một người khác nhau; dữ liệu không có mã bệnh nhân bổ sung để xác minh.

### 2.2.4 Phân phối biến số và ngoại lai

Hình 2.3 (03_histograms_train.png) gồm histogram của 11 biến số theo nhãn. Hình 2.4 (04_boxplots_train.png) trình bày trung vị, khoảng tứ phân vị và các điểm ngoài râu hộp. Các hình này sử dụng Train trước điền khuyết; giá trị thiếu không được đưa vào histogram hoặc boxplot.

Với mỗi biến, tính IQR = Q3 − Q1 và hai ngưỡng Q1 − 1,5×IQR, Q3 + 1,5×IQR. Bảng outliers_iqr_train.csv lưu ngưỡng, số điểm bị đánh dấu, số điểm quan sát và tỷ lệ. Mẫu số của tỷ lệ ngoại lai là số giá trị không thiếu của biến, không phải mặc định 280.

Các điểm ngoài ngưỡng là dấu hiệu cần rà soát thống kê. Quy trình giữ lại chúng vì chưa có thông tin nguồn để phân biệt chắc chắn giá trị bệnh lý, sai số đo và lỗi nhập liệu. Không khẳng định rằng mọi điểm lớn đều hợp lệ về lâm sàng. Nếu nghiên cứu thêm winsorization hoặc biến đổi log, cần chọn cách xử lý bằng CV trên Train và giữ Test độc lập.

Trong lần chạy hiện tại (trên tập Train 280 mẫu), bgr có 33/248 giá trị quan sát bị đánh dấu (13,31%), sc có 32/265 (12,08%), bp có 28/272 (10,29%), bu có 26/262 (9,92%). Các giá trị lớn nhất trên Train lần lượt gồm sc = 76, bgr = 490 và bu = 322; sod có giá trị nhỏ nhất 4,5. Đây là các quan sát cần truy vết nếu có hồ sơ nguồn. Histogram của bgr, bu và sc cho thấy phần đuôi phía giá trị cao, giải thích vì sao cần xem phân vị và boxplot thay vì chỉ đọc trung bình.

### 2.2.5 Tương quan và các biến phân loại

Hình 2.5 (05_correlations_train.png) gồm Pearson và Spearman cho 11 biến số. Pearson đo mức liên hệ tuyến tính; Spearman dựa trên thứ hạng và phản ánh quan hệ đơn điệu. Dùng hai cách đo giúp quan sát các tình huống hệ số tuyến tính chịu ảnh hưởng bởi điểm cực đoan.

Các hệ số được tính trước điền khuyết, theo các dòng có đủ dữ liệu của từng cặp. Vì vậy, bảng correlation_pair_counts_train.csv phải được đọc cùng heatmap: một cặp có hệ số lớn nhưng ít quan sát vẫn cần thận trọng khi diễn giải. Giá trị tương quan không thể được xem như bằng chứng nhân quả hoặc độ quan trọng của mô hình.

Hình 2.6 (06_categorical_train.png) biểu diễn tỷ lệ nhãn trong từng mức của 13 biến phân loại, bao gồm một nhóm riêng cho giá trị thiếu. Biểu đồ dùng chuẩn hóa theo hàng, nên mỗi cột cộng thành 100%; đây là tỷ lệ trong từng mức, không phải số bệnh nhân. Không chọn hoặc loại đặc trưng tự động dựa trên các hình này.

Các hệ số nổi bật của lần chạy được xuất tự động vào reports/KET_QUA_THUC_CHAY.md để số liệu báo cáo khớp với notebook.

Kết quả Pearson trên Train (280 mẫu) cho thấy hemo–pcv đạt 0,898 với 231 cặp quan sát, pcv–rc đạt 0,817 với 191 cặp, hemo–rc đạt 0,811 với 190 cặp. Cặp sc–sod có Pearson −0,815 với 211 cặp nhưng Spearman chỉ khoảng −0,532 (xấp xỉ −0,53). Ngoài ra, cặp bu–hemo đạt Pearson −0,601 với 232 cặp (Spearman −0,587) và bu–rc đạt Pearson −0,596 với 184 cặp. Chênh lệch giữa hai cách đo gợi ý cần xem xét hình dạng phân phối và các điểm cực đoan trước khi kết luận về quan hệ của cặp biến. Những kết quả này được dùng để mô tả dữ liệu, không dùng làm quy tắc chẩn đoán.

## 2.3 Chiến lược làm sạch và Pipeline

### 2.3.1 Chuẩn hóa kiểu dữ liệu

Lớp CKDCleaner nhận pandas DataFrame có đúng 24 tên cột quy định, từ chối tên cột thiếu, thừa hoặc trùng và sắp xếp về thứ tự thống nhất. Lớp này không ước lượng phân phối dữ liệu. Các biến số được chuyển thành float; biến phân loại dùng chuỗi thống nhất và np.nan cho giá trị thiếu.

Đưa CKDCleaner vào Pipeline giúp xử lý cùng một cách ở notebook, quy trình huấn luyện và ứng dụng Web. Vì đối tượng này là lớp tùy chỉnh, file ckd_preprocessing.py phải đi kèm preprocessor.joblib khi chuyển sang môi trường khác.

### 2.3.2 Điền khuyết

Nhánh số sử dụng SimpleImputer(strategy="median"). Với mỗi cột, median chỉ tính trên các giá trị không thiếu của Train rồi dùng để thay NaN cho cả Train và Test. Test không được tự tính median riêng.

Median được chọn làm phương án cơ sở vì đơn giản, tái lập và ít nhạy với điểm cực đoan hơn mean. KNNImputer có thể khai thác quan hệ giữa các biến nhưng phụ thuộc số láng giềng, thang đo và độ tương đồng giữa các bản ghi. Chương này chưa chạy đối sánh KNNImputer, do đó không kết luận phương pháp nào có độ chính xác tốt hơn.

Nhánh phân loại sử dụng SimpleImputer(strategy="most_frequent"). Giá trị mode học trên Train, không tính riêng theo nhãn CKD. Điền theo nhãn sẽ tạo sự phụ thuộc vào thông tin không có khi dự đoán người mới. Notebook kiểm tra Train không có cột thiếu hoàn toàn; nếu điều kiện này không thỏa, phải xem lại chiến lược thay vì tiếp tục mặc định.

### 2.3.3 Ngoại lai

Không xóa các dòng bị đánh dấu bởi IQR và không áp dụng clipping. Điều này bảo toàn các quan sát cực đoan để nhóm đánh giá mô hình sau đó, đồng thời để lại dấu vết trong bảng thống kê. StandardScaler chỉ đổi thang đo, không loại bỏ tác động của ngoại lai; đây là hạn chế cần lưu ý khi so sánh các mô hình dựa trên khoảng cách.

### 2.3.4 Mã hóa phân loại

Mười biến nhị phân dùng OneHotEncoder với hai mức quy định trước và drop="if_binary". Bỏ mức tham chiếu đầu tiên tạo một cột cho mỗi biến:

| Biến | Mức tham chiếu mã 0 | Mức mã 1 |
|---|---|---|
| rbc, pc | normal | abnormal |
| pcc, ba | notpresent | present |
| htn, dm, cad, pe, ane | no | yes |
| appet | good | poor |

sg được One-Hot thành 5 cột theo các mức 1.005, 1.01, 1.015, 1.02, 1.025. al và su mỗi biến có 6 cột theo các mức 0–5. Không bỏ mức của ba biến nhiều mức. Các miền này được xác định trước theo đặc tả, không lấy danh sách mức từ Test.

Mức phân loại không hợp lệ bị từ chối. Vì vậy, API cần báo lỗi nhập liệu tương ứng, không âm thầm hiểu giá trị lạ như mức tham chiếu.

### 2.3.5 Chuẩn hóa thang đo

Sau điền khuyết, mỗi biến số được đổi theo z = (x − μ_train)/σ_train. StandardScaler tính μ và σ trên Train đã điền median; σ tương ứng ddof=0. Những thống kê này được giữ nguyên khi biến đổi Test.

Chỉ 11 cột số được chuẩn hóa. Các cột One-Hot giữ giá trị 0/1. Bảng learned_numeric_parameters.csv lưu median, mean và scale; learned_modes.csv lưu mode của các cột phân loại. Nếu một cột số có phương sai bằng 0, StandardScaler dùng scale bằng 1 để tránh chia cho 0.

### 2.3.6 Chia tập phân tầng và chống rò rỉ dữ liệu

Dữ liệu được chia Train 70%, Test 30%, random_state=42 và stratify=y. Kết quả với file hiện tại:

| Tập | notckd | ckd | Tổng |
|---|---:|---:|---:|
| Train | 105 | 175 | 280 |
| Test | 45 | 75 | 120 |

Chỉ các phép làm sạch xác định sẵn được áp dụng trước chia tập. EDA phân phối và các phép fit thống kê thực hiện sau chia tập. preprocessor.fit_transform chỉ gọi trên Train; Test chỉ gọi transform.

Đối với 5-fold CV của Chương 3, còn một mức chống rò rỉ khác: không dùng ma trận đã được fit tiền xử lý trên toàn bộ 280 dòng trước CV. Trong từng fold, Pipeline phải được fit lại chỉ trên phần huấn luyện của fold (224 mẫu huấn luyện và 56 mẫu kiểm định mỗi fold, gồm 35 CKD và 21 Not CKD). Bộ bàn giao vì vậy có cả dữ liệu sạch cú pháp chưa điền khuyết để Hồng Vũ thực hiện đúng quy trình.

## 2.4 Mã nguồn và quy chuẩn bàn giao

### 2.4.1 Cấu trúc Pipeline

Hàm make_preprocessor tạo một đối tượng mới, chưa fit, gồm hai bước: clean và columns. clean là CKDCleaner; columns là ColumnTransformer với ba nhánh num, binary và multi. Mỗi nhánh tự thực hiện điền khuyết rồi biến đổi phù hợp.

Việc tạo mới Pipeline rất cần thiết trong kiểm chứng chéo. Ở Chương 3, ghép make_preprocessor() và classifier thành một Pipeline, sau đó truyền cả Pipeline vào cross_validate hoặc GridSearchCV. Ví dụ tham số C của bộ phân lớp mang tên model__C. Sau khi chọn tham số bằng CV, fit Pipeline trên toàn Train rồi mới đánh giá Test ở Chương 4.

### 2.4.2 Hợp đồng thứ tự đặc trưng

Thứ tự 24 đầu vào là:

~~~text
age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane
~~~

Cần phân biệt 24 thuộc tính đầu vào với số chiều sau mã hóa. Trong cách xử lý này, đầu ra gồm 38 cột: 11 số chuẩn hóa, 10 nhị phân, 5 cột sg, 6 cột al và 6 cột su. Do đó, cách diễn đạt “24 thuộc tính đầu ra” trong đề cương cần hiểu lại là 24 thuộc tính đầu vào trước mã hóa.

Đầu ra xếp theo nhánh num, binary, multi. Tên chính xác lấy từ get_feature_names_out và được ghi trong feature_contract.json, không tự đặt lại tên theo suy đoán. JSON còn chứa nhãn mã hóa, seed, tỷ lệ chia, SHA-256 và phiên bản thư viện.

### 2.4.3 Các file bàn giao

| File | Nội dung | Cách sử dụng |
|---|---|---|
| Chuong2_EDA_HuynhCongTao.ipynb | Code, diễn giải, bảng và hình EDA | Chạy toàn bộ trên Kaggle |
| preprocessor.joblib | Pipeline fit trên 280 dòng Train | Chỉ transform dữ liệu mới |
| ckd_preprocessing.py | Lớp làm sạch và hàm tạo Pipeline | Bắt buộc có khi nạp joblib |
| data/processed/X_train.csv | 280 × 38, không NaN/Inf | Huấn luyện trực tiếp trên tập Train cố định |
| data/processed/X_test.csv | 120 × 38, không NaN/Inf | Cho classifier học từ ma trận đã xử lý |
| data/processed/X_train_raw.csv | 280 × 24, sạch cú pháp, còn NaN | Đầu vào CV với Pipeline đầy đủ |
| data/processed/X_test_raw.csv | 120 × 24, sạch cú pháp, còn NaN | Đánh giá Pipeline đầy đủ ở Chương 4 |
| data/processed/y_train.csv và y_test.csv | Nhãn tương ứng theo dòng | Không sắp xếp lại độc lập |
| data/processed/split_manifest.csv | id gốc, source_row, split, position | Truy vết bản ghi; position bắt đầu từ 0 |
| feature_contract.json | Tên cột, miền giá trị, phiên bản | Thống nhất huấn luyện và Web |
| requirements_runtime.txt | Phiên bản thư viện lúc xuất joblib | Tái lập môi trường |
| verification.json | Kết quả kiểm tra tự động | Minh chứng bàn giao |
| reports/ | Hình, bảng, kết luận tự động | Minh chứng nội dung Chương 2 |

Hai cách sử dụng mô hình không được trộn lẫn. Classifier học trên X_train.csv cần nhận X_test.csv. Pipeline chứa cả preprocessor và classifier phải nhận X_train_raw/X_test_raw hoặc 24 trường đầu vào ban đầu. Nếu đưa ma trận 38 cột đã xử lý vào Pipeline đầy đủ thì dữ liệu sẽ bị xử lý sai hoặc bị từ chối.

Notebook kiểm tra tải lại joblib, so sánh với ma trận đã xuất, kiểm tra đảo thứ tự cột đầu vào, kiểm tra median Train và kiểm tra fit riêng ở 5 fold. Kiểm tra này xác nhận tính nhất quán kỹ thuật, chưa chứng minh mô hình có khả năng dự đoán tốt trên dữ liệu ngoài bộ mẫu.

## Kết luận chương

Quy trình làm sạch đã thống nhất nhãn, nhận diện các giá trị khuyết thiếu ẩn và tổ chức 24 thuộc tính thành ba nhánh tiền xử lý. Dữ liệu được chia phân tầng thành 280 dòng Train và 120 dòng Test. Pipeline học median, mode và tham số chuẩn hóa trên Train, tạo đầu ra 38 chiều và lưu đủ thông tin để sử dụng lại.

Các hạn chế còn tồn tại gồm quy mô dữ liệu nhỏ, tỷ lệ thiếu cao ở một số thuộc tính, chưa xác minh được nguồn gốc từng ngoại lai và chưa có đánh giá trên dữ liệu thu thập độc lập. Các bước huấn luyện, lựa chọn siêu tham số và đánh giá Test thuộc Chương 3–4.

## Tài liệu tham khảo

1. Rubini, L., Soundarapandian, P., và Eswaran, P. (2015). [Chronic Kidney Disease](https://doi.org/10.24432/C5G020). UCI Machine Learning Repository.
2. scikit-learn. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html).
3. scikit-learn. [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
4. scikit-learn. [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
