# Kết quả Chương 2

- Dữ liệu: 400 dòng và 24 thuộc tính dự báo; id bị loại khỏi mô hình.
- Nhãn: 250 CKD và 150 notCKD.
- Sau chuẩn hóa có 1012 ô thiếu; 158 dòng đủ 24 thuộc tính.
- Các cột thiếu nhiều nhất: rbc: 38.00%, rc: 32.75%, wc: 26.50%.
- Train/Test: 320/80; random_state=42.
- Một số cặp Pearson mạnh trên Train: hemo–pcv: r=0.897; pcv–rc: r=0.815; hemo–rc: r=0.814; sc–sod: r=-0.812; bu–rc: r=-0.612.
- 24 đầu vào → 38 đầu ra; không còn NaN hoặc Inf.
- Kiểm tra tải lại joblib và fit riêng ở 5 fold: đạt.
- Chưa huấn luyện hay báo cáo độ chính xác mô hình trong Chương 2.

Tương quan chỉ mô tả dữ liệu quan sát. Giữ ngoại lai không có nghĩa đã xác minh lâm sàng.
Quy mô mẫu nhỏ và tỷ lệ thiếu cao là hạn chế cần ghi nhận.
