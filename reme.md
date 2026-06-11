# 🛡️ Ứng dụng Web AI Phát hiện Giao dịch Gian lận & Rủi ro Tín dụng

Dự án này chuyển đổi thành công từ Notebook học máy thực nghiệm (`phat_hien_giao_dich_gian_lan.ipynb`) sang một ứng dụng Web tương tác Dashboard hoàn chỉnh và hiện đại bằng thư viện **Streamlit**. Hệ thống hỗ trợ đắc lực cho các chuyên viên quản trị rủi ro kiểm thử, phân tích dữ liệu và chạy mô hình chấm điểm tín dụng/gian lận thời gian thực.

## ✨ Tính năng chính của ứng dụng
1. **Sidebar động**: Cho phép tải tệp dữ liệu linh hoạt, lựa chọn thuật toán huấn luyện thích hợp (`Random Forest`, `Decision Tree`, `Logistic Regression`) và điều biến siêu tham số mô hình một cách trực quan.
2. **Khám phá phân tích (EDA)**: Thống kê mô tả dữ liệu thô và cấu trúc phân bổ các thuộc tính qua hệ thống biểu đồ tương tác cao cấp Plotly (Histogram, Boxplot, Pie).
3. **Đánh giá hiệu năng trực quan**: Tái hiện chính xác các chỉ số đo lường nghiệp vụ tài chính từ Notebook gốc như: *Accuracy, Precision, Recall, F1-Score* đi kèm Ma trận nhầm lẫn (Confusion Matrix).
4. **Dự báo linh hoạt (2 Chế độ)**:
   - **Chế độ đơn lẻ**: Nhập các tham số giao dịch trực tiếp trên form để chấm điểm ngay.
   - **Chế độ Batch**: Tải danh sách hồ sơ khách hàng mới (CSV/Excel) và xuất file kết quả phân loại rủi ro đóng dấu cảnh báo tự động.

## 🛠️ Hướng dẫn cài đặt và khởi chạy

**Bước 1: Di chuyển vào thư mục dự án chứa các file ứng dụng và cài đặt các thư viện phụ thuộc:**
```bash
pip install -r requirements.txt
