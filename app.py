import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# LỆNH STREAMLIT ĐẦU TIÊN: Cấu hình trang giao diện rộng
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="😎Hệ thống Phát hiện Gian lận tại Agribank😎",
    page_icon="❤️"
)

# ==============================================================================
# HÀM NẠP DỮ LIỆU ĐƯỢC CACHE (Nhận bytes để đảm bảo hashable)
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# ==============================================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH & TẢI DỮ LIỆU
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải file dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (CSV hoặc Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Chọn tệp dữ liệu có cấu trúc chứa các đặc trưng từ X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    
    # 2. Lựa chọn thuật toán (Trích xuất chính xác từ bộ 3 mô hình trong notebook)
    st.subheader("🤖 Cấu hình mô hình AI")
    model_choice = st.selectbox(
        "Chọn thuật toán huấn luyện:",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        index=0,
        help="Chọn một trong ba thuật toán mà bạn đã thử nghiệm trong Notebook để huấn luyện và so sánh."
    )
    
    # 3. Cấu hình tham số động dựa theo thuật toán được chọn
    params = {}
    if model_choice == "Random Forest":
        params['n_estimators'] = st.slider(
            "Số lượng cây (n_estimators)", 
            min_value=10, max_value=500, value=100, step=10,
            help="Số lượng cây quyết định trong rừng. Mặc định trong notebook là 100."
        )
        params['random_state'] = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=32, step=1,
            help="Hạt giống tạo số ngẫu nhiên để cố định kết quả. Mặc định là 32."
        )
    elif model_choice == "Decision Tree":
        params['random_state'] = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=32, step=1,
            help="Hạt giống tạo số ngẫu nhiên cho cây quyết định. Mặc định là 32."
        )
    elif model_choice == "Logistic Regression":
        params['max_iter'] = st.slider(
            "Số vòng lặp tối đa (max_iter)", 
            min_value=100, max_value=2000, value=1000, step=100,
            help="Số vòng lặp tối đa cho thuật toán tối ưu hội tụ. Notebook cảnh báo chưa hội tụ ở giới hạn cũ nên cần tăng thêm."
        )
        
    st.divider()
    
    # 4. Duy nhất nút kích hoạt huấn luyện đặt tại đây
    trigger_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==============================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG ỨNG DỤNG
# ==============================================================================
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận & Rủi ro Mặc định")
st.caption("Ứng dụng hỗ trợ phân tích dữ liệu giao dịch tài chính, huấn luyện các mô hình học máy (Logistic Regression, Decision Tree, Random Forest) để phân loại rủi ro gian lận dựa trên tập thuộc tính ẩn danh.")

# Kiểm tra trạng thái dữ liệu đầu vào độc lập
if uploaded_file is None:
    st.info("💡 Vui lòng tải tập dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu khám phá hệ thống.")
    st.stop()

# Đọc dữ liệu khi đã upload thành công
df = load_data(uploaded_file, uploaded_file.name)
if df is None or df.empty:
    st.error("❌ Tệp dữ liệu không hợp lệ hoặc không có nội dung. Vui lòng kiểm tra lại.")
    st.stop()

st.caption(f"💾 Đang sử dụng tệp dữ liệu: `{uploaded_file.name}`")
st.divider()

# ==============================================================================
# KHỐI XỬ LÝ CHÍNH: HUẤN LUYỆN KHI ẤN NÚT VÀ LƯU VÀO SESSION STATE
# ==============================================================================
if trigger_train:
    with st.spinner("🔄 Đang xử lý dữ liệu và huấn luyện mô hình, vui lòng đợi..."):
        # Bước 1: Xác định biến đầu vào và đầu ra dựa trên notebook
        target_col = 'default'
        if target_col not in df.columns:
            st.error(f"❌ Không tìm thấy cột mục tiêu '{target_col}' trong tập dữ liệu của bạn!")
        else:
            y = df[target_col]
            X = df.drop(target_col, axis=1)
            
            # Bước 2: Chia tập dữ liệu thành train và test (Tỷ lệ 80/20, random_state=32 tương tự notebook)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
            
            # Bước 3 & 4: Khởi tạo thuật toán tương ứng và Huấn luyện
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier(random_state=params['random_state'])
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=params['max_iter'])
            
            model.fit(X_train, y_train)
            
            # Bước 5: Dự báo trên tập kiểm định để lấy chỉ số
            y_pred = model.predict(X_test)
            
            # Lưu trữ toàn bộ trạng thái vào st.session_state để dùng chung cho các tab mà không phải train lại
            st.session_state['trained_model'] = model
            st.session_state['features_name'] = list(X.columns)
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['y_pred'] = y_pred
            st.session_state['model_name'] = model_choice
            
            st.success(f"🎉 Huấn luyện thành công mô hình {model_choice}!")

# ==============================================================================
# ĐỊNH NGHĨA KHỐI TABS TRÊN GIAO DIỆN CHÍNH
# ==============================================================================
tab_summary, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả kiểm định mô hình", 
    "🔮 Sử dụng dự báo"
])

# ------------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_summary:
    st.subheader("📋 Số liệu thống kê tổng quan")
    col1, col2, col3 = st.columns(3)
    col1.metric("Số lượng bản ghi (Rows)", f"{df.shape[0]:,}")
    col2.metric("Số lượng biến (Columns)", f"{df.shape[1]:,}")
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col3.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
    
    st.write("#### 🔍 Xem trước dữ liệu thô (5 dòng đầu tiên)")
    st.dataframe(df.head(5), use_container_width=True)
    
    st.write("#### 📐 Bảng mô tả thống kê các đặc trưng đưa vào mô hình")
    # Lấy các biến X và y từ notebook (Loại bỏ cột không liên quan nếu có, ở đây lấy toàn bộ cột đặc trưng)
    feature_cols = [c for c in df.columns if c != 'default']
    st.dataframe(df[feature_cols].describe(), use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_viz:
    st.subheader("📊 Phân tích trực quan các biến của mô hình")
    
    # Thiết lập phân bố dạng lưới 2x2 cho biểu đồ
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Biểu đồ biến mục tiêu (Ưu tiên số 1 - Bài toán học máy có giám sát)
    if 'default' in df.columns:
        with row1_col1:
            target_counts = df['default'].value_counts().reset_index()
            target_counts.columns = ['Trạng thái Rủi ro', 'Số lượng']
            target_counts['Trạng thái Rủi ro'] = target_counts['Trạng thái Rủi ro'].map({0: '0: Bình thường', 1: '1: Gian lận/Mặc định'})
            fig1 = px.bar(target_counts, x='Trạng thái Rủi ro', y='Số lượng', 
                          title="Phân phối của biến mục tiêu (default)",
                          color='Trạng thái Rủi ro', color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig1, use_container_width=True)
            
    # 2. Trực quan hóa một số biến đặc trưng ngẫu nhiên liên tục quan trọng từ dữ liệu
    available_features = [c for c in df.columns if c != 'default']
    
    if len(available_features) >= 1:
        with row1_col2:
            feat = available_features[0] # Biến X_1
            fig2 = px.histogram(df, x=feat, color='default' if 'default' in df.columns else None,
                                title=f"Phân phối tần suất của biến {feat}",
                                barmode='overlay', marginal='box')
            st.plotly_chart(fig2, use_container_width=True)
            
    if len(available_features) >= 2:
        with row2_col1:
            feat = available_features[5] # Biến X_6 có độ phân tán cao std~44
            fig3 = px.box(df, y=feat, x='default' if 'default' in df.columns else None,
                          title=f"Biểu đồ hộp (Boxplot) phát hiện ngoại lai của biến {feat}",
                          color='default' if 'default' in df.columns else None)
            st.plotly_chart(fig3, use_container_width=True)
            
    if len(available_features) >= 3:
        with row2_col2:
            feat = available_features[8] # Biến X_9 có giá trị max cực đại >77000
            fig4 = px.histogram(df, x=feat, title=f"Biểu đồ phân phối mật độ logarit của biến {feat}", 
                                log_y=True, color_discrete_sequence=['#9A60B4'])
            st.plotly_chart(fig4, use_container_width=True)
            
    # Mở rộng nếu người dùng muốn tự chọn trực quan hóa thêm các biến khác (>4 biến)
    st.divider()
    st.write("#### 🔄 Tùy chọn phân tích chuyên sâu các đặc trưng biến khác")
    selected_feature = st.selectbox("Chọn thuộc tính cụ thể cần kiểm tra phân phối:", options=available_features)
    fig_custom = px.violin(df, y=selected_feature, x='default' if 'default' in df.columns else None, 
                           box=True, points="all", title=f"Biểu đồ Violin chi tiết cho {selected_feature}")
    st.plotly_chart(fig_custom, use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------------------------------------------
with tab_metrics:
    st.subheader("🎯 Đánh giá hiệu năng và kiểm định thuật toán")
    
    # Kiểm tra xem mô hình đã được fit chưa
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Mô hình chưa được huấn luyện. Hãy chọn cấu hình thuật toán và bấm nút 'Huấn luyện mô hình' ở Sidebar trái.")
    else:
        model_name = st.session_state['model_name']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        
        st.write(f"### 📊 Kết quả kiểm định thuật toán: **{model_name}**")
        
        # 1. Hiển thị các chỉ tiêu vô hướng qua st.metric nhóm phân loại
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy (Độ chính xác tổng thể)", f"{accuracy_score(y_test, y_pred):.4f}")
        m_col2.metric("Precision (Độ chuẩn xác lớp 1)", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        m_col3.metric("Recall (Độ phủ lớp 1)", f"{recall_score(y_test, y_pred, zero_division=0):.4f}")
        m_col4.metric("F1-Score (Điểm cân bằng)", f"{f1_score(y_test, y_pred, zero_division=0):.4f}")
        
        st.divider()
        
        col_layout1, col_layout2 = st.columns([1, 1])
        
        with col_layout1:
            st.write("#### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred)
            
            # Trực quan hóa ma trận nhầm lẫn đẹp mắt bằng Plotly Heatmap
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng bản ghi"),
                x=['0 (Bình thường)', '1 (Gian lận)'],
                y=['0 (Bình thường)', '1 (Gian lận)'],
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_layout2:
            st.write("#### 📄 Báo cáo chi tiết (Classification Report)")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='YlGnBu', axis=0), use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH DỰ BÁO"
# ------------------------------------------------------------------------------
with tab_inference:
    st.subheader("🔮 Chẩn đoán & Dự báo rủi ro gian lận cho bản ghi mới")
    
    if 'trained_model' not in st.session_state:
        st.info("⚠️ Không thể chạy dự báo do chưa tìm thấy thực thể mô hình. Vui lòng bấm kích hoạt huấn luyện ở Sidebar.")
    else:
        model = st.session_state['trained_model']
        features_name = st.session_state['features_name']
        
        # Chọn chế độ nhập trực tiếp hoặc tải file hàng loạt
        mode = st.radio("Chọn hình thức nhập dữ liệu đầu vào:", 
                        options=["📍 Chế độ 1: Nhập trực tiếp chỉ số đơn lẻ", "📂 Chế độ 2: Tải file danh sách Excel/CSV hàng loạt"])
        
        # CHẾ ĐỘ 1: NHẬP TRỰC TIẾP QUA ST.FORM
        if mode == "📍 Chế độ 1: Nhập trực tiếp chỉ số đơn lẻ":
            st.write("#### Điền thông số đo lường của giao dịch cần kiểm tra:")
            
            # Sử dụng st.form để tối ưu hóa, tránh tự động rerun khi đang chỉnh sửa các ô số
            with st.form("inference_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                # Tạo tự động 14 widget dựa vào đúng số lượng biến đặc trưng đầu vào X từ notebook
                for idx, feat_name in enumerate(features_name):
                    col_to_use = form_cols[idx % 4]
                    with col_to_use:
                        # Gán giá trị mặc định dựa trên quan sát mẫu từ file data test trong notebook
                        if feat_name in ['X_1', 'X_2', 'X_3', 'X_4']:
                            default_val = 0.05
                        elif feat_name in ['X_5', 'X_6', 'X_7', 'X_8', 'X_14']:
                            default_val = 1.00
                        elif feat_name == 'X_13':
                            default_val = 100.0
                        else:
                            default_val = 0.00
                            
                        input_data[feat_name] = st.number_input(
                            f"Thông số {feat_name}", 
                            value=default_val, 
                            format="%.6f"
                        )
                
                submit_predict = st.form_submit_button("🔍 Thực hiện phân tích rủi ro", use_container_width=True)
                
            if submit_predict:
                # Chuyển đổi dữ liệu input sang cấu trúc DataFrame khớp chính xác tên cột
                X_new_single = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán
                prediction = model.predict(X_new_single)[0]
                
                st.write("### 📌 Kết quả chẩn đoán từ hệ thống AI:")
                if prediction == 1:
                    st.error("🚨 **CẢNH BÁO nguy hiểm:** Hệ thống phát hiện giao dịch này có đặc điểm rất trùng khớp với các mẫu **GIAN LẬN/MẶC ĐỊNH RỦI RO (Lớp 1)**. Cần tiến hành phong tỏa hoặc hậu kiểm lập tức.")
                else:
                    st.success("✅ **AN TOÀN:** Giao dịch được xác định nằm trong ngưỡng **BÌNH THƯỜNG (Lớp 0)**. Độ tin cậy cao.")
                    
                # Hiện xác suất dự đoán nếu mô hình có hỗ trợ predict_proba (Như RandomForest và Logistic Regression)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X_new_single)[0]
                    st.write(f"📊 *Chi tiết tỷ lệ xác suất: An toàn ({probabilities[0]*100:.2f}%) | Nguy cơ rủi ro ({probabilities[1]*100:.2f}%)*")

        # CHẾ ĐỘ 2: TẢI FILE DANH SÁCH BATCH HÀNG LOẠT 
        elif mode == "📂 Chế độ 2: Tải file danh sách Excel/CSV hàng loạt":
            st.write("#### Tải lên danh sách hồ sơ cần chấm điểm hàng loạt (File cấu trúc chứa đủ từ X_1 đến X_14):")
            batch_file = st.file_uploader("Chọn file dữ liệu mới cần dự đoán", type=["csv", "xlsx"])
            
            if batch_file is not None:
                df_batch = load_data(batch_file, batch_file.name)
                
                if df_batch is not None:
                    # Kiểm tra tính khớp của Schema cột đặc trưng (Thừa/Thiếu cột)
                    missing_cols = [col for col in features_name if col not in df_batch.columns]
                    
                    if missing_cols:
                        st.error(f"❌ File tải lên không đúng định dạng mẫu. Thiếu các cột đặc trưng sau: {missing_cols}")
                    else:
                        # Đảm bảo thứ tự cột truyền vào mô hình chính xác tuyệt đối như lúc train
                        X_batch_clean = df_batch[features_name]
                        
                        # Dự báo hàng loạt
                        batch_predictions = model.predict(X_batch_clean)
                        
                        # Tạo bảng kết quả kết hợp
                        df_result = df_batch.copy()
                        df_result['Dự_Đoán_Rủi_Rao_default'] = batch_predictions
                        
                        st.success(f"⚡ Đã thực hiện dự báo tự động thành công cho tất cả {len(df_result)} bản ghi dữ liệu.")
                        
                        # Hiển thị biểu đồ phân tích nhanh cơ cấu kết quả dự báo batch
                        res_counts = df_result['Dự_Đoán_Rủi_Rao_default'].value_counts().reset_index()
                        res_counts.columns = ['Kết quả dự báo', 'Số lượng']
                        res_counts['Kết quả dự báo'] = res_counts['Kết quả dự báo'].map({0: 'Bình thường', 1: 'Cảnh báo Gian lận'})
                        fig_batch = px.pie(res_counts, names='Kết quả dự báo', values='Số lượng', hole=0.4,
                                           title="Tỷ lệ phân phối rủi ro trong tệp dữ liệu mới tải lên",
                                           color_discrete_sequence=['#2ca02c', '#d62728'])
                        st.plotly_chart(fig_batch, use_container_width=True)
                        
                        # Preview bảng kết quả kèm nút download xuất file
                        st.write("#### 📋 Xem trước bảng dữ liệu kết quả chấm điểm:")
                        st.dataframe(df_result, use_container_width=True)
                        
                        # Xuất dữ liệu dạng CSV mã hóa utf-8-sig chống lỗi font tiếng Việt
                        csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống kết quả dự báo toàn bộ (.CSV)",
                            data=csv_data,
                            file_name="ket_qua_du_bao_gian_lan_hang_loat.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
