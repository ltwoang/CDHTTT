import streamlit as st
import cv2
import tempfile
import os
from tracker import ObjectCounter
from PIL import Image
import pandas as pd
import time
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="Vehicle Detection & Counting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .vehicle-info {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .vehicle-info.in {
        border-left-color: #4CAF50;
    }
    .vehicle-info.out {
        border-left-color: #f44336;
    }
    .vehicle-info p {
        margin: 5px 0;
        font-size: 14px;
    }
    .vehicle-info strong {
        color: #1f1f1f;
    }
    .vehicle-time {
        color: #666;
        font-size: 12px;
        margin-bottom: 8px;
    }
    .vehicle-direction {
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .direction-in {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .direction-out {
        background-color: #ffebee;
        color: #c62828;
    }
    .vehicle-details {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .detail-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .color-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .company-logo {
        width: 20px;
        height: 20px;
        margin-right: 5px;
    }
    .stat-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f1f1f;
    }
    .stat-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .in-stat {
        border-left: 5px solid #4CAF50;
    }
    .out-stat {
        border-left: 5px solid #f44336;
    }
    .gemini-result {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .gemini-result h4 {
        color: #1f1f1f;
        margin: 0 0 10px 0;
        font-size: 16px;
    }
    .gemini-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }
    .gemini-table th, .gemini-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #e0e0e0;
    }
    .gemini-table th {
        background-color: #f5f5f5;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.title("🚗 Hệ thống Phát hiện & Đếm Phương tiện")

# Khởi tạo session state
if 'in_count' not in st.session_state:
    st.session_state.in_count = 0
if 'out_count' not in st.session_state:
    st.session_state.out_count = 0
if 'vehicle_data' not in st.session_state:
    st.session_state.vehicle_data = []

# Sidebar
with st.sidebar:
    st.header("📊 Thống kê thời gian thực")
    
    # Hiển thị số liệu thống kê với giao diện đẹp hơn
    st.markdown("""
        <style>
        .stat-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #1f1f1f;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        .in-stat {
            border-left: 5px solid #4CAF50;
        }
        .out-stat {
            border-left: 5px solid #f44336;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Thống kê xe vào
    st.markdown(f"""
        <div class="stat-card in-stat">
            <div class="stat-value">🚗 {st.session_state.in_count}</div>
            <div class="stat-label">Xe vào</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Thống kê xe ra
    st.markdown(f"""
        <div class="stat-card out-stat">
            <div class="stat-value">🚗 {st.session_state.out_count}</div>
            <div class="stat-label">Xe ra</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị tổng số xe
    total_vehicles = st.session_state.in_count + st.session_state.out_count
    st.markdown(f"""
        <div class="stat-card" style="background-color: #e3f2fd; border-left: 5px solid #2196F3;">
            <div class="stat-value">📈 {total_vehicles}</div>
            <div class="stat-label">Tổng số xe</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hiển thị kết quả phân tích từ Gemini
    st.subheader("🤖 Kết quả phân tích AI")
    gemini_results = st.empty()
    
    # Danh sách xe
    st.subheader("📝 Danh sách xe ra/vào")
    vehicle_list = st.empty()

# Phần chính
col1, col2 = st.columns([2, 1])

with col1:
    # Upload video
    uploaded_file = st.file_uploader(
        "Chọn video để phát hiện phương tiện",
        type=["mp4", "avi", "mov"],
        help="Hỗ trợ các định dạng: MP4, AVI, MOV"
    )

    if uploaded_file is not None:
        # Lưu video tạm
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()

        # Hiển thị video gốc
        st.video(video_path)

        # Nút bắt đầu
        if st.button("▶️ Bắt đầu phát hiện & đếm xe"):
            stframe = st.empty()
            cap = cv2.VideoCapture(video_path)
            region_points = [(3, 412), (1015, 412)]

            counter = ObjectCounter(
                region=region_points,
                model="yolo12n.pt",
                show_in=True,
                show_out=True,
                line_width=2
            )

            last_in_count = 0
            last_out_count = 0
            
            try:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 2 != 0:
                        continue

                    frame = cv2.resize(frame, (1020, 500))
                    results = counter.process(frame)
                    
                    # Cập nhật số lượng xe
                    if counter.in_count != last_in_count:
                        st.session_state.in_count = counter.in_count
                        last_in_count = counter.in_count
                    
                    if counter.out_count != last_out_count:
                        st.session_state.out_count = counter.out_count
                        last_out_count = counter.out_count
                    
                    # Xử lý thông tin xe mới
                    for track_id, info in counter.gemini_results.items():
                        if track_id not in [x['track_id'] for x in st.session_state.vehicle_data]:
                            direction = None
                            for class_name, counts in counter.classwise_counts.items():
                                if track_id in counter.counted_ids:
                                    if counts['IN'] > 0:
                                        direction = 'IN'
                                    elif counts['OUT'] > 0:
                                        direction = 'OUT'
                                    break
                            
                            current_time = datetime.now().strftime("%H:%M:%S")
                            
                            # Lấy thông tin màu và hãng xe từ kết quả Gemini
                            vehicle_color = info.get('color', 'Unknown')
                            vehicle_company = info.get('company', 'Unknown')
                            
                            new_vehicle = {
                                'track_id': track_id,
                                'color': vehicle_color,
                                'company': vehicle_company,
                                'direction': direction if direction else 'Unknown',
                                'time': current_time
                            }
                            
                            st.session_state.vehicle_data.append(new_vehicle)
                            
                            # Cập nhật danh sách xe trong sidebar
                            with vehicle_list:
                                direction_class = "direction-in" if direction == 'IN' else "direction-out"
                                direction_text = "Vào" if direction == 'IN' else "Ra" if direction == 'OUT' else "Unknown"
                                vehicle_class = "vehicle-info in" if direction == 'IN' else "vehicle-info out"
                                
                                st.markdown(f"""
                                    <div class="{vehicle_class}">
                                        <div class="vehicle-time">⏰ {current_time}</div>
                                        <div class="vehicle-details">
                                            <div class="detail-item">
                                                <strong>🎨 Màu xe:</strong>
                                                <span class="color-dot" style="background-color: {vehicle_color.lower()};"></span>
                                                <span>{vehicle_color}</span>
                                            </div>
                                            <div class="detail-item">
                                                <strong>🏢 Hãng xe:</strong>
                                                <span>{vehicle_company}</span>
                                            </div>
                                            <div class="detail-item">
                                                <strong>↔️ Hướng:</strong>
                                                <span class="vehicle-direction {direction_class}">{direction_text}</span>
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Cập nhật kết quả phân tích Gemini
                            with gemini_results:
                                st.markdown(f"""
                                    <div class="gemini-result">
                                        <h4>🔍 Phân tích xe #{track_id}</h4>
                                        <table class="gemini-table">
                                            <tr>
                                                <th>Màu xe</th>
                                                <td>{vehicle_color}</td>
                                            </tr>
                                            <tr>
                                                <th>Hãng xe</th>
                                                <td>{vehicle_company}</td>
                                            </tr>
                                            <tr>
                                                <th>Hướng</th>
                                                <td>{direction_text}</td>
                                            </tr>
                                            <tr>
                                                <th>Thời gian</th>
                                                <td>{current_time}</td>
                                            </tr>
                                        </table>
                                    </div>
                                """, unsafe_allow_html=True)

                    # Hiển thị frame
                    img = cv2.cvtColor(results.plot_im, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    stframe.image(pil_img, channels="RGB", use_container_width=True)
                    time.sleep(0.01)

            finally:
                cap.release()
                time.sleep(1)
                try:
                    os.unlink(video_path)
                except:
                    pass

            st.success("✅ Hoàn thành phát hiện & đếm xe!")

with col2:
    # Hiển thị bảng thống kê
    if st.session_state.vehicle_data:
        st.subheader("📊 Bảng thống kê chi tiết")
        df = pd.DataFrame(st.session_state.vehicle_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )