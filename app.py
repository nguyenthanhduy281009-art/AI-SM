import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình giao diện rộng và đẹp hơn
st.set_page_config(page_title="BNB AI Pro - Nguyen Thanh Duy", layout="wide")

st.title("🤖 AI BNB PREDICTOR PRO")
st.markdown("---")

# Chia cột để giao diện gọn gàng
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("⚙️ Cấu hình")
    t_time = st.number_input("Thời gian dự đoán (phút):", min_value=1, value=5)
    # Lấy giá thực tế tự động để Duy đỡ phải nhập tay
    btn_analyze = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

with col2:
    if btn_analyze:
        with st.spinner('AI đang quét dữ liệu thị trường và tính toán chỉ báo...'):
            # 1. Tải dữ liệu
            data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
            df = data[['Close']].copy()
            
            # 2. Nâng cấp bộ chỉ báo (Features) để tăng độ chính xác
            df['MA5'] = df['Close'].rolling(5).mean()
            df['MA20'] = df['Close'].rolling(20).mean()
            # RSI: Chỉ số sức mạnh tương đối
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['Volatility'] = df['Close'].diff()
            
            df.dropna(inplace=True)

            # 3. Huấn luyện AI với Random Forest (Tăng n_estimators để ổn định hơn)
            X = df[['Close', 'MA5', 'MA20', 'RSI', 'Volatility']]
            y = (df['Close'].shift(-t_time) > df['Close']).astype(int)
            
            # Bỏ dòng cuối bị NaN do shift
            X_train = X[:-t_time]
            y_train = y[:-t_time]
            
            model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
            model.fit(X_train, y_train)

            # 4. Dự đoán ván hiện tại
            current_price = df['Close'].iloc[-1]
            last_features = np.array([df[['Close', 'MA5', 'MA20', 'RSI', 'Volatility']].iloc[-1]])
            pred = model.predict(last_features)[0]
            prob = model.predict_proba(last_features)[0]

            # --- HIỂN THỊ GIAO DIỆN CHUYÊN NGHIỆP ---
            c1, c2, c3 = st.columns(3)
            c1.metric("Giá BNB Hiện Tại", f"${current_price:.2f}")
            
            res_text = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
            color = "green" if pred == 1 else "red"
            c2.markdown(f"### Dự đoán: <span style='color:{color}'>{res_text}</span>", unsafe_allow_html=True)
            
            confidence = prob[pred] * 100
            c3.metric("Độ tin cậy AI", f"{confidence:.2f}%")

            # 5. Vẽ biểu đồ giá trực quan
            st.markdown("### 📊 Biểu đồ diễn biến 7 ngày")
            st.line_chart(df['Close'])

            st.info(f"💡 Lưu ý: Hệ thống đã học từ {len(df)} mẫu dữ liệu gần nhất để đưa ra kết quả này.")
    else:
        st.write("👈 Nhấn nút bên trái để bắt đầu soi cầu BNB!")

st.markdown("---")
st.caption("Thiết kế bởi Nguyen Thanh Duy - Dữ liệu thực tế từ Yahoo Finance")
