import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI BNB Predictor", page_icon="📈")
st.title("🤖 AI Dự Đoán Giá BNB")

# --- PHẦN NHẬP LIỆU ---
with st.sidebar:
    st.header("Cài đặt mô hình")
    t_time = st.number_input("Thời gian dự đoán (phút)", min_value=1, value=5)
    st.info("AI sẽ học các mẫu nến trong quá khứ để đoán xu hướng sau t phút.")

user_price = st.number_input("Nhập giá BNB hiện tại từ sàn (USD):", min_value=0.0, format="%.2f")

if st.button("Bắt đầu Phân tích & Dự đoán"):
    if user_price <= 0:
        st.warning("Vui lòng nhập giá BNB hiện tại để AI có mốc so sánh.")
    else:
        with st.spinner('AI đang quét dữ liệu thị trường 7 ngày qua...'):
            # 1. Tải dữ liệu 1 phút
            data = yf.download("BNB-USD", period="7d", interval="1m")
            
            if data.empty:
                st.error("Không thể kết nối với dữ liệu sàn. Thử lại sau.")
            else:
                df = data[['Close']].copy()
                
                # 2. Tạo các chỉ báo để tăng độ chính xác
                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['Price_Change'] = df['Close'].diff()
                
                # Nhãn: 1 nếu sau t_time giá tăng, 0 nếu giảm
                df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)
                df.dropna(inplace=True)

                # 3. Chia dữ liệu để tính Tỉ lệ đúng (Accuracy)
                features = ['Close', 'MA5', 'MA20', 'Price_Change']
                X = df[features]
                y = df['Target']
                
                # Dùng 80% để học, 20% để kiểm tra độ chính xác
                split = int(len(df) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Huấn luyện mô hình
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Tính tỉ lệ đúng trên tập kiểm tra
                y_pred_test = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred_test) * 100

                # 4. Dự đoán ván hiện tại
                current_data = np.array([[user_price, df['MA5'].iloc[-1], df['MA20'].iloc[-1], df['Price_Change'].iloc[-1]]])
                final_pred = model.predict(current_data)

                # --- HIỂN THỊ KẾT QUẢ ---
                st.divider()
                
                # Hiển thị Tỉ lệ đúng
                st.subheader("📊 Độ tin cậy của mô hình")
                if acc > 50:
                    st.success(f"Tỉ lệ đúng hiện tại: **{acc:.2f}%**")
                else:
                    st.warning(f"Tỉ lệ đúng hiện tại: **{acc:.2f}%** (Thị trường đang biến động nhiễu)")

                # Hiển thị Dự đoán
                st.subheader(f"🔮 Dự đoán sau {t_time} phút")
                if final_pred[0] == 1:
                    st.header("XU HƯỚNG: TĂNG 📈")
                else:
                    st.header("XU HƯỚNG: GIẢM 📉")
                
                st.caption(f"Dữ liệu được huấn luyện dựa trên {len(X_train)} nến 1 phút gần nhất.")
