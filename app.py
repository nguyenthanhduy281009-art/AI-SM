import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình giao diện
st.set_page_config(page_title="BNB AI Sentinel - Duy", layout="wide")

st.title("🤖 BNB AI MARKET SENTINEL PRO")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Cấu hình AI")
    t_time = st.slider("Dự đoán sau (phút):", 1, 60, 15)
    analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

if analyze_btn:
    with st.spinner('AI đang quét sóng...'):
        # 1. Tải dữ liệu và xử lý lỗi Multi-index
        data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
        
        if data.empty:
            st.error("Không lấy được dữ liệu. Duy kiểm tra lại kết nối nhé!")
            st.stop()
            
        # SỬA LỖI: Đảm bảo các cột dữ liệu ở dạng đơn giản
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Tính toán chỉ báo
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # 3. Huấn luyện AI
        features = ['Close', 'MA10', 'MA30', 'RSI']
        X = df[features]
        y = df['Target']
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        # 4. Dự đoán ván hiện tại (Sửa lỗi TypeError tại đây)
        last_row = X.iloc[[-1]]
        # Ép kiểu float để f-string :.2f không bị lỗi
        current_price = float(df['Close'].values[-1])
        
        pred = model.predict(last_row)[0]
        prob = model.predict_proba(last_row)[0]

        # --- HIỂN THỊ ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Giá BNB Hiện Tại", f"${current_price:.2f}")
        
        res = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
        color = "#00FF00" if pred == 1 else "#FF4B4B"
        c2.markdown(f"### Dự báo: <span style='color:{color}'>{res}</span>", unsafe_allow_html=True)
        c3.metric("Độ tin cậy AI", f"{prob[pred]*100:.1f}%")

        # 5. Biểu đồ nến chuyên nghiệp
        st.markdown("---")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index[-60:],
            open=df['Open'][-60:], high=df['High'][-60:],
            low=df['Low'][-60:], close=df['Close'][-60:]
        )])
        fig.update_layout(template="plotly_dark", height=450, title="Diễn biến 60 phút gần nhất")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Nhấn nút để AI bắt đầu làm việc!")
