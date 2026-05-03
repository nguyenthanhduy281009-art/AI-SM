import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình giao diện
st.set_page_config(page_title="BNB AI Sentinel", layout="wide")

st.title("🤖 BNB AI MARKET SENTINEL PRO")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    t_time = st.slider("Dự đoán sau (phút):", 1, 60, 15)
    analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

if analyze_btn:
    with st.spinner('AI đang quét sóng...'):
        # 1. Tải dữ liệu
        data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
        if data.empty:
            st.error("Lỗi kết nối dữ liệu!")
            st.stop()
            
        df = data[['Close', 'Open', 'High', 'Low']].copy()
        
        # 2. Chỉ báo kỹ thuật
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['Volatility'] = df['Close'].diff()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # 3. Huấn luyện AI
        features = ['Close', 'MA5', 'MA20', 'RSI', 'Volatility']
        X = df[features]
        y = df['Target']
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        # 4. Dự đoán
        last_val = X.iloc[[-1]]
        pred = model.predict(last_val)[0]
        prob = model.predict_proba(last_val)[0]

        # --- HIỂN THỊ ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Giá BNB Hiện Tại", f"${df['Close'].iloc[-1]:.2f}")
        
        res = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
        color = "#00FF00" if pred == 1 else "#FF4B4B"
        c2.markdown(f"### Dự báo: <span style='color:{color}'>{res}</span>", unsafe_allow_html=True)
        c3.metric("Độ tin cậy", f"{prob[pred]*100:.1f}%")

        # Biểu đồ
        st.markdown("---")
        fig = go.Figure(data=[go.Scatter(x=df.index[-100:], y=df['Close'][-100:], name="Giá", line=dict(color='#f3ba2f'))])
        fig.update_layout(template="plotly_dark", height=400, title="Biểu đồ 100 phút gần nhất")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Nhấn nút 'PHÂN TÍCH NGAY' để bắt đầu.")
