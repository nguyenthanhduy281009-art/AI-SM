import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình trang
st.set_page_config(page_title="BNB AI Predictor - Duy", layout="wide")

# Tùy chỉnh CSS để giao diện đẹp hơn
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ BNB AI MARKET SENTINEL")
st.caption("Phiên bản nâng cấp bởi Nguyen Thanh Duy")

# Sidebar cấu hình
with st.sidebar:
    st.header("🎮 Điều khiển")
    t_min = st.slider("Dự đoán sau (phút):", 1, 60, 15)
    analyze_btn = st.button("🔥 QUÉT THỊ TRƯỜNG", use_container_width=True)

if analyze_btn:
    with st.spinner('AI đang học lệnh...'):
        # 1. Lấy dữ liệu
        data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
        df = data[['Close']].copy()
        
        # 2. Chỉ báo kỹ thuật (Tăng độ chính xác)
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        # Tính RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        df.dropna(inplace=True)

        # 3. Training AI
        X = df[['Close', 'MA10', 'MA30', 'RSI']]
        y = (df['Close'].shift(-t_min) > df['Close']).astype(int)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X[:-t_min], y[:-t_min])

        # 4. Predict
        curr_feat = X.iloc[[-1]]
        pred = model.predict(curr_feat)[0]
        prob = model.predict_proba(curr_feat)[0]

        # --- GIAO DIỆN KẾT QUẢ ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Giá Hiện Tại", f"${df['Close'].iloc[-1]:.2f}")
        
        with col2:
            sentiment = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
            color = "#00FF00" if pred == 1 else "#FF4B4B"
            st.markdown(f"### Dự đoán: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
            
        with col3:
            st.metric("Độ tin cậy", f"{prob[pred]*100:.1f}%")

        # 5. Biểu đồ nến chuyên nghiệp
        fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], name="Giá BNB", line=dict(color='#f3ba2f'))])
        fig.update_layout(template="plotly_dark", title=f"Biểu đồ BNB (7 ngày qua)", height=400)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 Duy ơi, nhấn 'QUÉT THỊ TRƯỜNG' để xem AI dự đoán nhé!")
