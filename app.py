import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình giao diện
st.set_page_config(page_title="BNB AI Sentinel - Duy", layout="wide")

# CSS làm đẹp
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    h1 { color: #f3ba2f; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 BNB AI MARKET SENTINEL PRO")
st.markdown("---")

# --- SIDEBAR CẤU HÌNH ---
with st.sidebar:
    st.header("⚙️ Cấu hình AI")
    t_time = st.slider("Dự đoán sau (phút):", 1, 60, 15)
    
    # Ô NHẬP GIÁ CHO DUY
    st.markdown("---")
    st.subheader("💰 Nhập giá")
    user_price = st.number_input("Giá BNB muốn soi (USD):", min_value=0.0, value=0.0, help="Để 0.0 nếu muốn lấy giá thị trường thực tế")
    
    analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

if analyze_btn:
    with st.spinner('AI đang tính toán...'):
        # 1. Tải dữ liệu và xử lý Multi-index
        data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
        
        if data.empty:
            st.error("Không lấy được dữ liệu thị trường!")
            st.stop()
            
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Chỉ báo kỹ thuật
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA30'] = df['Close'].rolling(30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        df['Volatility'] = df['Close'].diff()
        df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # 3. Huấn luyện AI
        features = ['Close', 'MA10', 'MA30', 'RSI', 'Volatility']
        X = df[features]
        y = df['Target']
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)

        # 4. Dự đoán (Sử dụng giá Duy nhập hoặc giá thị trường)
        market_price = float(df['Close'].values[-1])
        price_to_predict = user_price if user_price > 0 else market_price
        
        # Tạo input cho model dựa trên giá được chọn
        current_features = np.array([[
            price_to_predict, 
            df['MA10'].iloc[-1], 
            df['MA30'].iloc[-1], 
            df['RSI'].iloc[-1],
            df['Volatility'].iloc[-1]
        ]])
        
        pred = model.predict(current_features)[0]
        prob = model.predict_proba(current_features)[0]

        # --- HIỂN THỊ ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Giá Đang Soi", f"${price_to_predict:.2f}")
        
        res = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
        color = "#00FF00" if pred == 1 else "#FF4B4B"
        c2.markdown(f"### Dự báo sau {t_time}p: <span style='color:{color}'>{res}</span>", unsafe_allow_html=True)
        c3.metric("Độ tin cậy AI", f"{prob[pred]*100:.1f}%")

        # 5. Biểu đồ nến
        st.markdown("---")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index[-60:],
            open=df['Open'][-60:], high=df['High'][-60:],
            low=df['Low'][-60:], close=df['Close'][-60:]
        )])
        fig.update_layout(template="plotly_dark", height=450, title="Diễn biến thị trường 60p gần nhất")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Nhập giá (nếu cần) và nhấn nút để bắt đầu phân tích!")
