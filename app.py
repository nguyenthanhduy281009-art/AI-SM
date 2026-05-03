import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# Cấu hình trang rộng và đẹp hơn
st.set_page_config(page_title="BNB AI Sentinel - Duy", layout="wide")

# CSS tùy chỉnh giao diện
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1 { color: #f3ba2f; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 BNB AI MARKET SENTINEL PRO")
st.caption("Phiên bản nâng cấp tối ưu - Thiết kế bởi Duy")

# Sidebar cấu hình
with st.sidebar:
    st.header("⚙️ Cấu hình AI")
    t_time = st.slider("Dự đoán sau (phút):", 1, 60, 15)
    st.markdown("---")
    st.write("Mô hình sử dụng: **Random Forest Classifier**")
    analyze_btn = st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True)

if analyze_btn:
    with st.spinner('AI đang quét sóng và học dữ liệu...'):
        # 1. Tải dữ liệu 7 ngày, khung 1 phút
        data = yf.download("BNB-USD", period="7d", interval="1m", progress=False)
        if data.empty:
            st.error("Không kết nối được dữ liệu thị trường!")
            st.stop()
            
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 2. Tăng cường chỉ báo kỹ thuật (Features)
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain/loss)))
        
        # Bollinger Bands
        df['STD'] = df['Close'].rolling(20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)
        
        df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)
        df.dropna(inplace=True)

        # 3. Huấn luyện AI chuyên sâu
        features = ['Close', 'MA5', 'MA20', 'RSI', 'Upper', 'Lower']
        X = df[features]
        y = df['Target']
        
        split = int(len(df) * 0.85) # Dùng 85% để học
        model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
        model.fit(X[:split], y[:split])
        
        acc = model.score(X[split:], y[split:]) * 100

        # 4. Dự đoán ván hiện tại
        last_data = X.iloc[[-1]]
        pred = model.predict(last_data)[0]
        prob = model.predict_proba(last_data)[0]

        # --- HIỂN THỊ KẾT QUẢ ---
        st.markdown("### 📊 Kết quả phân tích")
        col1, col2, col3 = st.columns(3)
        
        current_price = df['Close'].iloc[-1]
        col1.metric("Giá BNB Hiện Tại", f"${current_price:.2f}")
        
        res_text = "TĂNG 📈" if pred == 1 else "GIẢM 📉"
        res_color = "#00FF00" if pred == 1 else "#FF4B4B"
        col2.markdown(f"#### Dự báo sau {t_time}p: <br> <h2 style='color:{res_color}'>{res_text}</h2>", unsafe_allow_html=True)
        
        col3.metric("Độ tin cậy của AI", f"{prob[pred]*100:.1f}%")

        # 5. Biểu đồ nến chuyên nghiệp
        st.markdown("---")
        st.subheader("📈 Biểu đồ kỹ thuật (100 phút gần nhất)")
        chart_df = df.tail(100)
        fig = go.Figure(data=[go.Candlestick(x=chart_df.index,
                        open=chart_df['Open'], high=chart_df['High'],
                        low=chart_df['Low'], close=chart_df['Close'], name="BNB")])
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Upper'], line=dict(color='rgba(173, 216, 230, 0.5)'), name="Bollinger Upper"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Lower'], line=dict(color='rgba(173, 216, 230, 0.5)'), name="Bollinger Lower"))
        
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"Mô hình đạt độ chính xác thực tế {acc:.2f}% trong tuần qua.")

else:
    st.info("👋 Chào Duy! Hãy nhấn nút ở Sidebar để AI bắt đầu quét thị trường nhé.")
