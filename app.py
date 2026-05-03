import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="AI BNB Predictor", layout="wide")

st.title("🚀 AI Dự Đoán Giá BNB (Stable Version)")

t_time = st.slider("⏱ Dự đoán sau bao nhiêu phút?", 1, 30, 5)

# ====== LẤY DATA ======
@st.cache_data(ttl=60)
def get_binance_data():
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BNBUSDT",
            "interval": "1m",
            "limit": 500
        }
        res = requests.get(url, params=params, timeout=10)
        data = res.json()

        # kiểm tra lỗi API
        if not isinstance(data, list):
            return None

        df = pd.DataFrame(data)
        df = df[[0, 4, 5]]
        df.columns = ['time', 'close', 'volume']

        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        return df

    except:
        return None


df = get_binance_data()

if df is None or len(df) < 50:
    st.error("❌ Không lấy được dữ liệu Binance. Thử lại sau.")
    st.stop()

# ====== INDICATORS ======
def add_indicators(df):
    df = df.copy()

    df['EMA10'] = df['close'].ewm(span=10).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()

    delta = df['close'].diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()

    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Volatility'] = df['close'].pct_change()

    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()

    return df


df = add_indicators(df)

# ====== TARGET ======
df['Target'] = (df['close'].shift(-t_time) > df['close']).astype(int)

df = df.dropna()

if len(df) < 100:
    st.error("❌ Dữ liệu không đủ để train model.")
    st.stop()

# ====== TRAIN ======
features = ['close', 'volume', 'EMA10', 'EMA20', 'RSI', 'Volatility', 'MACD']

X = df[features]
y = df['Target']

split = int(len(df) * 0.8)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(n_estimators=120)
model.fit(X_scaled[:split], y[:split])

acc = model.score(X_scaled[split:], y[split:]) * 100

# ====== PREDICT ======
latest = X.iloc[-1:].values
latest_scaled = scaler.transform(latest)

proba = model.predict_proba(latest_scaled)[0]

up_prob = float(proba[1] * 100)
down_prob = float(proba[0] * 100)

# ====== UI ======
col1, col2 = st.columns(2)

with col1:
    st.metric("📊 Accuracy (ước tính)", f"{acc:.2f}%")

    if up_prob > down_prob:
        st.success(f"📈 Tăng ({up_prob:.1f}%)")
    else:
        st.error(f"📉 Giảm ({down_prob:.1f}%)")

with col2:
    st.subheader("📉 Biểu đồ giá")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA10'], name='EMA10'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA20'], name='EMA20'))

    st.plotly_chart(fig, use_container_width=True)

st.warning("⚠️ AI chỉ mang tính tham khảo. Không đảm bảo thắng khi trade.")
