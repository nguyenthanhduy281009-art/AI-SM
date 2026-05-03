import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

st.set_page_config(page_title="AI BNB Predictor", layout="wide")

st.title("🚀 AI Dự Đoán Giá BNB (Ultimate Version)")

# ===== INPUT =====
t_time = st.slider("⏱ Dự đoán sau bao nhiêu phút?", 1, 30, 5)

user_price = st.number_input(
    "💰 Nhập giá BNB hiện tại (có thể bỏ trống để dùng giá realtime)",
    min_value=0.0,
    value=0.0
)

# ===== GET DATA (CÓ BACKUP API) =====
@st.cache_data(ttl=60)
def get_data():
    # ---- thử Binance ----
    try:
        url = "https://api1.binance.com/api/v3/klines"
        headers = {"User-Agent": "Mozilla/5.0"}

        params = {
            "symbol": "BNBUSDT",
            "interval": "1m",
            "limit": 500
        }

        res = requests.get(url, params=params, headers=headers, timeout=10)
        data = res.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
            df = df[[0, 4, 5]]
            df.columns = ['time', 'close', 'volume']

            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            return df, "Binance"

    except:
        pass

    # ---- fallback CoinGecko ----
    try:
        url = "https://api.coingecko.com/api/v3/coins/binancecoin/market_chart"
        params = {"vs_currency": "usd", "days": 1}

        res = requests.get(url, timeout=10)
        data = res.json()

        prices = data["prices"]

        df = pd.DataFrame(prices, columns=["time", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        df["volume"] = 0

        return df, "CoinGecko"

    except:
        return None, None


df, source = get_data()

if df is None or len(df) < 50:
    st.error("❌ Không lấy được dữ liệu từ mọi nguồn.")
    st.stop()

st.success(f"✅ Đang dùng dữ liệu từ: {source}")

# ===== INDICATORS =====
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

# ===== TARGET =====
df['Target'] = (df['close'].shift(-t_time) > df['close']).astype(int)
df = df.dropna()

if len(df) < 100:
    st.error("❌ Không đủ dữ liệu train.")
    st.stop()

# ===== TRAIN =====
features = ['close', 'volume', 'EMA10', 'EMA20', 'RSI', 'Volatility', 'MACD']

X = df[features]
y = df['Target']

split = int(len(df) * 0.8)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingClassifier(n_estimators=120)
model.fit(X_scaled[:split], y[:split])

acc = model.score(X_scaled[split:], y[split:]) * 100

# ===== GIÁ HIỆN TẠI =====
real_price = df['close'].iloc[-1]

current_price = user_price if user_price > 0 else real_price

# ===== PREDICT =====
latest = X.iloc[-1:].copy()

# thay giá nếu user nhập
latest['close'] = current_price

latest_scaled = scaler.transform(latest)

proba = model.predict_proba(latest_scaled)[0]

up_prob = float(proba[1] * 100)
down_prob = float(proba[0] * 100)

# ===== UI =====
col1, col2 = st.columns(2)

with col1:
    st.metric("💰 Giá hiện tại", f"{current_price:.2f} USD")

    st.metric("📊 Accuracy (ước tính)", f"{acc:.2f}%")

    if up_prob > down_prob:
        st.success(f"📈 Tăng ({up_prob:.1f}%)")
    else:
        st.error(f"📉 Giảm ({down_prob:.1f}%)")

with col2:
    st.subheader("📉 Biểu đồ")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA10'], name='EMA10'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA20'], name='EMA20'))

    st.plotly_chart(fig, use_container_width=True)

# ===== NOTE =====
st.info("👉 Nếu không nhập giá, hệ thống sẽ dùng giá realtime.")
st.warning("⚠️ AI chỉ mang tính tham khảo. Không đảm bảo thắng khi trade.")
