import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier



st.title("🤖 AI Dự Đoán Giá BNB")



# Nhập dữ liệu từ người dùng

t_time = st.number_input("Bạn muốn dự đoán sau bao nhiêu phút?", min_value=1, value=5)

user_price = st.number_input("Nhập giá BNB hiện tại (USD):", min_value=0.0)



if st.button("Phân tích ngay"):

    with st.spinner('Đang học dữ liệu thị trường...'):

        # Lấy dữ liệu 7 ngày gần nhất, khung 1 phút

        data = yf.download("BNB-USD", period="7d", interval="1m")

        df = data[['Close']].copy()

        

        # TẠO CHIẾN THUẬT TĂNG TỶ LỆ ĐÚNG: Thêm các chỉ báo kỹ thuật

        df['MA5'] = df['Close'].rolling(5).mean()

        df['MA20'] = df['Close'].rolling(20).mean()

        df['Volatility'] = df['Close'].diff()

        

        # Gán nhãn: 1 là Tăng, 0 là Giảm sau t_time

        df['Target'] = (df['Close'].shift(-t_time) > df['Close']).astype(int)

        df.dropna(inplace=True)



        # Huấn luyện AI

        X = df[['Close', 'MA5', 'MA20', 'Volatility']]

        y = df['Target']

        

        split = int(len(df) * 0.8)

        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

        model.fit(X[:split], y[:split])

        

        # Tính tỷ lệ đúng thực tế

        acc = model.score(X[split:], y[split:]) * 100

        

        # Dự đoán ván hiện tại

        current_val = np.array([[user_price if user_price > 0 else df['Close'].iloc[-1], 

                                 df['MA5'].iloc[-1], df['MA20'].iloc[-1], df['Volatility'].iloc[-1]]])

        pred = model.predict(current_val)



        # Hiển thị kết quả

        st.success(f"Tỷ lệ đúng của mô hình trong 7 ngày qua: {acc:.2f}%")

        result = "TĂNG 📈" if pred[0] == 1 else "GIẢM 📉"

        st.header(f"Dự đoán sau {t_time} phút: {result}")
