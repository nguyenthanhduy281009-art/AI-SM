import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def predict_bnb(t_minutes=15):
    # 1. Lấy dữ liệu 5 ngày gần nhất, khoảng cách 1 phút
    data = yf.download(tickers='BNB-USD', period='5d', interval='1m')
    
    # 2. Tính toán biến động để làm dữ liệu huấn luyện (Features)
    data['Return'] = data['Close'].pct_change()
    data['Target'] = (data['Close'].shift(-t_minutes) > data['Close']).astype(int)
    
    # Chuẩn bị X (dữ liệu quá khứ) và y (kết quả sau t phút)
    data = data.dropna()
    X = data[['Close', 'High', 'Low', 'Volume', 'Return']].values
    y = data['Target'].values
    
    # 3. Huấn luyện mô hình nhanh (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:-1], y[:-1]) # Huấn luyện trừ dòng cuối cùng
    
    # 4. Dự đoán cho thời điểm hiện tại
    last_point = X[-1].reshape(1, -1)
    prediction = model.predict(last_point)[0]
    probability = model.predict_proba(last_point)[0]
    
    # 5. Kết quả
    trend = "TĂNG 🟢" if prediction == 1 else "GIẢM 🔴"
    confidence = probability[prediction] * 100
    
    print(f"--- DỰ ĐOÁN GIÁ BNB SAU {t_minutes} PHÚT ---")
    print(f"Xu hướng: {trend}")
    print(f"Tỷ lệ tin cậy: {confidence:.2f}%")

# Chạy thử
predict_bnb(t_minutes=15)
