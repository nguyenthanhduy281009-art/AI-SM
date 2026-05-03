import yfinance as yf
import pandas as pd
import numpy as np

def predict_bnb_fast(t_minutes=15):
    try:
        # 1. Lấy dữ liệu ngắn gọn hơn để tránh lag DevTools
        # Khoảng cách 2 ngày, mỗi phút 1 dòng
        data = yf.download('BNB-USD', period='2d', interval='1m', progress=False)
        
        if len(data) < 60:
            print("Đang lấy thêm dữ liệu, Duy đợi xíu nhé...")
            return

        # 2. Tính toán kỹ thuật đơn giản (SMA)
        # Tính đường trung bình động 10 phút
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        
        # Lấy giá hiện tại và giá 10 phút trước
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-10]
        
        # 3. Logic dự đoán dựa trên xu hướng (Trend Following)
        diff = current_price - prev_price
        change_pct = (diff / prev_price) * 100
        
        # Tính tỷ lệ xác suất dựa trên độ mạnh của xu hướng
        # Giả định cơ bản: Nếu đang tăng mạnh thì 65% tiếp tục tăng trong ngắn hạn
        confidence = 50 + min(abs(change_pct) * 10, 35) 
        
        trend = "TĂNG 📈" if diff > 0 else "GIẢM 📉"
        
        print(f"--- KẾT QUẢ CHO DUY ---")
        print(f"Giá BNB hiện tại: ${current_price:.2f}")
        print(f"Dự đoán sau {t_minutes}p: {trend}")
        print(f"Tỷ lệ tin cậy: {confidence:.1f}%")
        print("-----------------------")

    except Exception as e:
        print(f"Lỗi rồi Duy ơi: {e}")

# Chạy lệnh
predict_bnb_fast(15)
