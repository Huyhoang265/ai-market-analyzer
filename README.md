# AI Market Analyzer V2.1

## Mục tiêu
Theo lựa chọn của người dùng:
- Cả BTC/Crypto và cổ phiếu Việt Nam
- Ưu tiên phân tích kỹ thuật
- Mở app và xem từng mã

## Chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Có gì mới so với V2
- Chọn thị trường và mã ngay trên sidebar.
- Có sẵn danh sách một số mã VN và BTC/ETH.
- Có thể nhập ticker Yahoo Finance khác.
- Tự tải dữ liệu giá.
- Technical engine: EMA20/50/200, RSI14, MACD, ATR14, volume ratio.
- Hỗ trợ/kháng cự 60 kỳ.
- Stop/Target tham chiếu theo ATR.
- Dashboard từng mã, không cần nhập dữ liệu thủ công.
- Giữ module đọc PDF cơ bản.

## Lưu ý
Đây là prototype phân tích, không phải hệ thống dự báo chắc chắn. Stop/Target là mức tham chiếu tính từ ATR, không phải khuyến nghị.
