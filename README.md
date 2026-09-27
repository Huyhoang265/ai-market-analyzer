# AI Market Analyzer V2.2

V2.2 tập trung vào đúng lựa chọn:
- BTC + cổ phiếu Việt Nam
- Ưu tiên phân tích kỹ thuật
- Xem từng mã
- Có thêm lớp AI giải thích tín hiệu và xây dựng các kịch bản.

## Chạy
pip install -r requirements.txt
streamlit run app.py

## Cấu hình AI khi deploy
Thêm secret:
OPENAI_API_KEY = "your_api_key"

Nếu chưa có API key, phần phân tích kỹ thuật quy tắc vẫn chạy; nút AI sẽ báo chưa cấu hình.

Model AI hiện dùng: gpt-5.6-luna.
