
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="AI Market Analyzer V2.2", page_icon="📈", layout="wide")

VN = {
    "FPT": "FPT.VN", "VCB": "VCB.VN", "HPG": "HPG.VN", "MWG": "MWG.VN",
    "VHM": "VHM.VN", "VIC": "VIC.VN", "SSI": "SSI.VN", "MBB": "MBB.VN",
    "TCB": "TCB.VN", "VNM": "VNM.VN", "GAS": "GAS.VN", "CTG": "CTG.VN",
    "ACB": "ACB.VN", "STB": "STB.VN", "HDB": "HDB.VN",
}
CRYPTO = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}

def get_data(ticker, period, interval):
    if yf is None:
        raise RuntimeError("Thiếu yfinance.")
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"Không có dữ liệu cho {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [x[0] for x in df.columns]
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"]).copy()

def add_indicators(df):
    d = df.copy()
    d["EMA20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["EMA50"] = d.Close.ewm(span=50, adjust=False).mean()
    d["EMA200"] = d.Close.ewm(span=200, adjust=False).mean()
    delta = d.Close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - 100/(1+rs)
    e12 = d.Close.ewm(span=12, adjust=False).mean()
    e26 = d.Close.ewm(span=26, adjust=False).mean()
    d["MACD"] = e12-e26
    d["MACDSignal"] = d.MACD.ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        d.High-d.Low,
        (d.High-d.Close.shift()).abs(),
        (d.Low-d.Close.shift()).abs()
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["VolMA20"] = d.Volume.rolling(20).mean()
    d["VolRatio"] = d.Volume/d.VolMA20
    return d

def engine(d):
    x = d.dropna().iloc[-1]
    score = 0
    reasons = []
    if x.Close > x.EMA20:
        score += 1; reasons.append("Giá trên EMA20")
    else:
        score -= 1; reasons.append("Giá dưới EMA20")
    if x.EMA20 > x.EMA50:
        score += 1; reasons.append("EMA20 trên EMA50")
    else:
        score -= 1; reasons.append("EMA20 dưới EMA50")
    if x.Close > x.EMA200:
        score += 1; reasons.append("Giá trên EMA200")
    else:
        score -= 1; reasons.append("Giá dưới EMA200")
    if x.RSI14 >= 55:
        score += 1; reasons.append("RSI tích cực")
    elif x.RSI14 <= 45:
        score -= 1; reasons.append("RSI yếu")
    else:
        reasons.append("RSI trung tính")
    if x.MACD > x.MACDSignal:
        score += 1; reasons.append("MACD trên Signal")
    else:
        score -= 1; reasons.append("MACD dưới Signal")
    if pd.notna(x.VolRatio):
        reasons.append(f"Volume = {x.VolRatio:.2f}x trung bình 20 kỳ")
    if score >= 4:
        state = "TÍCH CỰC"
    elif score >= 2:
        state = "TÍCH CỰC NHẸ"
    elif score <= -4:
        state = "TIÊU CỰC"
    elif score <= -2:
        state = "TIÊU CỰC NHẸ"
    else:
        state = "TRUNG TÍNH"
    return int(score), state, reasons

def levels(d):
    x = d.dropna().iloc[-1]
    w = d.tail(60)
    support = float(w.Low.min())
    resistance = float(w.High.max())
    atr = float(x.ATR14)
    return support, resistance, float(x.Close-1.5*atr), float(x.Close+2*atr)

def rule_analysis(ticker, d):
    score, state, reasons = engine(d)
    x = d.dropna().iloc[-1]
    support, resistance, stop, target = levels(d)
    scenarios = []
    if x.Close > resistance * 0.99:
        scenarios.append("Giá đang gần vùng kháng cự 60 kỳ; cần theo dõi phản ứng giá và volume.")
    if x.Close > x.EMA50:
        scenarios.append("Giữ trên EMA50 giúp duy trì cấu trúc tăng trung hạn.")
    else:
        scenarios.append("Giá dưới EMA50; cần xác nhận lại xu hướng trước khi kỳ vọng tăng tiếp.")
    if x.MACD > x.MACDSignal:
        scenarios.append("MACD đang ủng hộ động lượng tăng.")
    else:
        scenarios.append("MACD chưa xác nhận động lượng tăng.")
    if x.RSI14 > 70:
        scenarios.append("RSI trên 70: động lượng cao nhưng rủi ro quá mua tăng.")
    elif x.RSI14 < 30:
        scenarios.append("RSI dưới 30: quá bán theo RSI, nhưng không đồng nghĩa chắc chắn đảo chiều.")
    return {
        "score": score, "state": state, "reasons": reasons,
        "support": support, "resistance": resistance,
        "stop": stop, "target": target,
        "scenarios": scenarios
    }

def ai_analysis(ticker, market, d, result):
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        return None, "Chưa cấu hình OPENAI_API_KEY. Bản phân tích quy tắc bên dưới vẫn dùng được."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        x = d.dropna().iloc[-1]
        recent = d.tail(20)[["Close","EMA20","EMA50","EMA200","RSI14","MACD","MACDSignal","ATR14","VolRatio"]]
        prompt = f"""
Bạn là trợ lý phân tích thị trường. Phân tích mô tả kỹ thuật, không hứa hẹn lợi nhuận,
không khẳng định chắc chắn giá sẽ tăng/giảm và không tự động đặt lệnh.

Tài sản: {ticker}
Thị trường: {market}
Điểm kỹ thuật quy tắc: {result['score']}/5
Trạng thái: {result['state']}
Giá hiện tại: {x.Close}
RSI14: {x.RSI14}
MACD: {x.MACD}
MACD Signal: {x.MACDSignal}
EMA20: {x.EMA20}
EMA50: {x.EMA50}
EMA200: {x.EMA200}
ATR14: {x.ATR14}
Volume ratio: {x.VolRatio}
Hỗ trợ 60 kỳ: {result['support']}
Kháng cự 60 kỳ: {result['resistance']}
Stop tham chiếu ATR: {result['stop']}
Target tham chiếu ATR: {result['target']}

Dữ liệu 20 kỳ gần nhất:
{recent.to_string()}

Viết bằng tiếng Việt, theo cấu trúc:
1. Xu hướng
2. Động lượng
3. Volume
4. Hỗ trợ/kháng cự
5. Kịch bản tích cực nếu...
6. Kịch bản tiêu cực nếu...
7. Điều cần xác nhận
Kết thúc bằng câu: "Đây là phân tích hỗ trợ quyết định, không phải dự báo chắc chắn."
"""
        resp = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
        )
        return resp.output_text, None
    except Exception as e:
        return None, f"Không gọi được AI: {e}"

# Sidebar
st.title("📈 AI Market Analyzer V2.2")
st.caption("Phân tích từng mã · ưu tiên kỹ thuật · BTC + cổ phiếu Việt Nam")

market = st.sidebar.radio("Thị trường", ["Cổ phiếu Việt Nam", "Bitcoin / Crypto"])
options = VN if market == "Cổ phiếu Việt Nam" else CRYPTO
name = st.sidebar.selectbox("Chọn mã", list(options.keys()))
ticker = st.sidebar.text_input("Ticker khác (tùy chọn)", "").strip().upper() or options[name]
period = st.sidebar.selectbox("Khoảng dữ liệu", ["6mo","1y","2y","5y"], index=1)
interval = st.sidebar.selectbox("Khung thời gian", ["1d","1wk"], index=0)
run = st.sidebar.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)

if run:
    try:
        raw = get_data(ticker, period, interval)
        data = add_indicators(raw)
        st.session_state.data = data
        st.session_state.ticker = ticker
        st.session_state.run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(str(e))

if "data" not in st.session_state:
    st.info("Chọn mã rồi bấm **PHÂN TÍCH**.")
    st.stop()

data = st.session_state.data
ticker = st.session_state.ticker
result = rule_analysis(ticker, data)
x = data.dropna().iloc[-1]
prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]
change = (x.Close/prev.Close-1)*100 if prev.Close else 0

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Giá", f"{x.Close:,.2f}", f"{change:+.2f}%")
c2.metric("RSI", f"{x.RSI14:.1f}")
c3.metric("MACD", f"{x.MACD:.3f}")
c4.metric("Điểm kỹ thuật", f"{result['score']:+d}/5")
c5.metric("ATR", f"{x.ATR14:,.2f}")

st.subheader(f"Trạng thái: {result['state']}")
st.write(" · ".join(result["reasons"]))

st.subheader("📊 Biểu đồ")
st.line_chart(data[["Close","EMA20","EMA50","EMA200"]].dropna().tail(300), height=420)

st.subheader("🎯 Mức tham chiếu")
a,b,c,e = st.columns(4)
a.metric("Hỗ trợ", f"{result['support']:,.2f}")
b.metric("Kháng cự", f"{result['resistance']:,.2f}")
c.metric("Stop ATR", f"{result['stop']:,.2f}")
e.metric("Target ATR", f"{result['target']:,.2f}")

st.caption("Stop/Target chỉ là phép tính tham chiếu từ ATR, không phải khuyến nghị mua/bán.")

st.subheader("🧠 Kịch bản kỹ thuật")
for s in result["scenarios"]:
    st.write("• " + s)

st.subheader("🤖 AI đọc tín hiệu")
if st.button("✨ Phân tích bằng AI", use_container_width=True):
    with st.spinner("AI đang đọc dữ liệu kỹ thuật..."):
        text, err = ai_analysis(ticker, market, data, result)
    if text:
        st.markdown(text)
    else:
        st.warning(err)
        st.info("Bạn vẫn có thể dùng phần phân tích kỹ thuật bên trên.")

with st.expander("🔎 Dữ liệu chỉ báo"):
    st.dataframe(
        data[["Close","EMA20","EMA50","EMA200","RSI14","MACD","MACDSignal","ATR14","VolRatio"]].tail(30),
        use_container_width=True
    )

st.divider()
st.caption(
    f"Cập nhật: {st.session_state.get('run_at','N/A')} · "
    "Nguồn giá: Yahoo Finance qua yfinance."
)
