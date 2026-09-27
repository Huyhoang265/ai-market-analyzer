
import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None

st.set_page_config(page_title="AI Market Analyzer V2.1", page_icon="📈", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.small-note {font-size: 0.85rem; color: #666;}
.signal-box {padding: 1rem; border: 1px solid #ddd; border-radius: 12px; margin: .5rem 0;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Market Analyzer V2.1")
st.caption("BTC + Cổ phiếu Việt Nam | Ưu tiên phân tích kỹ thuật | Xem từng mã")

# ---------- Data ----------
VN_PRESETS = {
    "FPT": "FPT.VN", "VCB": "VCB.VN", "HPG": "HPG.VN", "MWG": "MWG.VN",
    "VHM": "VHM.VN", "VIC": "VIC.VN", "SSI": "SSI.VN", "MBB": "MBB.VN",
    "TCB": "TCB.VN", "VNM": "VNM.VN", "GAS": "GAS.VN", "CTG": "CTG.VN",
    "ACB": "ACB.VN", "STB": "STB.VN", "HDB": "HDB.VN",
}
BTC_PRESETS = {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}

def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def download_data(ticker, period, interval):
    if yf is None:
        raise RuntimeError("Chưa cài yfinance.")
    df = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"Không có dữ liệu cho {ticker}.")
    df = flatten_columns(df)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Thiếu cột {c}.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Close"]).copy()

def indicators(df):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACDHist"] = d["MACD"] - d["MACDSignal"]

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - d["Close"].shift()).abs(),
        (d["Low"] - d["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    d["VolSMA20"] = d["Volume"].rolling(20).mean()
    d["VolRatio"] = d["Volume"] / d["VolSMA20"]

    mid = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["BBMid"] = mid
    d["BBUpper"] = mid + 2 * std
    d["BBLower"] = mid - 2 * std
    return d

def support_resistance(df, window=60):
    x = df.tail(window)
    support = float(x["Low"].min())
    resistance = float(x["High"].max())
    return support, resistance

def technical_engine(d):
    x = d.dropna().iloc[-1]
    score = 0
    reasons = []

    # Trend
    if x.Close > x.EMA20:
        score += 1; reasons.append("Giá > EMA20")
    else:
        score -= 1; reasons.append("Giá < EMA20")

    if x.EMA20 > x.EMA50:
        score += 1; reasons.append("EMA20 > EMA50")
    else:
        score -= 1; reasons.append("EMA20 < EMA50")

    if x.Close > x.EMA200:
        score += 1; reasons.append("Giá > EMA200")
    else:
        score -= 1; reasons.append("Giá < EMA200")

    # Momentum
    if x.RSI14 >= 55:
        score += 1; reasons.append("RSI trên 55")
    elif x.RSI14 <= 45:
        score -= 1; reasons.append("RSI dưới 45")
    else:
        reasons.append("RSI trung tính")

    if x.MACD > x.MACDSignal:
        score += 1; reasons.append("MACD > Signal")
    else:
        score -= 1; reasons.append("MACD < Signal")

    # Volume confirmation
    if pd.notna(x.VolRatio):
        if x.VolRatio >= 1.2:
            reasons.append("Volume cao hơn trung bình")
        elif x.VolRatio < 0.7:
            reasons.append("Volume thấp")

    if score >= 4:
        state = "XU HƯỚNG TÍCH CỰC"
    elif score >= 2:
        state = "TÍCH CỰC NHẸ"
    elif score <= -4:
        state = "XU HƯỚNG TIÊU CỰC"
    elif score <= -2:
        state = "TIÊU CỰC NHẸ"
    else:
        state = "TRUNG TÍNH"

    return score, state, reasons

def risk_levels(d, entry=None):
    x = d.dropna().iloc[-1]
    price = float(x.Close if entry is None else entry)
    atr = float(x.ATR14)
    support, resistance = support_resistance(d)
    stop_atr = price - 1.5 * atr
    target_atr = price + 2.0 * atr
    return support, resistance, stop_atr, target_atr, atr

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Chọn thị trường")
market = st.sidebar.radio("Thị trường", ["Cổ phiếu Việt Nam", "Bitcoin / Crypto"])

if market == "Cổ phiếu Việt Nam":
    name = st.sidebar.selectbox("Chọn mã", list(VN_PRESETS.keys()), index=0)
    ticker = VN_PRESETS[name]
else:
    name = st.sidebar.selectbox("Tài sản", list(BTC_PRESETS.keys()), index=0)
    ticker = BTC_PRESETS[name]

custom = st.sidebar.text_input("Hoặc nhập ticker Yahoo Finance", "")
if custom.strip():
    ticker = custom.strip().upper()

period = st.sidebar.selectbox("Dữ liệu", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Khung", ["1d", "1wk"], index=0)
run = st.sidebar.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("V2.1 tập trung vào phân tích kỹ thuật từng mã. Không tự đặt lệnh.")

# ---------- Main ----------
st.subheader(f"{name}  ·  {ticker}")

if run:
    try:
        raw = download_data(ticker, period, interval)
        data = indicators(raw)
        st.session_state["data"] = data
        st.session_state["ticker"] = ticker
        st.session_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(str(e))

if "data" not in st.session_state:
    st.info("Chọn mã ở bên trái rồi bấm **PHÂN TÍCH**.")
    st.stop()

data = st.session_state["data"]
score, state, reasons = technical_engine(data)
support, resistance, stop_atr, target_atr, atr = risk_levels(data)

last = data.iloc[-1]
prev = data.iloc[-2] if len(data) > 1 else last
change_pct = (last.Close / prev.Close - 1) * 100 if prev.Close else np.nan

# ---------- Header metrics ----------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Giá", f"{last.Close:,.2f}", f"{change_pct:+.2f}%")
c2.metric("RSI14", f"{last.RSI14:.1f}")
c3.metric("MACD", f"{last.MACD:.3f}")
c4.metric("Điểm kỹ thuật", f"{score:+d}/5")
c5.metric("ATR14", f"{atr:,.2f}")

st.markdown(f"### {state}")
st.write(" · ".join(reasons))

# ---------- Chart ----------
st.subheader("📊 Biểu đồ xu hướng")
chart_df = data[["Close", "EMA20", "EMA50", "EMA200"]].dropna().tail(300)
st.line_chart(chart_df, height=430)

# ---------- Levels ----------
st.subheader("🎯 Các mức tham chiếu")
l1, l2, l3, l4 = st.columns(4)
l1.metric("Hỗ trợ 60 kỳ", f"{support:,.2f}")
l2.metric("Kháng cự 60 kỳ", f"{resistance:,.2f}")
l3.metric("Stop tham chiếu ATR", f"{stop_atr:,.2f}")
l4.metric("Target tham chiếu ATR", f"{target_atr:,.2f}")

st.caption(
    "Các mức Stop/Target chỉ là phép tính tham chiếu từ ATR, không phải dự báo giá "
    "hay khuyến nghị giao dịch."
)

# ---------- Decision framework ----------
st.subheader("🧠 Đọc tín hiệu")
items = []
if last.Close > last.EMA50:
    items.append("Xu hướng trung hạn đang nằm trên EMA50.")
else:
    items.append("Giá đang dưới EMA50, cần thận trọng với xu hướng trung hạn.")

if last.Close > last.EMA200:
    items.append("Giá đang trên EMA200.")
else:
    items.append("Giá đang dưới EMA200.")

if 55 <= last.RSI14 <= 70:
    items.append("RSI thuộc vùng động lượng tích cực nhưng chưa vào vùng quá mua mạnh.")
elif last.RSI14 > 70:
    items.append("RSI trên 70: động lượng cao, cần xem thêm khả năng quá mua.")
elif last.RSI14 < 30:
    items.append("RSI dưới 30: đang quá bán theo RSI, nhưng quá bán không đồng nghĩa chắc chắn đảo chiều.")
else:
    items.append("RSI chưa cho tín hiệu động lượng mạnh.")

if last.MACD > last.MACDSignal:
    items.append("MACD đang trên đường Signal.")
else:
    items.append("MACD đang dưới đường Signal.")

for i in items:
    st.write("• " + i)

# ---------- Data ----------
with st.expander("🔎 Xem dữ liệu chỉ báo"):
    cols = ["Close","EMA20","EMA50","EMA200","RSI14","MACD","MACDSignal","MACDHist","ATR14","VolRatio"]
    st.dataframe(data[cols].tail(30), use_container_width=True)

st.divider()
st.caption(
    f"Cập nhật phiên phân tích: {st.session_state.get('last_run', 'N/A')} · "
    "Nguồn giá: Yahoo Finance qua yfinance. Dữ liệu có thể trễ hoặc có giới hạn theo nguồn."
)

# ---------- Optional PDF ----------
with st.expander("📄 Đọc nhanh BCTC PDF (giữ lại từ V2)"):
    st.write("Upload PDF để trích text cơ bản. Module AI đọc bảng/OCR sẽ nâng cấp ở V2.2.")
    pdf = st.file_uploader("PDF BCTC", type=["pdf"], key="pdf_v21")
    if pdf:
        try:
            import pypdf
            reader = pypdf.PdfReader(pdf)
            txt = "\n".join((p.extract_text() or "") for p in reader.pages)
            st.success(f"Đọc được {len(txt):,} ký tự.")
            st.text(txt[:15000])
        except Exception as e:
            st.error(f"Không đọc được PDF: {e}")
