"""
TradeLab Pro • Cash Market Scanner & Risk Engine (v2)
------------------------------------------------------
চালানোর নিয়ম:
    pip install streamlit pandas numpy yfinance requests
    streamlit run tradelab_pro.py

ডেটা সোর্স: Upstox (Analytics Token) | yfinance | CSV আপলোড | Demo (অফলাইন)
Educational only. Not SEBI-registered advice.
"""
import gzip
import io
import json
import os
import time
import zlib
from datetime import date, timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# ১. ইন্ডিকেটর (Wilder smoothing = TradingView-এর সাথে মিলবে)
# ============================================================
def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def calc_rsi(s, p=14):
    d = s.diff()
    avg_g = d.clip(lower=0).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    avg_l = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()
    rsi = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    return rsi.where(avg_l != 0, 100.0)


def calc_macd(s, fast=12, slow=26, sig=9):
    line = calc_ema(s, fast) - calc_ema(s, slow)
    signal = calc_ema(line, sig)
    return line, signal, line - signal


def true_range(df):
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)


def calc_atr(df, p=14):
    return true_range(df).ewm(alpha=1 / p, adjust=False, min_periods=p).mean()


def calc_adx(df, p=14):
    up, down = df["High"].diff(), -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = calc_atr(df, p)
    pdi = 100 * plus_dm.ewm(alpha=1 / p, adjust=False, min_periods=p).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / p, adjust=False, min_periods=p).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / p, adjust=False, min_periods=p).mean(), pdi, mdi


def calc_supertrend(df, p=10, mult=3.0):
    """রিটার্ন: direction সিরিজ (+1 = আপট্রেন্ড, -1 = ডাউনট্রেন্ড, 0 = ডেটা নেই)"""
    atr = calc_atr(df, p).to_numpy()
    hl2 = ((df["High"] + df["Low"]) / 2).to_numpy()
    c = df["Close"].to_numpy()
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    n = len(df)
    fub, flb, dr = ub.copy(), lb.copy(), np.zeros(n)
    start = int(np.argmax(~np.isnan(atr))) if (~np.isnan(atr)).any() else n
    for i in range(start, n):
        if i == start:
            dr[i] = 1 if c[i] > hl2[i] else -1
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if dr[i - 1] == 1:
            dr[i] = -1 if c[i] < flb[i] else 1
        else:
            dr[i] = 1 if c[i] > fub[i] else -1
    return pd.Series(dr, index=df.index)


# ============================================================
# ২. ডেটা প্রোভাইডার (সব একই আউটপুট দেয়: Open/High/Low/Close/Volume, DatetimeIndex)
# ============================================================
LOOKBACK_DAYS = {"6 Months": 183, "1 Year": 365, "2 Years": 730, "5 Years": 1825}
YF_PERIOD = {"6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
YF_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Hourly": "1h"}
UP_UNIT = {"Daily": ("days", 1), "Weekly": ("weeks", 1), "Hourly": ("hours", 1)}
DEFAULT_SYMBOLS = (
    "RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, SBIN, BHARTIARTL, ITC, LT, AXISBANK, "
    "KOTAKBANK, TATAMOTORS, MARUTI, SUNPHARMA, M&M, NTPC, POWERGRID, TITAN, ULTRACEMCO, ADANIPORTS"
)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def clean(df):
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[[c for c in OHLCV if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"]).sort_index()[~df.index.duplicated()]


def demo_data(symbol, bars=520):
    """একই সিম্বলে সবসময় একই ডেটা (deterministic)।"""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    drift = rng.uniform(-0.0002, 0.0009)
    ret = rng.normal(drift, rng.uniform(0.008, 0.02), bars)
    close = rng.uniform(150, 3000) * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.012, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.012, bars))
    vol = rng.uniform(5e5, 5e6, bars) * (1 + 2 * (np.abs(ret) > 0.02))
    idx = pd.bdate_range(end=date.today(), periods=bars)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@st.cache_data(ttl=3600, show_spinner=False)
def load_instruments(exchange):
    """Upstox instruments ফাইল → {trading_symbol: instrument_key}. URL ঠিক আছে কিনা Upstox ডক থেকে যাচাই করুন।"""
    import requests
    url = f"https://assets.upstox.com/market-assets/instruments/exchange/{exchange}.json.gz"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    rows = json.loads(gzip.decompress(r.content))
    seg = f"{exchange}_EQ"
    return {x["trading_symbol"]: x["instrument_key"] for x in rows if x.get("segment") == seg and x.get("instrument_type") == "EQ"}


def upstox_fetch(key, timeframe, lookback, token):
    import requests
    unit, interval = UP_UNIT[timeframe]
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    end, start = date.today(), date.today() - timedelta(days=LOOKBACK_DAYS[lookback])
    step = 85 if unit == "hours" else 100000  # ঘণ্টার ডেটা ছোট ছোট টুকরোয় আনা হয়
    frames, hi = [], end
    while hi > start:
        lo = max(start, hi - timedelta(days=step))
        url = f"https://api.upstox.com/v3/historical-candle/{quote(key, safe='')}/{unit}/{interval}/{hi}/{lo}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Upstox {r.status_code}: {r.text[:150]}")
        candles = r.json().get("data", {}).get("candles", [])
        if candles:
            d = pd.DataFrame(candles).iloc[:, :6]
            d.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]
            d.index = pd.to_datetime(d.pop("ts"), utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            frames.append(d)
        hi = lo - timedelta(days=1)
        time.sleep(0.12)  # নরম রেট-লিমিট
    if not frames:
        raise RuntimeError("Upstox: কোনো ক্যান্ডেল পাওয়া যায়নি")
    return clean(pd.concat(frames))


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(provider, symbol, exchange, timeframe, lookback, token=""):
    """ব্যর্থ হলে exception দেয় (ফলে ভুল ফলাফল ক্যাশ হয় না)।"""
    if provider == "Deterministic Demo (Offline)":
        return demo_data(symbol)
    if provider == "Upstox API":
        key = symbol if "|" in symbol else load_instruments(exchange).get(symbol)
        if not key:
            raise RuntimeError(f"'{symbol}' Upstox ইন্সট্রুমেন্ট লিস্টে নেই")
        return upstox_fetch(key, timeframe, lookback, token)
    if provider == "Live yfinance API":
        import yfinance as yf
        tk = symbol if symbol.startswith("^") else symbol + (".NS" if exchange == "NSE" else ".BO")
        df = yf.Ticker(tk).history(period=YF_PERIOD[lookback], interval=YF_INTERVAL[timeframe], auto_adjust=True)
        if df.empty:
            raise RuntimeError(f"yfinance: {tk} এর ডেটা নেই")
        return clean(df)
    raise RuntimeError("অজানা প্রোভাইডার")


def parse_csv(f):
    df = pd.read_csv(f)
    df.columns = [c.strip().title() for c in df.columns]
    df = df.set_index(pd.to_datetime(df.pop("Date")))
    return clean(df)


# ============================================================
# ৩. রুল ইঞ্জিন
# ============================================================
RULES = ["EMA Crossover", "Close vs EMA200", "RSI Zone", "MACD", "Supertrend", "ADX + DI", "Volume Spike", "Breakout", "Pullback to EMA"]
PRESETS = {
    "Trend Follower 🚀": ["EMA Crossover", "Close vs EMA200", "Supertrend", "ADX + DI"],
    "Breakout Scanner ⚡": ["Breakout", "Volume Spike", "ADX + DI"],
    "Pullback Entry 🎯": ["Close vs EMA200", "RSI Zone", "Pullback to EMA"],
}


def build_indicators(df, c):
    ind = {
        "ema_f": calc_ema(df["Close"], c["fast"]), "ema_s": calc_ema(df["Close"], c["slow"]),
        "ema50": calc_ema(df["Close"], 50), "ema200": calc_ema(df["Close"], 200),
        "rsi": calc_rsi(df["Close"]), "atr": calc_atr(df),
        "st": calc_supertrend(df), "vol_avg": df["Volume"].rolling(20).mean(),
    }
    ind["macd"], ind["macd_sig"], ind["macd_hist"] = calc_macd(df["Close"])
    ind["adx"], ind["pdi"], ind["mdi"] = calc_adx(df)
    return ind


def rule_signals(df, ind, c):
    """हर रুল → (bull, bear) boolean সিরিজ।"""
    cl, hi, lo, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    no = pd.Series(False, index=df.index)
    strong = ind["adx"] >= c["adx_min"]
    spike = vol > c["vol_mult"] * ind["vol_avg"]
    return {
        "EMA Crossover": (ind["ema_f"] > ind["ema_s"], ind["ema_f"] < ind["ema_s"]),
        "Close vs EMA200": (cl > ind["ema200"], cl < ind["ema200"]),
        "RSI Zone": ((ind["rsi"] >= c["rsi_lo"]) & (ind["rsi"] <= c["rsi_hi"]),
                     (ind["rsi"] <= 100 - c["rsi_lo"]) & (ind["rsi"] >= 100 - c["rsi_hi"])),
        "MACD": (ind["macd"] > ind["macd_sig"], ind["macd"] < ind["macd_sig"]),
        "Supertrend": (ind["st"] == 1, ind["st"] == -1),
        "ADX + DI": (strong & (ind["pdi"] > ind["mdi"]), strong & (ind["mdi"] > ind["pdi"])),
        "Volume Spike": (spike, spike),
        "Breakout": (cl > hi.rolling(c["brk_n"]).max().shift(), cl < lo.rolling(c["brk_n"]).min().shift()),
        "Pullback to EMA": ((cl > ind["ema200"]) & (lo <= ind["ema_s"] * 1.005) & (cl > ind["ema_s"]), no),
    }


def combine(sig, active, logic, n):
    if not active:
        f = pd.Series(False, index=next(iter(sig.values()))[0].index)
        return f, f, pd.Series(0, index=f.index), pd.Series(0, index=f.index)
    bull = pd.concat([sig[r][0] for r in active], axis=1).sum(axis=1)
    bear = pd.concat([sig[r][1] for r in active], axis=1).sum(axis=1)
    m = len(active)
    need = m if logic == "All Enabled" else 1 if logic == "ANY Enabled" else min(n, m)
    return bull >= need, bear >= need, bull, bear


# ============================================================
# ৪. স্কোর + পজিশন সাইজিং
# ============================================================
def quality_score(df, ind, side, bench, c):
    s = 1 if side == "BUY" else -1
    last = lambda x: float(x.iloc[-1])
    cl = last(df["Close"])
    trend = np.mean([s * (cl - last(ind["ema50"])) > 0, s * (last(ind["ema_f"]) - last(ind["ema50"])) > 0, s * (cl - last(ind["ema200"])) > 0])
    rsi = last(ind["rsi"])
    mom = 0.6 * np.clip((s * (rsi - 50) + 10) / 30, 0, 1) + 0.4 * float(s * last(ind["macd_hist"]) > 0)
    vol = np.clip(last(df["Volume"]) / max(last(ind["vol_avg"]), 1) / 2, 0, 1)
    rs = 0.5
    k = c["rs_n"]
    if bench is not None and len(bench) > k and len(df) > k:
        diff = (cl / df["Close"].iloc[-k - 1] - 1) - (bench["Close"].iloc[-1] / bench["Close"].iloc[-k - 1] - 1)
        rs = float(np.clip(0.5 + s * diff / 0.2, 0, 1))
    atr_pct = last(ind["atr"]) / cl
    vola = 1 - float(np.clip(abs(atr_pct - 0.02) / 0.03, 0, 1))
    w = np.array([c["w_trend"], c["w_mom"], c["w_vol"], c["w_rs"], c["w_vola"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(5, 0.2)
    return float(100 * np.dot(w, [trend, mom, vol, rs, vola]))


def position_plan(price, atr, side, c):
    risk_ps = atr * c["stop_m"]
    if not np.isfinite(risk_ps) or risk_ps <= 0:
        return None
    s = 1 if side == "BUY" else -1
    stop, target = price - s * risk_ps, price + s * atr * c["tgt_m"]
    cost = price * c["cost_pct"] / 100
    net_rr = (atr * c["tgt_m"] - cost) / (risk_ps + cost)
    qty = int(min((c["capital"] * c["risk_pct"] / 100) // (risk_ps + cost), (c["capital"] * c["max_exp"] / 100) // price))
    return dict(stop=stop, target=target, net_rr=net_rr, qty=max(qty, 0))


def analyze(sym, df, bench, c):
    if len(df) < 60:
        return None
    ind = build_indicators(df, c)
    sig = rule_signals(df, ind, c)
    buy, sell, nb, ns = combine(sig, c["active"], c["logic"], c["n_min"])
    b, s_ = bool(buy.iloc[-1]), bool(sell.iloc[-1])
    if c["direction"] == "BUY":
        s_ = False
    elif c["direction"] == "SELL":
        b = False
    signal = "BUY 🟢" if b and not s_ else "SELL 🔴" if s_ and not b else "NEUTRAL ⚪"
    side = "SELL" if signal.startswith("SELL") else "BUY"
    price = float(df["Close"].iloc[-1])
    score = quality_score(df, ind, side, bench, c)
    plan = position_plan(price, float(ind["atr"].iloc[-1]), side, c)
    turnover = float((df["Close"] * df["Volume"]).tail(20).mean() / 1e7)
    reasons = []
    if signal.startswith("NEUTRAL"):
        reasons.append("সিগন্যাল নেই")
    if score < c["min_score"]:
        reasons.append("স্কোর কম")
    if plan is None:
        reasons.append("ATR নেই")
    else:
        if plan["net_rr"] < c["min_rr"]:
            reasons.append("R:R কম")
        if plan["qty"] == 0:
            reasons.append("মূলধন/রিস্ক অপর্যাপ্ত")
    return {
        "Symbol": sym, "Price (₹)": round(price, 2), "Signal": signal, "Score": round(score),
        "Rules": f"{int(nb.iloc[-1] if side == 'BUY' else ns.iloc[-1])}/{len(c['active'])}",
        "Qty": plan["qty"] if plan else 0, "Stop": round(plan["stop"], 2) if plan else None,
        "Target": round(plan["target"], 2) if plan else None,
        "Net R:R": round(plan["net_rr"], 2) if plan else None,
        "Turnover (Cr)": round(turnover, 1), "Status": "✅ PASS" if not reasons else "❌ " + ", ".join(reasons),
    }


# ============================================================
# ৫. ব্যাকটেস্ট (long-only, পরের বারের Open-এ এন্ট্রি → look-ahead নেই)
# ============================================================
def backtest(sym, df, c, max_hold=20):
    ind = build_indicators(df, c)
    buy = combine(rule_signals(df, ind, c), c["active"], c["logic"], c["n_min"])[0].to_numpy()
    o, h, l, cl = (df[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    atr = ind["atr"].to_numpy()
    idx, trades, i, n = df.index, [], 0, len(df)
    while i < n - 1:
        if not (buy[i] and np.isfinite(atr[i])):
            i += 1
            continue
        j, entry = i + 1, o[i + 1]
        risk = atr[i] * c["stop_m"]
        stop, target = entry - risk, entry + atr[i] * c["tgt_m"]
        exit_k, exit_p, why = min(j + max_hold, n - 1), None, "Time"
        for k in range(j, min(j + max_hold, n - 1) + 1):
            if k > j and o[k] <= stop:
                exit_k, exit_p, why = k, o[k], "Stop(gap)"; break
            if l[k] <= stop:
                exit_k, exit_p, why = k, stop, "Stop"; break  # একই বারে দুটোই ছুঁলে Stop ধরা হয় (রক্ষণশীল)
            if h[k] >= target:
                exit_k, exit_p, why = k, target, "Target"; break
        if exit_p is None:
            exit_p = cl[exit_k]
        r = (exit_p - entry - entry * c["cost_pct"] / 100) / risk
        trades.append({"Symbol": sym, "Entry": idx[j], "Exit": idx[exit_k], "EntryPx": round(entry, 2),
                       "ExitPx": round(exit_p, 2), "Reason": why, "R": round(r, 2)})
        i = exit_k + 1
    return trades


def bt_stats(tr):
    if tr.empty:
        return {}
    r = tr["R"].to_numpy()
    eq = np.cumsum(r)
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return {"Trades": len(r), "Win %": round(100 * (r > 0).mean(), 1), "Avg R (Expectancy)": round(r.mean(), 3),
            "Profit Factor": round(gp / gl, 2) if gl > 0 else float("inf"), "Total R": round(eq[-1], 2),
            "Max Drawdown (R)": round(float((eq - np.maximum.accumulate(eq)).min()), 2)}


# ============================================================
# ===== UI =====
# ============================================================
def get_token():
    for getter in (lambda: st.secrets["UPSTOX_TOKEN"], lambda: os.environ["UPSTOX_TOKEN"]):
        try:
            return getter()
        except Exception:
            pass
    return ""


def run_self_test():
    df = demo_data("TEST")
    c = dict(fast=9, slow=21)
    ind = build_indicators(df, c)
    checks = {
        "RSI 0–100": bool(ind["rsi"].dropna().between(0, 100).all()),
        "ATR > 0": bool((ind["atr"].dropna() > 0).all()),
        "ADX 0–100": bool(ind["adx"].dropna().between(0, 100).all()),
        "Supertrend ±1": bool(set(ind["st"].unique()) <= {-1, 0, 1}),
        "Demo ডেটা deterministic": bool(demo_data("TEST").equals(df)),
    }
    return checks


def main():
    st.set_page_config(page_title="TradeLab Pro", page_icon="⚡", layout="wide")
    st.title("⚡ TradeLab Pro • Cash Market Scanner & Risk Engine")
    st.caption("Educational only. Not SEBI-registered advice. সিগন্যাল মানেই লাভের নিশ্চয়তা নয়।")
    ss = st.session_state
    tabs = st.tabs(["1️⃣ Universe", "2️⃣ Rules", "3️⃣ Risk & Sizing", "4️⃣ Data & Score", "5️⃣ Results & Backtest"])
    c = {}

    with tabs[0]:
        a, b = st.columns(2)
        with a:
            symbols_txt = st.text_area("Symbols (কমা দিয়ে; NSE সিম্বল, বা Upstox key যেমন NSE_EQ|INE002A01018)", DEFAULT_SYMBOLS, height=120)
            exchange = st.selectbox("Exchange", ["NSE", "BSE"])
        with b:
            timeframe = st.selectbox("Timeframe", list(YF_INTERVAL))
            lookback = st.selectbox("Lookback", list(LOOKBACK_DAYS), index=1)
            min_price = st.number_input("Min Price (₹)", 0.0, value=0.0, step=10.0)
            max_price = st.number_input("Max Price (₹)", 1.0, value=100000.0, step=100.0)
            min_turn = st.number_input("Min 20-bar Avg Turnover (₹ Cr)", 0.0, value=0.0, step=1.0)
        st.info("Market cap / ASM-GSM ফিল্টার আপাতত নেই, কারণ এর জন্য আলাদা ডেটা সোর্স লাগে। সেগুলো পরের ধাপে যোগ করা যাবে।")

    with tabs[1]:
        presets = st.multiselect("Strategy Presets", list(PRESETS), default=["Trend Follower 🚀"])
        default_rules = [r for r in RULES if any(r in PRESETS[p] for p in presets)]
        active = st.multiselect("সক্রিয় রুল", RULES, default=default_rules, key="rules_" + "|".join(sorted(presets)))
        r1, r2, r3 = st.columns(3)
        c["logic"] = r1.selectbox("Rule Logic", ["Win-N-of-M", "All Enabled", "ANY Enabled"])
        c["n_min"] = r2.number_input("N (Win-N-of-M)", 1, 10, 2)
        c["direction"] = {"BOTH (Buy & Sell)": "BOTH", "BUY Only 🟢": "BUY", "SELL Only 🔴": "SELL"}[
            r3.selectbox("Direction", ["BOTH (Buy & Sell)", "BUY Only 🟢", "SELL Only 🔴"])]
        st.caption("ক্যাশ মার্কেটে SELL = হোল্ডিং এক্সিট অথবা ইন্ট্রাডে (MIS) শর্ট। ডেলিভারিতে শর্ট হয় না।")
        with st.expander("⚙️ প্যারামিটার"):
            p1, p2, p3 = st.columns(3)
            c["fast"] = p1.number_input("Fast EMA", 2, 100, 9)
            c["slow"] = p1.number_input("Slow EMA", 3, 300, 21)
            c["rsi_lo"] = p2.slider("RSI Lower (Bullish)", 0, 100, 50)
            c["rsi_hi"] = p2.slider("RSI Upper (Bullish)", 0, 100, 70)
            c["adx_min"] = p3.slider("Min ADX", 10, 60, 20)
            c["vol_mult"] = p3.number_input("Volume Multiplier (x)", 1.0, 5.0, 1.5, 0.1)
            c["brk_n"] = p3.number_input("Breakout Lookback (bars)", 5, 100, 20)
    c["active"] = active

    with tabs[2]:
        k1, k2, k3 = st.columns(3)
        c["capital"] = k1.number_input("Account Capital (₹)", 1000, value=25000, step=1000)
        c["risk_pct"] = k1.number_input("Risk Per Trade (%)", 0.1, 10.0, 1.0, 0.1)
        c["max_exp"] = k1.number_input("Max Exposure per Trade (% of Capital)", 10, 500, 100)
        c["stop_m"] = k2.number_input("ATR Stop Multiplier", 0.5, 10.0, 2.0, 0.5)
        c["tgt_m"] = k2.number_input("ATR Target Multiplier", 1.0, 20.0, 4.0, 0.5)
        c["min_rr"] = k2.number_input("Minimum Net R:R", 0.1, 10.0, 1.5, 0.1)
        c["cost_pct"] = k3.number_input("Round-Trip Cost (%)", 0.0, 2.0, 0.15, 0.01)
        c["min_score"] = k3.slider("Minimum Quality Score", 0, 100, 50)
        c["rs_n"] = k3.number_input("Relative Strength Lookback (bars)", 5, 250, 20)

    with tabs[3]:
        d1, d2 = st.columns(2)
        with d1:
            provider = st.radio("Data Source", ["Upstox API", "Live yfinance API", "Upload Local CSV Files", "Deterministic Demo (Offline)"], index=3)
            token = ""
            if provider == "Upstox API":
                token = st.text_input("Upstox Analytics Token (read-only)", value=get_token(), type="password",
                                      help="Upstox Developer Apps → Analytics ট্যাব। কখনো কাউকে শেয়ার করবেন না।")
                bench_sym = st.text_input("Benchmark (Upstox key)", "NSE_INDEX|Nifty 50")
            else:
                bench_sym = st.text_input("Benchmark Symbol", "^NSEI")
            uploads = st.file_uploader("CSV (Date, Open, High, Low, Close, Volume)", accept_multiple_files=True, type="csv") \
                if provider == "Upload Local CSV Files" else []
        with d2:
            st.markdown("**Score Weights** (নিজে থেকেই ১০০%-এ নরমালাইজ হয়)")
            c["w_trend"] = st.slider("Trend", 0, 100, 30)
            c["w_mom"] = st.slider("Momentum", 0, 100, 25)
            c["w_vol"] = st.slider("Volume", 0, 100, 20)
            c["w_rs"] = st.slider("Relative Strength", 0, 100, 15)
            c["w_vola"] = st.slider("Volatility", 0, 100, 10)
        with st.expander("🧪 Self-test"):
            if st.button("Run self-test"):
                for name, ok in run_self_test().items():
                    st.write(("✅ " if ok else "❌ ") + name)

    with tabs[4]:
        if not c["active"]:
            st.warning("Rules ট্যাবে অন্তত একটি রুল সক্রিয় করুন।")
        if st.button("🚀 Run Market Scanner", type="primary", disabled=not c["active"]):
            frames, errors = {}, {}
            if provider == "Upload Local CSV Files":
                for f in uploads:
                    try:
                        frames[os.path.splitext(f.name)[0].upper()] = parse_csv(f)
                    except Exception as e:
                        errors[f.name] = str(e)
            else:
                syms = [s.strip().upper() for s in symbols_txt.replace("\n", ",").split(",") if s.strip()]
                bar = st.progress(0.0, text="ডেটা আনা হচ্ছে…")
                for i, s in enumerate(syms):
                    try:
                        frames[s] = fetch_ohlcv(provider, s, exchange, timeframe, lookback, token)
                    except Exception as e:
                        errors[s] = str(e)[:160]
                    bar.progress((i + 1) / len(syms), text=f"{s} ({i + 1}/{len(syms)})")
                bar.empty()
            bench = None
            try:
                if provider != "Upload Local CSV Files":
                    bench = fetch_ohlcv(provider, bench_sym, exchange, timeframe, lookback, token)
            except Exception:
                st.caption("⚠️ Benchmark পাওয়া যায়নি, Relative Strength নিরপেক্ষ (০.৫) ধরা হয়েছে।")
            rows = []
            for s, df in frames.items():
                if not (min_price <= df["Close"].iloc[-1] <= max_price):
                    continue
                r = analyze(s, df, bench, c)
                if r and r["Turnover (Cr)"] >= min_turn:
                    rows.append(r)
            ss["scan"] = pd.DataFrame(rows)
            ss["frames"], ss["errors"], ss["cfg"] = frames, errors, dict(c)

        if "scan" in ss:
            if provider == "Deterministic Demo (Offline)":
                st.warning("⚠️ এটি Demo (নকল) ডেটা। এর সিগন্যাল দিয়ে ট্রেড করবেন না।")
            res = ss["scan"]
            show_all = st.checkbox("রিজেক্টেড সহ সব দেখাও", False)
            if res.empty:
                st.info("কোনো ফলাফল নেই।")
            else:
                view = res if show_all else res[res["Status"].str.startswith("✅")]
                st.dataframe(view.sort_values("Score", ascending=False), hide_index=True)
                st.download_button("⬇️ CSV", res.to_csv(index=False).encode(), "scan_results.csv", "text/csv")
            if ss.get("errors"):
                with st.expander(f"⚠️ {len(ss['errors'])}টি সিম্বলে ত্রুটি"):
                    st.json(ss["errors"])

            st.divider()
            st.subheader("Backtest (সর্বশেষ স্ক্যানের ডেটা ও সেটিংসে)")
            hold = st.number_input("সর্বোচ্চ হোল্ড (বার)", 2, 100, 20)
            if st.button("▶️ Run Backtest"):
                allt = []
                for s, df in ss["frames"].items():
                    if len(df) > 100:
                        allt += backtest(s, df, ss["cfg"], hold)
                tr = pd.DataFrame(allt)
                if tr.empty:
                    st.info("কোনো ট্রেড হয়নি। রুল ঢিলা করে দেখুন।")
                else:
                    tr = tr.sort_values("Exit").reset_index(drop=True)
                    st.dataframe(pd.DataFrame([bt_stats(tr)]), hide_index=True)
                    st.line_chart(tr["R"].cumsum().rename("Cumulative R"))
                    st.dataframe(tr, hide_index=True)
                    st.caption("মডেল: long-only, সিগন্যালের পরের বারের Open-এ এন্ট্রি, ATR Stop/Target, খরচ সহ। "
                               "একই বারে Stop ও Target ছুঁলে Stop ধরা হয়। Survivorship bias আছে (আজকের সিম্বল লিস্ট)।")


if __name__ == "__main__":
    main()
