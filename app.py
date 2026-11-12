from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import time

BASE_URL = "https://api.toobit.com"
CONTRACTS = {
    "BTC": "BTC-SWAP-USDT",
    "ETH": "ETH-SWAP-USDT",
    "SOL": "SOL-SWAP-USDT",
}
INTERVALS = {"5m", "15m"}
app = FastAPI(title="NEXUS MVP", version="0.2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

session = requests.Session()
session.headers.update({"User-Agent": "NEXUS-MVP/0.2"})

def fetch_klines(symbol: str, interval: str, limit: int = 250) -> pd.DataFrame:
    if interval not in INTERVALS:
        raise ValueError("Unsupported interval")
    url = f"{BASE_URL}/quote/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(max(limit, 80), 1000)}
    response = session.get(url, params=params, timeout=12)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list) or len(rows) < 60:
        raise ValueError(f"Insufficient candle data for {symbol} {interval}")
    # Toobit futures kline schema: [open_time, open, high, low, close, volume, ...]
    df = pd.DataFrame(rows)
    if df.shape[1] < 6:
        raise ValueError("Unexpected kline schema from Toobit")
    df = df.iloc[:, :6].copy()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    # Exclude the currently-forming candle to avoid using incomplete data.
    if len(df) > 2:
        df = df.iloc[:-1].reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    x["vol_avg20"] = x["volume"].rolling(20).mean()
    x["prev_high20"] = x["high"].shift(1).rolling(20).max()
    x["prev_low20"] = x["low"].shift(1).rolling(20).min()
    x["mss_high5"] = x["high"].shift(1).rolling(5).max()
    x["mss_low5"] = x["low"].shift(1).rolling(5).min()
    return x

def analyze_symbol(asset: str):
    contract = CONTRACTS[asset]
    d5 = add_indicators(fetch_klines(contract, "5m", 250))
    d15 = add_indicators(fetch_klines(contract, "15m", 250))
    a = d5.iloc[-1]
    b = d15.iloc[-1]

    trend_up = bool(b["ema20"] > b["ema50"] and b["close"] > b["ema50"])
    trend_down = bool(b["ema20"] < b["ema50"] and b["close"] < b["ema50"])
    trend = "صعودی" if trend_up else ("نزولی" if trend_down else "خنثی")

    # Recent liquidity sweep: current completed candle sweeps the prior 20-bar extreme
    # and closes back inside the prior range.
    bull_sweep = bool(a["low"] < a["prev_low20"] and a["close"] > a["prev_low20"])
    bear_sweep = bool(a["high"] > a["prev_high20"] and a["close"] < a["prev_high20"])
    bull_mss = bool(a["close"] > a["mss_high5"])
    bear_mss = bool(a["close"] < a["mss_low5"])
    volume_ok = bool(a["vol_avg20"] > 0 and a["volume"] > a["vol_avg20"] * 1.3)
    ema_long = bool(a["close"] > a["ema20"])
    ema_short = bool(a["close"] < a["ema20"])

    # Score is a research heuristic, not a calibrated probability.
    long_parts = {
        "روند ۱۵ دقیقه": 20 if trend_up else 0,
        "جمع‌آوری نقدینگی": 20 if bull_sweep else 0,
        "شکست ساختار": 20 if bull_mss else 0,
        "تأیید حجم": 15 if volume_ok else 0,
        "تأیید EMA": 10 if ema_long else 0,
        "کیفیت موقعیت": 15 if (bull_sweep and bull_mss and a["close"] > a["open"]) else 0,
    }
    short_parts = {
        "روند ۱۵ دقیقه": 20 if trend_down else 0,
        "جمع‌آوری نقدینگی": 20 if bear_sweep else 0,
        "شکست ساختار": 20 if bear_mss else 0,
        "تأیید حجم": 15 if volume_ok else 0,
        "تأیید EMA": 10 if ema_short else 0,
        "کیفیت موقعیت": 15 if (bear_sweep and bear_mss and a["close"] < a["open"]) else 0,
    }
    long_score = int(sum(long_parts.values()))
    short_score = int(sum(short_parts.values()))

    signal = "WAIT"
    score = max(long_score, short_score)
    direction = None
    entry = stop = target = None
    reasons = []

    atr = float(a["atr14"]) if pd.notna(a["atr14"]) else 0.0
    if trend_up and bull_sweep and bull_mss and volume_ok and ema_long and long_score >= 85:
        signal, direction = "LONG", "خرید"
        entry = float(a["close"])
        stop = min(float(a["low"]), float(a["prev_low20"])) - 0.15 * atr
        risk = entry - stop
        if risk > 0:
            target = entry + 1.8 * risk
        else:
            signal, direction = "WAIT", None
    elif trend_down and bear_sweep and bear_mss and volume_ok and ema_short and short_score >= 85:
        signal, direction = "SHORT", "فروش"
        entry = float(a["close"])
        stop = max(float(a["high"]), float(a["prev_high20"])) + 0.15 * atr
        risk = stop - entry
        if risk > 0:
            target = entry - 1.8 * risk
        else:
            signal, direction = "WAIT", None

    if trend_up: reasons.append("روند ۱۵ دقیقه صعودی است")
    elif trend_down: reasons.append("روند ۱۵ دقیقه نزولی است")
    else: reasons.append("روند ۱۵ دقیقه خنثی است")
    if bull_sweep: reasons.append("شکار نقدینگی کف شناسایی شد")
    elif bear_sweep: reasons.append("شکار نقدینگی سقف شناسایی شد")
    else: reasons.append("شکار نقدینگی تأیید نشده")
    if bull_mss or bear_mss: reasons.append("شکست ساختار ۵ دقیقه تأیید شد")
    else: reasons.append("شکست ساختار تأیید نشده")
    reasons.append("حجم بالاتر از ۱.۳ برابر میانگین است" if volume_ok else "حجم کافی نیست")

    return {
        "asset": asset,
        "contract": contract,
        "signal": signal,
        "direction": direction,
        "score": score,
        "long_score": long_score,
        "short_score": short_score,
        "trend": trend,
        "price": float(a["close"]),
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": 1.8 if direction else None,
        "atr": atr,
        "volume_ratio": float(a["volume"] / a["vol_avg20"]) if a["vol_avg20"] else None,
        "candle_time": a["time"].isoformat(),
        "reasons": reasons,
        "score_breakdown": long_parts if long_score >= short_score else short_parts,
        "warning": "امتیاز، احتمال موفقیت نیست؛ این نسخه آزمایشی و بدون سفارش‌گذاری است."
    }

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/health")
def health():
    return {"ok": True, "app": "NEXUS", "version": "0.2.0", "mode": "analysis_only"}

@app.get("/api/analysis")
def analysis(asset: str = Query("BTC", pattern="^(BTC|ETH|SOL)$")):
    try:
        return analyze_symbol(asset)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"خطا در دریافت داده از Toobit: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطای تحلیل: {str(e)}")

@app.get("/api/overview")
def overview():
    result = []
    errors = []
    for asset in CONTRACTS:
        try:
            item = analyze_symbol(asset)
            result.append(item)
        except Exception as e:
            errors.append({"asset": asset, "error": str(e)})
    return {
        "items": result,
        "errors": errors,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "analysis_only"
    }
