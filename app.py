import io
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dave Elliott Wave Analyzer", page_icon="📈", layout="wide")
st.title("📈 Dave Elliott Wave Multi-Degree Analyzer")
st.caption("Upload OHLCV CSV files and generate Elliott Wave impulse/correction charts with outlook, targets, and invalidation levels.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").title() for c in df.columns]
    col_map = {}
    for c in df.columns:
        low = c.lower()
        if low in ["date", "time", "datetime", "timestamp"]:
            col_map[c] = "Date"
        elif low in ["open", "o"]:
            col_map[c] = "Open"
        elif low in ["high", "h"]:
            col_map[c] = "High"
        elif low in ["low", "l"]:
            col_map[c] = "Low"
        elif low in ["close", "c", "last"]:
            col_map[c] = "Close"
        elif low in ["volume", "vol", "v"]:
            col_map[c] = "Volume"
    df = df.rename(columns=col_map)
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")
    if "Volume" not in df.columns:
        df["Volume"] = 0
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def pct_move(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), 1e-9) * 100.0


def safe_ratio(a: float, b: float) -> float:
    return a / max(abs(b), 1e-9)


def in_range(value, low, high, tol=0.0) -> bool:
    if value is None:
        return False
    try:
        if np.isnan(value):
            return False
    except Exception:
        pass
    return (low - tol) <= value <= (high + tol)


def format_price(x) -> str:
    if x is None:
        return "n/a"
    try:
        if np.isnan(x):
            return "n/a"
    except Exception:
        pass
    x = float(x)
    return f"{x:.6f}" if abs(x) < 1 else f"{x:.2f}"


def fib_targets(start, end, correction_end, direction):
    impulse_len = abs(end - start)
    if direction == "Bullish":
        return {
            "T1 0.618": correction_end + impulse_len * 0.618,
            "T2 1.000": correction_end + impulse_len * 1.000,
            "T3 1.272": correction_end + impulse_len * 1.272,
            "T4 1.618": correction_end + impulse_len * 1.618,
        }
    if direction == "Bearish":
        return {
            "T1 0.618": correction_end - impulse_len * 0.618,
            "T2 1.000": correction_end - impulse_len * 1.000,
            "T3 1.272": correction_end - impulse_len * 1.272,
            "T4 1.618": correction_end - impulse_len * 1.618,
        }
    return {}


def detect_pivots(data, pivot_strength: int, min_swing_pct: float):
    highs = data["High"].values
    lows = data["Low"].values
    if len(data) < pivot_strength * 3:
        return []
    high_idx = argrelextrema(highs, np.greater_equal, order=pivot_strength)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=pivot_strength)[0]
    pivots = []
    for i in high_idx:
        pivots.append({"bar": int(i), "date": data.index[i], "price": float(data["High"].iloc[i]), "type": "H"})
    for i in low_idx:
        pivots.append({"bar": int(i), "date": data.index[i], "price": float(data["Low"].iloc[i]), "type": "L"})
    pivots = sorted(pivots, key=lambda x: x["bar"])
    clean = []
    for p in pivots:
        if not clean:
            clean.append(p)
        else:
            last = clean[-1]
            if p["type"] == last["type"]:
                if p["type"] == "H" and p["price"] > last["price"]:
                    clean[-1] = p
                elif p["type"] == "L" and p["price"] < last["price"]:
                    clean[-1] = p
            else:
                move = pct_move(last["price"], p["price"])
                if move >= min_swing_pct:
                    clean.append(p)
    return clean


def score_bullish_impulse(seq, min_total_impulse_pct, min_wave1_share, min_wave3_vs_wave1):
    if len(seq) < 6:
        return -999
    types = [p["type"] for p in seq[:6]]
    prices = [p["price"] for p in seq[:6]]
    if types != ["L", "H", "L", "H", "L", "H"]:
        return -999
    start, w1, w2, w3, w4, w5 = prices
    if not (w1 > start and w2 > start and w3 > w1 and w4 > w2 and w5 > w3):
        return -999
    if w4 <= w1:
        return -999
    wave1_len = abs(w1 - start)
    wave3_len = abs(w3 - w2)
    wave5_len = abs(w5 - w4)
    total_len = abs(w5 - start)
    total_impulse_pct = pct_move(start, w5)
    wave1_share = wave1_len / max(total_len, 1e-9)
    if total_impulse_pct < min_total_impulse_pct:
        return -999
    if wave1_share < min_wave1_share:
        return -999
    if wave3_len < wave1_len * min_wave3_vs_wave1:
        return -999
    if wave3_len < min(wave1_len, wave5_len):
        return -999
    wave3_extension_score = safe_ratio(wave3_len, wave1_len) * 10
    wave5_score = safe_ratio(wave5_len, wave1_len) * 5
    recency_score = seq[0]["bar"] * 0.05
    return total_impulse_pct * 2.0 + wave3_extension_score + wave5_score + recency_score


def score_bearish_impulse(seq, min_total_impulse_pct, min_wave1_share, min_wave3_vs_wave1):
    if len(seq) < 6:
        return -999
    types = [p["type"] for p in seq[:6]]
    prices = [p["price"] for p in seq[:6]]
    if types != ["H", "L", "H", "L", "H", "L"]:
        return -999
    start, w1, w2, w3, w4, w5 = prices
    if not (w1 < start and w2 < start and w3 < w1 and w4 < w2 and w5 < w3):
        return -999
    if w4 >= w1:
        return -999
    wave1_len = abs(start - w1)
    wave3_len = abs(w2 - w3)
    wave5_len = abs(w4 - w5)
    total_len = abs(start - w5)
    total_impulse_pct = pct_move(start, w5)
    wave1_share = wave1_len / max(total_len, 1e-9)
    if total_impulse_pct < min_total_impulse_pct:
        return -999
    if wave1_share < min_wave1_share:
        return -999
    if wave3_len < wave1_len * min_wave3_vs_wave1:
        return -999
    if wave3_len < min(wave1_len, wave5_len):
        return -999
    wave3_extension_score = safe_ratio(wave3_len, wave1_len) * 10
    wave5_score = safe_ratio(wave5_len, wave1_len) * 5
    recency_score = seq[0]["bar"] * 0.05
    return total_impulse_pct * 2.0 + wave3_extension_score + wave5_score + recency_score


def find_best_impulse(pivots, min_total_impulse_pct, min_wave1_share, min_wave3_vs_wave1):
    best = None
    best_score = -999
    best_direction = None
    for i in range(0, len(pivots) - 5):
        seq = pivots[i:i+6]
        bull_score = score_bullish_impulse(seq, min_total_impulse_pct, min_wave1_share, min_wave3_vs_wave1)
        bear_score = score_bearish_impulse(seq, min_total_impulse_pct, min_wave1_share, min_wave3_vs_wave1)
        if bull_score > best_score:
            best_score = bull_score
            best = seq
            best_direction = "Bullish"
        if bear_score > best_score:
            best_score = bear_score
            best = seq
            best_direction = "Bearish"
    if best is None or best_score <= 0:
        return None
    return {"direction": best_direction, "score": best_score, "points": best, "labels": ["Start", "1", "2", "3", "4", "5"]}


def classify_abc_after_impulse(pivots, impulse, correction_tol):
    if impulse is None:
        return None
    pts = impulse["points"]
    direction = impulse["direction"]
    w5_bar = pts[5]["bar"]
    after = [p for p in pivots if p["bar"] > w5_bar]
    if len(after) < 3:
        return {"type": "Correction Incomplete", "confidence": 35, "points": after, "labels": ["A", "B", "C"][:len(after)], "b_retrace": None, "c_vs_a": None, "status": "Need more pivots after wave 5"}
    abc = after[:3]
    types = [p["type"] for p in abc]
    prices = [p["price"] for p in abc]
    w5 = pts[5]["price"]
    A, B, C = prices
    if direction == "Bullish":
        if types != ["L", "H", "L"]:
            return None
        if not (A < w5 and B > A and C < B):
            return None
    if direction == "Bearish":
        if types != ["H", "L", "H"]:
            return None
        if not (A > w5 and B < A and C > B):
            return None
    a_len = abs(w5 - A)
    b_len = abs(B - A)
    c_len = abs(B - C)
    if a_len == 0:
        return None
    b_retrace = b_len / a_len
    c_vs_a = c_len / a_len
    correction_type = "Unclassified ABC"
    confidence = 45
    if direction == "Bullish":
        b_breaks_start = B > w5
        b_does_not_break_start = B <= w5
        b_near_start = in_range(b_retrace, 0.85, 1.15, correction_tol)
        c_breaks_a = C < A
        c_near_a = abs(C - A) / max(abs(A), 1e-9) <= 0.05
        c_fails_a = C > A
        if b_does_not_break_start and in_range(b_retrace, 0.30, 0.90, correction_tol) and in_range(c_vs_a, 0.60, 1.90, correction_tol) and c_breaks_a:
            correction_type = "Deep Zigzag" if c_vs_a >= 1.50 else "Zigzag"
            confidence = 88 if c_vs_a >= 1.50 else 85
        elif b_breaks_start and in_range(b_retrace, 1.00, 1.618, correction_tol) and in_range(c_vs_a, 0.90, 2.24, correction_tol) and c_breaks_a:
            correction_type = "Expanded Flat"
            confidence = 82
        elif b_near_start and in_range(c_vs_a, 0.70, 1.35, correction_tol) and c_near_a:
            correction_type = "Regular Flat"
            confidence = 78
        elif b_breaks_start and in_range(b_retrace, 1.00, 1.618, correction_tol) and in_range(c_vs_a, 0.50, 1.15, correction_tol) and c_fails_a:
            correction_type = "Running Flat"
            confidence = 72
        elif b_does_not_break_start and in_range(b_retrace, 0.35, 0.85, correction_tol) and in_range(c_vs_a, 0.35, 0.90, correction_tol) and c_fails_a:
            correction_type = "Contracting Flat / 3-3-5"
            confidence = 70
    elif direction == "Bearish":
        b_breaks_start = B < w5
        b_does_not_break_start = B >= w5
        b_near_start = in_range(b_retrace, 0.85, 1.15, correction_tol)
        c_breaks_a = C > A
        c_near_a = abs(C - A) / max(abs(A), 1e-9) <= 0.05
        c_fails_a = C < A
        if b_does_not_break_start and in_range(b_retrace, 0.30, 0.90, correction_tol) and in_range(c_vs_a, 0.60, 1.90, correction_tol) and c_breaks_a:
            correction_type = "Deep Zigzag" if c_vs_a >= 1.50 else "Zigzag"
            confidence = 88 if c_vs_a >= 1.50 else 85
        elif b_breaks_start and in_range(b_retrace, 1.00, 1.618, correction_tol) and in_range(c_vs_a, 0.90, 2.24, correction_tol) and c_breaks_a:
            correction_type = "Expanded Flat"
            confidence = 82
        elif b_near_start and in_range(c_vs_a, 0.70, 1.35, correction_tol) and c_near_a:
            correction_type = "Regular Flat"
            confidence = 78
        elif b_breaks_start and in_range(b_retrace, 1.00, 1.618, correction_tol) and in_range(c_vs_a, 0.50, 1.15, correction_tol) and c_fails_a:
            correction_type = "Running Flat"
            confidence = 72
        elif b_does_not_break_start and in_range(b_retrace, 0.35, 0.85, correction_tol) and in_range(c_vs_a, 0.35, 0.90, correction_tol) and c_fails_a:
            correction_type = "Contracting Flat / 3-3-5"
            confidence = 70
    return {"type": correction_type, "confidence": confidence, "points": abc, "labels": ["A", "B", "C"], "b_retrace": b_retrace, "c_vs_a": c_vs_a, "status": "Completed ABC"}


def detect_double_zigzag_after_impulse(pivots, impulse):
    if impulse is None:
        return None
    direction = impulse["direction"]
    w5_bar = impulse["points"][5]["bar"]
    after = [p for p in pivots if p["bar"] > w5_bar]
    if len(after) < 7:
        return None
    seq = after[:7]
    types = [p["type"] for p in seq]
    prices = [p["price"] for p in seq]
    if direction == "Bullish":
        if types != ["L", "H", "L", "H", "L", "H", "L"]:
            return None
        if not (prices[2] < prices[0] and prices[6] < prices[2]):
            return None
    if direction == "Bearish":
        if types != ["H", "L", "H", "L", "H", "L", "H"]:
            return None
        if not (prices[2] > prices[0] and prices[6] > prices[2]):
            return None
    return {"type": "Double Zigzag W-X-Y", "confidence": 72, "points": seq, "labels": ["W-A", "W-B", "W-C", "X", "Y-A", "Y-B", "Y-C"], "status": "Completed W-X-Y candidate"}


def detect_triangle_candidate(pivots):
    if len(pivots) < 5:
        return None
    seq = pivots[-5:]
    highs = [p["price"] for p in seq if p["type"] == "H"]
    lows = [p["price"] for p in seq if p["type"] == "L"]
    if len(highs) < 2 or len(lows) < 2:
        return None
    contracting_highs = highs[-1] < highs[0]
    contracting_lows = lows[-1] > lows[0]
    expanding_highs = highs[-1] > highs[0]
    expanding_lows = lows[-1] < lows[0]
    if contracting_highs and contracting_lows:
        return {"type": "Contracting Triangle A-B-C-D-E Candidate", "confidence": 65, "points": seq, "labels": ["A", "B", "C", "D", "E"], "status": "Triangle candidate"}
    if expanding_highs and expanding_lows:
        return {"type": "Expanding Triangle A-B-C-D-E Candidate", "confidence": 60, "points": seq, "labels": ["A", "B", "C", "D", "E"], "status": "Triangle candidate"}
    return None


def choose_best_correction(abc, wxy, tri):
    candidates = [x for x in [abc, wxy, tri] if x is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.get("confidence", 0))


def build_outlook(impulse, correction, data):
    last_close = float(data["Close"].iloc[-1])
    if impulse is None:
        return {"bias": "No clean impulse found", "outlook": "Wait for a clearer structure.", "confirmation": None, "invalidation": None, "targets": {}}
    direction = impulse["direction"]
    points = impulse["points"]
    start_price = points[0]["price"]
    w5_price = points[5]["price"]
    if correction is None or len(correction.get("points", [])) < 3:
        return {"bias": f"{direction} impulse detected", "outlook": "Correction may still be forming. Watch for A-B-C completion.", "confirmation": w5_price, "invalidation": points[4]["price"], "targets": {}}
    c_points = correction["points"]
    c_end = c_points[-1]["price"]
    if direction == "Bullish":
        confirmation = max([p["price"] for p in c_points])
        invalidation = c_end
        targets = fib_targets(start_price, w5_price, c_end, "Bullish")
        if last_close > confirmation:
            outlook = "Bullish continuation already confirmed above correction high."
        elif last_close <= invalidation:
            outlook = "Caution: price is at/below C. Correction may be extending."
        else:
            outlook = "Bullish recovery watch. Confirmation improves above B/correction high."
        return {"bias": "Bullish", "outlook": outlook, "confirmation": confirmation, "invalidation": invalidation, "targets": targets}
    if direction == "Bearish":
        confirmation = min([p["price"] for p in c_points])
        invalidation = c_end
        targets = fib_targets(start_price, w5_price, c_end, "Bearish")
        if last_close < confirmation:
            outlook = "Bearish continuation already confirmed below correction low."
        elif last_close >= invalidation:
            outlook = "Caution: price is at/above C. Correction may be extending."
        else:
            outlook = "Bearish continuation watch. Confirmation improves below B/correction low."
        return {"bias": "Bearish", "outlook": outlook, "confirmation": confirmation, "invalidation": invalidation, "targets": targets}
    return {"bias": "Neutral", "outlook": "No clear outlook.", "confirmation": None, "invalidation": None, "targets": {}}


def correction_text(c):
    if c is None:
        return "None"
    br = c.get("b_retrace")
    ca = c.get("c_vs_a")
    if br is not None and ca is not None:
        return f"{c['type']} | Conf {c['confidence']} | B {br:.3f} | C/A {ca:.3f}"
    return f"{c['type']} | Conf {c.get('confidence', 0)}"


def impulse_text(i):
    if i is None:
        return "None"
    return f"{i['direction']} | Score {i['score']:.1f}"


@dataclass
class Settings:
    bars_to_show: int
    major_pivot: int
    intermediate_pivot: int
    minor_pivot: int
    min_swing_major: float
    min_swing_intermediate: float
    min_swing_minor: float
    min_total_major: float
    min_total_intermediate: float
    min_total_minor: float
    min_wave1_share: float
    min_wave3_vs_wave1: float
    correction_tol: float
    show_major: bool
    show_intermediate: bool
    show_minor: bool
    show_major_correction: bool
    show_intermediate_correction: bool
    show_minor_correction: bool
    show_info_box: bool


def analyze_dataframe(df: pd.DataFrame, settings: Settings):
    data = df.copy() if settings.bars_to_show == 0 or settings.bars_to_show >= len(df) else df.tail(settings.bars_to_show).copy()
    major_pivots = detect_pivots(data, settings.major_pivot, settings.min_swing_major)
    intermediate_pivots = detect_pivots(data, settings.intermediate_pivot, settings.min_swing_intermediate)
    minor_pivots = detect_pivots(data, settings.minor_pivot, settings.min_swing_minor)
    major_impulse = find_best_impulse(major_pivots, settings.min_total_major, settings.min_wave1_share, settings.min_wave3_vs_wave1)
    intermediate_impulse = find_best_impulse(intermediate_pivots, settings.min_total_intermediate, settings.min_wave1_share, settings.min_wave3_vs_wave1)
    minor_impulse = find_best_impulse(minor_pivots, settings.min_total_minor, settings.min_wave1_share, settings.min_wave3_vs_wave1)
    major_best = choose_best_correction(classify_abc_after_impulse(major_pivots, major_impulse, settings.correction_tol), detect_double_zigzag_after_impulse(major_pivots, major_impulse), detect_triangle_candidate(major_pivots))
    intermediate_best = choose_best_correction(classify_abc_after_impulse(intermediate_pivots, intermediate_impulse, settings.correction_tol), detect_double_zigzag_after_impulse(intermediate_pivots, intermediate_impulse), detect_triangle_candidate(intermediate_pivots))
    minor_best = choose_best_correction(classify_abc_after_impulse(minor_pivots, minor_impulse, settings.correction_tol), detect_double_zigzag_after_impulse(minor_pivots, minor_impulse), detect_triangle_candidate(minor_pivots))
    major_outlook = build_outlook(major_impulse, major_best, data)
    intermediate_outlook = build_outlook(intermediate_impulse, intermediate_best, data)
    minor_outlook = build_outlook(minor_impulse, minor_best, data)
    primary_outlook = major_outlook if major_impulse is not None else intermediate_outlook
    return {"data": data, "major_impulse": major_impulse, "intermediate_impulse": intermediate_impulse, "minor_impulse": minor_impulse, "major_correction": major_best, "intermediate_correction": intermediate_best, "minor_correction": minor_best, "major_outlook": major_outlook, "intermediate_outlook": intermediate_outlook, "minor_outlook": minor_outlook, "primary_outlook": primary_outlook}


def draw_wave(ax, structure, degree_name, color, linewidth, label_size, y_offset_pct=0.015):
    if structure is None:
        return
    points = structure.get("points", [])
    labels = structure.get("labels", [])
    for i, p in enumerate(points):
        x, y = p["bar"], p["price"]
        label = labels[i] if i < len(labels) else ""
        ax.scatter(x, y, s=70, zorder=5, color=color)
        y_offset = y * y_offset_pct
        vertical = "bottom" if p["type"] == "L" else "top"
        y_text = y - y_offset if p["type"] == "H" else y + y_offset
        ax.text(x, y_text, f"{degree_name} {label}", fontsize=label_size, fontweight="bold", color=color, verticalalignment=vertical, zorder=6)
        if i > 0:
            prev = points[i - 1]
            ax.plot([prev["bar"], p["bar"]], [prev["price"], p["price"]], linewidth=linewidth, color=color, alpha=0.95, zorder=4)


def draw_correction(ax, correction, degree_name, color, linewidth, label_size, y_offset_pct=0.012):
    if correction is None:
        return
    points = correction.get("points", [])
    labels = correction.get("labels", [])
    for i, p in enumerate(points):
        x, y = p["bar"], p["price"]
        label = labels[i] if i < len(labels) else ""
        ax.scatter(x, y, s=60, zorder=5, color=color)
        y_offset = y * y_offset_pct
        vertical = "bottom" if p["type"] == "L" else "top"
        y_text = y - y_offset if p["type"] == "H" else y + y_offset
        ax.text(x, y_text, f"{degree_name} {label}", fontsize=label_size, fontweight="bold", color=color, verticalalignment=vertical, zorder=6)
        if i > 0:
            prev = points[i - 1]
            ax.plot([prev["bar"], p["bar"]], [prev["price"], p["price"]], linewidth=linewidth, color=color, linestyle="--", alpha=0.90, zorder=4)


def make_chart(symbol: str, analysis: dict, settings: Settings):
    data = analysis["data"]
    fig, axes = mpf.plot(data, type="candle", volume=True, title=f"{symbol} Elliott Wave Multi-Degree Analyzer", style="yahoo", figsize=(18, 10), returnfig=True)
    ax = axes[0]
    if settings.show_major:
        draw_wave(ax, analysis["major_impulse"], "(M)", "purple", 3.0, 11)
    if settings.show_major_correction:
        draw_correction(ax, analysis["major_correction"], "(M)", "purple", 2.5, 10)
    if settings.show_intermediate:
        draw_wave(ax, analysis["intermediate_impulse"], "Int", "blue", 2.2, 9)
    if settings.show_intermediate_correction:
        draw_correction(ax, analysis["intermediate_correction"], "Int", "blue", 1.8, 8)
    if settings.show_minor:
        draw_wave(ax, analysis["minor_impulse"], "Min", "darkgreen", 1.6, 8)
    if settings.show_minor_correction:
        draw_correction(ax, analysis["minor_correction"], "Min", "darkgreen", 1.3, 7)
    primary = analysis["primary_outlook"]
    targets = primary.get("targets", {})
    target_text = "Targets: n/a"
    if targets:
        target_text = f"T1: {format_price(targets.get('T1 0.618'))}  T2: {format_price(targets.get('T2 1.000'))}\nT3: {format_price(targets.get('T3 1.272'))}  T4: {format_price(targets.get('T4 1.618'))}"
    info = f"{symbol} Multi-Degree Elliott Wave\nMajor: {impulse_text(analysis['major_impulse'])} | {correction_text(analysis['major_correction'])}\nIntermediate: {impulse_text(analysis['intermediate_impulse'])} | {correction_text(analysis['intermediate_correction'])}\nMinor: {impulse_text(analysis['minor_impulse'])} | {correction_text(analysis['minor_correction'])}\n\nPrimary Bias: {primary.get('bias')}\nOutlook: {primary.get('outlook')}\nConfirmation: {format_price(primary.get('confirmation'))}\nInvalidation: {format_price(primary.get('invalidation'))}\n{target_text}"
    if settings.show_info_box:
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10, fontweight="bold", verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.88))
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf


def summary_row(symbol, analysis):
    primary = analysis["primary_outlook"]
    return {"Symbol": symbol, "Major Impulse": impulse_text(analysis["major_impulse"]), "Major Correction": correction_text(analysis["major_correction"]), "Intermediate Impulse": impulse_text(analysis["intermediate_impulse"]), "Intermediate Correction": correction_text(analysis["intermediate_correction"]), "Minor Impulse": impulse_text(analysis["minor_impulse"]), "Minor Correction": correction_text(analysis["minor_correction"]), "Primary Bias": primary.get("bias"), "Outlook": primary.get("outlook"), "Confirmation": primary.get("confirmation"), "Invalidation": primary.get("invalidation")}


with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    st.subheader("History")
    history_mode = st.radio("Chart history", ["Last N bars", "Full CSV"], index=0, horizontal=True)
    bars_to_show = st.number_input("Bars to show", min_value=50, max_value=5000, value=1250, step=50)
    st.subheader("Pivot Strength")
    major_pivot = st.slider("Major Pivot Strength", 5, 80, 24)
    intermediate_pivot = st.slider("Intermediate Pivot Strength", 3, 50, 12)
    minor_pivot = st.slider("Minor Pivot Strength", 2, 30, 6)
    st.subheader("Filters")
    min_swing_major = st.number_input("Major Minimum Swing %", min_value=0.1, max_value=50.0, value=4.0, step=0.1)
    min_swing_intermediate = st.number_input("Intermediate Minimum Swing %", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
    min_swing_minor = st.number_input("Minor Minimum Swing %", min_value=0.1, max_value=50.0, value=0.8, step=0.1)
    min_total_major = st.number_input("Major Minimum Total Impulse %", min_value=0.1, max_value=100.0, value=15.0, step=0.5)
    min_total_intermediate = st.number_input("Intermediate Minimum Total Impulse %", min_value=0.1, max_value=100.0, value=8.0, step=0.5)
    min_total_minor = st.number_input("Minor Minimum Total Impulse %", min_value=0.1, max_value=100.0, value=4.0, step=0.5)
    min_wave1_share = st.number_input("Minimum Wave 1 Share of Total", min_value=0.01, max_value=0.50, value=0.08, step=0.01)
    min_wave3_vs_wave1 = st.number_input("Minimum Wave 3 vs Wave 1", min_value=0.10, max_value=3.0, value=0.80, step=0.05)
    correction_tol = st.number_input("Correction Tolerance", min_value=0.0, max_value=0.50, value=0.08, step=0.01)
    st.subheader("Display")
    show_major = st.checkbox("Show Major Waves", value=True)
    show_intermediate = st.checkbox("Show Intermediate Waves", value=True)
    show_minor = st.checkbox("Show Minor Waves", value=False)
    show_major_correction = st.checkbox("Show Major Corrections", value=True)
    show_intermediate_correction = st.checkbox("Show Intermediate Corrections", value=True)
    show_minor_correction = st.checkbox("Show Minor Corrections", value=False)
    show_info_box = st.checkbox("Show Outlook Box", value=True)

settings = Settings(bars_to_show=0 if history_mode == "Full CSV" else int(bars_to_show), major_pivot=int(major_pivot), intermediate_pivot=int(intermediate_pivot), minor_pivot=int(minor_pivot), min_swing_major=float(min_swing_major), min_swing_intermediate=float(min_swing_intermediate), min_swing_minor=float(min_swing_minor), min_total_major=float(min_total_major), min_total_intermediate=float(min_total_intermediate), min_total_minor=float(min_total_minor), min_wave1_share=float(min_wave1_share), min_wave3_vs_wave1=float(min_wave3_vs_wave1), correction_tol=float(correction_tol), show_major=show_major, show_intermediate=show_intermediate, show_minor=show_minor, show_major_correction=show_major_correction, show_intermediate_correction=show_intermediate_correction, show_minor_correction=show_minor_correction, show_info_box=show_info_box)

if not uploaded_files:
    st.info("Upload one or more CSV files from TradingView or your data source to begin.")
    st.markdown("""
**CSV required columns:** Date, Open, High, Low, Close. Volume is optional.

Recommended starting settings:
- Major Pivot Strength: 24
- Intermediate Pivot Strength: 12
- Minor Pivot Strength: 6
- Show Minor Waves: Off for cleaner full-history charts
""")
else:
    file_names = [f.name for f in uploaded_files]
    selected_name = st.selectbox("Choose chart to display", file_names)
    summaries = []
    chart_zip_buffer = io.BytesIO()
    selected_fig = None
    selected_png = None
    import zipfile
    with zipfile.ZipFile(chart_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for uf in uploaded_files:
            try:
                raw_df = pd.read_csv(uf)
                df = normalize_columns(raw_df)
                symbol = os.path.splitext(uf.name)[0].upper()
                analysis = analyze_dataframe(df, settings)
                summaries.append(summary_row(symbol, analysis))
                fig = make_chart(symbol, analysis, settings)
                png_buf = fig_to_png_bytes(fig)
                zf.writestr(f"{symbol}_Elliott_Wave_Chart.png", png_buf.getvalue())
                if uf.name == selected_name:
                    selected_fig = fig
                    selected_png = png_buf
                plt.close(fig)
            except Exception as e:
                summaries.append({"Symbol": uf.name, "Major Impulse": "Error", "Major Correction": str(e), "Intermediate Impulse": "", "Intermediate Correction": "", "Minor Impulse": "", "Minor Correction": "", "Primary Bias": "", "Outlook": "", "Confirmation": "", "Invalidation": ""})
    summary_df = pd.DataFrame(summaries)
    st.subheader(f"Chart: {selected_name}")
    if selected_fig is not None:
        st.pyplot(selected_fig)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download displayed chart PNG", data=selected_png, file_name=f"{os.path.splitext(selected_name)[0]}_Elliott_Wave_Chart.png", mime="image/png")
        with col2:
            chart_zip_buffer.seek(0)
            st.download_button("Download all charts ZIP", data=chart_zip_buffer, file_name="elliott_wave_charts.zip", mime="application/zip")
    st.subheader("Summary")
    st.dataframe(summary_df, use_container_width=True)
    csv_buf = io.StringIO()
    summary_df.to_csv(csv_buf, index=False)
    st.download_button("Download summary CSV", data=csv_buf.getvalue(), file_name="elliott_wave_summary.csv", mime="text/csv")
    st.warning("This is algorithmic Elliott Wave analysis. Treat it as a research aid, not financial advice. Always verify counts manually.")
