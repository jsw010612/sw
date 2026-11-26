"""
조합 탐색용 데이터 만들기 — 바닥 신호마다 지표 70여 개를 계산해서 state/feature_trades.csv 로 저장

신호: 1일/주봉 RSI 다이버전스 · 52주 -30% 이상 · 5파 깊이 1.0↑ (넓게 모음)
결과: 수익형 매도(5파 저점 손절 → 저항선 절반 → 최고가 -20%) · 안정형 매도(5파+후행스팬 손절 → 같음)
특징은 전부 신호가 확정된 날(det) 종가까지의 정보만 씀 (매수는 다음 날 시가)

묶음
  거래량  : 3파·5파 저점 거래량, 신호일 거래량, 5일/20일 거래량, 5파 구간 vs 3파 구간 거래량, OBV 다이버전스·기울기
  RSI     : 일·주·월 RSI, 3파/5파 RSI, 차이, 5파 뒤 RSI 반등폭, 스토캐스틱
  캔들    : 5파 저점 캔들, 신호일 캔들, 아래꼬리·몸통 비율, 종가 위치
  MACD    : PPO 산 다이버전스, MACD 점 다이버전스, 히스토그램 상승 여부, MACD>시그널
  일목    : 구름 위/안/아래, 구름 두께, 전환>기준, 기준선 거리, 후행스팬 vs 26일 전, 앞 구름 색
  이평선  : 5/20/60/120일선 거리, 정배열·역배열, 20일선 기울기, 주·월 60/120선 거리, 이평선 수렴
  피보    : 5파 깊이, 4파 되돌림, 2년 저점~52주 고점 되돌림 비율
  다이버전스: 주봉 여부, 3파~5파 간격, 최근 60일 다이버전스 횟수, 주봉 RSI 다이버전스 동시 여부
  시장    : 코스피 120일선 거리, 시장 RSI, 시장 20일 수익률
  종목    : 주가, 시총 순위, 52주 하락폭, 손절폭, ATR%, 볼린저 %B·폭, 20일 수익률, 52주 고점 이후 일수
"""
import numpy as np
import pandas as pd

import scan
import stop_backtest as SBT
import hybrid_backtest as HB
from analysis import candle_code, rsi
from ma_backtest import log

MODES = ["A 5파 | 추적20", "B 5파+후행 | 추적20"]


def _prep(d1, t):
    O, H, L, C, V = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
    Cs, Hs, Ls, Vs = pd.Series(C), pd.Series(H), pd.Series(L), pd.Series(V)
    f = {}
    f["rsi"] = np.asarray(rsi(C, 14), float)
    lo14, hi14 = Ls.rolling(14).min(), Hs.rolling(14).max()
    k = (Cs - lo14) / (hi14 - lo14) * 100
    f["stoch"] = k.rolling(3).mean().to_numpy()
    for n in (5, 20, 60, 120):
        f[f"ma{n}"] = Cs.rolling(n).mean().to_numpy()
    f["vma20"] = Vs.rolling(20).mean().to_numpy()
    f["vma5"] = Vs.rolling(5).mean().to_numpy()
    obv = (np.sign(Cs.diff().fillna(0)) * Vs).cumsum()
    f["obv"] = obv.to_numpy()
    tr = pd.concat([Hs - Ls, (Hs - Cs.shift()).abs(), (Ls - Cs.shift()).abs()], axis=1).max(axis=1)
    f["atr"] = tr.rolling(14).mean().to_numpy()
    sd20 = Cs.rolling(20).std()
    f["bbu"], f["bbl"] = (Cs.rolling(20).mean() + 2 * sd20).to_numpy(), (Cs.rolling(20).mean() - 2 * sd20).to_numpy()
    ten = (Hs.rolling(9).max() + Ls.rolling(9).min()) / 2
    kij = (Hs.rolling(26).max() + Ls.rolling(26).min()) / 2
    sa_now = (ten + kij) / 2                                   # 오늘 계산한 선행1 (26일 뒤에 그려짐)
    sb_now = (Hs.rolling(52).max() + Ls.rolling(52).min()) / 2
    f["ten"], f["kij"] = ten.to_numpy(), kij.to_numpy()
    f["sa"], f["sb"] = sa_now.shift(26).to_numpy(), sb_now.shift(26).to_numpy()   # 오늘 위치 구름
    f["sa_f"], f["sb_f"] = sa_now.to_numpy(), sb_now.to_numpy()                   # 앞 구름 (26일 뒤)
    f["lo500"] = Ls.rolling(500, min_periods=120).min().to_numpy()
    f["hi250"] = Hs.rolling(250, min_periods=60).max().to_numpy()
    f["argmax250"] = Hs.rolling(250, min_periods=60).apply(lambda a: len(a) - 1 - int(np.argmax(a)), raw=True).to_numpy()
    f.update(O=O, H=H, L=L, C=C, V=V)
    t._feat, t._fid = f, id(d1)


def _r(a, b):
    return float(a / b - 1) * 100 if b and np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan


def extra(d1, i, snap, wb, ei, entry, l5, t):
    if getattr(t, "_fid", None) != id(d1):
        _prep(d1, t)
    f = t._feat
    O, H, L, C, V = f["O"], f["H"], f["L"], f["C"], f["V"]
    c = C[i]
    rng = H[i] - L[i]
    body = abs(C[i] - O[i])
    lw = min(O[i], C[i]) - L[i]
    x = {}
    # 거래량
    x["v_today"] = V[i] / f["vma20"][i] if f["vma20"][i] else np.nan
    x["v5_20"] = f["vma5"][i] / f["vma20"][i] if f["vma20"][i] else np.nan
    l3i, l5i = snap.get("_l3i"), snap.get("_l5i")
    daily = not wb and l3i is not None and l5i is not None
    if daily:
        a3 = V[max(0, l3i - 3):l3i + 4].mean()
        a5 = V[max(0, l5i - 3):min(i, l5i + 3) + 1].mean()
        x["v_w5_w3"] = a5 / a3 if a3 else np.nan
        x["obv_div"] = float(f["obv"][l5i] > f["obv"][l3i])
        x["rsi_bounce"] = f["rsi"][i] - f["rsi"][l5i]
        x["gap_bars"] = l5i - l3i
        x["l5_wick"] = ((min(O[l5i], C[l5i]) - L[l5i]) / (H[l5i] - L[l5i])) if H[l5i] > L[l5i] else np.nan
        x["since_l5"] = i - l5i
    else:
        for k in ("v_w5_w3", "obv_div", "rsi_bounce", "gap_bars", "l5_wick", "since_l5"):
            x[k] = np.nan
    x["obv_slope20"] = (f["obv"][i] - f["obv"][i - 20]) / (f["vma20"][i] * 20) if i >= 20 and f["vma20"][i] else np.nan
    # RSI
    x["rsi_d"] = f["rsi"][i]
    x["l3r"], x["l5r"] = float(snap["l3r"]), float(snap["l5r"])
    x["stoch"] = f["stoch"][i]
    x["l5c"] = int(snap.get("l5c") or 0)
    # 캔들
    x["today_c"] = candle_code(O, C, H, L, i)
    x["wick_low"] = lw / rng if rng > 0 else np.nan
    x["body_ratio"] = body / rng if rng > 0 else np.nan
    x["close_pos"] = (C[i] - L[i]) / rng if rng > 0 else np.nan
    x["bull_day"] = float(C[i] > O[i])
    # MACD (PPO 히스토그램은 stop_backtest 에서 t.ph 로 계산됨)
    x["hist_up"] = float(t.ph[i] > t.ph[i - 1]) if i >= 1 else np.nan
    x["macd_above"] = float(t.macd[i] > t.sig[i])
    x["macd_pct"] = t.macd[i] / c * 100
    # 일목
    top, bot = np.fmax(f["sa"][i], f["sb"][i]), np.fmin(f["sa"][i], f["sb"][i])
    x["cloud_pos"] = 1.0 if c > top else (-1.0 if c < bot else 0.0) if np.isfinite(top) else np.nan
    x["cloud_thick"] = (top - bot) / c * 100 if np.isfinite(top) else np.nan
    x["cloud_dist"] = _r(c, bot) if np.isfinite(bot) else np.nan
    x["tk_cross"] = float(f["ten"][i] > f["kij"][i])
    x["kijun_dist"] = _r(c, f["kij"][i])
    x["chikou"] = _r(c, C[i - 26]) if i >= 26 else np.nan
    x["future_cloud_up"] = float(f["sa_f"][i] > f["sb_f"][i])
    # 이평선
    for n in (5, 20, 60, 120):
        x[f"d_ma{n}"] = _r(c, f[f"ma{n}"][i])
    x["ma5_20"] = float(f["ma5"][i] > f["ma20"][i])
    x["ma_bear"] = float(f["ma20"][i] < f["ma60"][i] < f["ma120"][i])
    x["ma20_slope"] = _r(f["ma20"][i], f["ma20"][i - 5]) if i >= 5 else np.nan
    m3 = np.array([f["ma5"][i], f["ma20"][i], f["ma60"][i]])
    x["ma_conv"] = np.std(m3) / c * 100 if np.all(np.isfinite(m3)) else np.nan
    for nm, arr in zip(("w60", "w120", "m60", "m120"), t.res):
        x[f"d_{nm}"] = _r(c, arr[i])
    # 피보
    x["retr4"] = float(snap.get("retr4") or np.nan)
    lo2, hi = f["lo500"][i], f["hi250"][i]
    x["fib_retr"] = (hi - c) / (hi - lo2) if np.isfinite(lo2) and hi > lo2 else np.nan
    x["since_high"] = f["argmax250"][i]
    # 다이버전스 겹침
    x["div_cnt60"] = sum(1 for k in getattr(t, "_divs", []) if i - 60 <= k < i)
    x["w_rsi_div_now"] = float(getattr(t, "_wdiv_on", {}).get(i, False))
    # 종목·변동성
    x["atr_pct"] = f["atr"][i] / c * 100
    x["bb_pctb"] = (c - f["bbl"][i]) / (f["bbu"][i] - f["bbl"][i]) if f["bbu"][i] > f["bbl"][i] else np.nan
    x["bb_width"] = (f["bbu"][i] - f["bbl"][i]) / c * 100
    x["ret20"] = _r(c, C[i - 20]) if i >= 20 else np.nan
    x["mkt_ret20"] = getattr(t, "_mret20", np.full(len(C), np.nan))[i]
    x["cap_rank"] = getattr(t, "_rank", np.nan)
    return x


def report(sigs, n, t0):
    import time
    from datetime import datetime
    df = pd.DataFrame([{k: v for k, v in s.items() if k not in MODES} for s in sigs])
    for m in MODES:
        df[m] = [s[m]["ret"] for s in sigs]
        df[m + " 일수"] = [len(s[m]["vals"]) for s in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    df.to_csv(scan.STATE / "feature_trades.csv", index=False)
    L = ["# 조합 탐색용 데이터", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 · 신호 {len(df)}건 · 특징 {len(df.columns)}개",
         "- `state/feature_trades.csv` 에 저장했어요. 조합 탐색은 이 파일로 따로 돌려요.", "",
         "| 연도 | 신호 수 | 수익형 평균 | 안정형 평균 |", "|---|---|---|---|"]
    for y, g in df.groupby("year"):
        L.append(f"| {y} | {len(g)} | {g[MODES[0]].mean():+.2f}% | {g[MODES[1]].mean():+.2f}% |")
    L += ["", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    log("데이터 저장")


_orig_sigs = SBT.stock_sigs


def stock_sigs(s, start, kp, mk):
    """다이버전스 목록·주봉 다이버전스 날짜·시장 20일 수익률·시총 순위를 미리 넣어 둠"""
    t = SBT._tl
    t._rank = s.get("_rank", np.nan)
    idx = mk.get(s["market"])
    t._mret20 = None
    t._divs, t._wdiv_on = [], {}
    try:
        from analysis import analyze_frame, to_week
        d1 = SBT.EB.cached_daily(s["code"], 2000, s["market"])
        d1 = d1[~d1.index.duplicated()].sort_index()
        if idx is not None and len(d1):
            r20 = (idx / idx.shift(20) - 1) * 100
            t._mret20 = r20.reindex(d1.index, method="ffill").to_numpy()
        ev = []
        analyze_frame(d1, 1, SBT.P, ev)
        t._divs = [i for i, _ in ev]
        wk = to_week(d1)
        wev = []
        analyze_frame(wk, 1, SBT.P, wev)
        for j, _ in wev:
            a = d1.index.searchsorted(wk.index[j] + pd.Timedelta(days=6), side="right") - 1
            for k in range(max(0, a), min(len(d1), a + 10)):     # 주봉 다이버전스 확정 뒤 10일 동안 "켜짐"
                t._wdiv_on[k] = True
    except Exception:
        pass
    if t._mret20 is None:
        t._mret20 = np.full(4000, np.nan)
    return _orig_sigs(s, start, kp, mk)


def main():
    SBT.MODES = MODES
    SBT.path = HB.path
    SBT.report = report
    SBT.EXTRA = extra
    SBT.stock_sigs = stock_sigs
    _fu = scan.fetch_universe

    def ranked():
        u = _fu()
        for r, s in enumerate(u, 1):
            s["_rank"] = r
        return u
    scan.fetch_universe = ranked
    SBT.main()


if __name__ == "__main__":
    main()
