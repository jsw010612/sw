"""
손절 방식 비교 — 같은 바닥 신호, 같은 익절(저항선 절반 + 최고가 -20%)에서 "언제 손절하느냐"만 바꿈

손절 방식 (S0 말고는 전부 매수가 -20% 종가 이탈 시 비상 손절을 같이 둠)
  S0 5파 저점          : 5파 저점 종가 이탈 (지금 방식)
  S1 5파 + 거래량      : 5파 저점 이탈이 거래량 동반(20일 평균 1.5배↑)이면 손절, 거래량 없으면 5파 -8%까지 기다림
  S2 5파 -5% 여유      : 5파 저점 -5% 종가 이탈 (고정으로 널널하게)
  S3 5·20 데드크로스   : 매수 뒤 5일선이 20일선을 아래로 뚫는 날
  S4 후행스팬 캔들     : 후행스팬(오늘 종가)이 26일 전 캔들 저가 아래로 내려가는 날
  S5 후행스팬 20선     : 후행스팬이 26일 전 20일선 아래로 내려가는 날
  S6 MACD 데드크로스   : MACD가 시그널선을 아래로 뚫는 날
  S7 MACD 0선          : MACD가 0선 아래로 내려가는 날
  S8 5파 + 시장 과매도 : 5파 이탈해도 그 시장(코스피/코스닥) RSI가 30 아래면 손절 미룸
  S9 종합              : 5파 이탈 + (거래량 동반 또는 후행스팬이 캔들 아래)일 때만 손절, 아니면 5파 -8%까지 기다림
  S10 5파 + 후행스팬   : 5파 이탈 또는 후행스팬 캔들 하향 통과 중 먼저

신호 묶음: 예전 A등급 / 예전 A + 새 점수 강 / 넓은 신호 전체
매수 쪽 비교: MACD 히스토그램 다이버전스(5파 저점 히스토그램 > 3파 저점) · 시장 RSI 과매도에서 산 신호
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import re
import requests

import scan
import strategy as S
import deep_backtest as DB
import exit_backtest as EB
import score_backtest as SB
from analysis import Params, analyze_frame, rsi, to_week
from ma_backtest import htf_ma, log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
BUY_COST, SELL_COST = DB.BUY_COST, DB.SELL_COST
MODES = ["S0 5파 저점", "S1 5파+거래량", "S2 5파 -5% 여유", "S3 5·20 데드크로스", "S4 후행스팬 캔들",
         "S5 후행스팬 20선", "S6 MACD 데드크로스", "S7 MACD 0선", "S8 5파+시장 과매도", "S9 종합", "S10 5파+후행스팬"]
DISASTER = 0.80
P = replace(Params(), piv_r=2)
_tl = threading.local()


def index_series(sym, yf_sym):
    try:
        r = requests.get(f"https://fchart.stock.naver.com/sise.nhn?symbol={sym}&timeframe=day&count=2500&requestType=0",
                         headers=scan.UA, timeout=20)
        rows = [(pd.Timestamp(it.split("|")[0]), float(it.split("|")[4]))
                for it in re.findall(r'data="([^"]+)"', r.content.decode("euc-kr", "ignore"))]
        if len(rows) > 300:
            return pd.Series(dict(rows)).sort_index()
    except Exception:
        pass
    try:
        import yfinance as yf
        d = yf.download(yf_sym, period="10y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def path(O, H, L, C, ei, stop, mode):
    t = _tl
    V, vma, ma5, ma20 = t.V, t.vma, t.ma5, t.ma20
    entry = O[ei]
    n = len(C)
    last = min(n - 1, ei + S.HOLD_MAX)
    held, cash, tp, peak = 1.0, -BUY_COST, False, entry
    ups = [m[ei] for m in t.res if np.isfinite(m[ei]) and m[ei] >= entry * 1.10]
    rline = min(min(ups), entry * 1.5) if ups else entry * 1.5
    vals = []

    def lag(a, j):
        return a[j - 26] if j - 26 >= 0 else np.nan

    for j in range(ei, last + 1):
        peak = max(peak, H[j])
        if not tp and H[j] >= rline:
            px = max(O[j], rline) if j > ei else rline
            cash += 0.5 * px / entry * (1 - SELL_COST); held = 0.5; tp = True
        why = None
        c, pc = C[j], C[j - 1] if j > 0 else C[j]
        below5 = c < stop
        volx = V[j] >= 1.5 * vma[j] if np.isfinite(vma[j]) and vma[j] > 0 else False
        chik_below = np.isfinite(lag(L, j)) and c < lag(L, j) and not (pc < lag(L, j - 1) if j - 27 >= 0 else False)
        chik_under = np.isfinite(lag(L, j)) and c < lag(L, j)
        if mode == "S0 5파 저점":
            if below5: why = "5파 이탈"
        elif mode == "S1 5파+거래량":
            if below5 and volx: why = "5파 이탈(거래량)"
            elif c < stop * 0.92: why = "5파 -8%"
        elif mode == "S2 5파 -5% 여유":
            if c < stop * 0.95: why = "5파 -5%"
        elif mode == "S3 5·20 데드크로스":
            if j > ei and ma5[j] < ma20[j] and ma5[j - 1] >= ma20[j - 1]: why = "데드크로스"
        elif mode == "S4 후행스팬 캔들":
            if j > ei and chik_below: why = "후행스팬 캔들 아래"
        elif mode == "S5 후행스팬 20선":
            m26, m27 = lag(ma20, j), lag(ma20, j - 1) if j - 27 >= 0 else np.nan
            if j > ei and np.isfinite(m26) and c < m26 and not (np.isfinite(m27) and pc < m27): why = "후행스팬 20선 아래"
        elif mode == "S6 MACD 데드크로스":
            if j > ei and t.macd[j] < t.sig[j] and t.macd[j - 1] >= t.sig[j - 1]: why = "MACD 데드크로스"
        elif mode == "S7 MACD 0선":
            if j > ei and t.macd[j] < 0 <= t.macd[j - 1]: why = "MACD 0선 아래"
        elif mode == "S8 5파+시장 과매도":
            mr = t.mrsi[j]
            if below5 and not (np.isfinite(mr) and mr < 30): why = "5파 이탈"
        elif mode == "S9 종합":
            if below5 and (volx or chik_under): why = "5파 이탈(확인)"
            elif c < stop * 0.92: why = "5파 -8%"
        elif mode == "S10 5파+후행스팬":
            if below5: why = "5파 이탈"
            elif j > ei and chik_below: why = "후행스팬 캔들 아래"
        if why is None and mode != "S0 5파 저점" and c < entry * DISASTER:
            why = "비상 -20%"
        if why is None and tp and c < peak * 0.80:
            why = "추적 매도"
        if why or j == last:
            cash += held * c / entry * (1 - SELL_COST)
            vals.append(cash)
            return np.array(vals), why or "기간 만료", j
        vals.append(held * c / entry + cash)
    return np.array(vals), "기간 만료", last


def stock_sigs(s, start, kp, mk):
    try:
        d1 = EB.cached_daily(s["code"], 2000, s["market"])
        if len(d1) < 400:
            return []
        d1 = d1[~d1.index.duplicated()].sort_index()
        n = len(d1)
        O, H, L, C, V = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
        Cs = pd.Series(C)
        t = _tl
        t.V, t.vma = V, pd.Series(V).rolling(20).mean().shift(1).to_numpy()
        t.ma5, t.ma20 = Cs.rolling(5).mean().to_numpy(), Cs.rolling(20).mean().to_numpy()
        e12, e26 = Cs.ewm(span=12, adjust=False).mean(), Cs.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        t.macd, t.sig = macd.to_numpy(), macd.ewm(span=9, adjust=False).mean().to_numpy()
        hist = t.macd - t.sig
        t.res = [htf_ma(d1, f, m) for f, m in (("W", 60), ("W", 120), ("M", 60), ("M", 120))]
        idx = mk.get(s["market"])
        if idx is not None:
            mrs = pd.Series(rsi(idx.to_numpy(float), 14), index=idx.index)
            t.mrsi = mrs.reindex(d1.index, method="ffill").to_numpy()
        else:
            t.mrsi = np.full(n, np.nan)
        kg = None
        if kp is not None:
            k = kp.reindex(d1.index, method="ffill")
            kg = ((k / k.rolling(120).mean() - 1) * 100).to_numpy()
        cands = []
        dev = []
        analyze_frame(d1, 1, P, dev)
        cands += [(i, sn, False) for i, sn in dev]
        wk = to_week(d1)
        wev = []
        analyze_frame(wk, 1, P, wev)
        for j, sn in wev:
            k = d1.index.searchsorted(wk.index[j] + pd.Timedelta(days=6), side="right") - 1
            if k >= 0:
                cands.append((int(k), sn, True))
        out, seen = [], set()
        for det_i, snap, wb in cands:
            ei = det_i + 1
            if ei >= n or d1.index[ei] < start or (det_i, wb) in seen:
                continue
            seen.add((det_i, wb))
            entry, l5 = O[ei], float(snap["l5p"])
            if not entry > l5 or not np.isfinite(snap["depth"]) or snap["depth"] < 1.0:
                continue
            hi = H[max(0, det_i - 249):det_i + 1].max()
            if hi <= 0 or (1 - C[det_i] / hi) * 100 < 30 or (1 - l5 / entry) * 100 > 40:
                continue
            kgap = float(kg[det_i]) if kg is not None and np.isfinite(kg[det_i]) else 0.0
            f = SB.features_at(d1, det_i, snap, wb, kgap, entry)
            old = S.score(kgap < 0, f["mrsi"], wb, f["l3vr"])
            old_a = S.qualifies(f["dd52"], f["depth"], entry, l5) and old >= 85
            new = SB.score_of(f, RULES) if RULES else 0
            macd_div = None
            if not wb:
                l3i, l5i = snap.get("_l3i"), snap.get("_l5i")
                if l3i is not None and l5i is not None:
                    macd_div = bool(hist[l5i] > hist[l3i])
            rec = dict(code=s["code"], name=s["name"], market=s["market"], date=d1.index[ei], old_a=old_a,
                       strong=bool(RULES and new >= RULES["strong"]), new_score=new, macd_div=macd_div,
                       mkt_rsi=float(t.mrsi[det_i]) if np.isfinite(t.mrsi[det_i]) else np.nan, **f)
            for m in MODES:
                v, why, xj = path(O, H, L, C, ei, l5, m)
                rec[m] = dict(dates=d1.index[ei:ei + len(v)], vals=v, reason=why, ret=(v[-1] - 1) * 100)
            out.append(rec)
        return out
    finally:
        with EB._lock:
            EB._cache.pop(s["code"], None)


def stat(v):
    return SB.stat(v)


def row(lab, v):
    return SB.row(lab, v)


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in t.items() if k not in MODES} for t in sigs])
    for m in MODES:
        df[m] = [t[m]["ret"] for t in sigs]
        df[m + " 일수"] = [len(t[m]["vals"]) for t in sigs]
        df[m + " 사유"] = [t[m]["reason"] for t in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    groups = {"예전 A + 새 점수 강": df.old_a & df.strong, "예전 A": df.old_a, "넓은 신호 전체": df.old_a | ~df.old_a}
    years = sorted(df.year.unique())
    L = ["# 손절 방식 비교 — 바닥 반등 신호", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 · 신호 {len(df)}건",
         "- 익절은 전부 같음: 위쪽 주·월봉 60/120선(없으면 +50%)에서 절반 · 나머지 최고가 -20%",
         "- S0 말고는 매수가 -20% 종가 이탈 시 비상 손절을 같이 둠",
         "- 하루당 = 수익 합 ÷ 보유일 합 (돈 회전) · 손실 평균은 손실 난 거래만", ""]
    for gname, m in groups.items():
        g = df[m]
        L += [f"## {gname} ({len(g)}건)", "",
              "| 손절 방식 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF | 평균 보유일 | 하루당 | 비상 손절 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for md in MODES:
            s = stat(g[md])
            if not s:
                continue
            dd = g[md + " 일수"]
            emer = (g[md + " 사유"] == "비상 -20%").mean() * 100
            L.append(f"| {md} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | {SB.f2(s['rr'])} | "
                     f"{SB.f2(s['pf'])} | {dd.mean():.0f} | {g[md].sum() / dd.sum():.3f}% | {emer:.0f}% |")
        L += ["", "연도별 PF", "", "| 손절 방식 | " + " | ".join(map(str, years)) + " |", "|---|" + "---|" * len(years)]
        for md in MODES:
            cells = []
            for y in years:
                s = stat(g[g.year == y][md])
                cells.append("-" if not s or s["n"] < 5 else SB.f2(s["pf"]))
            L.append(f"| {md} | " + " | ".join(cells) + " |")
        L.append("")
    g = df[df.old_a]
    L += ["## 매수 쪽: MACD 히스토그램 다이버전스 · 시장 과매도 (예전 A, S0 기준)", "", SB.HEAD,
          row("MACD 다이버전스 있음", g[g.macd_div == True]["S0 5파 저점"]),
          row("MACD 다이버전스 없음", g[g.macd_div == False]["S0 5파 저점"]),
          row("시장 RSI 30 미만에서 매수", g[g.mkt_rsi < 30]["S0 5파 저점"]),
          row("시장 RSI 30~45", g[(g.mkt_rsi >= 30) & (g.mkt_rsi < 45)]["S0 5파 저점"]),
          row("시장 RSI 45 이상", g[g.mkt_rsi >= 45]["S0 5파 저점"]), ""]
    g = df
    L += ["### 넓은 신호 전체로 보면", SB.HEAD,
          row("MACD 다이버전스 있음", g[g.macd_div == True]["S0 5파 저점"]),
          row("MACD 다이버전스 없음", g[g.macd_div == False]["S0 5파 저점"]),
          row("시장 RSI 30 미만", g[g.mkt_rsi < 30]["S0 5파 저점"]),
          row("시장 RSI 30~45", g[(g.mkt_rsi >= 30) & (g.mkt_rsi < 45)]["S0 5파 저점"]),
          row("시장 RSI 45 이상", g[g.mkt_rsi >= 45]["S0 5파 저점"]), "",
          "## 주의", "- 상장폐지 종목이 빠져 있어서 실제보다 좋게 나와요. 손절을 늦추는 방식일수록 이 영향이 커요 (상폐 종목은 끝까지 떨어지니까).",
          "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    keep = [c for c in df.columns if c not in ("dates",)]
    df[keep].to_csv(scan.STATE / "stop_trades.csv", index=False)
    log("보고서 저장")


RULES = None


def main():
    global RULES
    t0 = time.time()
    try:
        RULES = json.loads((scan.STATE / "score_rules.json").read_text(encoding="utf-8"))
    except Exception:
        RULES = None
    log(f"새 점수표: {'있음' if RULES else '없음'}")
    scan.probe_sources()
    scan.naver_daily = EB.cached_daily
    uni = scan.fetch_universe()[:TOP_N]
    kp = DB.kospi()
    mk = {"KS": kp, "KQ": index_series("KOSDAQ", "^KQ11")}
    log(f"코스피: {'있음' if kp is not None else '없음'} · 코스닥: {'있음' if mk['KQ'] is not None else '없음'}")
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    sigs, done = [], [0]

    def job(s):
        try:
            return stock_sigs(s, start, kp, mk)
        except Exception as e:
            log("오류", s["name"], repr(e))
            return []
        finally:
            done[0] += 1
            if done[0] % 200 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            sigs += r
    log(f"신호 {len(sigs)}건")
    if not sigs:
        (scan.STATE / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    report(sigs, len(uni), t0)


if __name__ == "__main__":
    main()
