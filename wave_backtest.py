"""
파동 · 피보나치 · 하모닉 패턴 백테스트

목표: 연 30~50%를 노릴 수 있는 셋업이 있는지, 그리고 그게 특정 해에만 맞은 건지 확인

꼭짓점 찾기 (지그재그)
  고점에서 th% 이상 내려오면 그 고점 확정, 저점에서 th% 이상 오르면 그 저점 확정
  확정된 날 이후에만 사용 (미래 정보 없음)
  단기: th 8%  (2주 안팎 짧은 스윙)
  스윙: th 18% (몇 달짜리 큰 파동)

셋업 (전부 매수, 마지막 확정 꼭짓점이 고점이고 지금 내려오는 중일 때)
  가틀리   : AB = XA의 0.618, BC = AB의 0.382~0.886, D = XA의 0.786
  뱃       : AB = XA의 0.382~0.5, D = XA의 0.886
  버터플라이: AB = XA의 0.786, D = XA의 1.27
  크랩     : AB = XA의 0.382~0.618, D = XA의 1.618
  ABCD     : BC = AB의 0.618~0.786, CD = AB의 1.0~1.272
  3파 진입 : 1파(120일 최저점에서 출발, 거래량 증가) 후 2파 되돌림 0.5~0.786 구간에서 반등
  3파 돌파 : 2파 저점 확정 후 1파 고점을 종가로 돌파
  5파 진입 : 1·2·3파 규칙(2파 > 0, 3파가 1파보다 김) 확인 후 4파 되돌림 0.236~0.5 구간에서 반등 (1파 고점 아래로 가면 무효)
  피보 되돌림: 큰 상승(th의 3배 이상) 뒤 0.5~0.65 구간에서 반등

진입: 구간 안에 들어온 뒤 15일 안에 반전 신호(종가가 전날 고가 돌파, 또는 5일선 상향 회복) → 다음날 시가
손절: 셋업별 무효 가격 (하모닉: 구간 하단 -3% · 3파: 0파 저점 · 5파: 1파 고점 · 피보: 0.786 아래)
목표: 하모닉 T1 = AD의 0.382, T2 = 0.618 / 3파 T1 = 1파 길이만큼, T2 = 1.618배 / 5파 T1 = 1파의 0.618, T2 = 1파 길이 / 피보 T1 = 이전 고점, T2 = 1.272 확장

매도 비교
  T  목표가   : T1에서 절반 + 손절선을 본전으로, T2에서 나머지
  R  절반+추적: T1에서 절반 + 본전, 나머지는 최고 종가 대비 -15% 또는 T2의 1.618배 (최대 250일)

함께 기록하는 조건 (효과 비교용)
  RSI 다이버전스(이번 저점이 전 저점보다 낮은데 RSI는 높음) · 조정 구간 거래량 감소 ·
  주봉/월봉 60·120선이 진입 구간 ±3% 안에 겹침 · 코스피 정배열 · 시장 폭

과최적화 방지: 2021~2024년으로만 "쓸 만한 셋업"을 고르고, 2025~2026년 성과를 따로 보여줌
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import rsi
from ma_backtest import HEAD, htf_ma, log, row, st

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
BUY_COST, SELL_COST = 0.0015, 0.0015
DEGREES = {"단기": (0.08, 40), "스윙": (0.18, 120)}       # (지그재그 %, T방식 최대 보유일)
SETUPS = ["가틀리", "뱃", "버터플라이", "크랩", "ABCD", "3파 진입", "3파 돌파", "5파 진입", "피보 되돌림"]
EXITS = ["T 목표가", "R 절반+추적"]
TRIG_WIN = 15
SPLIT_YEAR = 2025          # 이 해 이전 = 고르는 기간, 이후 = 검증 기간


# ───────────────────────── 지그재그 ─────────────────────────
def zigzag(H, L, th):
    """[(꼭짓점 위치, 가격, 'H'/'L', 확정된 날)]"""
    n = len(H)
    out = []
    trend, hi, lo = 0, 0, 0
    for j in range(1, n):
        if trend == 0:
            if H[j] > H[hi]: hi = j
            if L[j] < L[lo]: lo = j
            if H[j] >= L[lo] * (1 + th) and lo < j:
                out.append((lo, L[lo], "L", j)); trend, hi = 1, j
            elif L[j] <= H[hi] * (1 - th) and hi < j:
                out.append((hi, H[hi], "H", j)); trend, lo = -1, j
        elif trend == 1:
            if H[j] > H[hi]:
                hi = j
            elif L[j] <= H[hi] * (1 - th):
                out.append((hi, H[hi], "H", j)); trend, lo = -1, j
        else:
            if L[j] < L[lo]:
                lo = j
            elif H[j] >= L[lo] * (1 + th):
                out.append((lo, L[lo], "L", j)); trend, hi = 1, j
    return out


def inr(v, a, b, tol=0.04):
    return a - tol <= v <= b + tol


# ───────────────────────── 셋업 만들기 ─────────────────────────
def setups_from(pv, k, O, H, L, C, V):
    """pv[k] = 방금 확정된 고점. 거기서 시작하는 하락 구간에 걸 수 있는 셋업들"""
    out = []
    if pv[k][2] != "H":
        return out

    def P(i):
        return pv[k - i][1] if k - i >= 0 else None

    def I(i):
        return pv[k - i][0]

    C_ = P(0)
    # ── 하모닉 XABCD (X=L, A=H, B=L, C=H) ──
    if k >= 3:
        X, A, B = P(3), P(2), P(1)
        XA, AB = A - X, A - B
        if XA > 0 and AB > 0 and C_ < A and B > X * 0.5:
            ab, bc = AB / XA, (C_ - B) / AB
            defs = [("가틀리", (0.618, 0.618), 0.786, (1.272, 1.618)),
                    ("뱃", (0.382, 0.5), 0.886, (1.618, 2.618)),
                    ("버터플라이", (0.786, 0.786), 1.27, (1.618, 2.24)),
                    ("크랩", (0.382, 0.618), 1.618, (2.24, 3.618))]
            for name, (a1, a2), dx, (c1, c2) in defs:
                if not inr(ab, a1, a2) or not inr(bc, 0.382, 0.886, 0.02):
                    continue
                D = A - dx * XA
                lo_bc, hi_bc = C_ - c2 * (C_ - B), C_ - c1 * (C_ - B)     # BC 연장으로 본 D 범위
                if not (lo_bc * 0.97 <= D <= hi_bc * 1.03) or D <= 0:
                    continue
                stop = (min(D, X) if dx < 1 else D) * 0.97
                out.append(dict(setup=name, ztop=D * 1.03, zbot=D * 0.97, stop=stop, A=A, mode="harm",
                                prev_low=B, prev_low_i=I(1)))
    # ── ABCD (A=H, B=L, C=H) ──
    if k >= 2:
        A, B = P(2), P(1)
        AB = A - B
        if AB > 0 and C_ < A:
            bc = (C_ - B) / AB
            if inr(bc, 0.618, 0.786, 0.02):
                d1, d2 = C_ - 1.0 * AB, C_ - 1.272 * AB
                if d2 > 0:
                    out.append(dict(setup="ABCD", ztop=d1 * 1.02, zbot=d2 * 0.98, stop=d2 * 0.95, A=A, mode="harm",
                                    prev_low=B, prev_low_i=I(1)))
    # ── 엘리엇 3파 진입 (0=L, 1=H) ──
    if k >= 1:
        p0, p1, i0, i1 = P(1), C_, I(1), I(0)
        w1 = p1 - p0
        if w1 / p0 >= 0.25 and i0 >= 120 and p0 <= L[i0 - 120:i0].min():
            v_up = V[i0:i1 + 1].mean()
            v_bef = V[max(0, i0 - 60):i0].mean()
            if v_bef > 0 and v_up >= 1.3 * v_bef:
                out.append(dict(setup="3파 진입", ztop=p1 - 0.5 * w1, zbot=p1 - 0.786 * w1, stop=p0 * 0.98,
                                w1=w1, p1=p1, mode="w3", prev_low=p0, prev_low_i=i0, v_up=v_up))
    # ── 엘리엇 5파 진입 (0=L,1=H,2=L,3=H) ──
    if k >= 3:
        p0, p1, p2, p3 = P(3), P(2), P(1), C_
        i0 = I(3)
        w1, w3 = p1 - p0, p3 - p2
        if (w1 > 0 and p2 > p0 and p3 > p1 and w3 >= w1 and inr((p1 - p2) / w1, 0.382, 0.786)
                and i0 >= 120 and p0 <= L[i0 - 120:i0].min()):
            ztop, zbot = p3 - 0.236 * w3, max(p3 - 0.5 * w3, p1 * 1.01)
            if ztop > zbot:
                out.append(dict(setup="5파 진입", ztop=ztop, zbot=zbot, stop=p1 * 0.99, w1=w1, p3=p3, mode="w5",
                                prev_low=p2, prev_low_i=I(1)))
    # ── 피보 되돌림 (큰 상승 뒤) ──
    if k >= 1:
        p0, p1 = P(1), C_
        sw = p1 - p0
        out.append(dict(setup="피보 되돌림", ztop=p1 - 0.5 * sw, zbot=p1 - 0.65 * sw, stop=(p1 - 0.786 * sw) * 0.98,
                        sw=sw, p0=p0, p1=p1, mode="fib", prev_low=p0, prev_low_i=I(1), need_big=True))
    return out


def setup_3b(pv, k, L, V):
    """3파 돌파: pv[k] = 방금 확정된 2파 저점"""
    if k < 2 or pv[k][2] != "L":
        return None
    p0, p1, p2 = pv[k - 2][1], pv[k - 1][1], pv[k][1]
    i0 = pv[k - 2][0]
    w1 = p1 - p0
    if w1 <= 0 or w1 / p0 < 0.25 or not (p2 > p0 and inr((p1 - p2) / w1, 0.382, 0.786)):
        return None
    if i0 < 120 or p0 > L[i0 - 120:i0].min():
        return None
    return dict(setup="3파 돌파", level=p1, stop=p2 * 0.98, t1=p2 + 1.272 * w1, t2=p2 + 1.618 * w1)


# ───────────────────────── 매도 시뮬레이션 ─────────────────────────
def simulate(O, H, L, C, ei, stop, t1, t2, mode, hold):
    entry = O[ei]
    if t1 <= entry * 1.01:
        t1 = entry * 1.03
    t2 = max(t2, t1 * 1.03)
    if mode == "R 절반+추적":
        t2 = entry + (t2 - entry) * 1.618
        hold = 250
    last = min(len(C) - 1, ei + hold)
    held, cash, half, peak = 1.0, -BUY_COST, False, entry
    reason, j = "기간 만료", ei
    for j in range(ei, last + 1):
        if L[j] <= stop:
            px = min(O[j], stop) if j > ei else stop
            cash += held * px / entry * (1 - SELL_COST); held = 0
            reason = "본전 정리" if half else "손절"
            break
        if not half and H[j] >= t1:
            px = max(O[j], t1) if j > ei else t1
            cash += 0.5 * px / entry * (1 - SELL_COST); held, half = 0.5, True
            stop = max(stop, entry * 1.003)
        if half and H[j] >= t2:
            px = max(O[j], t2) if j > ei else t2
            cash += held * px / entry * (1 - SELL_COST); held = 0; reason = "목표 도달"
            break
        peak = max(peak, C[j])
        if mode == "R 절반+추적" and half and C[j] < peak * 0.85:
            cash += held * C[j] / entry * (1 - SELL_COST); held = 0; reason = "추적 매도"
            break
    if held > 0:
        cash += held * C[j] / entry * (1 - SELL_COST)
    return round((cash - 1) * 100, 2), reason, j - ei


# ───────────────────────── 종목별 ─────────────────────────
def stock_trades(s, start):
    d1 = scan.naver_daily(s["code"], 3200, s["market"])
    if len(d1) < 400:
        return [], None
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C, V = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
    R = np.asarray(rsi(d1["close"], 14), float)
    ma5 = pd.Series(C).rolling(5).mean().to_numpy()
    ma60 = pd.Series(C).rolling(60).mean().to_numpy()
    mas = [htf_ma(d1, f, m) for f, m in (("W", 60), ("W", 120), ("M", 60), ("M", 120))]
    first = d1.index.searchsorted(start)
    dates = d1.index
    out = []

    def trig(j):
        return j >= 1 and (C[j] > H[j - 1] or (C[j] > ma5[j] and C[j - 1] <= ma5[j - 1]))

    for deg, (th, hold) in DEGREES.items():
        pv = zigzag(H, L, th)
        busy = {}                                          # 셋업별: 이 날까지 보유 중
        for k in range(len(pv)):
            conf = pv[k][3]
            if conf < first or conf >= n - 2:
                continue
            # 3파 돌파
            if pv[k][2] == "L":
                b = setup_3b(pv, k, L, V)
                if b and busy.get(b["setup"], -1) < conf:
                    nxt = pv[k + 1][3] if k + 1 < len(pv) else n - 1
                    for j in range(conf, min(n - 1, conf + 60)):
                        if L[j] <= b["stop"]:
                            break
                        if C[j] > b["level"] and C[j - 1] <= b["level"]:
                            ei = j + 1
                            if ei >= n:
                                break
                            rec = base_rec(s, deg, b["setup"], dates, ei, O, b["stop"], R, V, C, ma60, mas,
                                           j, None, None, b["level"])
                            if rec is None:
                                break
                            for ex in EXITS:
                                r, why, days = simulate(O, H, L, C, ei, b["stop"], b["t1"], b["t2"], ex, hold)
                                rec[ex], rec[ex + " 사유"], rec[ex + " 일수"] = r, why, days
                            out.append(rec)
                            busy[b["setup"]] = ei + rec["T 목표가 일수"]
                            break
                continue
            # 고점이 확정됨 → 지금 하락 구간에 걸리는 셋업들
            nxt_conf = pv[k + 1][3] if k + 1 < len(pv) else n - 1
            for su in setups_from(pv, k, O, H, L, C, V):
                if su.get("need_big") and su["sw"] / su["p0"] < 3 * th:
                    continue
                if busy.get(su["setup"], -1) >= conf:
                    continue
                entered, lo = None, np.inf
                end = min(n - 1, nxt_conf + 1, conf + 120)
                for j in range(conf, end + 1):
                    lo = min(lo, L[j])
                    if C[j] < su["zbot"] * (0.97 if su["mode"] == "harm" else 1.0) or L[j] <= su["stop"]:
                        break                                       # 구간 아래로 무너짐 → 무효
                    if entered is None and L[j] <= su["ztop"]:
                        entered = j
                    if entered is not None:
                        if j - entered > TRIG_WIN:
                            break
                        if trig(j):
                            ei = j + 1
                            if ei >= n:
                                break
                            D = lo
                            if su["mode"] == "harm":
                                t1, t2 = D + 0.382 * (su["A"] - D), D + 0.618 * (su["A"] - D)
                            elif su["mode"] == "w3":
                                t1, t2 = D + su["w1"], D + 1.618 * su["w1"]
                            elif su["mode"] == "w5":
                                t1, t2 = D + 0.618 * su["w1"], D + su["w1"]
                            else:
                                t1, t2 = su["p1"], su["p0"] + 1.272 * su["sw"]
                            rec = base_rec(s, deg, su["setup"], dates, ei, O, su["stop"], R, V, C, ma60, mas, j,
                                           su, lo, (su["ztop"] + su["zbot"]) / 2, conf_i=pv[k][0])
                            if rec is None:
                                break
                            for ex in EXITS:
                                r, why, days = simulate(O, H, L, C, ei, su["stop"], t1, t2, ex, hold)
                                rec[ex], rec[ex + " 사유"], rec[ex + " 일수"] = r, why, days
                            rec["t1_pct"] = round((t1 / O[ei] - 1) * 100, 1)
                            out.append(rec)
                            busy[su["setup"]] = ei + rec["T 목표가 일수"]
                            break
    above60 = pd.Series(C > ma60, index=d1.index)
    return out, above60


def base_rec(s, deg, setup, dates, ei, O, stop, R, V, C, ma60, mas, j, su, lo, zone_mid, conf_i=None):
    entry = O[ei]
    sd = (entry - stop) / entry * 100
    if sd <= 0.5 or sd > 35:
        return None
    rec = dict(code=s["code"], name=s["name"], deg=deg, setup=setup, date=dates[ei].strftime("%Y-%m-%d"),
               year=dates[ei].year, stop_dist=round(sd, 1))
    # RSI 다이버전스: 이번 하락 저점 vs 직전 저점
    div = False
    if su is not None and lo is not None:
        pi = su["prev_low_i"]
        k0 = conf_i if conf_i is not None else j
        low_i = k0 + int(np.argmin(C[k0:j + 1])) if j >= k0 else j
        if lo < su["prev_low"] and R[low_i] > R[pi] + 2:
            div = True
    rec["rsi_div"] = div
    # 조정 구간 거래량 감소
    k0 = conf_i if conf_i is not None else max(0, j - 10)
    v_dn = V[k0:j + 1].mean() if j >= k0 else V[j]
    v_up = V[max(0, k0 - 20):k0 + 1].mean()
    rec["vol_dry"] = bool(v_up > 0 and v_dn < 0.7 * v_up)
    # 큰 이평선 겹침
    ref = zone_mid if zone_mid else entry
    rec["ma_conf"] = bool(any(np.isfinite(m[ei]) and abs(m[ei] / ref - 1) <= 0.03 for m in mas))
    rec["above_ma60"] = bool(np.isfinite(ma60[ei]) and C[j] > ma60[j])
    return rec


# ───────────────────────── 계좌 ─────────────────────────
def account(g, ex, risk=0.01, cap=0.20, max_open=8, total=0.80):
    """복리 계좌. 연말 자산을 같이 돌려줌"""
    tr = g.sort_values("date").to_dict("records")
    for t in tr:
        t["exit"] = (pd.Timestamp(t["date"]) + pd.tseries.offsets.BDay(int(t[ex + " 일수"]))).strftime("%Y-%m-%d")
    ev = sorted([(t["date"], 1, k) for k, t in enumerate(tr)] + [(t["exit"], 0, k) for k, t in enumerate(tr)],
                key=lambda e: (e[0], e[1]))
    cash, op, peak, mdd, taken = 100.0, {}, 100.0, 0.0, 0
    codes, ye, cur_y = set(), {}, None
    for d, typ, k in ev:
        y = int(d[:4])
        if cur_y is not None and y != cur_y:
            ye[cur_y] = cash + sum(op.values())
        cur_y = y
        t = tr[k]
        if typ == 0 and k in op:
            cash += op.pop(k) * (1 + t[ex] / 100); codes.discard(t["code"])
            eq = cash + sum(op.values()); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(op.values())
            w = min(cap, risk / max(t["stop_dist"] / 100, 0.02))
            if t["code"] in codes or len(op) >= max_open or (sum(op.values()) / eq + w) > total or eq * w > cash:
                continue
            cash -= eq * w; op[k] = eq * w; taken += 1; codes.add(t["code"])
    for k, a in op.items():
        cash += a * (1 + tr[k][ex] / 100)
    if cur_y is not None:
        ye[cur_y] = cash
    yr, prev = {}, 100.0
    for y in sorted(ye):
        yr[y] = (ye[y] / prev - 1) * 100; prev = ye[y]
    cagr = ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100
    return cash, cagr, mdd * 100, taken, yr


# ───────────────────────── 보고서 ─────────────────────────
def report(df, n, t0):
    years = sorted(df.year.unique())
    L = ["# 파동 · 피보나치 · 하모닉 백테스트", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 (현재 상장 기준) · 비용 왕복 0.3%",
         "- 꼭짓점: 지그재그 (단기 8% · 스윙 18%), 확정된 뒤에만 사용",
         "- 진입: 셋업 구간 안에서 반전 신호(전날 고가 돌파 또는 5일선 회복) 다음날 시가",
         f"- 고르는 기간 {years[0]}~{SPLIT_YEAR - 1} · 검증 기간 {SPLIT_YEAR}~", ""]

    L += ["## 1. 셋업별 성과 (전체 기간)", ""]
    for ex in EXITS:
        L += [f"### {ex}", "| 셋업 | 파동 크기 " + HEAD.split("| 구분 ")[1].split("\n")[0],
              "|---|---|---|---|---|---|---|---|"]
        for su in SETUPS:
            for deg in DEGREES:
                g = df[(df.setup == su) & (df.deg == deg)]
                if len(g):
                    L.append(row(f"{su} | {deg}", g[ex]))
        L.append("")

    # 연도별 PF
    L += ["## 2. 연도별 PF (T 목표가) — 해마다 고르게 1 넘는 게 진짜", "",
          "| 셋업 | 크기 | " + " | ".join(str(y) for y in years) + " |",
          "|---|---|" + "---|" * len(years)]
    for su in SETUPS:
        for deg in DEGREES:
            g = df[(df.setup == su) & (df.deg == deg)]
            if len(g) < 30:
                continue
            cells = []
            for y in years:
                s = st(g[g.year == y]["T 목표가"])
                cells.append("-" if not s or s["n"] < 5 else (f"{s['pf']:.2f}" if s["pf"] != float("inf") else "∞") + f" ({s['n']})")
            L.append(f"| {su} | {deg} | " + " | ".join(cells) + " |")
    L.append("")

    # 겹치는 조건
    L += ["## 3. 같이 겹치면 좋아지는 조건 (T 목표가, 전체 셋업)", "", HEAD,
          row("전체", df["T 목표가"]),
          row("RSI 다이버전스 있음", df[df.rsi_div]["T 목표가"]), row("RSI 다이버전스 없음", df[~df.rsi_div]["T 목표가"]),
          row("조정 때 거래량 감소", df[df.vol_dry]["T 목표가"]), row("거래량 감소 아님", df[~df.vol_dry]["T 목표가"]),
          row("주·월 60/120선 겹침", df[df.ma_conf]["T 목표가"]), row("겹침 없음", df[~df.ma_conf]["T 목표가"]),
          row("60일선 위", df[df.above_ma60]["T 목표가"]), row("60일선 아래", df[~df.above_ma60]["T 목표가"]),
          row("코스피 정배열", df[df.kospi_up == True]["T 목표가"]), row("코스피 정배열 아님", df[df.kospi_up == False]["T 목표가"]),
          row("시장 폭 50% 이상", df[df.breadth >= 50]["T 목표가"]), row("시장 폭 50% 미만", df[df.breadth < 50]["T 목표가"]),
          row("다이버전스 + 거래량 감소", df[df.rsi_div & df.vol_dry]["T 목표가"]),
          row("다이버전스 + 이평선 겹침", df[df.rsi_div & df.ma_conf]["T 목표가"]),
          row("셋 다", df[df.rsi_div & df.vol_dry & df.ma_conf]["T 목표가"]), ""]

    # 고르는 기간으로 선택 → 검증
    tr, te = df[df.year < SPLIT_YEAR], df[df.year >= SPLIT_YEAR]
    picks = []
    flt = {"전체": lambda d: d, "다이버전스": lambda d: d[d.rsi_div], "거래량 감소": lambda d: d[d.vol_dry],
           "이평선 겹침": lambda d: d[d.ma_conf], "코스피 정배열": lambda d: d[d.kospi_up == True],
           "코스피 정배열 아님": lambda d: d[d.kospi_up == False]}
    for su in SETUPS:
        for deg in DEGREES:
            for fn, f in flt.items():
                for ex in EXITS:
                    g = f(tr[(tr.setup == su) & (tr.deg == deg)])
                    s = st(g[ex])
                    if s and s["n"] >= 40 and s["pf"] >= 1.15:
                        yrs = [st(g[g.year == y][ex]) for y in sorted(g.year.unique())]
                        good = sum(1 for v in yrs if v and v["n"] >= 5 and v["pf"] >= 1.0)
                        tot = sum(1 for v in yrs if v and v["n"] >= 5)
                        if tot and good >= max(2, tot - 1):
                            picks.append((su, deg, fn, ex, s))
    L += ["## 4. 고르는 기간(~2024)에 PF 1.15 이상 · 거래 40건 이상 · 거의 매년 1 이상인 조합 → 2025~ 검증", ""]
    if not picks:
        L += ["조건을 넘는 조합이 없어요. (= 이 셋업들만으로는 해마다 꾸준히 이기는 게 없음)", ""]
    else:
        L += ["| 셋업 | 크기 | 필터 | 매도 | 고르는 기간 거래 | 고르는 기간 PF | 검증 거래 | 검증 승률 | 검증 평균 | 검증 PF |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for su, deg, fn, ex, s in sorted(picks, key=lambda p: -p[4]["pf"])[:25]:
            g2 = flt[fn](te[(te.setup == su) & (te.deg == deg)])
            v = st(g2[ex])
            vt = f"{v['n']} | {v['win']:.0f}% | {v['avg']:+.2f}% | {v['pf']:.2f}" if v else "0 | - | - | -"
            L.append(f"| {su} | {deg} | {fn} | {ex} | {s['n']} | {s['pf']:.2f} | {vt} |")
        L.append("")

    # 계좌
    L += ["## 5. 계좌 시뮬레이션 (복리, 원금 100 · 종목당 최대 20% · 보유 8 · 총 80%)", "",
          "목표 기준: 연 30% = 5년 371 · 연 50% = 5년 759", "",
          "| 묶음 | 매도 | 1회 손실 | 최종 | 연평균 | 최대 낙폭 | 거래 | " + " | ".join(str(y) for y in years) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(years)]
    groups = {}
    if picks:
        keys = {(su, deg, fn) for su, deg, fn, ex, s in picks}
        parts = [flt[fn](df[(df.setup == su) & (df.deg == deg)]) for su, deg, fn in keys]
        groups["고른 조합 전부"] = pd.concat(parts).drop_duplicates(["code", "date"])
    for su in SETUPS:
        g = df[df.setup == su]
        if len(g) >= 100:
            groups[su] = g
    groups["다이버전스 붙은 것 전부"] = df[df.rsi_div].drop_duplicates(["code", "date"])
    for name, g in groups.items():
        for ex in EXITS:
            for rk in (0.01, 0.02):
                if name != "고른 조합 전부" and rk == 0.02:
                    continue
                fin, cagr, mdd, tk, yr = account(g, ex, risk=rk)
                cells = " | ".join(f"{yr[y]:+.0f}%" if y in yr else "-" for y in years)
                L.append(f"| {name} | {ex} | {rk*100:.0f}% | **{fin:.1f}** | {cagr:+.1f}% | {mdd:.1f}% | {tk} | {cells} |")
    L.append("")

    best = df.sort_values("T 목표가", ascending=False).head(10)
    L += ["### 수익 상위 10건 (T 목표가)", "| 진입 | 종목 | 셋업 | 크기 | 수익 | 사유 |", "|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        L.append(f"| {r.date} | {r['name']} | {r.setup} | {r.deg} | {r['T 목표가']:+.0f}% | {r['T 목표가 사유']} |")
    L += ["", "## 주의",
          "- 상장폐지 종목이 빠져 있어서 결과가 실제보다 좋게 나와요.",
          "- 파동 세는 법은 사람마다 달라요. 여기선 지그재그 규칙으로 기계적으로 셌어요.",
          "- 4번 표가 비어 있거나 검증 PF가 1 아래면, 그 셋업은 과거에만 맞은 거예요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "wave_trades.csv", index=False)
    log("보고서 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    allr, done = [], [0]

    def job(s):
        try:
            return stock_trades(s, start)
        except Exception as e:
            log("오류", s["name"], repr(e))
            return [], None
        finally:
            done[0] += 1
            if done[0] % 200 == 0:
                log(f"  {done[0]}/{len(uni)}")

    up_cnt = tot_cnt = None
    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r, a60 in ex.map(job, uni):
            allr += r
            if a60 is not None:
                a = a60.astype(float)
                up_cnt = a if up_cnt is None else up_cnt.add(a, fill_value=0)
                one = pd.Series(1.0, index=a60.index)
                tot_cnt = one if tot_cnt is None else tot_cnt.add(one, fill_value=0)
    df = pd.DataFrame(allr)
    log(f"거래 {len(df)}건")
    if df.empty:
        (scan.STATE / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    dts = pd.to_datetime(df.date)
    if up_cnt is not None:
        b = (up_cnt / tot_cnt * 100).sort_index().shift(1)
        df["breadth"] = b.reindex(dts, method="ffill").round(1).to_numpy()
    else:
        df["breadth"] = np.nan
    import deep_backtest
    kp = deep_backtest.kospi()
    if kp is not None:
        k = kp.sort_index()
        up = (k > k.rolling(20).mean()) & (k.rolling(20).mean() > k.rolling(60).mean()) & (k.rolling(60).mean() > k.rolling(120).mean())
        df["kospi_up"] = up.shift(1).reindex(dts, method="ffill").to_numpy()
    else:
        df["kospi_up"] = None
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
