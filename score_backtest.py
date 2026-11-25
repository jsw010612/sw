"""
점수 다시 만들기 — 지금까지 데이터로 "어떤 바닥 신호가 실제로 더 잘 올랐나"를 배워서 점수표를 만듦

1) 신호 모으기 (조건을 조금 넓혀서 비교 대상을 늘림)
   1일/주봉 RSI 다이버전스 · 52주 고점 대비 -30% 이상 · 5파 깊이 1.0배 이상 · 매수가 > 5파 저점
2) 특징마다 구간을 미리 정해 두고 (사람이 읽기 쉬운 경계값), 2021~2023년 신호만으로
   구간별 평균 수익(3분할 매도, +100% 넘는 건 100%로 잘라서 대박 몇 개에 휘둘리지 않게)을 계산
3) 차이가 큰 특징 6개까지 골라 구간마다 점수를 줌 → 합쳐서 0~100점
4) 2021~2023년 점수 분포로 강(상위 30%) · 보통(다음 30%) · 약(나머지) 경계를 정함
5) 2024~2026년(점수 만들 때 안 본 기간)에서 강 > 보통 > 약 순서가 유지되는지 확인
6) 계좌: 예전 A등급 균등 vs 새 점수(강·보통만) vs 차등 비중
7) 결과를 state/score_rules.json 으로 저장 → 스캐너와 앱이 다음 실행부터 이 점수표를 씀
"""
import json
import os
import zlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import strategy as S
import deep_backtest as DB
import exit_backtest as EB
from analysis import Params, analyze_frame, rsi, to_month, to_week
from ma_backtest import htf_ma, log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
TRAIN_END = 2023
EXITS = ["M3 3분할", "M4 저항 매도", "M0 지금 방식"]
TARGET = "M3 3분할"
P = replace(Params(), piv_r=2)

# 특징: (이름, 설명, 경계값). 구간은 [경계i, 경계i+1)
FEATS = [
    ("dd52", "52주 고점 대비 하락 %", [30, 45, 55, 65, 75, 1000]),
    ("depth", "5파 깊이 (4파 반등폭 대비)", [1.0, 1.27, 1.618, 2.0, 1000]),
    ("l3vr", "3파 저점 거래량 (평소 대비 배)", [0, 2, 4, 8, 1e9]),
    ("l5vr", "5파 저점 거래량 (평소 대비 배)", [0, 0.7, 1.0, 1.5, 1e9]),
    ("rgap", "다이버전스 RSI 차이 (5파-3파)", [-100, 3, 6, 10, 100]),
    ("mrsi", "월봉 RSI", [0, 25, 30, 35, 45, 101]),
    ("wrsi", "주봉 RSI", [0, 25, 30, 35, 45, 101]),
    ("kgap", "코스피 120일선 대비 %", [-100, -10, -5, 0, 5, 1000]),
    ("stop_dist", "손절폭 %", [0, 5, 8, 12, 20, 1000]),
    ("price", "주가 (원)", [0, 1000, 3000, 10000, 1e12]),
    ("weekly", "주봉 다이버전스", [0, 0.5, 2]),
]
MAX_FEATS = 6
MIN_N = 40
SHRINK = 100

_lock = threading.Lock()


def features_at(d1, det_i, snap, weekly, kgap, entry):
    C = d1["close"].to_numpy(float)
    H = d1["high"].to_numpy(float)
    hi = H[max(0, det_i - 249):det_i + 1].max()
    sub = d1.iloc[:det_i + 1]
    mth, wk = to_month(sub), to_week(sub)
    mr = rsi(mth["close"].to_numpy(float), 14)[-1] if len(mth) > 16 else np.nan
    wr = rsi(wk["close"].to_numpy(float), 14)[-1] if len(wk) > 16 else np.nan
    l5 = float(snap["l5p"])
    return dict(dd52=(1 - C[det_i] / hi) * 100 if hi > 0 else np.nan, depth=float(snap["depth"]),
                l3vr=float(snap.get("l3vr") or 0), l5vr=float(snap.get("l5vr") or 0),
                rgap=float(snap["l5r"]) - float(snap["l3r"]), mrsi=float(mr), wrsi=float(wr), kgap=kgap,
                stop_dist=(1 - l5 / entry) * 100, price=entry, weekly=1.0 if weekly else 0.0)


def stock_sigs(s, start, kp):
    try:
        d1 = EB.cached_daily(s["code"], 2000, s["market"])
        if len(d1) < 400:
            return []
        d1 = d1[~d1.index.duplicated()].sort_index()
        n = len(d1)
        O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
        EB._tl.res = [htf_ma(d1, f, m) for f, m in (("W", 60), ("W", 120), ("M", 60), ("M", 120))]
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
            f = features_at(d1, det_i, snap, wb, kgap, entry)
            if not np.isfinite(f["dd52"]) or f["dd52"] < 30 or f["stop_dist"] > 40:
                continue
            # 예전 점수 (비교용)
            old = S.score(kgap < 0, f["mrsi"], wb, f["l3vr"])
            old_a = S.qualifies(f["dd52"], f["depth"], entry, l5) and old >= 85
            rec = dict(code=s["code"], name=s["name"], date=d1.index[ei], old_a=old_a, old_score=old, **f)
            for ex in EXITS:
                v, reason, xj = EB.path(O, H, L, C, ei, l5, ex)
                rec[ex] = dict(dates=d1.index[ei:ei + len(v)], vals=v, reason=reason, ret=(v[-1] - 1) * 100)
            out.append(rec)
        return out
    finally:
        EB._tl.res = None
        with EB._lock:
            EB._cache.pop(s["code"], None)


# ───────────────────────── 점수표 학습 ─────────────────────────
def bucket(v, edges):
    if v is None or not np.isfinite(v):
        return None
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return None


def blabel(name, edges, i):
    a, b = edges[i], edges[i + 1]
    if name == "weekly":
        return "주봉" if i == 1 else "1일봉"
    fa = lambda x: f"{x:g}"
    if b >= 999:
        return f"{fa(a)} 이상"
    if a <= -99 or (a == 0 and name not in ("rgap",)):
        return f"{fa(b)} 미만"
    return f"{fa(a)}~{fa(b)}"


def _effects(sub, name, edges, base):
    b = sub[name].apply(lambda v: bucket(v, edges))
    eff = {}
    for i in range(len(edges) - 1):
        m = b == i
        k = int(m.sum())
        if k >= MIN_N // 2:
            eff[i] = ((sub.y[m].mean() - base) * k / (k + SHRINK), k)      # 건수가 적으면 0 쪽으로 줄임
    return eff


def learn(df):
    """배운 기간 안에서도 종목을 반으로 나눠 두 쪽 모두 같은 방향인 특징만 씀 (우연 걸러내기)"""
    tr = df[df.year <= TRAIN_END].copy()
    base = tr.y.mean()
    half = tr.code.apply(lambda c: zlib.crc32(str(c).encode()) % 2)      # 종목코드 끝자리는 대부분 0이라 해시로 나눔
    cand = []
    for name, desc, edges in FEATS:
        eff = {i: e for i, e in _effects(tr, name, edges, base).items() if e[1] >= MIN_N}
        if len(eff) < 2:
            continue
        a = _effects(tr[half == 0], name, edges, tr[half == 0].y.mean())
        b = _effects(tr[half == 1], name, edges, tr[half == 1].y.mean())
        common = [i for i in eff if i in a and i in b]
        agree = (np.corrcoef([a[i][0] for i in common], [b[i][0] for i in common])[0, 1]
                 if len(common) >= 3 else (1.0 if len(common) == 2 and np.sign(a[common[0]][0] - a[common[1]][0]) ==
                                           np.sign(b[common[0]][0] - b[common[1]][0]) else 0.0))
        vals = np.array([e for e, _ in eff.values()]); ns = np.array([c for _, c in eff.values()])
        imp = float(np.sqrt(np.average((vals - np.average(vals, weights=ns)) ** 2, weights=ns)))
        cand.append((imp, name, desc, edges, eff, float(np.nan_to_num(agree))))
    cand.sort(key=lambda c: -c[0])
    chosen = [c for c in cand if c[0] >= 0.8 and c[5] >= 0.3][:MAX_FEATS]
    rules = []
    for imp, name, desc, edges, eff, ag in chosen:
        pts = [round(eff[i][0], 2) if i in eff else 0.0 for i in range(len(edges) - 1)]
        rules.append(dict(name=name, label=desc, edges=edges, raw=pts, imp=round(imp, 2),
                          bins=[blabel(name, edges, i) for i in range(len(edges) - 1)]))
    raw = tr.apply(lambda r: raw_score(r, rules), axis=1)
    lo, hi = float(np.percentile(raw, 2)), float(np.percentile(raw, 98))
    span = hi - lo if hi > lo else 1.0
    for r in rules:
        r["pts"] = [int(round(p / span * 100)) for p in r["raw"]]
    offset = int(round(-lo / span * 100))
    tmp = dict(features=rules, offset=offset)
    sc = np.array([score_of(r, tmp) for _, r in tr.iterrows()])
    uniq = sorted(set(sc), reverse=True)
    strong = next((u for u in uniq if (sc >= u).mean() >= 0.25), uniq[-1])
    mid = next((u for u in uniq if (sc >= u).mean() >= 0.55), uniq[-1])
    if mid >= strong:
        mid = next((u for u in uniq if u < strong), strong)
    return dict(features=rules, offset=offset, strong=float(strong), mid=float(mid),
                all_imp=[(c[1], c[2], round(c[0], 2), round(c[5], 2)) for c in cand])


def raw_score(r, rules):
    t = 0.0
    for f in rules:
        i = bucket(r[f["name"]], f["edges"])
        if i is not None:
            t += f["raw"][i]
    return t


def score_of(r, R):
    t = R["offset"]
    for f in R["features"]:
        i = bucket(r[f["name"]], f["edges"])
        if i is not None:
            t += f["pts"][i]
    return float(np.clip(t, 0, 100))


def tier_of(sc, R):
    return "강" if sc >= R["strong"] else "보통" if sc >= R["mid"] else "약"


# ───────────────────────── 통계·계좌 ─────────────────────────
def stat(v):
    v = pd.Series(v, dtype=float).dropna()
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    aw, al = (w.mean() if len(w) else 0.0), (l.mean() if len(l) else 0.0)
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), aw=aw, al=al,
                rr=aw / abs(al) if al else float("inf"),
                pf=w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf"))


f2 = EB.f2
HEAD = "| 구분 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF |\n|---|---|---|---|---|---|---|---|"


def row(lab, v):
    s = stat(v)
    if not s:
        return f"| {lab} | 0 | - | - | - | - | - | - |"
    return (f"| {lab} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | "
            f"{f2(s['rr'])} | {f2(s['pf'])} |")


def account(sigs, ex, risk_of, cap=0.35, max_open=10, total=0.90, daily_new=3):
    tr = [t for t in sigs if risk_of(t) > 0]
    by_day = {}
    for t in tr:
        by_day.setdefault(t["date"], []).append(t)
    days = sorted({d for t in tr for d in t[ex]["dates"]})
    if not days:
        return None
    cash, opn, curve, taken = 100.0, [], [], []
    for d in days:
        still = []
        for p in opn:
            idx = p["pos"].get(d)
            if idx is not None:
                p["val"] = p["amt"] * p["t"][ex]["vals"][idx]
                if idx == len(p["t"][ex]["vals"]) - 1:
                    cash += p["val"]
                    continue
            still.append(p)
        opn = still
        eq = cash + sum(p["val"] for p in opn)
        held = {p["t"]["code"] for p in opn}
        cnt = 0
        for t in sorted(by_day.get(d, []), key=lambda t: -t["score"]):
            if cnt >= daily_new or len(opn) >= max_open or t["code"] in held:
                continue
            w = min(cap, risk_of(t) / max(t["stop_dist"] / 100, 0.02))
            amt = eq * w
            if (sum(p["val"] for p in opn) + amt) / eq > total or amt > cash or amt <= 0:
                continue
            cash -= amt
            v = t[ex]["vals"]
            p = dict(t=t, amt=amt, val=amt * v[0], pos={dd: k for k, dd in enumerate(t[ex]["dates"])})
            if len(v) == 1:
                cash += p["val"]
            else:
                opn.append(p)
            held.add(t["code"]); taken.append(t); cnt += 1
        curve.append((d, cash + sum(p["val"] for p in opn)))
    c = pd.Series(dict(curve))
    yrs = max((c.index[-1] - c.index[0]).days / 365, 0.5)
    ye = c.groupby(c.index.year).last()
    yr = (ye / ye.shift(1).fillna(100) - 1) * 100
    return dict(final=c.iloc[-1], cagr=((c.iloc[-1] / 100) ** (1 / yrs) - 1) * 100,
                mdd=((c / c.cummax()) - 1).min() * 100, yr=yr.to_dict(), s=stat([t[ex]["ret"] for t in taken]),
                n=len(taken))


# ───────────────────────── 보고서 ─────────────────────────
def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in t.items() if k not in EXITS} for t in sigs])
    for ex in EXITS:
        df[ex] = [t[ex]["ret"] for t in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    df["y"] = df[TARGET].clip(upper=100)
    R = learn(df)
    df["score"] = df.apply(lambda r: score_of(r, R), axis=1)
    df["tier"] = df.score.apply(lambda s: tier_of(s, R))
    for t, sc in zip(sigs, df.score):
        t["score"] = sc
    for t, tr_ in zip(sigs, df.tier):
        t["tier"] = tr_
    years = sorted(df.year.unique())
    te = df[df.year > TRAIN_END]
    trn = df[df.year <= TRAIN_END]

    L = ["# 점수 다시 만들기 — 데이터로 배운 바닥 반등 점수", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 · 신호 {len(df)}건 "
         f"(배운 기간 ~{TRAIN_END}: {len(trn)}건 · 검증 기간 {TRAIN_END + 1}~: {len(te)}건)",
         "- 신호 범위: 1일/주봉 다이버전스 · 52주 -30% 이상 · 5파 깊이 1.0↑ (예전 A등급보다 넓게)",
         f"- 기준 수익: {TARGET} 매도, +100% 넘는 건 100%로 잘라서 계산", ""]

    L += ["## 1. 특징별 영향 (배운 기간) — 숫자가 클수록 수익 차이를 많이 만든 특징", "",
          "영향 = 구간별 수익 차이(%p) · 일관성 = 종목을 반으로 나눴을 때 두 쪽이 같은 방향인 정도(1에 가까울수록 진짜)", "",
          "| 특징 | 영향 | 일관성 | 점수에 씀 |", "|---|---|---|---|"]
    used = {f["name"] for f in R["features"]}
    for name, desc, imp, ag in R["all_imp"]:
        L.append(f"| {desc} | {imp:.2f} | {ag:.2f} | {'✓' if name in used else ''} |")
    L.append("")

    L += ["## 2. 새 점수표", "", f"기본 {R['offset']:.0f}점에서 시작해 아래 점수를 더함 (0~100으로 자름)", ""]
    for f in R["features"]:
        L.append(f"**{f['label']}** — " + " · ".join(f"{b} {p:+.0f}" for b, p in zip(f["bins"], f["pts"])))
    L += ["", f"- 강: {R['strong']:.0f}점 이상 · 보통: {R['mid']:.0f}점 이상 · 약: 그 아래", ""]

    # 구간별 실제 성과 (배운 기간 vs 검증 기간)
    L += ["## 3. 특징 구간별 실제 평균 수익 (배운 기간 → 검증 기간)", "",
          "같은 방향이면 진짜 효과, 검증 기간에 뒤집히면 우연", ""]
    for f in R["features"]:
        cells = []
        for i, b in enumerate(f["bins"]):
            m1 = trn[f["name"]].apply(lambda v: bucket(v, f["edges"])) == i
            m2 = te[f["name"]].apply(lambda v: bucket(v, f["edges"])) == i
            a1 = trn.y[m1].mean() if m1.sum() >= 10 else np.nan
            a2 = te.y[m2].mean() if m2.sum() >= 10 else np.nan
            cells.append(f"{b}: {a1:+.1f}% → {a2:+.1f}% ({int(m2.sum())})" if np.isfinite(a1) or np.isfinite(a2) else f"{b}: -")
        L.append(f"- **{f['label']}** — " + " · ".join(cells))
    L.append("")

    # 4. 등급별
    L += ["## 4. 새 등급별 성과 (3분할 매도)", "", "### 배운 기간", HEAD]
    L += [row(t, trn[trn.tier == t][TARGET]) for t in ("강", "보통", "약")]
    L += [row("예전 A등급", trn[trn.old_a][TARGET]), "", "### 검증 기간 (점수 만들 때 안 본 기간) ← 이게 진짜 성적", HEAD]
    L += [row(t, te[te.tier == t][TARGET]) for t in ("강", "보통", "약")]
    L += [row("예전 A등급", te[te.old_a][TARGET]), ""]
    L += ["### 연도별 PF", "", "| 등급 | " + " | ".join(map(str, years)) + " |", "|---|" + "---|" * len(years)]
    for t in ("강", "보통", "약", "예전 A"):
        g = df[df.old_a] if t == "예전 A" else df[df.tier == t]
        cells = []
        for y in years:
            s = stat(g[g.year == y][TARGET])
            cells.append("-" if not s or s["n"] < 5 else f"{f2(s['pf'])} ({s['n']})")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L.append("")

    # 사는 등급 결정: 검증 기간 PF 1.3 이상 & 30건 이상
    buy = []
    for t in ("강", "보통"):
        s = stat(te[te.tier == t][TARGET])
        s0 = stat(trn[trn.tier == t][TARGET])
        if s and s0 and s["n"] >= 30 and s["pf"] >= 1.3 and s0["pf"] >= 1.3:
            buy.append(t)
    if not buy:
        buy = ["강"]
    sb = stat(te[te.tier.isin(buy)][TARGET])
    so = stat(te[te.old_a][TARGET])
    active = bool(sb and sb["n"] >= 30 and sb["pf"] >= 1.3 and (not so or sb["pf"] >= so["pf"] * 0.9))

    # 5. 계좌
    old = lambda t: 0.03 if t["old_a"] else 0
    plans = {
        "예전 A등급 · 균등 3%": old,
        "새 점수 강만 · 3%": lambda t: 0.03 if t["tier"] == "강" else 0,
        "새 점수 강·보통 · 균등 3%": lambda t: 0.03 if t["tier"] in ("강", "보통") else 0,
        "새 점수 차등 · 강 4% · 보통 2%": lambda t: {"강": 0.04, "보통": 0.02}.get(t["tier"], 0),
        "새 점수 차등 · 강 4% · 보통 2.5% · 약 1%": lambda t: {"강": 0.04, "보통": 0.025, "약": 0.01}[t["tier"]],
    }
    head = ("| 방식 | 매도 | 최종 | 연평균 | 최대 낙폭 | 거래 | 승률 | 손익비 | PF | " + " | ".join(map(str, years)) + " |\n"
            "|---|---|---|---|---|---|---|---|---|" + "---|" * len(years))
    L += ["## 5. 계좌 비교 (복리 · 매일 평가 · 보유 10 · 하루 신규 3 · 종목당 최대 35% · 같은 날은 점수 높은 순)", "", head]
    best = None
    for name, fn in plans.items():
        for ex in ("M3 3분할", "M4 저항 매도"):
            a = account(sigs, ex, fn)
            if not a:
                continue
            s = a["s"] or dict(win=0, rr=0, pf=0)
            cells = " | ".join(f"{a['yr'][y]:+.0f}%" if y in a["yr"] else "-" for y in years)
            L.append(f"| {name} | {ex} | **{a['final']:.1f}** | {a['cagr']:+.1f}% | {a['mdd']:.1f}% | {a['n']} | "
                     f"{s['win']:.0f}% | {f2(s['rr'])} | {f2(s['pf'])} | {cells} |")
            if name.startswith("새") and (best is None or a["cagr"] + a["mdd"] * 0.5 > best[0]):
                best = (a["cagr"] + a["mdd"] * 0.5, name, ex, a)
    L.append("")
    # 검증 기간만 따로 (2024~)
    sub = [t for t in sigs if t["date"].year > TRAIN_END]
    L += [f"### 검증 기간({TRAIN_END + 1}~)만 따로 돌린 계좌", "",
          "| 방식 | 매도 | 최종 | 연평균 | 최대 낙폭 | 거래 |", "|---|---|---|---|---|---|"]
    for name, fn in plans.items():
        for ex in ("M3 3분할", "M4 저항 매도"):
            a = account(sub, ex, fn)
            if a:
                L.append(f"| {name} | {ex} | **{a['final']:.1f}** | {a['cagr']:+.1f}% | {a['mdd']:.1f}% | {a['n']} |")
    L.append("")

    risk = {"강": 0.04, "보통": 0.02 if "보통" in buy else 0.0, "약": 0.0}
    ex_pick = best[2] if best else "M3 3분할"
    if best:
        L += ["## 6. 추천", "", f"- **{best[1]} · 매도 {best[2]}**",
              f"- 원금 100 → {best[3]['final']:.1f} · 연평균 {best[3]['cagr']:+.1f}% · 최대 낙폭 {best[3]['mdd']:.1f}%",
              f"- 매수 등급: {' · '.join(buy)} (검증 기간 PF 1.3 이상인 등급만)", ""]

    # 저장
    rules = dict(version=2, made=datetime.now().strftime("%Y-%m-%d"), train=f"~{TRAIN_END}",
                 offset=R["offset"], strong=R["strong"], mid=R["mid"], features=[
                     {k: f[k] for k in ("name", "label", "edges", "pts", "bins")} for f in R["features"]],
                 buy=buy, risk=risk, cap=0.35, exit=ex_pick, active=active,
                 min=dict(dd52=30, depth=1.0))
    (scan.STATE / "score_rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    L += [("- 점수표를 `state/score_rules.json` 으로 저장했어요. **검증을 통과해서** 다음 스캔부터 앱·알림에 새 점수가 쓰여요."
           if active else "- 점수표는 저장했지만 **검증 기간 성적이 예전 A등급보다 못해서 쓰지 않아요** (스캐너는 예전 점수 그대로)."), "",
          "## 주의", "- 상장폐지 종목이 빠져 있어서 실제보다 좋게 나와요.",
          "- 4번의 검증 기간 표에서 강 > 보통 > 약 순서가 안 지켜지면 점수표를 믿으면 안 돼요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    keep = ["code", "name", "date", "year", "old_a", "old_score", "score", "tier"] + [f[0] for f in FEATS] + EXITS
    out = df[keep].copy()
    for ex in EXITS:
        out[ex + " 일수"] = [len(t[ex]["vals"]) for t in sigs]
    out.to_csv(scan.STATE / "score_trades.csv", index=False)
    log("보고서·점수표 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    scan.naver_daily = EB.cached_daily
    uni = scan.fetch_universe()[:TOP_N]
    kp = DB.kospi()
    log(f"코스피: {'있음' if kp is not None else '없음'}")
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목")
    sigs, done = [], [0]

    def job(s):
        try:
            return stock_sigs(s, start, kp)
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
    if len(sigs) < 200:
        (scan.STATE / "backtest.md").write_text(f"# 신호가 너무 적어요 ({len(sigs)}건)", encoding="utf-8")
        return
    report(sigs, len(uni), t0)


if __name__ == "__main__":
    main()
