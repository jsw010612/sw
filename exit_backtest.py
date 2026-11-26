"""
매도 타이밍 분석 — 깊은 바닥 반등 신호를 그대로 두고 "언제 파느냐"만 바꿔서 비교

통합 백테스트에서 보인 문제
  - 바닥A 신호는 바닥 시기(22년 10월, 23년 10~11월, 24년 11~12월)에 한꺼번에 몰림 → 계좌는 10개만 담고 나머진 놓침
  - 손절 59%가 열흘 안에 나고, 수익 종목은 평균 48일 들고 감 → 그동안 돈이 묶여서 다음 바닥에 못 들어감
  - 추적 매도(최고가 -25%)는 고점에서 많이 토해냄

매도 방식 (손절은 전부 5파 저점 종가 이탈, 최대 250거래일)
  M0 지금 방식     : +30% 절반 · 최고가가 +30% 넘은 뒤 최고가 -25%면 나머지
  M1 추적 -15%    : +30% 절반 · 나머지 최고가 -15%
  M2 빠른 절반    : +20% 절반 · 나머지 최고가 -20% (+20% 넘은 뒤)
  M3 3분할        : +20% 1/3 · +40% 1/3 · 나머지 최고가 -20% (+40% 넘은 뒤)
  M4 저항 매도    : 위쪽 주·월 60/120선(매수가 +10% 이상)이나 +50% 중 먼저 닿는 곳에서 절반 · 나머지 최고가 -20%
  M5 시간 손절    : M0 + 20일 지나도 최고가가 +10% 못 넘으면 정리
  M6 본전 이동    : M0 + 최고가가 +15% 넘으면 손절선을 매수가로
  M7 눌림목       : +30% 절반 · +15% 넘은 뒤 눌림목 저점(좌우 3봉)이 생길 때마다 손절선을 그 저점으로
  M8 종합         : +20% 절반 · +15% 넘으면 본전 · 20일/+10% 시간 손절 · 나머지 최고가 -20%

계좌 (매일 종가로 평가 · 복리)
  비중 = min(종목당 최대, 1회 손실 한도 ÷ 손절폭) · 같은 날 신호가 많으면 고르는 순서 비교
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import strategy as S
import deep_backtest as DB
from ma_backtest import htf_ma, log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
BUY_COST, SELL_COST = DB.BUY_COST, DB.SELL_COST
MODES = ["M0 지금 방식", "M1 추적 -15%", "M2 빠른 절반", "M3 3분할", "M4 저항 매도",
         "M5 시간 손절", "M6 본전 이동", "M7 눌림목", "M8 종합"]

_tl = threading.local()
_orig_daily = scan.naver_daily
_cache, _lock = {}, threading.Lock()


def cached_daily(code, count=1500, market="KS"):
    with _lock:
        d = _cache.get(code)
    if d is None:
        d = _orig_daily(code, max(3200, int(250 * (YEARS + 3))), market)
        with _lock:
            _cache[code] = d
    return d.iloc[-count:] if count and len(d) > count else d


def path(O, H, L, C, ei, stop, mode):
    """1원어치의 날짜별 가치 · 청산 사유 · 마지막 위치 (deep_backtest.path 대신 끼워 넣음)"""
    res = getattr(_tl, "res", None)
    entry = O[ei]
    n = len(C)
    last = min(n - 1, ei + S.HOLD_MAX)
    held, cash = 1.0, -BUY_COST
    peak, be, tp1, tp2 = entry, False, False, False
    sl = stop
    vals, reason = [], "기간 만료"

    def sell(frac, px):
        nonlocal held, cash
        cash += frac * px / entry * (1 - SELL_COST); held -= frac

    # 저항선 (매수일 기준 위쪽에 있는 주·월 60/120선)
    rline = None
    if mode == "M4 저항 매도" and res is not None:
        ups = [m[ei] for m in res if np.isfinite(m[ei]) and m[ei] >= entry * 1.10]
        rline = min(ups) if ups else None
        rline = min(rline, entry * 1.5) if rline else entry * 1.5
    piv_i = ei
    for j in range(ei, last + 1):
        peak = max(peak, H[j])
        g = peak / entry - 1
        # 익절
        if mode in ("M0 지금 방식", "M1 추적 -15%", "M5 시간 손절", "M6 본전 이동", "M7 눌림목") and not tp1 and H[j] >= entry * 1.3:
            sell(0.5, max(O[j], entry * 1.3) if j > ei else entry * 1.3); tp1 = True
        if mode in ("M2 빠른 절반", "M8 종합") and not tp1 and H[j] >= entry * 1.2:
            sell(0.5, max(O[j], entry * 1.2) if j > ei else entry * 1.2); tp1 = True
        if mode == "M3 3분할":
            if not tp1 and H[j] >= entry * 1.2:
                sell(1 / 3, max(O[j], entry * 1.2) if j > ei else entry * 1.2); tp1 = True
            if tp1 and not tp2 and H[j] >= entry * 1.4:
                sell(1 / 3, max(O[j], entry * 1.4) if j > ei else entry * 1.4); tp2 = True
        if mode == "M4 저항 매도" and not tp1 and H[j] >= rline:
            sell(0.5, max(O[j], rline) if j > ei else rline); tp1 = True
        # 손절선 올리기
        if mode in ("M6 본전 이동", "M8 종합") and g >= 0.15:
            be = True
        if mode == "M7 눌림목" and g >= 0.15:
            k = j - 3
            if k - 3 > piv_i and L[k] <= L[k - 3:k + 4].min():
                sl = max(sl, L[k] * 0.99); piv_i = k
        # 청산 판단 (종가)
        why = None
        if C[j] < sl:
            why = "눌림목 이탈" if sl > stop else "손절"
        elif be and C[j] < entry:
            why = "본전"
        elif mode in ("M5 시간 손절", "M8 종합") and j - ei >= 20 and peak < entry * 1.10:
            why = "시간 손절"
        else:
            arm, gap = {"M0 지금 방식": (0.3, 0.25), "M5 시간 손절": (0.3, 0.25), "M6 본전 이동": (0.3, 0.25),
                        "M1 추적 -15%": (0.3, 0.15), "M2 빠른 절반": (0.2, 0.2), "M8 종합": (0.2, 0.2),
                        "M3 3분할": (0.4, 0.2), "M4 저항 매도": (None, 0.2), "M7 눌림목": (None, None)}[mode]
            if mode == "M4 저항 매도":
                arm = (rline / entry - 1) if rline else 0.5
            if arm is not None and gap is not None and g >= arm and C[j] < peak * (1 - gap):
                why = "추적 매도"
        if why or j == last:
            sell(held, C[j])
            vals.append(cash)
            return np.array(vals), why or "기간 만료", j
        vals.append(held * C[j] / entry + cash)
    return np.array(vals), reason, last


def stock_sigs(s, start, kp):
    try:
        d = cached_daily(s["code"], 2000, s["market"])
        if len(d) < 400:
            return []
        d = d[~d.index.duplicated()].sort_index()
        _tl.res = [htf_ma(d, f, m) for f, m in (("W", 60), ("W", 120), ("M", 60), ("M", 120))]
        out = DB.stock_signals(s, start, kp)
        H = d["high"].to_numpy(float)
        for t in out:
            ei = d.index.get_loc(t["date"])
            t["mfe"] = (H[ei:ei + S.HOLD_MAX + 1].max() / t["entry"] - 1) * 100       # 250일 안 최고 상승폭
        return [t for t in out if t["grade"] in ("A", "B")]
    finally:
        _tl.res = None
        with _lock:
            _cache.pop(s["code"], None)


# ───────────────────────── 통계 ─────────────────────────
def stat(v):
    v = pd.Series(v, dtype=float).dropna()
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    aw, al = (w.mean() if len(w) else 0.0), (l.mean() if len(l) else 0.0)
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), aw=aw, al=al,
                rr=aw / abs(al) if al else float("inf"),
                pf=w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf"))


def f2(x):
    return "∞" if x == float("inf") else f"{x:.2f}"


# ───────────────────────── 계좌 (매일 평가) ─────────────────────────
RANKS = {"점수순": lambda t: (-t["score"], -t["dd52"]),
         "손절폭 작은 순": lambda t: (t["stop_dist"],),
         "많이 빠진 순": lambda t: (-t["dd52"],),
         "월RSI 낮은 순": lambda t: (t["mrsi"] if t["mrsi"] is not None else 99,)}


def account(sigs, ex, risk=0.02, cap=0.30, max_open=10, total=0.90, daily_new=3, rank="점수순", grades=("A",)):
    tr = [t for t in sigs if t["grade"] in grades]
    by_day = {}
    for t in tr:
        by_day.setdefault(t["date"], []).append(t)
    days = sorted({d for t in tr for d in t[ex]["dates"]})
    if not days:
        return None
    key = RANKS[rank]
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
        new = sorted(by_day.get(d, []), key=lambda t: ((0 if t["grade"] == "A" else 1),) + key(t))
        for t in new:
            if cnt >= daily_new or len(opn) >= max_open or t["code"] in held:
                continue
            w = min(cap, risk / max(t["stop_dist"] / 100, 0.02))
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
    mdd = ((c / c.cummax()) - 1).min() * 100
    yrs = max((c.index[-1] - c.index[0]).days / 365, 0.5)
    final = c.iloc[-1]
    ye = c.groupby(c.index.year).last()
    yr = (ye / ye.shift(1).fillna(100) - 1) * 100
    s = stat([t[ex]["ret"] for t in taken])
    # 돈이 놀고 있던 비율
    return dict(final=final, cagr=((final / 100) ** (1 / yrs) - 1) * 100, mdd=mdd, yr=yr.to_dict(), s=s, n=len(taken))


def acct_row(lab, a, years):
    s = a["s"] or dict(win=0, rr=0, pf=0)
    cells = " | ".join(f"{a['yr'][y]:+.0f}%" if y in a["yr"] else "-" for y in years)
    return (f"| {lab} | **{a['final']:.1f}** | {a['cagr']:+.1f}% | {a['mdd']:.1f}% | {a['n']} | {s['win']:.0f}% | "
            f"{f2(s['rr'])} | {f2(s['pf'])} | {cells} |")


def report(sigs, n, t0):
    A = [t for t in sigs if t["grade"] == "A"]
    years = sorted({t["date"].year for t in sigs})
    L = ["# 매도 타이밍 분석 — 깊은 바닥 반등", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 · A등급 {len(A)}건 · B등급 {len(sigs) - len(A)}건",
         "- 손익비 = 평균 이익 ÷ 평균 손실 · PF = 이익 합 ÷ 손실 합 · 하루당 = 수익 합 ÷ 보유일 합 (돈 회전 효율)", ""]

    # 1. 거래 단위
    L += ["## 1. 매도 방식별 (A등급 거래 하나하나)", "",
          "| 매도 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF | 평균 보유일 | 하루당 | 큰 상승 잡은 비율 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in MODES:
        r = np.array([t[m]["ret"] for t in A])
        dd = np.array([len(t[m]["vals"]) for t in A])
        s = stat(r)
        big = [t for t in A if t["mfe"] >= 50]
        cap_ = np.mean([t[m]["ret"] / t["mfe"] for t in big]) * 100 if big else 0
        L.append(f"| {m} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | "
                 f"{f2(s['rr'])} | {f2(s['pf'])} | {dd.mean():.0f} | {r.sum() / dd.sum():.3f}% | {cap_:.0f}% |")
    big = [t for t in A if t["mfe"] >= 50]
    L += ["", f"- 큰 상승 잡은 비율: 250일 안에 +50% 이상 갔던 {len(big)}건에서, 최고 상승폭 중 실제로 먹은 비율", ""]

    # 2. 연도별
    L += ["## 2. 매도 방식별 연도별 PF (A등급)", "", "| 매도 | " + " | ".join(map(str, years)) + " |",
          "|---|" + "---|" * len(years)]
    for m in MODES:
        cells = []
        for y in years:
            s = stat([t[m]["ret"] for t in A if t["date"].year == y])
            cells.append("-" if not s or s["n"] < 5 else f"{f2(s['pf'])} ({s['n']})")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. 청산 사유
    L += ["## 3. 청산 사유 (A등급)", "", "| 매도 | 사유별 건수 · 평균 |", "|---|---|"]
    for m in MODES:
        dfm = pd.DataFrame([(t[m]["reason"], t[m]["ret"]) for t in A], columns=["r", "v"])
        g = dfm.groupby("r")["v"].agg(["count", "mean"])
        L.append(f"| {m} | " + " · ".join(f"{k} {int(v['count'])}건 {v['mean']:+.1f}%" for k, v in g.iterrows()) + " |")
    L.append("")

    head = ("| 구분 | 최종 | 연평균 | 최대 낙폭 | 거래 | 승률 | 손익비 | PF | " + " | ".join(map(str, years)) + " |\n"
            "|---|---|---|---|---|---|---|---|" + "---|" * len(years))
    # 4. 계좌: 매도별 (기본 설정)
    L += ["## 4. 계좌: 매도 방식별 (A등급 · 1회 손실 2% · 종목당 30% · 보유 10 · 하루 신규 3 · 점수순)", "", head]
    res = {}
    for m in MODES:
        a = account(sigs, m)
        res[m] = a
        L.append(acct_row(m, a, years))
    L.append("")
    top = sorted(MODES, key=lambda m: -(res[m]["cagr"] + res[m]["mdd"] * 0.5))[:3]

    # 5. 계좌: 상위 3개 매도 × 고르는 순서 × 보유 수 × 손실 한도
    L += [f"## 5. 계좌: 상위 매도 {', '.join(top)} × 설정", "", head.replace("| 구분 |", "| 매도 · 순서 · 보유 · 손실 한도 · 등급 |")]
    grid = []
    for m in top:
        for rank in RANKS:
            for mo in (10, 15):
                for rk, cp in ((0.02, 0.30), (0.03, 0.35)):
                    for gr in (("A",), ("A", "B")):
                        a = account(sigs, m, risk=rk, cap=cp, max_open=mo, rank=rank, grades=gr,
                                    daily_new=5 if mo == 15 else 3)
                        grid.append(((m, rank, mo, rk, "+".join(gr)), a))
    grid.sort(key=lambda x: -(x[1]["cagr"] + x[1]["mdd"] * 0.5))
    for (m, rank, mo, rk, gr), a in grid[:20]:
        L.append(acct_row(f"{m} · {rank} · {mo}개 · {rk*100:.0f}% · {gr}", a, years))
    L.append("")
    # 안정성: 같은 설정에서 고르는 순서만 바꿔도 결과가 크게 흔들리는지
    L += ["## 6. 결과가 운에 얼마나 좌우되나 (같은 매도·보유·한도에서 고르는 순서만 바꿨을 때 연평균 범위)", "",
          "| 매도 · 보유 · 손실 한도 · 등급 | 최저 | 최고 | 차이 |", "|---|---|---|---|"]
    by = {}
    for (m, rank, mo, rk, gr), a in grid:
        by.setdefault((m, mo, rk, gr), []).append(a["cagr"])
    for k, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1]))[:10]:
        L.append(f"| {k[0]} · {k[1]}개 · {k[2]*100:.0f}% · {k[3]} | {min(v):+.1f}% | {max(v):+.1f}% | {max(v) - min(v):.1f} |")
    L.append("")
    (m, rank, mo, rk, gr), a = grid[0]
    L += ["## 7. 추천", "",
          f"- **{m} · {rank} · 동시 보유 {mo}개 · 1회 손실 {rk*100:.0f}% · {gr}등급**",
          f"- 원금 100 → {a['final']:.1f} · 연평균 {a['cagr']:+.1f}% · 최대 낙폭 {a['mdd']:.1f}% (매일 평가 기준)",
          "- 연도별: " + " · ".join(f"{y} {v:+.0f}%" for y, v in a["yr"].items()), "",
          "## 주의", "- 상장폐지 종목이 빠져 있어서 실제보다 좋게 나와요.",
          "- 6번 표의 차이가 크면, 그 설정의 수익은 '어떤 종목을 골랐냐'는 운이 많이 섞인 거예요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    log("보고서 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    scan.naver_daily = cached_daily
    DB.path = path
    DB.EXITS = MODES
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
    if not sigs:
        (scan.STATE / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    report(sigs, len(uni), t0)


if __name__ == "__main__":
    main()
