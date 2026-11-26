"""
깊은 바닥 반등 전략 · 계좌 단위 백테스트

strategy.py 의 규칙 그대로 신호를 찾고, 실제 계좌처럼 굴려 봄
  - 하루 신규 3종목(점수 높은 순) · 동시 보유 8종목 · 총 보유 70% · 같은 종목 중복 매수 안 함
  - 매일 종가로 평가해서 최대 낙폭 계산
  - 비용: 살 때 0.05%, 팔 때 0.25% (수수료 + 거래세)

청산 방식 비교
  E1 추적       : 5파 저점 종가 이탈 손절 · +30% 넘으면 최고가 -25% 추적
  E2 절반+추적  : E1 + 처음 +30% 닿을 때 절반 익절
  E3 본전+추적  : E1 + 최고가가 +15% 넘으면 손절선을 매수가로 올림

등급 범위 비교: 전체(A·B·C) / A·B / A만
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import scan
import strategy as S
from analysis import Params, analyze_frame, rsi, to_month, to_week

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "1000") or 1000)
BUY_COST, SELL_COST = 0.0005, 0.0025
P = replace(Params(), piv_r=2)
OUT = scan.STATE
EXITS = ["E1 추적", "E2 절반+추적", "E3 본전+추적"]


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def kospi():
    try:
        r = requests.get("https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count=4000&requestType=0",
                         headers=scan.UA, timeout=20)
        rows = [(pd.Timestamp(it.split("|")[0]), float(it.split("|")[4]))
                for it in re.findall(r'data="([^"]+)"', r.content.decode("euc-kr", "ignore"))]
        if rows:
            return pd.Series(dict(rows)).sort_index()
    except Exception:
        pass
    try:
        import yfinance as yf
        d = yf.download("^KS11", period="max", interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s
    except Exception:
        return None


def path(O, H, L, C, ei, stop, mode):
    """1원어치 샀을 때 날짜별 가치 (청산 후엔 끝). 반환: 가치 배열, 청산 사유"""
    entry = O[ei]
    n = len(C)
    last = min(n - 1, ei + S.HOLD_MAX)
    held, cash = 1.0, -BUY_COST
    peak, half_done, be = entry, False, False
    vals, reason = [], "기간 만료"
    for j in range(ei, last + 1):
        peak = max(peak, H[j])
        if mode == "E2 절반+추적" and not half_done and H[j] >= entry * (1 + S.TRAIL_ARM):
            px = max(O[j], entry * (1 + S.TRAIL_ARM)) if j > ei else entry * (1 + S.TRAIL_ARM)
            cash += 0.5 * px / entry * (1 - SELL_COST)
            held, half_done = 0.5, True
        if mode == "E3 본전+추적" and peak >= entry * 1.15:
            be = True
        exit_now = False
        if C[j] < stop:
            exit_now, reason = True, "손절 5파 이탈"
        elif be and C[j] < entry:
            exit_now, reason = True, "본전 손절"
        elif S.trail_exit(peak, entry, C[j]):
            exit_now, reason = True, "추적 손절"
        if exit_now or j == last:
            cash += held * C[j] / entry * (1 - SELL_COST)
            vals.append(cash)
            return np.array(vals), reason, j
        vals.append(held * C[j] / entry + cash)
    return np.array(vals), reason, last


def stock_signals(s, start, kp):
    d1 = scan.naver_daily(s["code"], 2000, s["market"])
    if len(d1) < 400:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    hi250 = pd.Series(H).rolling(250, min_periods=60).max().to_numpy()
    weak = None
    if kp is not None:
        k = kp.reindex(d1.index, method="ffill")
        weak = (k < k.rolling(120).mean()).to_numpy()
    wk = to_week(d1)
    cands = []
    dev = []
    analyze_frame(d1, 1, P, dev)
    for i, snap in dev:
        cands.append((i, snap, False))
    wev = []
    analyze_frame(wk, 1, P, wev)
    for j, snap in wev:
        k = d1.index.searchsorted(wk.index[j] + pd.Timedelta(days=6), side="right") - 1
        if k >= 0:
            cands.append((int(k), snap, True))
    out = []
    for det_i, snap, wbase in cands:
        ei = det_i + 1
        if ei >= n or d1.index[ei] < start:
            continue
        entry, l5 = O[ei], float(snap["l5p"])
        h = hi250[det_i]
        dd = (1 - C[det_i] / h) * 100 if h and not np.isnan(h) else np.nan
        if not S.qualifies(dd, float(snap["depth"]), entry, l5):
            continue
        mth = to_month(d1.iloc[:det_i + 1])
        mr = rsi(mth["close"].to_numpy(float), P.rsi_len)[-1] if len(mth) > P.rsi_len + 1 else np.nan
        mw = bool(weak[det_i]) if weak is not None else False
        sc = S.score(mw, mr, wbase, float(snap["l3vr"]))
        g = S.grade(sc)
        if not g:
            continue
        rec = dict(code=s["code"], name=s["name"], date=d1.index[ei], score=sc, grade=g,
                   weekly=wbase, market_weak=mw, mrsi=None if np.isnan(mr) else round(float(mr), 1),
                   dd52=round(dd, 1), entry=entry, stop=l5, stop_dist=round((1 - l5 / entry) * 100, 2),
                   w=S.weight(g, entry, l5))
        for ex in EXITS:
            v, reason, xj = path(O, H, L, C, ei, l5, ex)
            rec[ex] = dict(dates=d1.index[ei:ei + len(v)], vals=v, reason=reason, ret=(v[-1] - 1) * 100)
        out.append(rec)
    return out


def portfolio(sigs, ex, grades):
    tr = [t for t in sigs if t["grade"] in grades]
    by_day = {}
    for t in tr:
        by_day.setdefault(t["date"], []).append(t)
    days = sorted({d for t in tr for d in t[ex]["dates"]})
    if not days:
        return None
    cash, opn, eq_curve, taken, skipped = 100.0, [], [], [], 0
    for d in days:
        # 평가 + 청산
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
        equity = cash + sum(p["val"] for p in opn)
        # 신규 진입
        new = sorted(by_day.get(d, []), key=lambda t: -t["score"])
        cnt = 0
        held = {p["t"]["code"] for p in opn}
        for t in new:
            if cnt >= S.DAILY_NEW or len(opn) >= S.MAX_OPEN or t["code"] in held:
                skipped += 1
                continue
            amt = equity * t["w"]
            if (sum(p["val"] for p in opn) + amt) / equity > S.MAX_EXPOSURE or amt > cash or amt <= 0:
                skipped += 1
                continue
            cash -= amt
            pos = {dd: k for k, dd in enumerate(t[ex]["dates"])}
            v0 = t[ex]["vals"][0]
            p = dict(t=t, amt=amt, val=amt * v0, pos=pos)
            if len(t[ex]["vals"]) == 1:
                cash += p["val"]
            else:
                opn.append(p)
            held.add(t["code"])
            taken.append(t)
            cnt += 1
        eq_curve.append((d, cash + sum(p["val"] for p in opn)))
    curve = pd.Series(dict(eq_curve))
    peak = curve.cummax()
    mdd = ((curve / peak) - 1).min() * 100
    final = curve.iloc[-1]
    years = max((curve.index[-1] - curve.index[0]).days / 365, 0.5)
    cagr = ((final / 100) ** (1 / years) - 1) * 100
    rets = np.array([t[ex]["ret"] for t in taken]) if taken else np.array([0.0])
    w, l = rets[rets > 0], rets[rets <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    yearly = curve.groupby(curve.index.year).last()
    yr = (yearly / yearly.shift(1).fillna(100) - 1) * 100
    return dict(final=final, cagr=cagr, mdd=mdd, n=len(taken), skipped=skipped, win=len(w) / len(rets) * 100,
                avg=rets.mean(), aw=w.mean() if len(w) else 0, al=l.mean() if len(l) else 0, pf=pf,
                yearly=yr, curve=curve, taken=taken)


def report(sigs, n_stocks, kp, t0):
    L = ["# 깊은 바닥 반등 전략 · 계좌 백테스트", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · 시총 상위 {n_stocks}종목 (현재 상장 기준)",
         f"- 조건: 52주 고점 대비 -{S.DD_MIN:g}% 이상 · 5파 깊이 {S.DEPTH_MIN}배 이상 · 1일/주봉 다이버전스 (저점 확정 2봉)",
         "- 점수: 기본 50 + 코스피 120일선 아래 20 + 월봉 RSI 35 미만 15 + 주봉 다이버전스 10 + 투매 거래량 5",
         "- 비중: A 1회 손실 1.25%·최대 20% / B 1%·15% / C 0.5%·10% · 하루 신규 3 · 보유 8 · 총 70%",
         f"- 조건을 만족한 신호: {len(sigs)}건 (A {sum(t['grade'] == 'A' for t in sigs)} · B {sum(t['grade'] == 'B' for t in sigs)} · C {sum(t['grade'] == 'C' for t in sigs)})", ""]
    L += ["## 1. 청산 방식 × 등급 범위 (원금 100)",
          "| 청산 | 등급 | 최종 자산 | 연평균 | 최대 낙폭 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | PF |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    results = {}
    for ex in EXITS:
        for lab, gs in (("A·B·C", "ABC"), ("A·B", "AB"), ("A만", "A")):
            r = portfolio(sigs, ex, gs)
            if not r:
                continue
            results[(ex, lab)] = r
            pf = f"{r['pf']:.2f}" if r["pf"] != float("inf") else "∞"
            L.append(f"| {ex} | {lab} | **{r['final']:.1f}** | {r['cagr']:+.1f}% | {r['mdd']:.1f}% | {r['n']} | "
                     f"{r['win']:.0f}% | {r['avg']:+.1f}% | {r['aw']:+.1f}% | {r['al']:+.1f}% | {pf} |")
    if kp is not None and results:
        any_r = next(iter(results.values()))
        c = any_r["curve"]
        k = kp.reindex(c.index, method="ffill").dropna()
        if len(k) > 1:
            kc = k / k.iloc[0] * 100
            kmdd = ((kc / kc.cummax()) - 1).min() * 100
            L += ["", f"비교: 같은 기간 **코스피 지수 그냥 보유** → {kc.iloc[-1]:.1f} (최대 낙폭 {kmdd:.1f}%)"]
    # 위험 대비 수익이 가장 좋은 조합 (연평균 ÷ |최대 낙폭|)
    if results:
        pool = [k for k in results if results[k]["n"] >= 20] or list(results)
        best = max(pool, key=lambda k: results[k]["cagr"] / max(abs(results[k]["mdd"]), 1))
        r = results[best]
        L += ["", f"## 2. 위험 대비 수익이 가장 좋은 조합: {best[0]} · 등급 {best[1]}", "",
              "### 연도별 수익", "| 연도 | 수익률 |", "|---|---|"]
        for y, v in r["yearly"].items():
            L.append(f"| {y} | {v:+.1f}% |")
        tk = r["taken"]
        L += ["", "### 등급별 (실제 들어간 거래)", "| 등급 | 거래 | 승률 | 평균 수익 |", "|---|---|---|---|"]
        for g in "ABC":
            v = np.array([t[best[0]]["ret"] for t in tk if t["grade"] == g])
            if len(v):
                L.append(f"| {g} | {len(v)} | {(v > 0).mean() * 100:.0f}% | {v.mean():+.1f}% |")
        L += ["", "### 청산 사유", "| 사유 | 횟수 | 평균 수익 |", "|---|---|---|"]
        rs = pd.DataFrame([dict(r=t[best[0]]["reason"], v=t[best[0]]["ret"]) for t in tk])
        if len(rs):
            for k2, g in rs.groupby("r"):
                L.append(f"| {k2} | {len(g)} | {g.v.mean():+.1f}% |")
        L += ["", "### 수익 상위 10건", "| 날짜 | 종목 | 등급 | 수익 |", "|---|---|---|---|"]
        for t in sorted(tk, key=lambda t: -t[best[0]]["ret"])[:10]:
            L.append(f"| {t['date'].date()} | {t['name']} | {t['grade']} | {t[best[0]]['ret']:+.1f}% |")
        L += ["", "### 손실 상위 10건", "| 날짜 | 종목 | 등급 | 수익 |", "|---|---|---|---|"]
        for t in sorted(tk, key=lambda t: t[best[0]]["ret"])[:10]:
            L.append(f"| {t['date'].date()} | {t['name']} | {t['grade']} | {t[best[0]]['ret']:+.1f}% |")
    L += ["", "## 주의",
          "- **상장폐지된 종목이 빠져 있어요.** 이 전략은 크게 빠진 종목을 사서 그 영향이 특히 커요. 실전에서는 재무 ⚠위험 필터가 꼭 켜져 있어야 해요 (백테스트에는 과거 재무 데이터가 없어서 못 넣었어요).",
          "- 같은 날 신호가 많으면 점수 높은 순으로 3종목만 샀어요.",
          "- 과거 결과가 앞으로도 같다는 보장은 없어요.", "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (OUT / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    rows = [dict(date=t["date"].date(), code=t["code"], name=t["name"], grade=t["grade"], score=t["score"],
                 dd52=t["dd52"], mrsi=t["mrsi"], weekly=t["weekly"], market_weak=t["market_weak"], stop_dist=t["stop_dist"],
                 weight=round(t["w"] * 100, 1), **{ex: round(t[ex]["ret"], 2) for ex in EXITS},
                 **{ex + " 사유": t[ex]["reason"] for ex in EXITS}) for t in sigs]
    pd.DataFrame(rows).to_csv(OUT / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    kp = kospi()
    log(f"코스피 지수: {'있음' if kp is not None else '없음'}")
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    sigs, done = [], [0]

    def job(s):
        try:
            return stock_signals(s, start, kp)
        except Exception as e:
            log("오류", s["name"], e)
            return []
        finally:
            done[0] += 1
            if done[0] % 100 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            sigs += r
    log(f"조건 만족 신호 {len(sigs)}건")
    if not sigs:
        (OUT / "backtest.md").write_text("# 결과 없음 (조건을 만족한 신호가 없어요)", encoding="utf-8")
        return
    report(sigs, len(uni), kp, t0)


if __name__ == "__main__":
    main()
