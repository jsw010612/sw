"""
단계별 RSI 다이버전스 비교 백테스트 (차트만, 재무 제외)

신호 (스캐너와 같은 규칙)
  - 1일봉 다이버전스 + 그 시점 1주 RSI 40 이상 → 매수 (기준 1일)
  - 주봉 다이버전스 + 1주 RSI 40 미만           → 매수 (기준 1주)
  - 1일봉 다이버전스 + 1주 RSI 40 미만          → 주봉 대기 (매수 안 함, 참고로만 집계)
  점수는 단기 타이밍(10점)을 뺀 90점 만점을 100점으로 환산

비교하는 규칙
  ① 지금 규칙   : 저점 확정 5봉 · 손절 -5% 또는 5파 저점 종가 이탈
  ② 추격 금지   : ① + 진입가가 5파 저점보다 3% 넘게 위면 안 삼
  ③ 빠른 확정   : ① + 저점 확정 2봉
  ④ 구조 손절   : ① 에서 -5% 손절을 빼고 5파 저점 종가 이탈만 손절
  ⑤ 전부 (2봉)  : ②+③+④
  ⑥ 전부 (3봉)  : ⑤ 에서 저점 확정만 3봉

공통 청산: 4파 고점 절반 익절 · 나머지는 120일선(아래에서 샀으면 도달, 위에서 샀으면 종가 이탈) ·
          1일 과매수 다이버전스 · 60거래일 · 왕복 비용 0.3%

차트만으로 정한 비중 등급 (원금 대비, 1회 최대 손실)
  S 30% / 1.5% : 주봉 다이버전스 기준 85점↑ · 손익비 2↑   또는  95점↑ · 손익비 2↑ · 진입가가 5파 저점 +2% 이내
  A 20% / 1%   : 90점↑ · 손익비 2↑
  B 10% / 0.75%: 85점↑ · 손익비 1.5↑
  C  5% / 0.5% : 80점↑ · 손익비 1.5↑
  실제 비중 = min(등급 비중, 1회 손실 한도 ÷ 손절폭) · 총 보유 70% 이하 · S는 동시에 2개까지
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import Params, analyze_frame, rsi, to_week

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "1000") or 1000)
HOLD_MAX = 60
COST = 0.003
HARD = 0.05
CHASE = 0.03
BASE_P = Params()
OUT = scan.STATE

VARIANTS = [  # 이름, 저점 확정 봉 수, 추격 금지, -5% 손절 사용
    ("① 지금 규칙", 5, False, True),
    ("② 추격 금지", 5, True, True),
    ("③ 빠른 확정", 2, False, True),
    ("④ 구조 손절", 5, False, False),
    ("⑤ 전부 (2봉)", 2, True, False),
    ("⑥ 전부 (3봉)", 3, True, False),
]
TIERS = {"S": (0.30, 0.015), "A": (0.20, 0.010), "B": (0.10, 0.0075), "C": (0.05, 0.005)}


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def comp_score(snap, wpts):
    raw = (20 + (10 if snap["v3"] else 0) + (10 if snap["vd"] else 0) + (15 if snap["deep"] else 0)
           + (10 if snap["zone"] else 0) + (10 if snap["ok4"] else 0) + wpts)
    return int(round(raw * 100 / 90))


def tier_of(base, score, rr, chase):
    if (base == "1주" and score >= 85 and rr >= 2) or (score >= 95 and rr >= 2 and chase <= 2):
        return "S"
    if score >= 90 and rr >= 2:
        return "A"
    if score >= 85 and rr >= 1.5:
        return "B"
    if score >= 80 and rr >= 1.5:
        return "C"
    return None


def simulate(O, H, L, C, ei, l5, h4, ma120, bear_idx, use_hard):
    entry = O[ei]
    hard = entry * (1 - HARD)
    above_ma = not np.isnan(ma120[ei]) and entry > ma120[ei]
    pos, ret, tp1, reason = 1.0, 0.0, False, "기간 만료"
    last = min(len(C) - 1, ei + HOLD_MAX)
    j = ei
    for j in range(ei, last + 1):
        if use_hard and L[j] <= hard:
            px = min(O[j], hard) if j > ei else hard
            ret += pos * (px / entry - 1); pos = 0; reason = "손절 -5%"; break
        if not tp1 and h4 and H[j] >= h4:
            px = max(O[j], h4) if j > ei else max(entry, h4)
            ret += 0.5 * (px / entry - 1); pos -= 0.5; tp1 = True
        if C[j] < l5:
            ret += pos * (C[j] / entry - 1); pos = 0; reason = "손절 5파 이탈"; break
        m = ma120[j]
        if not np.isnan(m) and j > ei:
            if not above_ma and H[j] >= m:
                ret += pos * (max(O[j], m) / entry - 1); pos = 0; reason = "120일선 도달"; break
            if above_ma and C[j] < m:
                ret += pos * (C[j] / entry - 1); pos = 0; reason = "120일선 이탈"; break
        if j in bear_idx and j > ei:
            ret += pos * (C[j] / entry - 1); pos = 0; reason = "과매수 다이버전스"; break
    if pos > 0:
        ret += pos * (C[j] / entry - 1)
        if reason == "기간 만료" and tp1:
            reason = "4파 익절 후 기간 만료"
    return ret - COST, reason, j - ei, j


def signals_for(d1, wk, P):
    """한 가지 저점 확정 설정으로 신호 목록 만들기: (종류, 기준, 확정 봉, snap, 주봉 RSI)"""
    out = []
    dev = []
    analyze_frame(d1, 1, P, dev)
    for i, snap in dev:
        w = to_week(d1.iloc[:i + 1])
        wr = rsi(w["close"].to_numpy(float), P.rsi_len)[-1] if len(w) > P.rsi_len + 1 else np.nan
        out.append(("대기" if (not np.isnan(wr) and wr < scan.W_WAIT) else "매수", "1일", i, snap, wr))
    wev = []
    analyze_frame(wk, 1, P, wev)
    wr_all = rsi(wk["close"].to_numpy(float), P.rsi_len)
    for j, snap in wev:
        if np.isnan(wr_all[j]) or wr_all[j] >= scan.W_WAIT:
            continue
        k = d1.index.searchsorted(wk.index[j] + pd.Timedelta(days=6), side="right") - 1
        if k >= 0:
            out.append(("매수", "1주", k, snap, wr_all[j]))
    bev = []
    analyze_frame(d1, -1, P, bev)
    return out, {i for i, _ in bev}


def stock_trades(s, start):
    d1 = scan.naver_daily(s["code"], 2000, s["market"])
    if len(d1) < 300:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    wk = to_week(d1)
    n = len(d1)
    O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    ma120 = pd.Series(C).rolling(120).mean().to_numpy()
    cache = {}
    trades = []
    for vname, pivr, chase_lim, use_hard in VARIANTS:
        if pivr not in cache:
            cache[pivr] = signals_for(d1, wk, replace(BASE_P, piv_r=pivr))
        sigs, bear_idx = cache[pivr]
        for kind, base, det_i, snap, wr in sigs:
            ei = det_i + 1
            if ei >= n or d1.index[ei] < start:
                continue
            entry = O[ei]
            l5 = float(snap["l5p"])
            h4 = snap.get("h4")
            h4 = float(h4) if h4 is not None and not np.isnan(h4) else None
            if entry <= l5:
                continue
            chase = (entry / l5 - 1) * 100
            if chase_lim and chase > CHASE * 100:
                continue
            stop = max(l5, entry * (1 - HARD)) if use_hard else l5
            dist = (entry - stop) / entry * 100
            rr = ((h4 - entry) / (entry - stop)) if (h4 and entry < h4 and entry > stop) else 0.0
            wpts = 15 if base == "1주" else (0 if kind == "대기" else (15 if (np.isnan(wr) or wr < 50) else 10))
            score = comp_score(snap, wpts)
            ret, reason, days, xi = simulate(O, H, L, C, ei, l5, h4, ma120, bear_idx, use_hard)
            trades.append(dict(
                variant=vname, code=s["code"], name=s["name"], kind=kind, base=base,
                date=d1.index[ei].strftime("%Y-%m-%d"), exit_date=d1.index[xi].strftime("%Y-%m-%d"),
                entry=round(entry, 2), chase=round(chase, 2), stop_dist=round(dist, 2), rr=round(rr, 2),
                score=score, wrsi=None if np.isnan(wr) else round(float(wr), 1), ret=round(ret * 100, 2),
                reason=reason, days=int(days), tier=tier_of(base, score, rr, chase) if kind == "매수" else None,
            ))
    return trades


# ===== 집계 =====
def stats(df):
    if len(df) == 0:
        return dict(n=0)
    w, l = df[df.ret > 0], df[df.ret <= 0]
    pf = w.ret.sum() / abs(l.ret.sum()) if len(l) and l.ret.sum() != 0 else float("inf")
    return dict(n=len(df), win=round(len(w) / len(df) * 100, 1), avg=round(df.ret.mean(), 2),
                avg_w=round(w.ret.mean(), 2) if len(w) else 0, avg_l=round(l.ret.mean(), 2) if len(l) else 0,
                pf=round(pf, 2) if pf != float("inf") else "∞", days=round(df.days.mean(), 1))


HEAD = "| 구분 | 거래 수 | 승률 | 평균 수익 | 평균 이익 | 평균 손실 | PF | 평균 보유일 |\n|---|---|---|---|---|---|---|---|"


def row(label, st):
    if st["n"] == 0:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    return (f"| {label} | {st['n']} | {st['win']}% | {st['avg']:+.2f}% | {st['avg_w']:+.2f}% | "
            f"{st['avg_l']:+.2f}% | {st['pf']} | {st['days']} |")


def portfolio(df):
    tr = df[df.tier.notna()].sort_values("date").to_dict("records")
    ev = sorted([(t["date"], 1, k) for k, t in enumerate(tr)] + [(t["exit_date"], 0, k) for k, t in enumerate(tr)],
                key=lambda e: (e[0], e[1]))
    cash, open_pos, taken, skipped = 100.0, {}, 0, 0
    peak, mdd = 100.0, 0.0
    for date, typ, k in ev:
        t = tr[k]
        if typ == 0 and k in open_pos:
            amt = open_pos.pop(k)
            cash += amt * (1 + t["ret"] / 100)
            eq = cash + sum(open_pos.values())
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(open_pos.values())
            mx, risk = TIERS[t["tier"]]
            w = min(mx, risk / max(t["stop_dist"] / 100, 0.005))
            s_cnt = sum(1 for kk in open_pos if tr[kk]["tier"] == "S")
            if sum(open_pos.values()) / eq + w > 0.70 or (t["tier"] == "S" and s_cnt >= 2) or eq * w > cash:
                skipped += 1
                continue
            cash -= eq * w
            open_pos[k] = eq * w
            taken += 1
    for k, amt in open_pos.items():
        cash += amt * (1 + tr[k]["ret"] / 100)
    cagr = ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100
    return dict(final=round(cash, 1), cagr=round(cagr, 1), mdd=round(mdd * 100, 1), taken=taken, skipped=skipped)


def detail(df, title):
    buy = df[df.kind == "매수"]
    L = [f"## {title}", "", "### 기준·등급별", HEAD,
         row("매수 전체", stats(buy)), row("· 1일 기준", stats(buy[buy.base == "1일"])),
         row("· 주봉 다이버전스 기준", stats(buy[buy.base == "1주"])), row("주봉 대기 (샀다면)", stats(df[df.kind == "대기"]))]
    for t in ("S", "A", "B", "C"):
        L.append(row(f"{t}등급", stats(buy[buy.tier == t])))
    L.append(row("등급 밖", stats(buy[buy.tier.isna()])))
    L += ["", "### 점수·손익비·진입 위치", HEAD]
    for lo, hi, lab in ((95, 999, "95점 이상"), (90, 95, "90~94점"), (85, 90, "85~89점"), (80, 85, "80~84점"), (0, 80, "80점 미만")):
        L.append(row(lab, stats(buy[(buy.score >= lo) & (buy.score < hi)])))
    L += [row("손익비 2 이상", stats(buy[buy.rr >= 2])), row("손익비 2 미만", stats(buy[buy.rr < 2])),
          row("진입가 5파 저점 +1% 이내", stats(buy[buy.chase <= 1])),
          row("+1~2%", stats(buy[(buy.chase > 1) & (buy.chase <= 2)])),
          row("+2~3%", stats(buy[(buy.chase > 2) & (buy.chase <= 3)])),
          row("+3% 초과", stats(buy[buy.chase > 3])),
          row("1주 RSI 40~50", stats(buy[(buy.wrsi >= 40) & (buy.wrsi < 50)])),
          row("1주 RSI 50 이상", stats(buy[buy.wrsi >= 50])), "", "### 연도별", HEAD]
    for y, g in buy.groupby(buy.date.str[:4]):
        L.append(row(y, stats(g)))
    L += ["", "### 청산 사유", "| 사유 | 횟수 | 평균 수익 |", "|---|---|---|"]
    for k, g in buy.groupby("reason"):
        L.append(f"| {k} | {len(g)} | {g.ret.mean():+.2f}% |")
    L.append("")
    return L


def report(df, n_stocks, t0):
    L = ["# 다이버전스 비교 백테스트", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 기간: 최근 {YEARS:g}년 · 대상: 시총 상위 {n_stocks}종목 (현재 상장 종목 기준)",
         "- 공통: 다음 날 시가 진입 · 4파 고점 절반 익절 · 120일선/과매수 다이버전스/60일 청산 · 왕복 비용 0.3% · 재무 제외",
         "- 점수: 단기 타이밍(10점) 제외 90점을 100점으로 환산 · 주봉 대기 신호는 사지 않음", "",
         "## 규칙별 비교 (매수 신호 전체)", HEAD.replace("구분", "규칙")]
    port = {}
    for vname, *_ in VARIANTS:
        g = df[(df.variant == vname) & (df.kind == "매수")]
        L.append(row(vname, stats(g)))
        port[vname] = portfolio(df[df.variant == vname])
    L += ["", "## 규칙별 계좌 시뮬레이션 (등급 비중 적용, 원금 100)",
          "| 규칙 | 최종 자산 | 연평균 | 최대 낙폭 | 들어간 거래 | 한도로 건너뜀 |", "|---|---|---|---|---|---|"]
    for vname, p in port.items():
        L.append(f"| {vname} | **{p['final']}** | {p['cagr']:+.1f}% | {p['mdd']}% | {p['taken']} | {p['skipped']} |")
    L.append("")
    best = max(port, key=lambda v: port[v]["final"])
    L += detail(df[df.variant == best], f"가장 좋은 규칙 상세: {best}")
    if best != "① 지금 규칙":
        L += detail(df[df.variant == "① 지금 규칙"], "비교용: ① 지금 규칙 상세")
    L += ["## 주의", "- **상장폐지된 종목은 빠져 있어서** 실제보다 결과가 좋게 나와요.",
          "- 같은 날 손절가와 익절가가 둘 다 닿으면 손절이 먼저 난 걸로 계산했어요 (보수적).",
          "- 구조 손절은 갭 하락이면 손실이 -5%보다 커질 수 있어요. 그래서 비중을 손절폭에 맞춰 줄여요.",
          "- 여러 규칙을 비교해서 가장 좋은 걸 고르면 과거에만 맞춘 결과가 될 위험이 있어요. 큰 차이가 날 때만 의미 있게 보세요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (OUT / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(OUT / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md, state/backtest_trades.csv")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"백테스트 시작: {len(uni)}종목, {start.date()} 이후 신호, 규칙 {len(VARIANTS)}가지")
    all_tr, done = [], [0]

    def job(s):
        try:
            return stock_trades(s, start)
        except Exception as e:
            log("오류", s["name"], e)
            return []
        finally:
            done[0] += 1
            if done[0] % 100 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for tr in ex.map(job, uni):
            all_tr += tr
    df = pd.DataFrame(all_tr)
    log(f"거래 {len(df)}건 (규칙 전체 합)")
    if df.empty:
        (OUT / "backtest.md").write_text("# 백테스트 결과\n\n신호가 하나도 없었어요.", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
