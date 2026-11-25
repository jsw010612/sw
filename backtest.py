"""
단계별 RSI 다이버전스 백테스트 (재무 제외, 1일봉 기준)

신호 (스캐너와 같은 규칙)
  - 1일봉 다이버전스 확정 + 그 시점 1주 RSI 40 이상 → 매수 (기준 1일)
  - 1일봉 다이버전스 확정 + 1주 RSI 40 미만         → 주봉 대기 (참고로 따로 집계)
  - 주봉 다이버전스 확정 + 1주 RSI 40 미만           → 매수 (기준 1주)
  점수는 단기 타이밍(10점)을 뺀 90점 만점을 100점으로 환산

매매 규칙
  진입: 신호 확정 다음 날 시가
  손절: 장중 -5% 닿으면 전량 / 5파 저점을 종가로 깨면 전량
  익절: 4파 고점에서 절반 / 나머지는 120일선 도달(아래에서 산 경우) 또는 120일선 종가 이탈(위에서 산 경우),
        1일 과매수 다이버전스, 60거래일 경과
  비용: 왕복 0.3% (수수료 + 거래세 대략)

등급·비중 (원금 대비)
  S 30% : 점수 95↑ · 직전 1년 RSI 30 이하 없음 · 손익비 2↑   · 1회 손실 1.5%
  A 20% : 점수 90↑ · 직전 6개월 RSI 30 이하 없음 · 손익비 2↑ · 1회 손실 1%
  B 10% : 점수 85↑ · 손익비 1.5↑                            · 1회 손실 0.75%
  C  5% : 점수 80↑ 또는 주봉 대기 80↑(타이밍 빼고 환산) · 손익비 1.5↑ · 1회 손실 0.5%
  실제 비중 = min(등급 비중, 1회 손실 한도 ÷ 손절폭) · 총 보유 70% 이하 · S는 동시에 2개까지
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import Params, analyze_frame, rsi, to_week

YEARS = float(os.getenv("BT_YEARS", "3"))
TOP_N = int(os.getenv("BT_TOP", "500"))
HOLD_MAX = 60
COST = 0.003
HARD_STOP = 0.05
P = Params()
OUT = scan.STATE

TIERS = {  # 등급: (최대 비중, 1회 손실 한도)
    "S": (0.30, 0.015), "A": (0.20, 0.010), "B": (0.10, 0.0075), "C": (0.05, 0.005),
}


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def comp_score(snap, wpts):
    raw = (20 + (10 if snap["v3"] else 0) + (10 if snap["vd"] else 0) + (15 if snap["deep"] else 0)
           + (10 if snap["zone"] else 0) + (10 if snap["ok4"] else 0) + wpts)
    return int(round(raw * 100 / 90))


def last_os_gap(r, l3i):
    """3파 저점 이전에 RSI 30 이하가 마지막으로 나온 뒤 몇 거래일이 지났는지 (없으면 큰 수)"""
    end = max(0, l3i - 15)  # 3파로 내려오는 구간은 빼고 봄
    seg = r[:end]
    idx = np.where(seg <= P.os_lvl)[0]
    return (end - idx[-1]) if len(idx) else 9999


def tier_of(kind, score, gap, rr):
    if kind == "대기":
        return "C" if score >= 80 and rr >= 1.5 else None
    if score >= 95 and gap >= 250 and rr >= 2:
        return "S"
    if score >= 90 and gap >= 125 and rr >= 2:
        return "A"
    if score >= 85 and rr >= 1.5:
        return "B"
    if score >= 80 and rr >= 1.5:
        return "C"
    return None


def simulate(d1, ei, l5, h4, ma120, bear_idx):
    """ei: 진입 봉 번호. 반환: 수익률(비용 포함), 청산 사유, 보유일, 청산 봉"""
    O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    entry = O[ei]
    hard = entry * (1 - HARD_STOP)
    above_ma = not np.isnan(ma120[ei]) and entry > ma120[ei]
    pos, ret, tp1, reason = 1.0, 0.0, False, "기간 만료"
    n = len(d1)
    last = min(n - 1, ei + HOLD_MAX)
    j = ei
    for j in range(ei, last + 1):
        # 1) 장중 -5%
        if L[j] <= hard:
            px = min(O[j], hard) if j > ei else hard
            ret += pos * (px / entry - 1); pos = 0; reason = "손절 -5%"; break
        # 2) 4파 고점 절반 익절
        if not tp1 and h4 and H[j] >= h4:
            px = max(O[j], h4) if j > ei else max(entry, h4)
            ret += 0.5 * (px / entry - 1); pos -= 0.5; tp1 = True
        # 3) 5파 저점 종가 이탈
        if C[j] < l5:
            ret += pos * (C[j] / entry - 1); pos = 0; reason = "손절 5파 이탈"; break
        # 4) 120일선
        m = ma120[j]
        if not np.isnan(m):
            if not above_ma and H[j] >= m and j > ei:
                ret += pos * (max(O[j], m) / entry - 1); pos = 0; reason = "120일선 도달"; break
            if above_ma and C[j] < m:
                ret += pos * (C[j] / entry - 1); pos = 0; reason = "120일선 이탈"; break
        # 5) 과매수 다이버전스
        if j in bear_idx and j > ei:
            ret += pos * (C[j] / entry - 1); pos = 0; reason = "과매수 다이버전스"; break
    if pos > 0:
        ret += pos * (C[j] / entry - 1)
        if reason == "기간 만료" and tp1:
            reason = "4파 익절 후 기간 만료"
    return ret - COST, reason, j - ei, j


def stock_trades(s, start):
    d1 = scan.naver_daily(s["code"], 1500, s["market"])
    if len(d1) < 300:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    C = d1["close"].to_numpy(float)
    r = rsi(C, P.rsi_len)
    ma120 = pd.Series(C).rolling(120).mean().to_numpy()
    idx_of = {t: k for k, t in enumerate(d1.index)}

    # 과매수 다이버전스(청산용)
    bev = []
    analyze_frame(d1, -1, P, bev)
    bear_idx = {i for i, _ in bev}

    trades = []

    def add(kind, base, det_i, snap, wr):
        ei = det_i + 1
        if ei >= n or d1.index[ei] < start:
            return
        entry = float(d1["open"].iloc[ei])
        l5 = float(snap["l5p"])
        h4 = snap.get("h4")
        h4 = float(h4) if h4 is not None and not np.isnan(h4) else None
        if entry <= l5:
            return
        stop = max(l5, entry * (1 - HARD_STOP))
        dist = (entry - stop) / entry
        rr = ((h4 - entry) / (entry - stop)) if (h4 and entry < h4 and entry > stop) else 0.0
        wpts = 15 if base == "1주" else (0 if kind == "대기" else (15 if (np.isnan(wr) or wr < 50) else 10))
        score = comp_score(snap, wpts)
        l3i = idx_of.get(pd.Timestamp(snap["l3t"]), None) if base == "1일" else \
            idx_of.get(d1.index[d1.index <= pd.Timestamp(snap["l3t"]) + pd.Timedelta(days=6)][-1], None)
        gap = last_os_gap(r, l3i) if l3i is not None else 0
        ret, reason, days, xi = simulate(d1, ei, l5, h4, ma120, bear_idx)
        trades.append(dict(
            code=s["code"], name=s["name"], kind=kind, base=base, date=d1.index[ei].strftime("%Y-%m-%d"),
            exit_date=d1.index[xi].strftime("%Y-%m-%d"), entry=round(entry, 2), stop_dist=round(dist * 100, 2),
            rr=round(rr, 2), score=score, gap=int(gap), wrsi=None if np.isnan(wr) else round(float(wr), 1),
            ret=round(ret * 100, 2), reason=reason, days=int(days), tier=tier_of(kind, score, gap, rr),
        ))

    # 1일봉 다이버전스
    dev = []
    analyze_frame(d1, 1, P, dev)
    for i, snap in dev:
        wk = to_week(d1.iloc[:i + 1])
        wr = rsi(wk["close"].to_numpy(float), P.rsi_len)[-1] if len(wk) > P.rsi_len + 1 else np.nan
        kind = "대기" if (not np.isnan(wr) and wr < scan.W_WAIT) else "매수"
        add(kind, "1일", i, snap, wr)

    # 주봉 다이버전스 (주봉 RSI 40 미만일 때 매수)
    wk = to_week(d1)
    wev = []
    analyze_frame(wk, 1, P, wev)
    wr_all = rsi(wk["close"].to_numpy(float), P.rsi_len)
    for j, snap in wev:
        if np.isnan(wr_all[j]) or wr_all[j] >= scan.W_WAIT:
            continue
        week_end = wk.index[j] + pd.Timedelta(days=6)
        k = d1.index.searchsorted(week_end, side="right") - 1  # 그 주 마지막 거래일
        if k < 0:
            continue
        add("매수", "1주", k, snap, wr_all[j])
    return trades


# ===== 집계 =====
def stats(df):
    if len(df) == 0:
        return dict(n=0)
    w = df[df.ret > 0]
    l = df[df.ret <= 0]
    pf = w.ret.sum() / abs(l.ret.sum()) if len(l) and l.ret.sum() != 0 else float("inf")
    return dict(n=len(df), win=round(len(w) / len(df) * 100, 1), avg=round(df.ret.mean(), 2),
                avg_w=round(w.ret.mean(), 2) if len(w) else 0, avg_l=round(l.ret.mean(), 2) if len(l) else 0,
                pf=round(pf, 2) if pf != float("inf") else "∞", days=round(df.days.mean(), 1))


def row(label, st):
    if st["n"] == 0:
        return f"| {label} | 0 | - | - | - | - | - | - |"
    return f"| {label} | {st['n']} | {st['win']}% | {st['avg']:+.2f}% | {st['avg_w']:+.2f}% | {st['avg_l']:+.2f}% | {st['pf']} | {st['days']} |"


HEAD = "| 구분 | 거래 수 | 승률 | 평균 수익 | 평균 이익 | 평균 손실 | 손익비(PF) | 평균 보유일 |\n|---|---|---|---|---|---|---|---|"


def portfolio(df):
    """등급 비중으로 실제 계좌처럼 굴려 봄 (원금 100 기준)"""
    tr = df[df.tier.notna()].sort_values("date").to_dict("records")
    events = []
    for k, t in enumerate(tr):
        events.append((t["date"], 1, k))
        events.append((t["exit_date"], 0, k))
    events.sort(key=lambda e: (e[0], e[1]))
    cash, equity = 100.0, 100.0
    open_pos = {}
    curve, taken, skipped = [], 0, 0
    for date, typ, k in events:
        t = tr[k]
        if typ == 0 and k in open_pos:
            amt = open_pos.pop(k)
            pnl = amt * t["ret"] / 100
            cash += amt + pnl
            equity = cash + sum(open_pos.values())
            curve.append((date, equity))
        elif typ == 1:
            mx, risk = TIERS[t["tier"]]
            dist = max(t["stop_dist"], 0.5) / 100
            w = min(mx, risk / dist)
            exposure = sum(open_pos.values()) / max(equity, 1e-9)
            s_cnt = sum(1 for kk in open_pos if tr[kk]["tier"] == "S")
            if exposure + w > 0.70 or (t["tier"] == "S" and s_cnt >= 2):
                skipped += 1
                continue
            amt = equity * w
            if amt > cash:
                skipped += 1
                continue
            cash -= amt
            open_pos[k] = amt
            taken += 1
    for k, amt in list(open_pos.items()):
        cash += amt * (1 + tr[k]["ret"] / 100)
    equity = cash
    if not curve:
        return dict(final=100.0, mdd=0, taken=0, skipped=skipped, cagr=0)
    peak, mdd = 100.0, 0.0
    for _, e in curve:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    cagr = (equity / 100) ** (1 / YEARS) - 1
    return dict(final=round(equity, 1), mdd=round(mdd * 100, 1), taken=taken, skipped=skipped,
                cagr=round(cagr * 100, 1), curve=curve)


def report(df, n_stocks, t0):
    L = [f"# 다이버전스 백테스트 결과", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 기간: 최근 {YEARS:g}년 · 대상: 시총 상위 {n_stocks}종목 (현재 상장 종목 기준)",
         f"- 규칙: 신호 다음 날 시가 진입 · -5% 또는 5파 저점 종가 이탈 손절 · 4파 고점 절반 익절 · 120일선/과매수 다이버전스/60일 청산 · 왕복 비용 0.3%",
         f"- 점수: 단기 타이밍(10점) 제외 90점 만점을 100점으로 환산 · 재무 제외", ""]
    buy = df[df.kind == "매수"]
    L += ["## 1. 전체", HEAD, row("매수 신호 전체", stats(buy)), row("· 1일 기준", stats(buy[buy.base == "1일"])),
          row("· 주봉 다이버전스 기준", stats(buy[buy.base == "1주"])), row("주봉 대기 신호 (샀다면)", stats(df[df.kind == "대기"])), ""]
    L += ["## 2. 등급별 (제안한 비중 등급)", HEAD]
    for t in ("S", "A", "B", "C"):
        L.append(row(f"{t}등급", stats(df[df.tier == t])))
    L.append(row("등급 밖 (제외 대상)", stats(buy[buy.tier.isna()])))
    L.append("")
    L += ["## 3. 점수 구간별 (매수 신호)", HEAD]
    for lo, hi in ((95, 101), (90, 95), (85, 90), (80, 85), (0, 80)):
        L.append(row(f"{lo}~{hi - 1}점" if hi < 101 else f"{lo}점 이상", stats(buy[(buy.score >= lo) & (buy.score < hi)])))
    L.append("")
    L += ["## 4. 필터 효과 (매수 신호)", HEAD,
          row("손익비 2 이상", stats(buy[buy.rr >= 2])), row("손익비 2 미만", stats(buy[buy.rr < 2])),
          row("첫 과매도 (직전 1년 RSI30 없음)", stats(buy[buy.gap >= 250])),
          row("직전 1년 안에 RSI30 있었음", stats(buy[buy.gap < 250])),
          row("1주 RSI 50 이상", stats(buy[buy.wrsi >= 50])), row("1주 RSI 40~50", stats(buy[(buy.wrsi >= 40) & (buy.wrsi < 50)])), ""]
    L += ["## 5. 연도별 (매수 신호)", HEAD]
    for y, g in buy.groupby(buy.date.str[:4]):
        L.append(row(y, stats(g)))
    L.append("")
    L += ["## 6. 청산 사유 (매수 신호)", "| 사유 | 횟수 | 평균 수익 |", "|---|---|---|"]
    for k, g in buy.groupby("reason"):
        L.append(f"| {k} | {len(g)} | {g.ret.mean():+.2f}% |")
    L.append("")
    pf = portfolio(df)
    L += ["## 7. 계좌 시뮬레이션 (등급 비중 적용, 원금 100)",
          f"- 최종 자산: **{pf['final']}** (연평균 {pf['cagr']:+.1f}%) · 최대 낙폭 {pf['mdd']}%",
          f"- 실제로 들어간 거래 {pf['taken']}건 · 한도 때문에 건너뛴 거래 {pf['skipped']}건",
          "- 총 보유 70% 이하, S등급 동시 2개까지, 비중 = min(등급 비중, 1회 손실 한도 ÷ 손절폭)", ""]
    L += ["## 주의", "- **상장폐지된 종목은 빠져 있어서** 실제보다 결과가 좋게 나와요.",
          "- 하루 안에서 손절가와 익절가가 둘 다 닿으면 손절이 먼저 난 걸로 계산했어요 (보수적).",
          "- 과거 성과가 앞으로도 같다는 보장은 없어요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (OUT / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(OUT / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md, state/backtest_trades.csv")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"백테스트 시작: {len(uni)}종목, {start.date()} 이후 신호")
    all_tr = []
    done = [0]

    def job(s):
        try:
            return stock_trades(s, start)
        except Exception as e:
            log("오류", s["name"], e)
            return []
        finally:
            done[0] += 1
            if done[0] % 50 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for tr in ex.map(job, uni):
            all_tr += tr
    df = pd.DataFrame(all_tr)
    log(f"신호 {len(df)}건")
    if df.empty:
        (OUT / "backtest.md").write_text("# 백테스트 결과\n\n신호가 하나도 없었어요.", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
