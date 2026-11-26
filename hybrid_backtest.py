"""
혼합 매도 — 단계별로 다른 지표를 씀

1단계 (1차 익절 전, 아직 바닥 확인 중) : 손실을 끊는 규칙
  A 5파      : 5파 저점 종가 이탈
  B 5파+후행 : 5파 저점 이탈 또는 후행스팬이 26일 전 캔들 아래로 내려가는 날
  C 5파+시장 : 5파 저점 이탈. 단 그 시장(코스피/코스닥) RSI가 30 아래면 손절을 미룸

1차 익절: 위쪽 주·월봉 60/120선(없으면 +50%)에서 절반

2단계 (절반 판 뒤 남은 절반) : 수익을 지키는 규칙 (1단계 손절선도 계속 유지)
  추적20   : 최고가 -20% 종가 이탈 (지금 방식)
  데드     : 5일선이 20일선을 아래로 뚫는 날
  후행캔들 : 후행스팬이 26일 전 캔들 아래로 내려가는 날
  후행20   : 후행스팬이 26일 전 20일선 아래로 내려가는 날
  MACD0    : MACD가 0선 아래로 내려가는 날
  MACD데드 : MACD가 시그널선 아래로 내려가는 날
  2/3      : [5일선<20일선, 후행스팬<캔들, MACD<시그널] 셋 중 둘이 켜지는 날

비상 손절: 매수가 -20% 종가 이탈 (1단계 C에서만 의미 있음)
매수 쪽: PPO 히스토그램 '산' 다이버전스 (5파 쪽 음(-) 산의 바닥이 3파 쪽보다 얕음)
"""
import time
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import strategy as S
import score_backtest as SB
import stop_backtest as SBT
from ma_backtest import log

P1 = ["A 5파", "B 5파+후행", "C 5파+시장"]
P2 = ["추적20", "데드", "후행캔들", "후행20", "MACD0", "MACD데드", "2/3"]
MODES = [f"{a} | {b}" for a in P1 for b in P2]
BASE = "A 5파 | 추적20"
BUY_COST, SELL_COST = SBT.BUY_COST, SBT.SELL_COST


def path(O, H, L, C, ei, stop, mode):
    t = SBT._tl
    p1, p2 = [x.strip() for x in mode.split("|")]
    ma5, ma20, macd, sig = t.ma5, t.ma20, t.macd, t.sig
    entry = O[ei]
    n = len(C)
    last = min(n - 1, ei + S.HOLD_MAX)
    held, cash, tp, peak, tp_j = 1.0, -BUY_COST, False, entry, None
    ups = [m[ei] for m in t.res if np.isfinite(m[ei]) and m[ei] >= entry * 1.10]
    rline = min(min(ups), entry * 1.5) if ups else entry * 1.5
    vals = []

    def lag(a, j):
        return a[j - 26] if j - 26 >= 0 else np.nan

    def chik_under(j):
        v = lag(L, j)
        return np.isfinite(v) and C[j] < v

    def cnt(j):
        return int(ma5[j] < ma20[j]) + int(chik_under(j)) + int(macd[j] < sig[j])

    for j in range(ei, last + 1):
        peak = max(peak, H[j])
        if not tp and H[j] >= rline:
            px = max(O[j], rline) if j > ei else rline
            cash += 0.5 * px / entry * (1 - SELL_COST); held = 0.5; tp = True; tp_j = j
        c = C[j]
        why = None
        # 1단계 규칙 (익절 뒤에도 손절선은 유지)
        below5 = c < stop
        if p1 == "A 5파":
            if below5: why = "5파 이탈"
        elif p1 == "B 5파+후행":
            if below5: why = "5파 이탈"
            elif not tp and j > ei and chik_under(j) and not chik_under(j - 1): why = "후행스팬 캔들 아래"
        elif p1 == "C 5파+시장":
            mr = t.mrsi[j]
            if below5 and not (np.isfinite(mr) and mr < 30): why = "5파 이탈"
            elif c < entry * 0.80: why = "비상 -20%"
        # 2단계 규칙 (익절 다음 날부터)
        if why is None and tp and j > tp_j:
            if p2 == "추적20" and c < peak * 0.80: why = "추적 -20%"
            elif p2 == "데드" and ma5[j] < ma20[j] and ma5[j - 1] >= ma20[j - 1]: why = "데드크로스"
            elif p2 == "후행캔들" and chik_under(j) and not chik_under(j - 1): why = "후행스팬 캔들 아래"
            elif p2 == "후행20":
                m26, m27 = lag(ma20, j), lag(ma20, j - 1)
                if np.isfinite(m26) and c < m26 and not (np.isfinite(m27) and C[j - 1] < m27): why = "후행스팬 20선 아래"
            elif p2 == "MACD0" and macd[j] < 0 <= macd[j - 1]: why = "MACD 0선 아래"
            elif p2 == "MACD데드" and macd[j] < sig[j] and macd[j - 1] >= sig[j - 1]: why = "MACD 데드크로스"
            elif p2 == "2/3" and cnt(j) >= 2 and cnt(j - 1) < 2: why = "3개 중 2개"
        if why or j == last:
            cash += held * c / entry * (1 - SELL_COST)
            vals.append(cash)
            return np.array(vals), why or "기간 만료", j
        vals.append(held * c / entry + cash)
    return np.array(vals), "기간 만료", last


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in t.items() if k not in MODES} for t in sigs])
    for m in MODES:
        df[m] = [t[m]["ret"] for t in sigs]
        df[m + " 일수"] = [len(t[m]["vals"]) for t in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    years = sorted(df.year.unique())
    groups = {"예전 A + 새 점수 강": df.old_a & df.strong, "예전 A": df.old_a}
    L = ["# 혼합 매도 — 1단계(익절 전) 손절 × 2단계(절반 익절 뒤) 수익 관리", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 · 신호 {len(df)}건",
         f"- 기준(지금 방식): {BASE}", ""]
    for gname, m in groups.items():
        g = df[m]
        L += [f"## {gname} ({len(g)}건)", "",
              "| 1단계 | 2단계 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF | 보유일 | 하루당 | 해마다 PF 1↑ |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for md in MODES:
            s = SB.stat(g[md])
            dd = g[md + " 일수"]
            ok = sum(1 for y in years if (lambda v: v and v["n"] >= 5 and v["pf"] >= 1)(SB.stat(g[g.year == y][md])))
            tot = sum(1 for y in years if (lambda v: v and v["n"] >= 5)(SB.stat(g[g.year == y][md])))
            rows.append((s["avg"], md, s, dd, ok, tot))
        for avg, md, s, dd, ok, tot in sorted(rows, key=lambda r: -r[0]):
            a, b = md.split(" | ")
            L.append(f"| {a} | {b}{' ◀ 지금' if md == BASE else ''} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | "
                     f"{s['al']:+.1f}% | {SB.f2(s['rr'])} | {SB.f2(s['pf'])} | {dd.mean():.0f} | {g[md].sum() / dd.sum():.3f}% | {ok}/{tot} |")
        L.append("")
    L += ["## 매수 쪽: PPO 히스토그램 '산' 다이버전스 (지금 방식 매도 기준)", "", SB.HEAD]
    for gname, m in [("예전 A + 강", df.old_a & df.strong), ("예전 A", df.old_a), ("넓은 신호 전체", df.old_a | ~df.old_a)]:
        g = df[m]
        L += [SB.row(f"{gname} · PPO 다이버전스 있음", g[g.ppo_div == True][BASE]),
              SB.row(f"{gname} · PPO 다이버전스 없음", g[g.ppo_div == False][BASE]),
              SB.row(f"{gname} · 판단 못함(주봉 신호 등)", g[g.ppo_div.isna()][BASE])]
    L += ["", "- 계좌 결과(운 범위 포함)는 hybrid_trades.csv 로 따로 계산해요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    keep = ["code", "name", "market", "date", "year", "old_a", "strong", "new_score", "ppo_div", "macd_div", "mkt_rsi",
            "stop_dist", "dd52", "depth"] + MODES + [m + " 일수" for m in MODES]
    df[keep].to_csv(scan.STATE / "hybrid_trades.csv", index=False)
    log("보고서 저장")


def main():
    SBT.MODES = MODES
    SBT.path = path
    SBT.report = report
    SBT.main()


if __name__ == "__main__":
    main()
