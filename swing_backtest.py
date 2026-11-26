"""
단기 스윙(2~10일) 백테스트 — 바닥 신호가 없는 동안 노는 돈을 굴릴 셋업 찾기

대상: 최근 5년 · 20일 평균 거래대금 30억 원 이상인 날만 (단기는 체결이 중요해서)
진입: 신호가 나온 다음 날 시가 · 같은 종목·같은 셋업은 5일 안에 다시 안 잡음
비용: 왕복 0.5% (수수료·세금 0.3% + 체결 밀림 0.2%) — 단기는 비용이 수익을 크게 깎아서 보수적으로

셋업
  T1 추세 눌림      : 60일선 > 120일선 · 20일선 상승 중 · 저가가 20일선 1% 안에 닿고 종가는 20일선 위
  T2 RSI2 과매도    : 종가 > 120일선 · 2일 RSI 5 미만 (짧게 크게 빠진 날)
  T3 거래량 돌파    : 종가가 20일 최고가 돌파 · 거래량 20일 평균 2배↑ · 종가가 하루 범위 위쪽 30% 안
  T4 볼린저 하단    : 종가 > 120일선 · 종가 < 볼린저 하단
  T5 3일 하락 반전  : 종가 > 60일선 · 3일 연속 하락 뒤 양봉 · 종가 > 전날 고가
  T6 기준선 지지    : 구름 위 · 저가가 기준선 2% 안에 닿고 종가는 기준선 위
  T7 골든크로스     : 5일선이 20일선 위로 · 거래량 1.5배↑ · 종가 > 60일선
  T8 단기 다이버전스: 종가 > 60일선 · 최근 10일 저가가 그 전 10일 저가보다 낮은데 RSI14는 높음 · 오늘 양봉

매도 (목표·손절은 장중 가격에 닿으면, 같은 날 둘 다 닿으면 손절로 침)
  E1 +5% / -3% / 5일
  E2 +8% / -4% / 10일
  E3 +10% / -5% / 10일
  E4 2일 RSI 70↑ 또는 종가 > 5일선이면 다음 날 시가 · 손절 -5% · 10일
  E5 +5% 넘으면 최고가 -3% 추적 · 손절 -4% · 10일
인버스: 코스피·코스닥 지수가 하락 추세일 때만 인버스(1배)·곱버스(2배)를 들고 있는 방식 4가지
결과는 state/swing_trades.csv.gz 에 특징과 같이 저장 (조합 탐색·계좌 계산은 따로)
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import deep_backtest as DB
import exit_backtest as EB
import stop_backtest as SBT
from analysis import rsi, candle_code
from ma_backtest import log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
MIN_VALUE = 3e9
COST = 0.005
INV = []
SETUPS = ["T1 추세 눌림", "T2 RSI2 과매도", "T3 거래량 돌파", "T4 볼린저 하단", "T5 3일 하락 반전",
          "T6 기준선 지지", "T7 골든크로스", "T8 단기 다이버전스"]
EXITS = {"E1 +5/-3/5일": (0.05, 0.03, 5), "E2 +8/-4/10일": (0.08, 0.04, 10), "E3 +10/-5/10일": (0.10, 0.05, 10),
         "E4 RSI2·5일선": (None, 0.05, 10), "E5 추적 -3%": ("trail", 0.04, 10)}


def sim(O, H, L, C, r2, ma5, ei, mode):
    tp, sl, hold = EXITS[mode]
    e = O[ei]
    stop = e * (1 - sl)
    last = min(len(C) - 1, ei + hold - 1)
    peak = e
    for j in range(ei, last + 1):
        if L[j] <= stop:
            px = min(O[j], stop) if j > ei else stop
            return px / e - 1 - COST, j - ei + 1, "손절"
        if isinstance(tp, float) and H[j] >= e * (1 + tp):
            px = max(O[j], e * (1 + tp)) if j > ei else e * (1 + tp)
            return px / e - 1 - COST, j - ei + 1, "목표"
        if tp == "trail":
            peak = max(peak, H[j])
            if peak >= e * 1.05:
                stop = max(stop, peak * 0.97)
        if tp is None and (r2[j] > 70 or C[j] > ma5[j]) and j < len(C) - 1:
            return O[j + 1] / e - 1 - COST, j - ei + 2, "조건 청산"
    return C[last] / e - 1 - COST, last - ei + 1, "기간 만료"


def stock_trades(s, start, kg, mr20):
    d1 = EB._orig_daily(s["code"], 1500, s["market"])
    if len(d1) < 300:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C, V = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
    Cs, Hs, Ls, Vs = pd.Series(C), pd.Series(H), pd.Series(L), pd.Series(V)
    ma = {k: Cs.rolling(k).mean().to_numpy() for k in (5, 20, 60, 120)}
    vma = Vs.rolling(20).mean().to_numpy()
    val = (Cs * Vs).rolling(20).mean().to_numpy()
    r2 = np.asarray(rsi(C, 2), float)
    r14 = np.asarray(rsi(C, 14), float)
    sd = Cs.rolling(20).std().to_numpy()
    bbl, bbu = ma[20] - 2 * sd, ma[20] + 2 * sd
    hi20 = Hs.rolling(20).max().shift(1).to_numpy()
    ten = (Hs.rolling(9).max() + Ls.rolling(9).min()) / 2
    kij = ((Hs.rolling(26).max() + Ls.rolling(26).min()) / 2).to_numpy()
    sa = ((ten + pd.Series(kij)) / 2).shift(26).to_numpy()
    sb = ((Hs.rolling(52).max() + Ls.rolling(52).min()) / 2).shift(26).to_numpy()
    ctop = np.fmax(sa, sb)
    atr = pd.concat([Hs - Ls, (Hs - Cs.shift()).abs(), (Ls - Cs.shift()).abs()], axis=1).max(axis=1).rolling(14).mean().to_numpy()
    hi250 = Hs.rolling(250, min_periods=60).max().to_numpy()
    lo10 = Ls.rolling(10).min().to_numpy()
    first = max(d1.index.searchsorted(start), 130)
    k_g = kg.reindex(d1.index, method="ffill").to_numpy() if kg is not None else np.full(n, np.nan)
    m_r = mr20.reindex(d1.index, method="ffill").to_numpy() if mr20 is not None else np.full(n, np.nan)
    out, lastsig = [], {}
    for i in range(first, n - 2):
        if not (val[i] >= MIN_VALUE):
            continue
        c = C[i]
        rng = H[i] - L[i]
        cp = (c - L[i]) / rng if rng > 0 else 0.5
        hits = []
        if ma[60][i] > ma[120][i] and ma[20][i] > ma[20][i - 5] and L[i] <= ma[20][i] * 1.01 and c > ma[20][i]:
            hits.append("T1 추세 눌림")
        if c > ma[120][i] and r2[i] < 5:
            hits.append("T2 RSI2 과매도")
        if c > hi20[i] and V[i] >= 2 * vma[i] and cp >= 0.7:
            hits.append("T3 거래량 돌파")
        if c > ma[120][i] and c < bbl[i]:
            hits.append("T4 볼린저 하단")
        if c > ma[60][i] and C[i - 1] < C[i - 2] < C[i - 3] < C[i - 4] and c > O[i] and c > H[i - 1]:
            hits.append("T5 3일 하락 반전")
        if np.isfinite(ctop[i]) and c > ctop[i] and L[i] <= kij[i] * 1.02 and c > kij[i]:
            hits.append("T6 기준선 지지")
        if ma[5][i] > ma[20][i] and ma[5][i - 1] <= ma[20][i - 1] and V[i] >= 1.5 * vma[i] and c > ma[60][i]:
            hits.append("T7 골든크로스")
        if i >= 20 and c > ma[60][i] and c > O[i]:
            a, b = lo10[i], L[i - 19:i - 9].min()
            if a < b:
                ia = i - 9 + int(np.argmin(L[i - 9:i + 1]))
                ib = i - 19 + int(np.argmin(L[i - 19:i - 9]))
                if r14[ia] > r14[ib] + 2:
                    hits.append("T8 단기 다이버전스")
        if not hits:
            continue
        ei = i + 1
        feat = dict(code=s["code"], name=s["name"], market=s["market"], date=d1.index[ei].strftime("%Y-%m-%d"),
                    year=d1.index[ei].year, val_eok=round(val[i] / 1e8, 1), price=c,
                    r2=r2[i], r14=r14[i], v_ratio=V[i] / vma[i] if vma[i] else np.nan, close_pos=cp,
                    candle=candle_code(O, C, H, L, i), atr_pct=atr[i] / c * 100,
                    d_ma20=(c / ma[20][i] - 1) * 100, d_ma60=(c / ma[60][i] - 1) * 100, d_ma120=(c / ma[120][i] - 1) * 100,
                    ma_up=float(ma[20][i] > ma[60][i] > ma[120][i]), above_cloud=float(np.isfinite(ctop[i]) and c > ctop[i]),
                    bb_pos=(c - bbl[i]) / (bbu[i] - bbl[i]) if bbu[i] > bbl[i] else np.nan,
                    ret5=(c / C[i - 5] - 1) * 100, ret20=(c / C[i - 20] - 1) * 100,
                    from_high=(1 - c / hi250[i]) * 100, gap_next=(O[ei] / c - 1) * 100,
                    kgap=k_g[i], mkt_ret20=m_r[i], cap_rank=s.get("_rank", np.nan))
        for su in hits:
            if i - lastsig.get(su, -99) < 5:
                continue
            lastsig[su] = i
            rec = dict(feat, setup=su)
            for ex in EXITS:
                r, d, why = sim(O, H, L, C, r2, ma[5], ei, ex)
                rec[ex], rec[ex + " 일수"] = round(r * 100, 2), d
            out.append(rec)
    return out


INV_RULES = {
    "I1 120일선 아래": lambda k, m20, m60, m120: k < m120,
    "I2 역배열(20<60<120)": lambda k, m20, m60, m120: (m20 < m60) & (m60 < m120),
    "I3 60일선 아래 + 20일선 하락": lambda k, m20, m60, m120: (k < m60) & (m20 < m20.shift(5)),
    "I4 120일선 아래 + 20일선 아래": lambda k, m20, m60, m120: (k < m120) & (k < m20),
}


def inverse_test(idx, name):
    """지수 인버스(1배)·곱버스(2배): 조건이 켜진 다음 날부터 들고, 꺼지면 다음 날 정리. 운용보수 연 0.6%, 매매 비용 0.1%"""
    k = idx.dropna().sort_index()
    m20, m60, m120 = k.rolling(20).mean(), k.rolling(60).mean(), k.rolling(120).mean()
    r = k.pct_change().fillna(0)
    start = k.index[-1] - pd.Timedelta(days=int(365 * YEARS))
    rows = []
    for lab, f in INV_RULES.items():
        on = f(k, m20, m60, m120).astype(float).shift(1).fillna(0)          # 오늘 신호 → 내일 보유
        trade = on.diff().abs().fillna(0)
        for lev in (1, 2):
            dr = on * (-lev * r - 0.006 / 250) - trade * 0.001
            dr = dr[dr.index >= start]
            eq = (1 + dr).cumprod() * 100
            mdd = (eq / eq.cummax() - 1).min() * 100
            yrs = (eq.index[-1] - eq.index[0]).days / 365
            ye = eq.groupby(eq.index.year).last()
            yr = (ye / ye.shift(1).fillna(100) - 1) * 100
            onr = on[on.index >= start].mean() * 100
            rows.append((f"{name} {'인버스' if lev == 1 else '곱버스'} · {lab}", eq.iloc[-1],
                         ((eq.iloc[-1] / 100) ** (1 / yrs) - 1) * 100, mdd, onr, yr.to_dict()))
    bh = k[k.index >= start]
    return rows, (bh.iloc[-1] / bh.iloc[0] - 1) * 100


def stat(v):
    v = pd.Series(v, dtype=float)
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), aw=w.mean() if len(w) else 0,
                al=l.mean() if len(l) else 0, pf=w.sum() / abs(l.sum()) if l.sum() != 0 else float("inf"))


def report(df, n, t0):
    years = sorted(df.year.unique())
    L = ["# 단기 스윙 백테스트 (2~10일)", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 · 거래대금 30억↑ · 신호 {len(df)}건",
         "- 비용 왕복 0.5% 포함 · 하루당 = 평균 수익 ÷ 평균 보유일", ""]
    for su in SETUPS:
        g = df[df.setup == su]
        if not len(g):
            continue
        L += [f"## {su} ({len(g)}건)", "",
              "| 매도 | 승률 | 평균 | 평균 이익 | 평균 손실 | PF | 보유일 | 하루당 | " + " | ".join(map(str, years)) + " |",
              "|---|---|---|---|---|---|---|---|" + "---|" * len(years)]
        for ex in EXITS:
            s = stat(g[ex])
            dd = g[ex + " 일수"].mean()
            cells = []
            for y in years:
                t = stat(g[g.year == y][ex])
                cells.append("-" if not t or t["n"] < 20 else f"{t['pf']:.2f}")
            L.append(f"| {ex} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | {s['pf']:.2f} | "
                     f"{dd:.1f} | {s['avg'] / dd:+.3f}% | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 시장 상황별 (E3 기준, 전체 셋업)", "", "| 구분 | 거래 | 승률 | 평균 | PF |", "|---|---|---|---|---|"]
    for lab, m in [("코스피 120일선 위", df.kgap > 0), ("코스피 120일선 아래", df.kgap <= 0),
                   ("시장 20일 +5%↑", df.mkt_ret20 > 5), ("시장 20일 -5%↓", df.mkt_ret20 < -5)]:
        s = stat(df[m]["E3 +10/-5/10일"])
        if s:
            L.append(f"| {lab} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['pf']:.2f} |")
    if INV:
        L += ["", "## 인버스·곱버스 (지수 하락 추세일 때만 보유)", "",
              "운용보수 연 0.6% · 매매 비용 0.1% · 보유 비율 = 기간 중 들고 있던 날의 비율", ""]
        for (rows, bh), nm in INV:
            ys = sorted({y for r in rows for y in r[5]})
            L += [f"{nm} 그냥 보유: 5년 {bh:+.0f}%", "",
                  "| 방식 | 최종 | 연평균 | 최대 낙폭 | 보유 비율 | " + " | ".join(map(str, ys)) + " |",
                  "|---|---|---|---|---|" + "---|" * len(ys)]
            for lab, fin, cagr, mdd, onr, yr in rows:
                L.append(f"| {lab} | **{fin:.1f}** | {cagr:+.1f}% | {mdd:.1f}% | {onr:.0f}% | "
                         + " | ".join(f"{yr[y]:+.0f}%" if y in yr else "-" for y in ys) + " |")
            L.append("")
    L += ["", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "swing_trades.csv.gz", index=False, compression="gzip")
    log("보고서 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    for r, s in enumerate(uni, 1):
        s["_rank"] = r
    kp = DB.kospi()
    kg = (kp / kp.rolling(120).mean() - 1) * 100 if kp is not None else None
    mk = {"KS": kp, "KQ": SBT.index_series("KOSDAQ", "^KQ11")}
    mr = {m: ((x / x.shift(20) - 1) * 100 if x is not None else None) for m, x in mk.items()}
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    INV.clear()
    for m, nm in (("KS", "코스피"), ("KQ", "코스닥")):
        if mk.get(m) is not None and len(mk[m]) > 400:
            INV.append((inverse_test(mk[m], nm), nm))
    log(f"시작: {len(uni)}종목")
    rows, done = [], [0]

    def job(s):
        try:
            return stock_trades(s, start, kg, mr.get(s["market"]))
        except Exception as e:
            log("오류", s["name"], repr(e))
            return []
        finally:
            done[0] += 1
            if done[0] % 200 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            rows += r
    df = pd.DataFrame(rows)
    log(f"신호 {len(df)}건")
    if df.empty:
        (scan.STATE / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
