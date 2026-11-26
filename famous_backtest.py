"""
유명 매매기법 검증 — 공개된 규칙 그대로 + 공식 숫자 바꿔가며, 상장폐지 종목 포함

기법 (원래 규칙을 일봉으로 옮김 · 신호 다음 날 시가 진입이 기본)
  LW 변동성 돌파 (래리 윌리엄스) : 오늘 시가 + K×(어제 고가-저가) 돌파 시 매수 → 다음 날 시가 매도
                                  K 0.3/0.5/0.7 · 이평 필터(시가 > 5일선) 있음/없음
  LW %R (래리 윌리엄스)          : 120일선 위 + 윌리엄스 %R(10) -90/-95 아래 → %R -20 위로 오면 매도 (최대 10일, 손절 -8%)
  LW Oops (래리 윌리엄스)        : 갭하락으로 어제 저가 아래 시가 → 어제 저가 회복 시 매수 → 당일 종가/다음 날 시가 매도
  터틀 (리처드 데니스)           : 20일 신고가 돌파 → 10일 신저가 이탈 매도 / 55일 → 20일 · 손절 2ATR
  미너비니 (트렌드 템플릿+VCP)   : 정배열(50>150>200) + 52주 저점 +25%↑ + 고점 -25% 이내 + 돌파 직전 20일 변동폭 15/25% 이하
                                  + 거래량 실린 20일 고가 돌파 → 손절 -8% · 50일선 이탈 매도
  다바스 박스                    : 52주 신고가 뒤 3일 박스 → 박스 상단 돌파 매수 · 박스 하단 손절 · 20일 신저가 이탈 매도
  와인스타인 2단계               : 150일선(≈30주선) 상향 돌파 + 150일선 상승 + 거래량 2배 → 150일선 -3% 이탈 매도
  홀리 그레일 (린다 라쉬케)      : ADX 30↑ 강한 추세에서 20일 지수이평 눌림 → 다음 날 전날 고가 돌파 매수 · 눌림 저점 손절 · 20일 고점 목표 (최대 20일)
  코너스 RSI2                    : 200일선 위 + 2일 RSI 5/10 아래 → 5일선 위 마감 시 매도
  볼린저 스퀴즈                  : 밴드 폭이 120일 중 최저 근처 → 상단 돌파 매수 · 20일선 이탈 매도
  모멘텀 (12-1개월)              : 매달 말 지난 12개월(최근 1개월 제외) 수익률 상위 20종목 균등 보유 1개월
  역모멘텀 (1개월 반전)          : 매달 말 지난 1개월 낙폭 상위 20종목 보유 1개월
대상: 그날 20일 평균 거래대금 30억↑ (모멘텀은 매달 기준) · 비용 왕복 0.4%
결과는 state/famous_trades.csv.gz 로도 저장 (계좌 계산은 따로)
"""
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import deep_backtest as DB
import exit_backtest as EB
import entry_backtest as ENB
from analysis import rsi
from ma_backtest import log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
COST = 0.004
MIN_VAL = 3e9
TRAIN_END = 2023
CL, VL = {}, {}
INFO = {}
_lock = threading.Lock()


def adx(H, L, C, n=14):
    Hs, Ls, Cs = pd.Series(H), pd.Series(L), pd.Series(C)
    up, dn = Hs.diff(), -Ls.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([Hs - Ls, (Hs - Cs.shift()).abs(), (Ls - Cs.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm).ewm(alpha=1 / n, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=1 / n, adjust=False).mean().to_numpy(), atr.to_numpy()


def run_exit(O, H, L, C, ei, entry, stop, kind, p, hold):
    """kind: nextopen / close / pr(%R) / lowN / ma / target / ma5"""
    n = len(C)
    last = min(n - 1, ei + hold)
    for j in range(ei, last + 1):
        if stop is not None and L[j] <= stop:
            px = min(O[j], stop) if j > ei else stop
            return px / entry - 1 - COST, j - ei + 1
        if kind == "close" and j == ei:
            return C[j] / entry - 1 - COST, 1
        if kind == "nextopen" and j == ei:
            return (O[j + 1] / entry - 1 - COST, 2) if j + 1 < n else (C[j] / entry - 1 - COST, 1)
        if kind == "pr" and p["pr"][j] > -20 and j + 1 < n:
            return O[j + 1] / entry - 1 - COST, j - ei + 2
        if kind == "lowN" and j > ei and C[j] < p["lowN"][j]:
            return C[j] / entry - 1 - COST, j - ei + 1
        if kind == "ma" and j > ei and C[j] < p["ma"][j] * p.get("mult", 1.0):
            return C[j] / entry - 1 - COST, j - ei + 1
        if kind == "ma5" and C[j] > p["ma"][j] and j + 1 < n:
            return O[j + 1] / entry - 1 - COST, j - ei + 2
        if kind == "target" and H[j] >= p["tgt"]:
            px = max(O[j], p["tgt"]) if j > ei else p["tgt"]
            return px / entry - 1 - COST, j - ei + 1
        if kind == "darvas" and j > ei and C[j] < p["lowN"][j]:
            return C[j] / entry - 1 - COST, j - ei + 1
    return C[last] / entry - 1 - COST, last - ei + 1


def stock_trades(s, start, kmap):
    if s.get("_dl"):
        d1, _ = ENB.delisted_daily(s["code"])
        if d1 is None:
            return []
    else:
        d1 = EB._orig_daily(s["code"], 1600, s["market"])
    if d1 is None or len(d1) < 320:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C, V = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
    Cs, Hs, Ls, Vs = pd.Series(C), pd.Series(H), pd.Series(L), pd.Series(V)
    with _lock:
        CL[s["code"]] = Cs.set_axis(d1.index).astype("float32")
        VL[s["code"]] = (Cs * Vs).set_axis(d1.index).astype("float32")
    ma = {k: Cs.rolling(k).mean().to_numpy() for k in (5, 20, 50, 60, 120, 150, 200)}
    ema20 = Cs.ewm(span=20, adjust=False).mean().to_numpy()
    val = (Cs * Vs).rolling(20).mean().to_numpy()
    vma50 = Vs.rolling(50).mean().to_numpy()
    vma5 = Vs.rolling(5).mean().to_numpy()
    hh = {k: Hs.rolling(k).max().shift(1).to_numpy() for k in (20, 55)}
    ll = {k: Ls.rolling(k).min().shift(1).to_numpy() for k in (10, 20)}
    lo252, hi252 = Ls.rolling(252, min_periods=200).min().to_numpy(), Hs.rolling(252, min_periods=200).max().to_numpy()
    r2 = np.asarray(rsi(C, 2), float)
    pr = ((Hs.rolling(10).max() - Cs) / (Hs.rolling(10).max() - Ls.rolling(10).min()) * -100).to_numpy()
    ad, atr = adx(H, L, C)
    sd = Cs.rolling(20).std()
    bw = (4 * sd / Cs.rolling(20).mean()).to_numpy()
    bw_min = pd.Series(bw).rolling(120).min().to_numpy()
    bbu = (Cs.rolling(20).mean() + 2 * sd).to_numpy()
    rng20 = ((Hs.rolling(20).max() - Ls.rolling(20).min()) / Cs).to_numpy()
    kup = kmap.reindex(d1.index, method="ffill").to_numpy() if kmap is not None else np.full(n, np.nan)
    first = max(d1.index.searchsorted(start), 260)
    out = []
    dates = d1.index
    liq = val >= MIN_VAL
    busy = {}

    def add(name, i, ei, entry, stop, kind, p, hold):
        if ei >= n or busy.get(name, -1) >= ei or not np.isfinite(entry) or entry <= 0:
            return
        r, d = run_exit(O, H, L, C, ei, entry, stop, kind, p, hold)
        busy[name] = ei if kind in ("nextopen", "close") else ei + d - 1
        out.append((name, s["code"], dates[ei].strftime("%Y-%m-%d"), dates[ei].year, round(r * 100, 3), d,
                    round((entry - stop) / entry * 100, 2) if stop else 8.0, round(val[i] / 1e8, 1),
                    float(kup[i]) if np.isfinite(kup[i]) else np.nan, bool(s.get("_dl"))))

    for i in range(first, n - 1):
        if not liq[i]:
            continue
        # 래리 윌리엄스 변동성 돌파: 당일 진입 (i+1 이 아니라 i 날 장중 돌파 → 다음 날 시가 매도)
        rg = H[i - 1] - L[i - 1]
        for k in (0.3, 0.5, 0.7):
            tgt = O[i] + k * rg
            if H[i] >= tgt and rg > 0:
                add(f"LW 변동성 돌파 K{k}", i, i, max(O[i], tgt), None, "nextopen", {}, 1)
                if O[i] > ma[5][i - 1]:
                    add(f"LW 변동성 돌파 K{k} +5일선", i, i, max(O[i], tgt), None, "nextopen", {}, 1)
        # Oops: 갭하락 후 어제 저가 회복
        if O[i] < L[i - 1] and H[i] >= L[i - 1]:
            add("LW Oops 당일 종가 매도", i, i, L[i - 1], None, "close", {}, 1)
            add("LW Oops 다음 날 시가 매도", i, i, L[i - 1], None, "nextopen", {}, 1)
        ei = i + 1
        # %R
        for th in (-90, -95):
            if C[i] > ma[120][i] and pr[i] < th:
                add(f"LW %R {th}", i, ei, O[ei], O[ei] * 0.92, "pr", {"pr": pr}, 10)
        # 터틀
        for en, ex in ((20, 10), (55, 20)):
            if H[i] > hh[en][i] and H[i - 1] <= hh[en][i - 1]:
                lvl = max(O[i], hh[en][i])
                add(f"터틀 {en}/{ex}", i, i, lvl, lvl - 2 * atr[i], "lowN", {"lowN": ll[ex]}, 250)
        # 미너비니
        tt = (C[i] > ma[50][i] > ma[150][i] > ma[200][i] and ma[200][i] > ma[200][i - 20]
              and C[i] >= 1.25 * lo252[i] and C[i] >= 0.75 * hi252[i])
        if tt and C[i] > hh[20][i] and V[i] > 1.5 * vma50[i]:
            for rr in (0.15, 0.25):
                if rng20[i - 1] <= rr:                # 돌파 직전 20일 변동폭(고가-저가)/종가
                    add(f"미너비니 VCP {int(rr * 100)}%", i, ei, O[ei], O[ei] * 0.92, "ma", {"ma": ma[50]}, 250)
        # 다바스: 52주 신고가 뒤 3일 고가 못 넘기면 박스 상단, 그 뒤 돌파
        if i >= 4 and H[i - 4] >= hi252[i - 4] * 0.999 and max(H[i - 3:i]) <= H[i - 4] and C[i] > H[i - 4]:
            bot = L[i - 4:i].min()
            add("다바스 박스", i, ei, O[ei], bot, "darvas", {"lowN": ll[20]}, 250)
        # 와인스타인
        if C[i] > ma[150][i] and C[i - 1] <= ma[150][i - 1] and ma[150][i] >= ma[150][i - 10] and vma5[i] > 2 * vma50[i]:
            add("와인스타인 2단계", i, ei, O[ei], O[ei] * 0.85, "ma", {"ma": ma[150], "mult": 0.97}, 250)
        # 홀리 그레일: 어제 눌림(저가가 EMA20 터치) + ADX>30 → 오늘 어제 고가 돌파 시 매수
        if ad[i - 1] > 30 and ema20[i - 1] > ema20[i - 6] and L[i - 1] <= ema20[i - 1] and H[i] > H[i - 1]:
            ent = max(O[i], H[i - 1])
            add("홀리 그레일", i, i, ent, L[i - 1], "target", {"tgt": max(hh[20][i], ent * 1.03)}, 20)
        # 코너스
        for th in (5, 10):
            if C[i] > ma[200][i] and r2[i] < th:
                add(f"코너스 RSI2<{th}", i, ei, O[ei], O[ei] * 0.90, "ma5", {"ma": ma[5]}, 15)
        # 볼린저 스퀴즈
        if bw[i - 1] <= bw_min[i - 1] * 1.1 and C[i] > bbu[i] and C[i - 1] <= bbu[i - 1]:
            add("볼린저 스퀴즈", i, ei, O[ei], O[ei] * 0.92, "ma", {"ma": ma[20]}, 120)
    return out


def momentum(start):
    codes = list(CL)
    if not codes:
        return []
    px = pd.DataFrame(CL).sort_index()
    vv = pd.DataFrame(VL).sort_index().rolling(20).mean()
    me = px.resample("ME").last().index
    me = [d for d in me if d >= start]
    res = []
    for nm, look, skip, asc in (("모멘텀 12-1개월", 252, 21, False), ("모멘텀 6-1개월", 126, 21, False),
                                ("역모멘텀 1개월 낙폭 상위", 21, 0, True)):
        rets = []
        prev = set()
        for a, b in zip(me[:-1], me[1:]):
            pa = px.loc[:a]
            if len(pa) < look + 5:
                continue
            p0, p1 = pa.iloc[-1 - skip], pa.iloc[-1 - look]
            score = (p0 / p1 - 1)
            liqd = vv.loc[:a].iloc[-1] >= MIN_VAL
            sc = score[liqd & p1.notna() & p0.notna()].dropna()
            if len(sc) < 40:
                continue
            pick = sc.sort_values(ascending=asc).head(20).index
            pb = px.loc[:b].iloc[-1][pick]
            pa_ = pa.iloc[-1][pick]
            r = (pb / pa_ - 1).fillna(-0.5)          # 중간에 데이터가 끊긴(상폐 등) 종목은 -50%
            turn = len(set(pick) - prev) / 20
            prev = set(pick)
            rets.append((b, float(r.mean()) - turn * COST))
        if rets:
            res.append((nm, pd.Series(dict(rets))))
    return res


def stat(v):
    v = pd.Series(v, dtype=float)
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), pf=w.sum() / abs(l.sum()) if l.sum() else float("inf"))


def report(df, mom, n, t0, kmap):
    years = sorted(df.year.unique())
    L = ["# 유명 매매기법 검증 (상장폐지 포함 · 비용 왕복 0.4% · 거래대금 30억↑)", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 (상폐 {INFO.get('dl', 0)}종목 포함) · 거래 {len(df)}건",
         f"- 배운 기간 ~{TRAIN_END} / 검증 기간 {TRAIN_END + 1}~ · 하루당 = 평균 ÷ 평균 보유일", "",
         "## 1. 기법별 (거래 하나하나)", "",
         "| 기법 | 거래 | 승률 | 평균 | PF | 배운 PF | 검증 PF | 보유일 | 하루당 | 코스피 120선 위 PF | 아래 PF | " + " | ".join(map(str, years)) + " |",
         "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(years)]
    rows = []
    for nm, g in df.groupby("name"):
        s = stat(g.ret)
        a, b = stat(g[g.year <= TRAIN_END].ret), stat(g[g.year > TRAIN_END].ret)
        up, dn = stat(g[g.kup > 0].ret), stat(g[g.kup <= 0].ret)
        cells = []
        for y in years:
            t = stat(g[g.year == y].ret)
            cells.append("-" if not t or t["n"] < 20 else f"{t['pf']:.2f}")
        f = lambda x: f"{x['pf']:.2f}" if x and x["n"] >= 20 else "-"
        rows.append((min(a["pf"] if a else 0, b["pf"] if b else 0),
                     f"| {nm} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['pf']:.2f} | {f(a)} | {f(b)} | "
                     f"{g.days.mean():.1f} | {s['avg'] / g.days.mean():+.3f}% | {f(up)} | {f(dn)} | " + " | ".join(cells) + " |"))
    L += [r for _, r in sorted(rows, key=lambda x: -x[0])]
    L += ["", "(배운 기간과 검증 기간 PF 중 낮은 쪽 기준으로 정렬 · 둘 다 1.2 넘어야 쓸 만함)", ""]
    if mom:
        L += ["## 2. 모멘텀 포트폴리오 (매달 20종목 균등 · 교체 비용 포함 · 상폐 종목 -50% 처리)", "",
              "| 방식 | 5년 결과 | 연평균 | 최대 낙폭 | 월 승률 | " + " | ".join(map(str, years)) + " |",
              "|---|---|---|---|---|" + "---|" * len(years)]
        for nm, r in mom:
            eq = (1 + r).cumprod() * 100
            yrs = max((r.index[-1] - r.index[0]).days / 365, 0.5)
            ye = eq.groupby(eq.index.year).last()
            yr = (ye / ye.shift(1).fillna(100) - 1) * 100
            L.append(f"| {nm} | **{eq.iloc[-1]:.0f}** | {((eq.iloc[-1] / 100) ** (1 / yrs) - 1) * 100:+.1f}% | "
                     f"{(eq / eq.cummax() - 1).min() * 100:.0f}% | {(r > 0).mean() * 100:.0f}% | "
                     + " | ".join(f"{yr[y]:+.0f}%" if y in yr else "-" for y in years) + " |")
        L.append("")
    L += [f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "famous_trades.csv.gz", index=False, compression="gzip")
    log("보고서 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    dl = ENB.fetch_delisted(start)
    have = {s["code"] for s in uni}
    for code, name, mk, dt, why in dl:
        if code not in have:
            uni.append({"code": code, "name": name, "market": mk, "_dl": True})
    INFO["dl"] = sum(1 for s in uni if s.get("_dl"))
    kp = DB.kospi()
    kmap = (kp / kp.rolling(120).mean() - 1) * 100 if kp is not None else None
    log(f"시작: {len(uni)}종목 (상폐 {INFO['dl']})")
    rows, done = [], [0]

    def job(s):
        try:
            return stock_trades(s, start, kmap)
        except Exception as e:
            log("오류", s.get("name"), repr(e))
            return []
        finally:
            done[0] += 1
            if done[0] % 300 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            rows += r
    df = pd.DataFrame(rows, columns=["name", "code", "date", "year", "ret", "days", "stop_dist", "val_eok", "kup", "delisted"])
    log(f"거래 {len(df)}건 · 모멘텀 계산")
    mom = momentum(start)
    report(df, mom, len(uni), t0, kmap)


if __name__ == "__main__":
    main()
