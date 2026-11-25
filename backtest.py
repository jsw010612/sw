"""
"50~100% 오를 자리" 찾기 백테스트 (차트만)

1) 신호마다 "손절(5파 저점 종가 이탈)에 걸리기 전까지 최대 몇 %까지 올랐나"를 잼
   → +30% / +50% / +100% 도달률
2) 신호 시점의 조건별로 +50% 도달률을 비교 (어떤 조건이 큰 반등 자리였나)
3) 짧은 손절 + 추적 손절로 실제 매매했을 때 성과
4) 기준선: 단순히 "일봉 RSI가 30을 아래에서 위로 돌파하면 산다"와 비교

신호 종류 (저점 확정 2봉)
  1일 매수 : 1일봉 다이버전스 + 1주 RSI 40 이상
  1일 대기 : 1일봉 다이버전스 + 1주 RSI 40 미만
  주봉 매수: 주봉 다이버전스 + 1주 RSI 40 미만
  RSI 돌파 : 일봉 RSI 30 상향 돌파 (기준선, 손절은 최근 10일 최저가)

청산 (공통: 손절은 5파 저점 종가 이탈, 최대 250거래일)
  추적15 : +20% 넘은 뒤 최고가 대비 -15% 종가 이탈
  추적25 : +30% 넘은 뒤 최고가 대비 -25% 종가 이탈
  주봉20 : +20% 넘은 뒤 주봉 20주선 종가 이탈 (주말 기준)
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import Params, analyze_frame, rsi, to_month, to_week

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "1000") or 1000)
HOLD = 250
COST = 0.003
P = replace(Params(), piv_r=2)
OUT = scan.STATE
EXITS = ["추적15", "추적25", "주봉20"]


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def comp_score(snap, wpts):
    raw = (20 + (10 if snap["v3"] else 0) + (10 if snap["vd"] else 0) + (15 if snap["deep"] else 0)
           + (10 if snap["zone"] else 0) + (10 if snap["ok4"] else 0) + wpts)
    return int(round(raw * 100 / 90))


# ===== 시장(코스피) =====
def kospi():
    import re
    import requests
    try:
        r = requests.get("https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count=2500&requestType=0",
                         headers=scan.UA, timeout=20)
        rows = []
        for it in re.findall(r'data="([^"]+)"', r.content.decode("euc-kr", "ignore")):
            d, o, h, l, c, v = it.split("|")
            rows.append((pd.Timestamp(d), float(c)))
        if rows:
            s = pd.Series(dict(rows)).sort_index()
            log(f"코스피 지수 {len(s)}일 (네이버)")
            return s
    except Exception:
        pass
    try:
        import yfinance as yf
        d = yf.download("^KS11", period="10y", interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        s = d["Close"].dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        log(f"코스피 지수 {len(s)}일 (야후)")
        return s
    except Exception:
        log("코스피 지수 못 받음 → 시장 조건 없이 진행")
        return None


# ===== 매매 계산 =====
def run_trade(O, H, L, C, wk_ma_daily, ei, stop):
    """반환: 최대상승(손절 전), +30/50/100 도달, +50까지 일수, 청산별 수익"""
    entry = O[ei]
    n = len(C)
    last = min(n - 1, ei + HOLD)
    peak = entry
    mfe, d50 = 0.0, None
    stopped_at = None
    for j in range(ei, last + 1):
        if C[j] < stop:
            stopped_at = j
            break
        peak = max(peak, H[j])
        g = peak / entry - 1
        if g > mfe:
            mfe = g
        if d50 is None and g >= 0.5:
            d50 = j - ei
    res = {"mfe": round(mfe * 100, 1), "d50": d50, "hit30": mfe >= 0.3, "hit50": mfe >= 0.5, "hit100": mfe >= 1.0}
    # 청산 규칙별
    for ex in EXITS:
        peak, armed, out, days = entry, False, None, 0
        for j in range(ei, last + 1):
            if C[j] < stop:
                out = C[j]; days = j - ei; break
            peak = max(peak, H[j])
            g = peak / entry - 1
            if ex == "추적15":
                armed = armed or g >= 0.20
                if armed and C[j] < peak * 0.85:
                    out = C[j]; days = j - ei; break
            elif ex == "추적25":
                armed = armed or g >= 0.30
                if armed and C[j] < peak * 0.75:
                    out = C[j]; days = j - ei; break
            else:
                armed = armed or g >= 0.20
                m = wk_ma_daily[j]
                if armed and not np.isnan(m) and C[j] < m:
                    out = C[j]; days = j - ei; break
        if out is None:
            out = C[last]; days = last - ei
        res[ex] = round((out / entry - 1 - COST) * 100, 2)
        res[ex + "_d"] = days
    return res


def stock_events(s, start, kp):
    d1 = scan.naver_daily(s["code"], 2000, s["market"])
    if len(d1) < 400:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    r = rsi(C, P.rsi_len)
    hi250 = pd.Series(H).rolling(250, min_periods=60).max().to_numpy()
    wk = to_week(d1)
    # 주봉 20주선을 일봉 날짜에 붙이기 (그 주가 끝난 뒤에만 반영)
    wma = wk["close"].rolling(20).mean()
    wma.index = wma.index + pd.Timedelta(days=7)
    wk_ma_daily = wma.reindex(d1.index, method="ffill").to_numpy()
    kp_ok = None
    if kp is not None:
        k = kp.reindex(d1.index, method="ffill")
        kp_ok = (k > k.rolling(120).mean()).to_numpy()

    dev = []
    analyze_frame(d1, 1, P, dev)
    dev_idx = np.array([i for i, _ in dev]) if dev else np.array([], int)
    wev = []
    analyze_frame(wk, 1, P, wev)
    wr_all = rsi(wk["close"].to_numpy(float), P.rsi_len)
    w_det_days = []
    for j, _ in wev:
        k = d1.index.searchsorted(wk.index[j] + pd.Timedelta(days=6), side="right") - 1
        w_det_days.append(k)
    w_det_days = np.array(w_det_days) if w_det_days else np.array([], int)

    out = []

    def feats(det_i, ei, stop):
        mth = to_month(d1.iloc[:det_i + 1])
        mr = rsi(mth["close"].to_numpy(float), P.rsi_len)[-1] if len(mth) > P.rsi_len + 1 else np.nan
        h = hi250[det_i]
        dd = (1 - C[det_i] / h) * 100 if h and not np.isnan(h) else np.nan
        since_hi = det_i - int(np.argmax(H[max(0, det_i - 250):det_i + 1])) - max(0, det_i - 250) if det_i > 0 else 0
        n_d = int(((dev_idx <= det_i) & (dev_idx >= det_i - 120)).sum())
        w_recent = bool(((w_det_days <= det_i) & (w_det_days >= det_i - 40)).any())
        return dict(mrsi=None if np.isnan(mr) else round(float(mr), 1), dd52=None if np.isnan(dd) else round(dd, 1),
                    since_hi=int(since_hi), n_daily_div=n_d, weekly_div_recent=w_recent,
                    market_up=None if kp_ok is None else bool(kp_ok[det_i]),
                    chase=round((O[ei] / stop - 1) * 100, 2) if ei < n else None)

    def add(kind, det_i, stop, extra):
        ei = det_i + 1
        if ei >= n or d1.index[ei] < start or O[ei] <= stop:
            return
        rec = dict(code=s["code"], name=s["name"], cap=s.get("cap", 0), kind=kind,
                   date=d1.index[ei].strftime("%Y-%m-%d"), entry=round(O[ei], 2),
                   stop_dist=round((1 - stop / O[ei]) * 100, 2), **extra, **feats(det_i, ei, stop))
        rec.update(run_trade(O, H, L, C, wk_ma_daily, ei, stop))
        out.append(rec)

    for i, snap in dev:
        w = to_week(d1.iloc[:i + 1])
        wr = rsi(w["close"].to_numpy(float), P.rsi_len)[-1] if len(w) > P.rsi_len + 1 else np.nan
        low_w = (not np.isnan(wr)) and wr < scan.W_WAIT
        wpts = 0 if low_w else (15 if wr < 50 else 10)
        add("1일 대기" if low_w else "1일 매수", i, float(snap["l5p"]),
            dict(wrsi=None if np.isnan(wr) else round(float(wr), 1), score=comp_score(snap, wpts),
                 l3vr=round(float(snap["l3vr"]), 2), depth=round(float(snap["depth"]), 2)))
    for (j, snap), k in zip(wev, w_det_days):
        if np.isnan(wr_all[j]) or wr_all[j] >= scan.W_WAIT or k < 0:
            continue
        add("주봉 매수", int(k), float(snap["l5p"]),
            dict(wrsi=round(float(wr_all[j]), 1), score=comp_score(snap, 15),
                 l3vr=round(float(snap["l3vr"]), 2), depth=round(float(snap["depth"]), 2)))
    # 기준선: RSI 30 상향 돌파 (20일에 한 번까지)
    last_b = -99
    for i in range(1, n - 1):
        if r[i - 1] < 30 <= r[i] and i - last_b >= 20:
            last_b = i
            stop = float(L[max(0, i - 10):i + 1].min())
            w = None
            add("RSI 돌파(기준선)", i, stop, dict(wrsi=w, score=None, l3vr=None, depth=None))
    return out


# ===== 보고서 =====
def pct(x):
    return f"{x * 100:.1f}%" if pd.notna(x) else "-"


def trade_stats(g, ex):
    v = g[ex]
    w, l = v[v > 0], v[v <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    return dict(n=len(v), win=len(w) / len(v) if len(v) else np.nan, avg=v.mean(), aw=w.mean() if len(w) else 0,
                al=l.mean() if len(l) else 0, pf=pf, days=g[ex + "_d"].mean())


def hit_row(label, g):
    if len(g) == 0:
        return f"| {label} | 0 | - | - | - | - | - |"
    return (f"| {label} | {len(g)} | {pct(g.hit30.mean())} | **{pct(g.hit50.mean())}** | {pct(g.hit100.mean())} | "
            f"{g.mfe.median():.1f}% | {g['추적25'].mean():+.2f}% |")


HIT_HEAD = "| 구분 | 신호 수 | +30% 도달 | +50% 도달 | +100% 도달 | 최대상승 중간값 | 추적25 평균수익 |\n|---|---|---|---|---|---|---|"


def buckets(df, col, edges, labels):
    rows = []
    for (lo, hi), lab in zip(edges, labels):
        g = df[(df[col] >= lo) & (df[col] < hi)]
        rows.append(hit_row(lab, g))
    return rows


def portfolio(df, ex, risk=0.01, cap_w=0.20, total=0.70):
    tr = df.sort_values("date").to_dict("records")
    for t in tr:
        t["exit_date"] = (pd.Timestamp(t["date"]) + pd.tseries.offsets.BDay(int(t[ex + "_d"]))).strftime("%Y-%m-%d")
    ev = sorted([(t["date"], 1, k) for k, t in enumerate(tr)] + [(t["exit_date"], 0, k) for k, t in enumerate(tr)],
                key=lambda e: (e[0], e[1]))
    cash, op, taken = 100.0, {}, 0
    peak, mdd = 100.0, 0.0
    for d, typ, k in ev:
        t = tr[k]
        if typ == 0 and k in op:
            cash += op.pop(k) * (1 + t[ex] / 100)
            eq = cash + sum(op.values()); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(op.values())
            w = min(cap_w, risk / max(t["stop_dist"] / 100, 0.01))
            if sum(op.values()) / eq + w > total or eq * w > cash:
                continue
            cash -= eq * w; op[k] = eq * w; taken += 1
    for k, a in op.items():
        cash += a * (1 + tr[k][ex] / 100)
    cagr = ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100
    return cash, cagr, mdd * 100, taken


def report(df, n_stocks, t0):
    div = df[df.kind != "RSI 돌파(기준선)"]
    L = ["# 50~100% 자리 찾기 백테스트", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · 시총 상위 {n_stocks}종목 (현재 상장 기준) · 저점 확정 2봉",
         "- 도달률 = 손절(5파 저점 종가 이탈, 기준선은 최근 10일 최저가)에 걸리기 전에 그만큼 오른 비율 · 최대 250거래일", "",
         "## 1. 신호 종류별 큰 반등 도달률", HIT_HEAD]
    for k in ("주봉 매수", "1일 매수", "1일 대기", "RSI 돌파(기준선)"):
        L.append(hit_row(k, df[df.kind == k]))
    L.append(hit_row("다이버전스 전체", div))
    L += ["", "## 2. 짧은 손절 + 추적 손절로 매매했을 때",
          "| 신호 | 청산 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | PF | 보유일 |", "|---|---|---|---|---|---|---|---|---|"]
    for k in ("주봉 매수", "1일 매수", "1일 대기", "RSI 돌파(기준선)"):
        g = df[df.kind == k]
        if len(g) == 0:
            continue
        for ex in EXITS:
            s = trade_stats(g, ex)
            pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "∞"
            L.append(f"| {k} | {ex} | {s['n']} | {pct(s['win'])} | {s['avg']:+.2f}% | {s['aw']:+.2f}% | {s['al']:+.2f}% | {pf} | {s['days']:.0f} |")
    L += ["", "## 3. 어떤 조건에서 +50%가 많이 나왔나 (다이버전스 신호 전체)", "", "### 52주 고점 대비 하락폭", HIT_HEAD]
    L += buckets(div, "dd52", [(0, 30), (30, 45), (45, 60), (60, 75), (75, 101)], ["30% 미만", "30~45%", "45~60%", "60~75%", "75% 이상"])
    L += ["", "### 월봉 RSI", HIT_HEAD]
    L += buckets(div, "mrsi", [(0, 30), (30, 40), (40, 50), (50, 101)], ["30 미만", "30~40", "40~50", "50 이상"])
    L += ["", "### 주봉 RSI", HIT_HEAD]
    L += buckets(div, "wrsi", [(0, 30), (30, 40), (40, 50), (50, 101)], ["30 미만", "30~40", "40~50", "50 이상"])
    L += ["", "### 3파 거래량 (20일 평균 대비 배수, 투매 정도)", HIT_HEAD]
    L += buckets(div, "l3vr", [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 999)], ["1.5배 미만", "1.5~2.5배", "2.5~4배", "4배 이상"])
    L += ["", "### 겹침: 최근 120일 안 일봉 다이버전스 개수 / 최근 주봉 다이버전스", HIT_HEAD]
    L += [hit_row("일봉 다이버전스 1개", div[div.n_daily_div <= 1]), hit_row("일봉 다이버전스 2개 이상", div[div.n_daily_div >= 2]),
          hit_row("최근 8주 안 주봉 다이버전스 있음", div[div.weekly_div_recent == True]),
          hit_row("주봉 다이버전스 없음", div[div.weekly_div_recent == False]),
          hit_row("일봉 2개 이상 + 주봉 있음", div[(div.n_daily_div >= 2) & (div.weekly_div_recent == True)])]
    L += ["", "### 하락 기간 (52주 고점 이후 거래일)", HIT_HEAD]
    L += buckets(div, "since_hi", [(0, 60), (60, 120), (120, 200), (200, 999)], ["3개월 미만", "3~6개월", "6~10개월", "10개월 이상"])
    if div.market_up.notna().any():
        L += ["", "### 시장 (코스피가 120일선 위/아래)", HIT_HEAD,
              hit_row("코스피 120일선 위", div[div.market_up == True]), hit_row("코스피 120일선 아래", div[div.market_up == False])]
    L += ["", "### 종목 크기 (시총 순위)", HIT_HEAD]
    rk = div.cap.rank(ascending=False, method="first") if div.cap.sum() > 0 else None
    if rk is not None:
        q = rk / rk.max()
        L += [hit_row("상위 1/3 (대형)", div[q <= 1 / 3]), hit_row("중간 1/3", div[(q > 1 / 3) & (q <= 2 / 3)]),
              hit_row("하위 1/3 (중소형)", div[q > 2 / 3])]
    L += ["", "### 점수 · 진입 위치 · 5파 깊이", HIT_HEAD]
    L += buckets(div, "score", [(95, 999), (85, 95), (0, 85)], ["95점 이상", "85~94점", "85점 미만"])
    L += buckets(div, "chase", [(0, 2), (2, 5), (5, 10), (10, 999)], ["저점 +2% 이내 진입", "+2~5%", "+5~10%", "+10% 초과"])
    L += buckets(div, "depth", [(0, 1.272), (1.272, 1.618), (1.618, 99)], ["5파 깊이 1.27배 미만", "1.27~1.62배", "1.62배 이상"])
    # +50% 간 신호 vs 못 간 신호 특징 중간값
    a, b = div[div.hit50], div[~div.hit50]
    L += ["", "## 4. +50% 간 신호 vs 못 간 신호 (중간값)", "| 항목 | +50% 간 신호 | 못 간 신호 |", "|---|---|---|"]
    for col, lab in (("dd52", "52주 고점 대비 하락폭 %"), ("mrsi", "월봉 RSI"), ("wrsi", "주봉 RSI"), ("l3vr", "3파 거래량 배수"),
                     ("since_hi", "고점 이후 거래일"), ("n_daily_div", "120일 내 일봉 다이버전스 수"), ("stop_dist", "손절폭 %"),
                     ("chase", "진입가-손절가 %"), ("depth", "5파 깊이 배수"), ("score", "점수")):
        L.append(f"| {lab} | {a[col].median():.1f} | {b[col].median():.1f} |" if len(a) and len(b) else f"| {lab} | - | - |")
    L.append(f"| 신호 수 | {len(a)} | {len(b)} |")
    L += ["", "## 5. 계좌 시뮬레이션 (1회 손실 원금 1% · 종목당 최대 20% · 총 70%)",
          "| 신호 | 청산 | 최종 자산 (원금 100) | 연평균 | 최대 낙폭 | 들어간 거래 |", "|---|---|---|---|---|---|"]
    for k, g in (("주봉 매수", df[df.kind == "주봉 매수"]), ("1일 매수", df[df.kind == "1일 매수"]),
                 ("주봉+1일 매수", df[df.kind.isin(["주봉 매수", "1일 매수"])]), ("1일 대기", df[df.kind == "1일 대기"]),
                 ("RSI 돌파(기준선)", df[df.kind == "RSI 돌파(기준선)"])):
        if len(g) == 0:
            continue
        for ex in EXITS:
            fin, cagr, mdd, tk = portfolio(g, ex)
            L.append(f"| {k} | {ex} | **{fin:.1f}** | {cagr:+.1f}% | {mdd:.1f}% | {tk} |")
    L += ["", "## 주의", "- **상장폐지된 종목은 빠져 있어서** 특히 큰 하락 후 반등 자리의 성과가 실제보다 좋게 나와요.",
          "- 3번 표는 조건 하나씩만 나눠 본 거예요. 칸이 작으면(신호 수 30개 미만) 우연일 수 있어요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (OUT / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(OUT / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    kp = kospi()
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    allr, done = [], [0]

    def job(s):
        try:
            return stock_events(s, start, kp)
        except Exception as e:
            log("오류", s["name"], e)
            return []
        finally:
            done[0] += 1
            if done[0] % 100 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            allr += r
    df = pd.DataFrame(allr)
    log(f"신호 {len(df)}건")
    if df.empty:
        (OUT / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
