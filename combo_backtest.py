"""
통합 백테스트 — 지금까지 검증된 신호만 묶어서 한 계좌로 굴려 보기

신호원
  바닥A        : 깊은 바닥 반등 A등급 (strategy.py 그대로) · 매도: 30% 절반 익절 + 최고가 -25% 추적
  바닥B        : 같은 전략 B등급 (지금은 관찰만 하는 등급) · 비교용
  다이버전스+거래량 : 파동·하모닉 셋업 자리에서 RSI 다이버전스 + 조정 때 거래량 감소
  다이버전스   : 파동·하모닉 셋업 자리에서 RSI 다이버전스 (전부)
  3파+거래량   : 엘리엇 3파 진입(단기) + 조정 때 거래량 감소
  뱃(스윙)     : 하모닉 뱃 (큰 파동)
  파동 계열 매도는 T(1차 목표 절반+본전, 2차 목표 전량)와 R(1차 절반+본전, 나머지 추적) 둘 다 봄

계좌 규칙 (복리)
  비중 = min(종목당 최대 비중, 1회 손실 한도 ÷ 손절폭) · 최대 비중은 1%→20%, 2%→30%, 3%→40% · 동시 보유 10 · 총 보유 85%
  같은 날 신호가 여러 개면 바닥A → 다이버전스+거래량 → 3파+거래량 → 뱃 → 다이버전스 순서
  1회 손실 한도 1% · 2% · 3% 비교
  낙폭은 청산 시점 기준이라 실제(매일 평가)보다 조금 작게 나올 수 있음
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
import wave_backtest as WB
from ma_backtest import log

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
CAPS = {0.01: 0.20, 0.02: 0.30, 0.03: 0.40}     # 손실 한도를 올리면 종목당 최대 비중도 같이 올림
PRIORITY = {"바닥A": 0, "다이버전스+거래량": 1, "3파+거래량": 2, "뱃(스윙)": 3, "다이버전스": 4, "바닥B": 5}

_orig_daily = scan.naver_daily
_cache, _lock = {}, threading.Lock()


def cached_daily(code, count=1500, market="KS"):
    with _lock:
        d = _cache.get(code)
    if d is None:
        d = _orig_daily(code, 3200, market)
        with _lock:
            _cache[code] = d
    return d.iloc[-count:] if count and len(d) > count else d


def stock_rows(s, start, kp):
    rows = []
    try:
        for t in DB.stock_signals(s, start, kp):
            if t["grade"] not in ("A", "B"):
                continue
            e = t["E2 절반+추적"]
            rec = dict(src="바닥" + t["grade"], code=t["code"], name=t["name"],
                       date=pd.Timestamp(t["date"]).strftime("%Y-%m-%d"), stop_dist=float(t["stop_dist"]),
                       days=len(e["vals"]) - 1, why=e["reason"])
            rec["T"] = rec["R"] = round(float(e["ret"]), 2)
            rows.append(rec)
        tr, _ = WB.stock_trades(s, start)
        for t in tr:
            srcs = []
            if t["rsi_div"]:
                srcs.append("다이버전스")
                if t["vol_dry"]:
                    srcs.append("다이버전스+거래량")
            if t["setup"] == "3파 진입" and t["deg"] == "단기" and t["vol_dry"]:
                srcs.append("3파+거래량")
            if t["setup"] == "뱃" and t["deg"] == "스윙":
                srcs.append("뱃(스윙)")
            for src in srcs:
                rows.append(dict(src=src, code=t["code"], name=t["name"], date=t["date"], stop_dist=t["stop_dist"],
                                 T=t["T 목표가"], R=t["R 절반+추적"], days_T=t["T 목표가 일수"],
                                 days_R=t["R 절반+추적 일수"], why=t["R 절반+추적 사유"]))
    finally:
        with _lock:
            _cache.pop(s["code"], None)
    return rows


# ───────────────────────── 통계 ─────────────────────────
def stat(v):
    v = pd.Series(v, dtype=float).dropna()
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    aw, al = (w.mean() if len(w) else 0.0), (l.mean() if len(l) else 0.0)
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    rr = aw / abs(al) if al else float("inf")
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), aw=aw, al=al, rr=rr, pf=pf)


def f2(x):
    return "∞" if x == float("inf") else f"{x:.2f}"


HEAD = ("| 구분 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF |\n"
        "|---|---|---|---|---|---|---|---|")


def row(lab, v):
    s = stat(v)
    if not s:
        return f"| {lab} | 0 | - | - | - | - | - | - |"
    return (f"| {lab} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | "
            f"{f2(s['rr'])} | {f2(s['pf'])} |")


# ───────────────────────── 계좌 ─────────────────────────
def account(g, ex, risk, cap=0.20, max_open=10, total=0.85):
    tr = g.copy()
    tr["ret"] = tr[ex]
    tr["hold"] = np.where(tr.src.str.startswith("바닥"), tr["days"],
                          tr["days_T"] if ex == "T" else tr["days_R"])
    tr["pri"] = tr.src.map(PRIORITY)
    tr = tr.sort_values(["date", "pri"]).to_dict("records")
    for t in tr:
        t["exit"] = (pd.Timestamp(t["date"]) + pd.tseries.offsets.BDay(int(t["hold"]))).strftime("%Y-%m-%d")
    ev = sorted([(t["date"], 1, t["pri"], k) for k, t in enumerate(tr)] +
                [(t["exit"], 0, 0, k) for k, t in enumerate(tr)], key=lambda e: (e[0], e[1], e[2]))
    cash, op, peak, mdd = 100.0, {}, 100.0, 0.0
    codes, ye, cur_y, taken = set(), {}, None, []
    for d, typ, _, k in ev:
        y = int(d[:4])
        if cur_y is not None and y != cur_y:
            ye[cur_y] = cash + sum(op.values())
        cur_y = y
        t = tr[k]
        if typ == 0 and k in op:
            cash += op.pop(k) * (1 + t["ret"] / 100); codes.discard(t["code"])
            eq = cash + sum(op.values()); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(op.values())
            w = min(cap, risk / max(t["stop_dist"] / 100, 0.02))
            if t["code"] in codes or len(op) >= max_open or (sum(op.values()) / eq + w) > total or eq * w > cash:
                continue
            cash -= eq * w; op[k] = eq * w; codes.add(t["code"]); taken.append(t)
    for k, a in op.items():
        cash += a * (1 + tr[k]["ret"] / 100)
    if cur_y is not None:
        ye[cur_y] = cash
    yr, prev = {}, 100.0
    for y in sorted(ye):
        yr[y] = (ye[y] / prev - 1) * 100; prev = ye[y]
    cagr = ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100
    s = stat([t["ret"] for t in taken])
    return dict(final=cash, cagr=cagr, mdd=mdd * 100, yr=yr, s=s,
                mix=pd.Series([t["src"] for t in taken]).value_counts().to_dict() if taken else {})


# ───────────────────────── 보고서 ─────────────────────────
def report(df, n, t0):
    years = sorted(int(y) for y in pd.to_datetime(df.date).dt.year.unique())
    df["year"] = pd.to_datetime(df.date).dt.year
    srcs = [s for s in PRIORITY if s in set(df.src)]
    L = ["# 통합 백테스트 — 검증된 신호만 묶은 계좌", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 (현재 상장 기준)",
         "- 손익비 = 평균 이익 ÷ 평균 손실 · PF = 이익 합 ÷ 손실 합 (1 넘어야 돈 범)", ""]

    L += ["## 1. 신호별 손익비 (거래 하나하나 기준)", ""]
    for ex, title in (("R", "R 절반+추적 (바닥은 30% 절반+추적)"), ("T", "T 목표가 (바닥은 같은 값)")):
        L += [f"### {title}", HEAD] + [row(s, df[df.src == s][ex]) for s in srcs] + [""]

    L += ["## 2. 신호별 연도별 PF (R 기준) — 해마다 1 넘는지", "",
          "| 신호 | " + " | ".join(map(str, years)) + " |", "|---|" + "---|" * len(years)]
    for s in srcs:
        g = df[df.src == s]
        cells = []
        for y in years:
            v = stat(g[g.year == y]["R"])
            cells.append("-" if not v or v["n"] < 5 else f"{f2(v['pf'])} ({v['n']})")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")

    combos = {
        "① 바닥A만": ["바닥A"],
        "② 바닥A + 다이버전스·거래량": ["바닥A", "다이버전스+거래량"],
        "③ ② + 3파·거래량": ["바닥A", "다이버전스+거래량", "3파+거래량"],
        "④ ③ + 뱃": ["바닥A", "다이버전스+거래량", "3파+거래량", "뱃(스윙)"],
        "⑤ 바닥A + 다이버전스 전부": ["바닥A", "다이버전스"],
        "⑥ 바닥A·B + 다이버전스·거래량": ["바닥A", "바닥B", "다이버전스+거래량"],
    }
    L += ["## 3. 계좌 시뮬레이션 (복리 · 원금 100 · 종목당 최대 20/30/40% · 보유 10 · 총 85%)", "",
          "목표: 연 30% = 5년 371 · 연 50% = 5년 759", "",
          "| 묶음 | 매도 | 1회 손실 | 최종 | 연평균 | 최대 낙폭 | 거래 | 승률 | 손익비 | PF | "
          + " | ".join(map(str, years)) + " |",
          "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(years)]
    best = None
    for name, ss in combos.items():
        g = df[df.src.isin(ss)].copy()
        g["pri"] = g.src.map(PRIORITY)
        g = g.sort_values("pri").drop_duplicates(["code", "date"])
        if g.empty:
            continue
        for ex in ("R", "T"):
            for rk in (0.01, 0.02, 0.03):
                a = account(g, ex, rk, cap=CAPS[rk])
                s = a["s"] or dict(n=0, win=0, rr=0, pf=0)
                cells = " | ".join(f"{a['yr'][y]:+.0f}%" if y in a["yr"] else "-" for y in years)
                L.append(f"| {name} | {ex} | {rk*100:.0f}% | **{a['final']:.1f}** | {a['cagr']:+.1f}% | "
                         f"{a['mdd']:.1f}% | {s['n']} | {s['win']:.0f}% | {f2(s['rr'])} | {f2(s['pf'])} | {cells} |")
                score = a["cagr"] + a["mdd"] * 0.5            # 수익 - 낙폭 절반 (낙폭은 음수)
                if a["mdd"] > -35 and (best is None or score > best[0]):
                    best = (score, name, ex, rk, a)
    L.append("")
    if best:
        _, name, ex, rk, a = best
        L += ["## 4. 추천 (연평균 − 최대낙폭÷2 가 가장 큰 것, 낙폭 -35% 안쪽)", "",
              f"- **{name} · 매도 {ex} · 1회 손실 {rk*100:.0f}%**",
              f"- 원금 100 → {a['final']:.1f} · 연평균 {a['cagr']:+.1f}% · 최대 낙폭 {a['mdd']:.1f}%",
              "- 연도별: " + " · ".join(f"{y} {v:+.0f}%" for y, v in a["yr"].items()),
              "- 실제 들어간 신호: " + " · ".join(f"{k} {v}건" for k, v in a["mix"].items()), ""]

    L += ["## 5. 신호별 수익 상위 5건", ""]
    for s in srcs:
        g = df[df.src == s].sort_values("R", ascending=False).head(5)
        L.append(f"**{s}** — " + " · ".join(f"{r['name']} {r.date[:7]} {r.R:+.0f}%" for _, r in g.iterrows()))
    L += ["", "## 주의",
          "- 상장폐지 종목이 빠져 있어서 실제보다 좋게 나와요. 손실 한도 3%는 이 오차도 3배로 커져요.",
          "- 바닥A는 1년에 몇 번 안 나와요. 신호가 없는 해는 현금으로 쉬는 것도 결과에 포함돼 있어요.",
          "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "combo_trades.csv", index=False)
    log("보고서 저장")


def main():
    t0 = time.time()
    scan.probe_sources()
    scan.naver_daily = cached_daily
    uni = scan.fetch_universe()[:TOP_N]
    kp = DB.kospi()
    log(f"코스피: {'있음' if kp is not None else '없음'}")
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    rows, done = [], [0]

    def job(s):
        try:
            return stock_rows(s, start, kp)
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
    for c in ("days", "days_T", "days_R"):
        if c not in df:
            df[c] = np.nan
    df["days"] = df["days"].fillna(0)
    df["days_T"] = df["days_T"].fillna(df["days"])
    df["days_R"] = df["days_R"].fillna(df["days"])
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
