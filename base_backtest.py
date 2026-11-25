"""
바닥 횡보 돌파 백테스트 (주봉 기준)

사용자가 실제로 큰 수익(2~10배)을 낸 자리를 규칙으로 옮긴 것
  1) 크게 빠진 종목: 최근 3년 최고가 대비 바닥이 -55% 이상
  2) 바닥에서 옆으로 기는 구간(박스): 최근 26주 고가/저가 폭이 좁고, 바닥이 8주 이상 안 깨짐
  3) 이평선 수렴: 주봉 5·10·20주선이 서로 가까이 모임 (납작해짐)
  4) 돌파: 주봉 종가가 박스 고점을 넘고, 거래량이 박스 평균의 N배 이상
  진입: 돌파 다음 주 첫 거래일 시가

비교하는 설정
  박스 폭 35% / 50%, 돌파 거래량 2배 / 3배, 손절 = 박스 바닥 / 돌파 주 저가
  청산 공통: 손절가 종가 이탈 · +50% 넘으면 최고 종가 대비 -30% 이탈 · 최대 2년
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

import scan
from analysis import to_week

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
BOX_W = 26          # 박스 길이 (주)
FLAT_W = 8          # 최근 몇 주 동안 바닥이 안 깨졌나
DD_MIN = 0.55       # 3년 최고가 대비 바닥 하락폭
MA_TIGHT = 0.10     # 5·10·20주선이 이 범위 안에 모여 있어야 함
ARM, GAP = 0.50, 0.30
HOLD_W = 104
BUY_COST, SELL_COST = 0.0005, 0.0025
CASES = {"025560": "미래산업", "263750": "필어비스", "090710": "휴림로봇", "225570": "넥슨게임즈"}

VARIANTS = []
for box, vol, stop in product((0.35, 0.50), (2.0, 3.0), ("박스 바닥", "돌파 주 저가")):
    VARIANTS.append((f"폭{int(box * 100)}%·거래량{vol:g}배·손절 {stop}", box, vol, stop))


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def simulate(O, H, L, C, ei, stop):
    """주봉으로 매매. 반환: 수익률%, 최대상승%, 보유 주수, 사유"""
    entry = O[ei]
    last = min(len(C) - 1, ei + HOLD_W)
    peak_c, mfe, reason = entry, 0.0, "기간 만료"
    j = ei
    for j in range(ei, last + 1):
        mfe = max(mfe, H[j] / entry - 1)
        if C[j] < stop:
            reason = "손절"
            break
        peak_c = max(peak_c, C[j])
        if peak_c >= entry * (1 + ARM) and C[j] < peak_c * (1 - GAP):
            reason = "추적 매도"
            break
    ret = C[j] / entry * (1 - SELL_COST) - 1 - BUY_COST
    return round(ret * 100, 2), round(mfe * 100, 1), j - ei, reason, j


def stock_signals(s, start):
    d1 = scan.naver_daily(s["code"], 2000, s["market"])
    if len(d1) < 400:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    wk = to_week(d1)
    n = len(wk)
    if n < 80:
        return []
    O, H, L, C, V = (wk[c].to_numpy(float) for c in ("open", "high", "low", "close", "volume"))
    ma = {k: pd.Series(C).rolling(k).mean().to_numpy() for k in (5, 10, 20, 60, 120)}
    hi156 = pd.Series(H).rolling(156, min_periods=52).max().to_numpy()
    out = []
    last_sig = {v[0]: -99 for v in VARIANTS}
    for t in range(BOX_W + 1, n - 1):
        if wk.index[t + 1] < start:
            continue
        a = t - BOX_W
        bh, bl = H[a:t].max(), L[a:t].min()
        if bl <= 0 or np.isnan(hi156[t - 1]):
            continue
        dd = 1 - bl / hi156[t - 1]
        if dd < DD_MIN:
            continue
        low_i = a + int(np.argmin(L[a:t]))
        if t - low_i < FLAT_W:                      # 최근에 바닥을 새로 깼으면 아직 횡보 아님
            continue
        m5, m10, m20 = ma[5][t - 1], ma[10][t - 1], ma[20][t - 1]
        if np.isnan(m20):
            continue
        spread = max(m5, m10, m20) / min(m5, m10, m20) - 1
        if spread > MA_TIGHT:
            continue
        if not C[t] > bh:
            continue
        vavg = V[a:t].mean()
        vr = V[t] / vavg if vavg > 0 else 0
        rng = bh / bl - 1
        above120 = not np.isnan(ma[120][t]) and C[t] > ma[120][t]
        for name, box, vol, stopk in VARIANTS:
            if rng > box or vr < vol or t - last_sig[name] < 8:
                continue
            ei = t + 1
            stop = bl if stopk == "박스 바닥" else L[t]
            if O[ei] <= stop:
                continue
            ret, mfe, held, reason, xj = simulate(O, H, L, C, ei, stop)
            last_sig[name] = t
            out.append(dict(variant=name, code=s["code"], name=s["name"], cap=s.get("cap", 0),
                            date=wk.index[ei].strftime("%Y-%m-%d"), exit=wk.index[xj].strftime("%Y-%m-%d"),
                            entry=round(O[ei], 2), stop=round(stop, 2), stop_dist=round((1 - stop / O[ei]) * 100, 1),
                            dd=round(dd * 100, 1), box=round(rng * 100, 1), vr=round(vr, 1), above120=above120,
                            ret=ret, mfe=mfe, weeks=held, reason=reason))
    return out


def stats(g):
    v = g.ret
    w, l = v[v > 0], v[v <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    return dict(n=len(g), win=(v > 0).mean() * 100, avg=v.mean(), med=v.median(), aw=w.mean() if len(w) else 0,
                al=l.mean() if len(l) else 0, pf=pf, h50=(g.mfe >= 50).mean() * 100, h100=(g.mfe >= 100).mean() * 100,
                wk=g.weeks.mean())


def row(lab, g):
    if len(g) == 0:
        return f"| {lab} | 0 | - | - | - | - | - | - | - | - |"
    s = stats(g)
    pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "∞"
    return (f"| {lab} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.1f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | {pf} | "
            f"{s['h50']:.0f}% | {s['h100']:.0f}% | {s['wk']:.0f} |")


HEAD = ("| 구분 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | PF | +50% 도달 | +100% 도달 | 평균 보유(주) |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def portfolio(g, risk=0.01, cap=0.20, max_open=8, total=0.80):
    tr = g.sort_values("date").to_dict("records")
    ev = sorted([(t["date"], 1, k) for k, t in enumerate(tr)] + [(t["exit"], 0, k) for k, t in enumerate(tr)],
                key=lambda e: (e[0], e[1]))
    cash, op, peak, mdd, taken = 100.0, {}, 100.0, 0.0, 0
    for d, typ, k in ev:
        t = tr[k]
        if typ == 0 and k in op:
            cash += op.pop(k) * (1 + t["ret"] / 100)
            eq = cash + sum(op.values()); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(op.values())
            w = min(cap, risk / max(t["stop_dist"] / 100, 0.02))
            if len(op) >= max_open or (sum(op.values()) / eq + w) > total or eq * w > cash:
                continue
            cash -= eq * w; op[k] = eq * w; taken += 1
    for k, a in op.items():
        cash += a * (1 + tr[k]["ret"] / 100)
    cagr = ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100
    return cash, cagr, mdd * 100, taken


def report(df, n, t0):
    L = ["# 바닥 횡보 돌파 백테스트 (주봉)", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 (현재 상장 기준)",
         f"- 조건: 3년 최고가 대비 바닥 -{DD_MIN * 100:.0f}% 이상 · 최근 {BOX_W}주 박스 · 바닥이 {FLAT_W}주 이상 안 깨짐 · "
         f"5·10·20주선 {MA_TIGHT * 100:.0f}% 안에 모임 · 박스 고점 종가 돌파 + 거래량 급증",
         f"- 청산: 손절가 종가 이탈 · +{ARM * 100:.0f}% 넘으면 최고 종가 대비 -{GAP * 100:.0f}% · 최대 2년 · 비용 왕복 0.3%",
         "- +50%/+100% 도달 = 보유 중 한 번이라도 그만큼 올랐던 비율", "",
         "## 1. 설정별 비교", HEAD]
    port = {}
    for name, *_ in VARIANTS:
        g = df[df.variant == name]
        L.append(row(name, g))
        if len(g):
            port[name] = portfolio(g)
    L += ["", "## 2. 계좌 시뮬레이션 (1회 손실 원금 1% · 종목당 최대 20% · 보유 8종목 · 총 80%, 원금 100)",
          "| 설정 | 최종 자산 | 연평균 | 최대 낙폭 | 들어간 거래 |", "|---|---|---|---|---|"]
    for k, (fin, cagr, mdd, tk) in port.items():
        L.append(f"| {k} | **{fin:.1f}** | {cagr:+.1f}% | {mdd:.1f}% | {tk} |")
    if port:
        best = max(port, key=lambda k: port[k][1] / max(abs(port[k][2]), 1))
        g = df[df.variant == best]
        L += ["", f"## 3. 위험 대비 가장 좋은 설정 상세: {best}", HEAD,
              row("전체", g), row("돌파 때 120주선 위", g[g.above120]), row("돌파 때 120주선 아래", g[~g.above120]),
              row("하락폭 55~70%", g[g.dd < 70]), row("하락폭 70% 이상", g[g.dd >= 70]),
              row("돌파 거래량 5배 이상", g[g.vr >= 5]), row("손절폭 15% 이하", g[g.stop_dist <= 15]),
              row("손절폭 15% 초과", g[g.stop_dist > 15])]
        L += ["", "### 연도별", HEAD]
        for y, gg in g.groupby(g.date.str[:4]):
            L.append(row(y, gg))
        L += ["", "### 수익 상위 15건", "| 진입 | 종목 | 수익 | 최대상승 | 보유(주) | 사유 |", "|---|---|---|---|---|---|"]
        for t in g.nlargest(15, "ret").itertuples():
            L.append(f"| {t.date} | {t.name} | {t.ret:+.0f}% | {t.mfe:+.0f}% | {t.weeks} | {t.reason} |")
    L += ["", "## 4. 보여 주신 사례 종목에서 나온 신호 (모든 설정)", "| 종목 | 설정 | 진입 | 수익 | 최대상승 | 사유 |", "|---|---|---|---|---|---|"]
    cs = df[df.code.isin(CASES)]
    if len(cs):
        for t in cs.sort_values(["code", "date"]).itertuples():
            L.append(f"| {t.name} | {t.variant} | {t.date} | {t.ret:+.0f}% | {t.mfe:+.0f}% | {t.reason} |")
    else:
        L.append("| (신호 없음) | | | | | |")
    L += ["", "## 주의", "- **상장폐지된 종목이 빠져 있어요.** 바닥에서 횡보하다 결국 폐지된 종목이 실제로는 꽤 있어서, 결과가 좋게 나와요. 실전에서는 재무 ⚠위험 필터가 꼭 필요해요.",
          "- 이런 종목은 테마주·투자경고 종목이 많아서 변동이 커요. 비중을 손절폭에 맞춰 작게 가져가야 해요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    have = {u["code"] for u in uni}
    for c, nm in CASES.items():
        if c not in have:
            uni.append({"code": c, "name": nm, "market": "KQ" if c != "025560" else "KS", "cap": 0})
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    allr, done = [], [0]

    def job(s):
        try:
            return stock_signals(s, start)
        except Exception as e:
            log("오류", s["name"], e)
            return []
        finally:
            done[0] += 1
            if done[0] % 200 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        for r in ex.map(job, uni):
            allr += r
    df = pd.DataFrame(allr)
    log(f"신호 {len(df)}건 (설정 전체 합)")
    if df.empty:
        (scan.STATE / "backtest.md").write_text("# 결과 없음 (조건을 만족한 신호가 없어요)", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
