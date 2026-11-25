"""
상위 프레임 이평선 지지 백테스트

아이디어 (사용자 방식)
  - 주봉·월봉 60선, 120선 = 가장 중요한 지지선
  - 가격이 위에서 내려와 그 선 근처(±3%)에 닿으면 → 일봉에서 타점을 잡아 매수
  - 위쪽에 있는 주봉·월봉 60·120선 = 저항 → 거기서 매도

지지선 (완성된 주봉·월봉으로만 계산해서 미래 정보 없음)
  주60 · 주120 · 월60 · 월120

닿음 판정
  일봉 저가가 지지선 +3% 이내로 내려오고, 종가는 지지선 -3% 위
  20일 전에는 지지선보다 5% 이상 위에 있었음 (위에서 내려온 경우만)
  같은 선에 20일 안에 다시 닿은 건 새 신호로 안 셈

일봉 타점 (닿은 뒤 10일 안에 처음 나온 것)
  닿은 날 바로 · 5일선 회복 · 5·20 골든크로스 · 일목 기준선 돌파 · 전환선·기준선 교차 · 후행스팬 돌파 ·
  반전 캔들(망치형·상승장악형) · RSI 다이버전스

진입 시점 상태 (타점과 함께 기록해서 효과 비교)
  구름 위치(위/안/아래) · 구름 두께(얇음/두꺼움) · 5·20·60선 수렴 · 이평선 수렴 + 구름 위 ·
  최근 20일 안에 위쪽 저항(큰 이평선·구름 하단)에 맞고 밀린 이력

손절: 지지선 -3% 아래로 종가 이탈
매도 비교
  X1 저항 매도   : 위쪽에 있는 가장 가까운 주·월 60/120선에 닿으면 전량
  X2 추적        : +30% 넘으면 최고 종가 대비 -20%
  X3 절반+추적   : 위쪽 저항에서 절반, 나머지 추적
  공통 최대 120거래일 · 비용 왕복 0.3%
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import Params, analyze_frame, candle_code, to_month, to_week

YEARS = float(os.getenv("BT_YEARS", "5") or 5)
TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
NEAR, BREAK, FROM_ABOVE = 0.03, 0.03, 0.05
WINDOW = 10
HOLD = 120
BUY_COST, SELL_COST = 0.0005, 0.0025
LEVELS = ["주60", "주120", "월60", "월120"]
TRIGGERS = ["닿은 날 바로", "5일선 회복", "5·20 골든크로스", "기준선 돌파", "전환·기준 교차", "후행스팬 돌파",
            "반전 캔들", "RSI 다이버전스"]
EXITS = ["X1 저항 매도", "X2 추적", "X3 절반+추적"]
P = replace(Params(), piv_r=2)


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def htf_ma(d1, frame, n):
    """완성된 주봉/월봉 이평선을 일봉 날짜에 붙임 (그 봉이 끝난 다음부터 반영)"""
    g = to_week(d1) if frame == "W" else to_month(d1)
    ma = g["close"].rolling(n).mean()
    if frame == "W":
        ma.index = ma.index + pd.Timedelta(days=7)
    else:
        ma.index = ma.index + pd.offsets.MonthBegin(1)
    return ma.reindex(d1.index, method="ffill").to_numpy()


def sim(O, H, L, C, ei, stop, resist, mode):
    entry = O[ei]
    last = min(len(C) - 1, ei + HOLD)
    held, cash = 1.0, -BUY_COST
    peak, half = entry, False
    reason = "기간 만료"
    j = ei
    for j in range(ei, last + 1):
        if resist and mode in ("X1 저항 매도", "X3 절반+추적") and not half and H[j] >= resist:
            px = max(O[j], resist) if j > ei else resist
            if mode == "X1 저항 매도":
                cash += held * px / entry * (1 - SELL_COST); held = 0; reason = "저항 매도"; break
            cash += 0.5 * px / entry * (1 - SELL_COST); held = 0.5; half = True
        if C[j] < stop:
            reason = "지지 이탈 손절"; break
        peak = max(peak, C[j])
        if mode != "X1 저항 매도" and peak >= entry * 1.3 and C[j] < peak * 0.8:
            reason = "추적 매도"; break
    if held > 0:
        cash += held * C[j] / entry * (1 - SELL_COST)
    if half and reason == "기간 만료":
        reason = "저항 절반 후 만료"
    return round((cash - 1) * 100, 2), reason, j - ei


def stock_trades(s, start):
    d1 = scan.naver_daily(s["code"], 3200, s["market"])
    if len(d1) < 700:
        return []
    d1 = d1[~d1.index.duplicated()].sort_index()
    n = len(d1)
    O, H, L, C = (d1[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    lv = {"주60": htf_ma(d1, "W", 60), "주120": htf_ma(d1, "W", 120),
          "월60": htf_ma(d1, "M", 60), "월120": htf_ma(d1, "M", 120)}
    ma5 = pd.Series(C).rolling(5).mean().to_numpy()
    ma20 = pd.Series(C).rolling(20).mean().to_numpy()
    Hs, Ls = pd.Series(H), pd.Series(L)
    tenkan_s = (Hs.rolling(9).max() + Ls.rolling(9).min()) / 2
    kijun_s = (Hs.rolling(26).max() + Ls.rolling(26).min()) / 2
    tenkan, kijun = tenkan_s.to_numpy(), kijun_s.to_numpy()
    span_a = ((tenkan_s + kijun_s) / 2).shift(26).to_numpy()            # 오늘 위치에 그려지는 선행스팬1
    span_b = ((Hs.rolling(52).max() + Ls.rolling(52).min()) / 2).shift(26).to_numpy()
    cloud_top, cloud_bot = np.fmax(span_a, span_b), np.fmin(span_a, span_b)
    ma60 = pd.Series(C).rolling(60).mean().to_numpy()
    dev = []
    analyze_frame(d1, 1, P, dev)
    div_days = {i for i, _ in dev}
    first = d1.index.searchsorted(start)

    def trig_ok(name, j):
        if name == "닿은 날 바로":
            return True
        if j < 1:
            return False
        if name == "5일선 회복":
            return C[j] > ma5[j] and C[j - 1] <= ma5[j - 1] and ma5[j] > ma5[j - 1]
        if name == "5·20 골든크로스":
            return ma5[j] > ma20[j] and ma5[j - 1] <= ma20[j - 1]
        if name == "기준선 돌파":
            return C[j] > kijun[j] and C[j - 1] <= kijun[j - 1]
        if name == "전환·기준 교차":
            return tenkan[j] > kijun[j] and tenkan[j - 1] <= kijun[j - 1]
        if name == "후행스팬 돌파":
            return j >= 27 and C[j] > H[j - 26] and C[j - 1] <= H[j - 27]
        if name == "반전 캔들":
            return candle_code(O, C, H, L, j) in (2, 7)
        if name == "RSI 다이버전스":
            return j in div_days
        return False

    out = []
    for lname, lvl in lv.items():
        last_touch = -99
        for t in range(max(first - 1, 25), n - 2):
            v = lvl[t]
            if np.isnan(v) or np.isnan(lvl[t - 20]):
                continue
            if not (L[t] <= v * (1 + NEAR) and C[t] >= v * (1 - BREAK) and C[t - 20] > lvl[t - 20] * (1 + FROM_ABOVE)):
                continue
            if t - last_touch < 20:
                last_touch = t
                continue
            last_touch = t
            others = [lv[k][t] for k in LEVELS if k != lname and not np.isnan(lv[k][t])]
            confl = sum(1 for o in others if abs(o / v - 1) <= 0.05)
            for tg in TRIGGERS:
                for j in range(t, min(n - 2, t + WINDOW) + 1):
                    if C[j] < lvl[j] * (1 - BREAK):
                        break
                    if trig_ok(tg, j):
                        ei = j + 1
                        if d1.index[ei] < start:
                            break
                        entry = O[ei]
                        stop = lvl[j] * (1 - BREAK)
                        if entry <= stop:
                            break
                        ups = [lv[k][j] for k in LEVELS if not np.isnan(lv[k][j]) and lv[k][j] > entry * 1.03]
                        resist = min(ups) if ups else None
                        ct, cb = cloud_top[j], cloud_bot[j]
                        cpos = "?" if np.isnan(ct) else ("위" if C[j] > ct else "아래" if C[j] < cb else "안")
                        thick = (ct - cb) / C[j] * 100 if not np.isnan(ct) else np.nan
                        mm = [ma5[j], ma20[j], ma60[j]]
                        conv = (not any(np.isnan(mm))) and (max(mm) / min(mm) - 1) <= 0.05
                        fail = False
                        for k in range(max(1, j - 20), j):
                            if resist and H[k] >= resist * 0.98 and C[k] < resist:
                                fail = True; break
                            if not np.isnan(cloud_bot[k]) and C[k] < cloud_bot[k] <= H[k]:
                                fail = True; break
                        rec = dict(code=s["code"], name=s["name"], level=lname, trigger=tg,
                                   cloud=cpos, thick=None if np.isnan(thick) else round(thick, 1),
                                   conv=bool(conv), conv_cloud=bool(conv and cpos == "위"), rej=fail,
                                   date=d1.index[ei].strftime("%Y-%m-%d"), confl=confl,
                                   stop_dist=round((1 - stop / entry) * 100, 2),
                                   resist_gap=round((resist / entry - 1) * 100, 1) if resist else None,
                                   above_w120=bool(not np.isnan(lv["주120"][j]) and C[j] > lv["주120"][j]))
                        for ex in EXITS:
                            r, why, days = sim(O, H, L, C, ei, stop, resist, ex)
                            rec[ex], rec[ex + " 사유"], rec[ex + " 일수"] = r, why, days
                        out.append(rec)
                        break
    return out


def st(v):
    v = pd.Series(v).dropna()
    if not len(v):
        return None
    w, l = v[v > 0], v[v <= 0]
    pf = w.sum() / abs(l.sum()) if len(l) and l.sum() != 0 else float("inf")
    return dict(n=len(v), win=(v > 0).mean() * 100, avg=v.mean(), aw=w.mean() if len(w) else 0,
                al=l.mean() if len(l) else 0, pf=pf)


def row(lab, v, extra=""):
    s = st(v)
    if not s:
        return f"| {lab} | 0 | - | - | - | - | - |{extra}"
    pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "∞"
    return f"| {lab} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | {s['al']:+.1f}% | {pf} |{extra}"


HEAD = "| 구분 | 거래 | 승률 | 평균 | 평균 이익 | 평균 손실 | PF |\n|---|---|---|---|---|---|---|"


def portfolio(g, ex, risk=0.01, cap=0.20, max_open=8, total=0.80):
    tr = g.sort_values("date").to_dict("records")
    for t in tr:
        t["exit"] = (pd.Timestamp(t["date"]) + pd.tseries.offsets.BDay(int(t[ex + " 일수"]))).strftime("%Y-%m-%d")
    ev = sorted([(t["date"], 1, k) for k, t in enumerate(tr)] + [(t["exit"], 0, k) for k, t in enumerate(tr)],
                key=lambda e: (e[0], e[1]))
    cash, op, peak, mdd, taken = 100.0, {}, 100.0, 0.0, 0
    held_codes = set()
    for d, typ, k in ev:
        t = tr[k]
        if typ == 0 and k in op:
            cash += op.pop(k) * (1 + t[ex] / 100); held_codes.discard(t["code"])
            eq = cash + sum(op.values()); peak = max(peak, eq); mdd = min(mdd, eq / peak - 1)
        elif typ == 1:
            eq = cash + sum(op.values())
            w = min(cap, risk / max(t["stop_dist"] / 100, 0.02))
            if t["code"] in held_codes or len(op) >= max_open or (sum(op.values()) / eq + w) > total or eq * w > cash:
                continue
            cash -= eq * w; op[k] = eq * w; taken += 1; held_codes.add(t["code"])
    for k, a in op.items():
        cash += a * (1 + tr[k][ex] / 100)
    return cash, ((cash / 100) ** (1 / YEARS) - 1) * 100 if cash > 0 else -100, mdd * 100, taken


def report(df, n, t0):
    L = ["# 상위 프레임 이평선 지지 백테스트", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · 최근 {YEARS:g}년 · {n}종목 (현재 상장 기준) · 비용 왕복 0.3%",
         "- 지지선: 완성된 주봉·월봉의 60·120 이평선 · 위에서 내려와 ±3% 안에 닿으면 일봉 타점을 10일 동안 기다림",
         "- 손절: 지지선 -3% 종가 이탈 · 최대 120거래일", ""]
    L += ["## 1. 매도 방식 비교 (전체)", HEAD]
    for ex in EXITS:
        L.append(row(ex, df[ex]))
    best_ex = max(EXITS, key=lambda e: (st(df[e]) or {"pf": 0})["pf"])
    L += ["", f"이하 표는 **{best_ex}** 기준이에요.", "", "## 2. 지지선별", HEAD]
    for lv in LEVELS:
        L.append(row(lv, df[df.level == lv][best_ex]))
    L += ["", "## 3. 일봉 타점별", HEAD]
    for tg in TRIGGERS:
        L.append(row(tg, df[df.trigger == tg][best_ex]))
    L += ["", "## 4. 지지선 × 타점 (거래 30건 이상, PF 순)", "| 지지선 | 타점 | 거래 | 승률 | 평균 | PF |", "|---|---|---|---|---|---|"]
    combos = []
    for (lv, tg), g in df.groupby(["level", "trigger"]):
        s = st(g[best_ex])
        if s and s["n"] >= 30:
            combos.append((s["pf"], lv, tg, s))
    for pf, lv, tg, s in sorted(combos, reverse=True)[:12]:
        L.append(f"| {lv} | {tg} | {s['n']} | {s['win']:.0f}% | {s['avg']:+.2f}% | {pf:.2f} |")
    L += ["", "## 5. 겹침·추세", HEAD,
          row("다른 큰 이평선과 5% 안에 겹침", df[df.confl >= 1][best_ex]),
          row("겹침 없음", df[df.confl == 0][best_ex]),
          row("주120선 위에서 지지 (상승 추세)", df[df.above_w120][best_ex]),
          row("주120선 아래", df[~df.above_w120][best_ex]),
          row("위쪽 저항까지 15% 이상 여유", df[df.resist_gap >= 15][best_ex]),
          row("위쪽 저항까지 15% 미만", df[df.resist_gap < 15][best_ex]),
          row("위쪽 저항 없음", df[df.resist_gap.isna()][best_ex])]
    L += ["", "## 6. 일목·이평선 상태 (진입 시점)", HEAD,
          row("구름 위", df[df.cloud == "위"][best_ex]), row("구름 안", df[df.cloud == "안"][best_ex]),
          row("구름 아래", df[df.cloud == "아래"][best_ex]),
          row("구름 얇음 (가격의 3% 미만)", df[df.thick < 3][best_ex]), row("구름 두꺼움 (8% 이상)", df[df.thick >= 8][best_ex]),
          row("구름 아래 + 두꺼운 구름 (막히는 선)", df[(df.cloud == "아래") & (df.thick >= 8)][best_ex]),
          row("구름 아래 + 얇은 구름 (뚫기 쉬운 선)", df[(df.cloud == "아래") & (df.thick < 3)][best_ex]),
          row("5·20·60선 수렴 (5% 안)", df[df.conv][best_ex]), row("수렴 없음", df[~df.conv][best_ex]),
          row("이평선 수렴 + 구름 위", df[df.conv_cloud][best_ex]),
          row("최근 20일 저항 맞고 밀린 이력 있음", df[df.rej][best_ex]), row("밀린 이력 없음", df[~df.rej][best_ex])]
    L += ["", "## 7. 좋은 상태끼리 겹쳤을 때", HEAD]
    good = df[(df.cloud == "위") & (~df.rej)]
    L += [row("구름 위 + 밀린 이력 없음", good[best_ex]),
          row("구름 위 + 밀린 이력 없음 + 주120선 위", good[good.above_w120][best_ex]),
          row("구름 위 + 밀린 이력 없음 + 큰 이평선 겹침", good[good.confl >= 1][best_ex]),
          row("이평선 수렴 + 구름 위 + 밀린 이력 없음", df[df.conv_cloud & ~df.rej][best_ex])]
    L += ["", "## 8. 연도별", HEAD]
    for y, g in df.groupby(df.date.str[:4]):
        L.append(row(y, g[best_ex]))
    # 계좌 시뮬레이션: 상위 조합
    L += ["", "## 9. 계좌 시뮬레이션 (상위 조합 · 1회 손실 원금 1% · 종목당 20% · 보유 8 · 총 80%, 원금 100)",
          "| 지지선 | 타점 | 최종 자산 | 연평균 | 최대 낙폭 | 들어간 거래 |", "|---|---|---|---|---|---|"]
    for pf, lv, tg, s in sorted(combos, reverse=True)[:5]:
        fin, cagr, mdd, tk = portfolio(df[(df.level == lv) & (df.trigger == tg)], best_ex)
        L.append(f"| {lv} | {tg} | **{fin:.1f}** | {cagr:+.1f}% | {mdd:.1f}% | {tk} |")
    fin, cagr, mdd, tk = portfolio(df[df.trigger == "닿은 날 바로"], best_ex)
    L.append(f"| 전체 | 닿은 날 바로 (기준) | {fin:.1f} | {cagr:+.1f}% | {mdd:.1f}% | {tk} |")
    L += ["", "## 주의", "- 상장폐지 종목이 빠져 있어서 결과가 실제보다 좋게 나와요.",
          "- 여러 조합 중 가장 좋은 걸 고르면 과거에만 맞춘 결과일 수 있어요. 거래 수가 많고 연도별로 고르게 좋은 조합만 믿으세요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "backtest_trades.csv", index=False, encoding="utf-8-sig")
    log("보고서 저장: state/backtest.md")


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    start = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * YEARS))
    log(f"시작: {len(uni)}종목, {start.date()} 이후")
    allr, done = [], [0]

    def job(s):
        try:
            return stock_trades(s, start)
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
    log(f"거래 {len(df)}건")
    if df.empty:
        (scan.STATE / "backtest.md").write_text("# 결과 없음", encoding="utf-8")
        return
    report(df, len(uni), t0)


if __name__ == "__main__":
    main()
