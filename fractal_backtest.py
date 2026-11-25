"""
프랙탈(차트 모양 유사도) 검증

1) 패턴 만들기 (앞 기간): 이후 1년 안에 2배 이상 오른 자리의 "직전 52주 주봉 모양"을 모아
   비슷한 것끼리 3개 그룹으로 묶음 (k-평균). 각 그룹 평균 모양 = 패턴 ①②③
   모양 = 주봉 종가(로그)를 평균 0, 표준편차 1로 맞춘 52개 값 + 거래량 흐름 13개 값
2) 검증 (뒤 기간): 모든 종목을 4주마다 최근 52주 모양과 패턴의 유사도(상관계수)로 점수화
   → 유사도 높은 자리가 그 뒤 26주·52주 동안 실제로 더 많이 올랐는지 비교
3) 지금 각 패턴과 가장 닮은 종목 목록

결과: state/fractal.md · state/fractal_patterns.svg · state/fractal_now.json
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

import scan
from analysis import to_week

TOP_N = int(os.getenv("BT_TOP", "3000") or 3000)
WIN = 52
K = 3
STEP = 4
SPLIT_YEARS = float(os.getenv("FR_SPLIT", "3"))      # 최근 몇 년을 검증 기간으로 쓸지
OUT = scan.STATE


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def shape(c, v):
    """52주 종가·거래량 → 비교용 모양 벡터 (65개)"""
    lc = np.log(np.maximum(c, 1e-9))
    p = (lc - lc.mean()) / (lc.std() + 1e-9)
    lv = np.log1p(np.maximum(v, 0))
    vv = lv.reshape(13, 4).mean(axis=1)
    vv = (vv - vv.mean()) / (vv.std() + 1e-9)
    return np.concatenate([p, vv * 0.5])


def corr_many(X, C):
    """X(n,d) 와 패턴 C(k,d) 의 상관계수 (n,k)"""
    Xz = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    Cz = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-9)
    return Xz @ Cz.T / X.shape[1]


def kmeans(X, k, iters=60, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = np.argmax(corr_many(X, C), axis=1)
        newC = np.array([X[lab == i].mean(0) if (lab == i).any() else C[i] for i in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C, lab


def stock_data(s, split_date):
    d1 = scan.naver_daily(s["code"], 2500, s["market"])
    if len(d1) < 400:
        return None
    d1 = d1[~d1.index.duplicated()].sort_index()
    wk = to_week(d1)
    if len(wk) < WIN + 60:
        return None
    C, V, L = wk["close"].to_numpy(float), wk["volume"].to_numpy(float), wk["low"].to_numpy(float)
    H = wk["high"].to_numpy(float)
    idx = wk.index
    n = len(wk)
    winners, samples = [], []
    last_w = -99
    for t in range(WIN - 1, n):
        seg_c, seg_v = C[t - WIN + 1:t + 1], V[t - WIN + 1:t + 1]
        if (seg_c <= 0).any():
            continue
        fut = H[t + 1:t + 53]
        fut26 = H[t + 1:t + 27]
        g52 = fut.max() / C[t] - 1 if len(fut) >= 26 else np.nan
        g26 = fut26.max() / C[t] - 1 if len(fut26) >= 13 else np.nan
        ret26 = C[min(n - 1, t + 26)] / C[t] - 1 if t + 26 < n else np.nan
        # 패턴 학습용: 앞 기간, 이후 1년 2배 이상, 52주 저점 근처(+20% 이내), 26주에 한 번
        if idx[t] < split_date and not np.isnan(g52) and g52 >= 1.0 and C[t] <= L[t - WIN + 1:t + 1].min() * 1.2 \
                and t - last_w >= 26:
            winners.append(shape(seg_c, seg_v))
            last_w = t
        # 검증용: 뒤 기간, 4주마다
        if idx[t] >= split_date and (t % STEP == 0) and not np.isnan(g26):
            samples.append((shape(seg_c, seg_v), g26, g52, ret26, idx[t].strftime("%Y-%m-%d")))
    now = shape(C[-WIN:], V[-WIN:]) if (C[-WIN:] > 0).all() else None
    return dict(code=s["code"], name=s["name"], winners=winners, samples=samples, now=now,
                curve=(C[-WIN:] / C[-WIN:].max()).tolist())


def svg_patterns(C):
    W, Hh, pad = 720, 220, 20
    cols = ["#36d399", "#ffcf5a", "#7aa9ff"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh * K + 20}" font-family="sans-serif">',
             f'<rect width="100%" height="100%" fill="#0c0f14"/>']
    for i in range(K):
        y0 = i * Hh + 10
        p = C[i][:WIN]
        lo, hi = p.min(), p.max()
        pts = " ".join(f"{pad + j * (W - 2 * pad) / (WIN - 1):.1f},{y0 + 30 + (hi - v) / (hi - lo + 1e-9) * (Hh - 60):.1f}"
                       for j, v in enumerate(p))
        vol = C[i][WIN:]
        bars = "".join(
            f'<rect x="{pad + b * (W - 2 * pad) / 13:.1f}" y="{y0 + Hh - 25 - (v - vol.min()) / (vol.max() - vol.min() + 1e-9) * 20:.1f}" '
            f'width="{(W - 2 * pad) / 13 - 4:.1f}" height="{4 + (v - vol.min()) / (vol.max() - vol.min() + 1e-9) * 20:.1f}" fill="#5d6877"/>'
            for b, v in enumerate(vol))
        parts.append(f'<text x="{pad}" y="{y0 + 20}" fill="{cols[i]}" font-size="15" font-weight="bold">패턴 {"①②③"[i]} (52주 주봉 모양 · 아래는 거래량)</text>')
        parts.append(bars)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{cols[i]}" stroke-width="3"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    t0 = time.time()
    scan.probe_sources()
    uni = scan.fetch_universe()[:TOP_N]
    split = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * SPLIT_YEARS))
    log(f"시작: {len(uni)}종목 · 패턴 학습 {split.date()} 이전 · 검증 {split.date()} 이후")
    data, done = [], [0]

    def job(s):
        try:
            return stock_data(s, split)
        except Exception as e:
            log("오류", s["name"], e)
            return None
        finally:
            done[0] += 1
            if done[0] % 300 == 0:
                log(f"  {done[0]}/{len(uni)}")

    with ThreadPoolExecutor(scan.WORKERS) as ex:
        data = [d for d in ex.map(job, uni) if d]
    W = np.array([w for d in data for w in d["winners"]])
    log(f"학습용 2배 상승 사례 {len(W)}개")
    if len(W) < 30:
        (OUT / "fractal.md").write_text("# 프랙탈 검증\n\n학습할 사례가 너무 적어요.", encoding="utf-8")
        return
    Cn, lab = kmeans(W, K)
    order = np.argsort([-(lab == i).sum() for i in range(K)])
    Cn = Cn[order]
    counts = [(lab == i).sum() for i in order]
    (OUT / "fractal_patterns.svg").write_text(svg_patterns(Cn), encoding="utf-8")
    np.save(OUT / "fractal_patterns.npy", Cn)

    X = np.array([s[0] for d in data for s in d["samples"]])
    g26 = np.array([s[1] for d in data for s in d["samples"]])
    g52 = np.array([s[2] if not np.isnan(s[2]) else np.nan for d in data for s in d["samples"]])
    r26 = np.array([s[3] for d in data for s in d["samples"]])
    dates = np.array([s[4] for d in data for s in d["samples"]])
    S = corr_many(X, Cn)
    best = S.max(1)
    which = S.argmax(1)
    L = ["# 프랙탈(차트 모양 유사도) 검증", "",
         f"- 실행: {datetime.now(scan.KST).strftime('%Y-%m-%d %H:%M')} · {len(data)}종목",
         f"- 패턴 학습: {split.date()} 이전, 이후 1년 안에 2배 이상 오른 자리 {len(W)}개의 직전 52주 모양을 3개로 묶음",
         f"- 검증: {split.date()} 이후 4주마다 모든 종목 {len(X):,}개 시점 · 유사도 = 패턴과의 상관계수 (1에 가까울수록 닮음)",
         "", "![패턴 모양](fractal_patterns.svg)", "",
         "## 1. 패턴 크기", "| 패턴 | 학습 사례 수 |", "|---|---|"]
    for i, c in enumerate(counts):
        L.append(f"| {'①②③'[i]} | {c} |")
    base26, base_h50 = np.nanmean(r26) * 100, np.mean(g26 >= 0.5) * 100

    def line(lab_, m):
        if m.sum() == 0:
            return f"| {lab_} | 0 | - | - | - | - |"
        return (f"| {lab_} | {m.sum():,} | {np.mean(g26[m] >= 0.3) * 100:.1f}% | {np.mean(g26[m] >= 0.5) * 100:.1f}% | "
                f"{np.nanmean(np.where(np.isnan(g52[m]), np.nan, g52[m] >= 1.0)) * 100:.1f}% | {np.nanmean(r26[m]) * 100:+.1f}% |")
    HEAD = "| 구분 | 시점 수 | 26주 안 +30% | 26주 안 +50% | 52주 안 +100% | 26주 뒤 평균 수익 |\n|---|---|---|---|---|---|"
    L += ["", "## 2. 유사도가 높을수록 실제로 더 올랐나 (검증 기간)", HEAD, line("전체 (기준)", np.ones(len(X), bool))]
    qs = np.quantile(best, [0.5, 0.8, 0.95, 0.99])
    L += [line("유사도 하위 50%", best < qs[0]), line("50~80%", (best >= qs[0]) & (best < qs[1])),
          line("80~95%", (best >= qs[1]) & (best < qs[2])), line("상위 5%", (best >= qs[2]) & (best < qs[3])),
          line("상위 1%", best >= qs[3])]
    L += ["", f"(상위 5% 기준 유사도 ≥ {qs[2]:.2f}, 상위 1% ≥ {qs[3]:.2f})", "", "## 3. 패턴별 (각 패턴과 닮은 정도 상위 5%)", HEAD]
    for i in range(K):
        qi = np.quantile(S[:, i], 0.95)
        L.append(line(f"패턴 {'①②③'[i]} (그 패턴 유사도 ≥ {qi:.2f})", S[:, i] >= qi))
    L += ["", "## 4. 연도별 (유사도 상위 5% vs 전체)", "| 연도 | 상위 5% 26주 +50% | 전체 26주 +50% | 상위 5% 평균 수익 | 전체 평균 수익 |", "|---|---|---|---|---|"]
    yrs = np.array([d[:4] for d in dates])
    for y in sorted(set(yrs)):
        m, a = (yrs == y) & (best >= qs[2]), yrs == y
        if m.sum():
            L.append(f"| {y} | {np.mean(g26[m] >= 0.5) * 100:.1f}% | {np.mean(g26[a] >= 0.5) * 100:.1f}% | "
                     f"{np.nanmean(r26[m]) * 100:+.1f}% | {np.nanmean(r26[a]) * 100:+.1f}% |")
    # 지금 닮은 종목
    now = [(d["code"], d["name"], d["now"]) for d in data if d["now"] is not None]
    if now:
        Sn = corr_many(np.array([x[2] for x in now]), Cn)
        res = {}
        L += ["", "## 5. 지금 각 패턴과 가장 닮은 종목 (참고)"]
        for i in range(K):
            top = np.argsort(-Sn[:, i])[:15]
            res["①②③"[i]] = [dict(code=now[j][0], name=now[j][1], sim=round(float(Sn[j, i]), 3)) for j in top]
            L += [f"### 패턴 {'①②③'[i]}", "| 종목 | 유사도 |", "|---|---|"]
            L += [f"| {now[j][1]} ({now[j][0]}) | {Sn[j, i]:.2f} |" for j in top]
        (OUT / "fractal_now.json").write_text(json.dumps({"updated": datetime.now(scan.KST).isoformat(timespec="minutes"),
                                                          "threshold_top5": round(float(qs[2]), 3), "patterns": res},
                                                         ensure_ascii=False), encoding="utf-8")
    L += ["", "## 읽는 법", f"- 2번 표에서 **상위 5%·1%가 기준(전체)보다 확실히 높아야** 유사도가 쓸모 있는 거예요. 기준선: 26주 안 +50% {base_h50:.1f}%, 26주 뒤 평균 {base26:+.1f}%.",
          "- 상장폐지 종목이 빠져 있어서 숫자가 실제보다 좋게 나와요.", "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (OUT / "fractal.md").write_text("\n".join(L), encoding="utf-8")
    log("보고서 저장: state/fractal.md")


if __name__ == "__main__":
    main()
