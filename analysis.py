"""
단계별 RSI 다이버전스 분석 (트레이딩뷰 파인 스크립트와 같은 계산)

- 최상위 프레임: 1시간 RSI가 과매도(30 이하)에서 시작해서
  위 프레임이 과매도 부근(35 이하)이면 한 칸씩 올라감.
- 3파 저점: 최근 저점 중 RSI 과매도 + 거래량 급증, 그중 RSI가 가장 낮은 저점
- 5파 저점: 3파보다 가격은 낮고 RSI는 높은 저점 (+ 4파 반등폭의 1.272배 이상 하락)
- 점수(100점): 단계x6 + 다이버전스15 + 3파거래량10 + 거래량다이버전스10
                + 5파깊이15 + 피보구간10 + 4파되돌림10
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Params:
    rsi_len: int = 14
    os_lvl: float = 30.0
    near_lvl: float = 35.0
    piv_l: int = 5
    piv_r: int = 5
    min_gap: int = 5
    max_gap: int = 80
    w3_look: int = 40
    fresh_n: int = 10
    vol_win: int = 1
    vol_mult: float = 1.5
    req_v3: bool = True
    fib_a: float = 1.272
    fib_b: float = 1.618
    fib_tol: float = 0.05
    req_deep: bool = True
    r4_min: float = 0.236
    r4_max: float = 0.618


FRAMES = ["1시간", "4시간", "1일", "1주", "1달"]
BULL_CANDLE = ["-", "도지", "망치형(아래꼬리)", "장대양봉", "장대음봉", "양봉", "음봉", "상승장악형"]
BEAR_CANDLE = ["-", "도지", "유성형(위꼬리)", "장대음봉", "장대양봉", "음봉", "양봉", "하락장악형"]


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """파인 스크립트 ta.rma와 같음 (처음 n개는 단순평균으로 시작)"""
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) == 0:
        return out
    s = valid[0]
    if len(x) - s < n:
        return out
    out[s + n - 1] = np.mean(x[s:s + n])
    a = 1.0 / n
    for i in range(s + n, len(x)):
        xi = x[i] if not np.isnan(x[i]) else 0.0
        out[i] = a * xi + (1 - a) * out[i - 1]
    return out


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    ch = np.diff(close, prepend=np.nan)
    up = rma(np.where(np.isnan(ch), np.nan, np.maximum(ch, 0)), n)
    dn = rma(np.where(np.isnan(ch), np.nan, -np.minimum(ch, 0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r[np.isnan(up) | np.isnan(dn)] = np.nan
    return r


def candle_code(o, c, h, l, p) -> int:
    oo, cc, hh, ll = o[p], c[p], h[p], l[p]
    body = abs(cc - oo)
    rng = hh - ll
    lw = min(oo, cc) - ll
    uw = hh - max(oo, cc)
    po = o[p - 1] if p >= 1 else np.nan
    pc = c[p - 1] if p >= 1 else np.nan
    if rng <= 0:
        return 0
    if body <= rng * 0.1:
        return 1
    if lw >= body * 2 and uw <= body:
        return 2
    if cc > oo and pc < po and cc >= po and oo <= pc:
        return 7
    if cc > oo and body >= rng * 0.6:
        return 3
    if cc < oo and body >= rng * 0.6:
        return 4
    if cc > oo:
        return 5
    return 6


def analyze_frame(df: pd.DataFrame, direction: int, P: Params) -> dict:
    """한 프레임을 처음부터 끝까지 봉 단위로 훑어서 마지막 봉 시점의 상태를 돌려줌.
    direction = 1: 과매도(매수) / -1: 과매수(익절, 가격·RSI를 뒤집어서 같은 계산)"""
    res = {"r": np.nan, "act": False}
    if df is None or len(df) < P.rsi_len + P.piv_l + P.piv_r + 5:
        return res
    d = direction
    O = df["open"].to_numpy(float)
    H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float)
    C = df["close"].to_numpy(float)
    V = df["volume"].to_numpy(float)
    T = df.index
    lo = L if d == 1 else -H
    hi = H if d == 1 else -L
    op = O if d == 1 else -O
    cl = C if d == 1 else -C
    rr = rsi(C, P.rsi_len)
    r = rr if d == 1 else 100 - rr
    vma = pd.Series(V).rolling(20).mean().to_numpy()

    n = len(df)
    pivs = []  # dict(p, r, v, vr, idx, c)
    dv = False
    broken = False
    dI = None
    snap = {}

    for i in range(n):
        p = i - P.piv_r
        if p - P.piv_l >= 0:
            left = lo[p - P.piv_l:p]
            right = lo[p + 1:i + 1]
            if lo[p] < left.min() and lo[p] <= right.min():
                a = max(0, p - P.vol_win)
                b = min(i, p + P.vol_win)
                vpiv = float(np.mean(V[a:b + 1]))
                vmap = vma[p]
                vr = vpiv / vmap if vmap and not np.isnan(vmap) and vmap > 0 else 0.0
                cpiv = candle_code(op, cl, hi, lo, p)
                while pivs and p - pivs[0]["idx"] > P.max_gap:
                    pivs.pop(0)
                best = None
                for pv in pivs:
                    if p - pv["idx"] >= P.min_gap and pv["r"] <= P.os_lvl:
                        if best is None or pv["r"] < best["r"]:
                            best = pv
                if best is not None:
                    l3i = best["idx"]
                    seg = slice(l3i + 1, p + 1)
                    mn = lo[seg].min()
                    hh, hti = np.nan, None
                    if p - 1 >= l3i + 1:
                        k = int(np.argmax(hi[l3i + 1:p])) + l3i + 1
                        hh, hti = hi[k], k
                    h2 = hi[max(0, l3i - P.w3_look):l3i + 1].max()
                    l3p = best["p"]
                    is_div = lo[p] < l3p and r[p] > best["r"] and lo[p] <= mn
                    w3len = h2 - l3p
                    b4 = hh - l3p
                    retr4 = b4 / w3len if w3len > 0 and not np.isnan(b4) else np.nan
                    depth = (hh - lo[p]) / b4 if not np.isnan(b4) and b4 > 0 else np.nan
                    ok4 = (not np.isnan(retr4)) and P.r4_min <= retr4 <= P.r4_max
                    deep = (not np.isnan(depth)) and depth >= P.fib_a - P.fib_tol
                    zone = deep and depth <= P.fib_b + P.fib_tol
                    v3ok = best["vr"] >= P.vol_mult
                    vdv = vpiv < best["v"]
                    if is_div and (not P.req_v3 or v3ok) and (not P.req_deep or deep):
                        dv, broken, dI = True, False, i
                        snap = dict(
                            v3=v3ok, vd=vdv, deep=deep, zone=zone, ok4=ok4,
                            depth=depth, retr4=retr4,
                            l3p=l3p * d, l3t=T[l3i], l3r=best["r"], l3vr=best["vr"], l3c=best["c"],
                            h4=hh * d, h4t=T[hti] if hti is not None else None,
                            l5p=lo[p] * d, l5t=T[p], l5r=r[p], l5vr=vr, l5c=cpiv,
                            zt=(hh - b4 * P.fib_a) * d, zb=(hh - b4 * P.fib_b) * d,
                            _l5raw=lo[p],
                        )
                pivs.append(dict(p=lo[p], r=r[p], v=vpiv, vr=vr, idx=p, c=cpiv))
        if dv and not broken and lo[i] < snap["_l5raw"]:
            broken = True

    act = dv and not broken and dI is not None and (n - 1 - dI) <= P.fresh_n
    res["r"] = r[-1]
    res["act"] = bool(act)
    if snap:
        res.update({k: v for k, v in snap.items() if not k.startswith("_")})
    return res


def top_frame(rs, P: Params) -> int:
    """rs: 5개 프레임의 (방향 맞춘) RSI 마지막 값"""
    if not (rs[0] <= P.os_lvl):
        return -1
    t = 0
    for i in range(1, 5):
        v = rs[i]
        if v is not None and not np.isnan(v) and v <= P.near_lvl:
            t = i
            if v > P.os_lvl:
                break
        else:
            break
    return t


def score_of(stage: int, f: dict) -> int:
    if not f.get("act"):
        return 0
    return (stage * 6 + 15 + (10 if f.get("v3") else 0) + (10 if f.get("vd") else 0)
            + (15 if f.get("deep") else 0) + (10 if f.get("zone") else 0) + (10 if f.get("ok4") else 0))


def grade(s: int) -> str:
    return "강" if s >= 80 else "중" if s >= 60 else "약"


def evaluate(frames: list, direction: int, P: Params) -> dict:
    """frames: [1시간, 4시간, 1일, 1주, 1달] DataFrame 목록"""
    results = [analyze_frame(f, direction, P) for f in frames]
    rs = [x["r"] for x in results]
    top = top_frame(rs, P)
    out = {"top": top, "stage": top + 1, "rsi": [(v if direction == 1 else 100 - v) for v in rs]}
    if top < 0:
        out["score"] = 0
        return out
    f = results[top]
    out["frame"] = f
    out["score"] = score_of(top + 1, f)
    return out


# ===== 준비 점수 (신호가 완성되지 않아도 "곧 나올 만한 순서") =====
RSI_W = [1, 2, 3, 4, 5]


def _pivot_lows(lo, P):
    n = len(lo)
    out = []
    for p in range(P.piv_l, n - P.piv_r):
        left = lo[p - P.piv_l:p]
        right = lo[p + 1:p + P.piv_r + 1]
        if lo[p] < left.min() and lo[p] <= right.min():
            out.append(p)
    return out


def forming_state(df: pd.DataFrame, P: Params, spark_n: int = 80) -> dict:
    """현재 진행 중인 3파→4파→5파 구조를 지금 가격 기준으로 봄 (확정 전)"""
    if df is None or len(df) < 40:
        return {}
    L = df["low"].to_numpy(float)
    H = df["high"].to_numpy(float)
    C = df["close"].to_numpy(float)
    V = df["volume"].to_numpy(float)
    O = df["open"].to_numpy(float)
    T = df.index
    n = len(df)
    r = rsi(C, P.rsi_len)
    vma = pd.Series(V).rolling(20).mean().to_numpy()

    def vavg(p):
        a, b = max(0, p - P.vol_win), min(n - 1, p + P.vol_win)
        return float(np.mean(V[a:b + 1]))

    best = None
    for p in _pivot_lows(L, P):
        if n - 1 - p > P.max_gap or n - 1 - p < 2:
            continue
        if not (r[p] <= P.os_lvl):
            continue
        if best is None or r[p] < r[best]:
            best = p
    st = {"spark": [round(float(x), 2) for x in C[-spark_n:]], "spark_t0": str(T[max(0, n - spark_n)].date())}
    if best is None:
        return st
    l3 = best
    l3v = vavg(l3)
    l3vr = l3v / vma[l3] if vma[l3] and not np.isnan(vma[l3]) else 0.0
    k = int(np.argmax(H[l3 + 1:])) + l3 + 1
    h4 = H[k]
    seg_lo = L[k:]
    j = int(np.argmin(seg_lo)) + k
    cur_low = L[j]
    below = cur_low < L[l3]
    rsi_div = bool(below and r[j] > r[l3])
    b4 = h4 - L[l3]
    depth = (h4 - cur_low) / b4 if b4 > 0 else 0.0
    vol_shrink = bool(below and vavg(j) < l3v)
    base = max(0, n - spark_n)
    st.update(dict(
        l3p=float(L[l3]), l3t=T[l3], l3r=float(r[l3]), l3vr=float(l3vr),
        l3c=candle_code(O, C, H, L, l3),
        h4=float(h4), h4t=T[k], low=float(cur_low), lowt=T[j], low_r=float(r[j]),
        below=bool(below), rsi_div=rsi_div, vol_shrink=vol_shrink,
        v3=bool(l3vr >= P.vol_mult), depth=float(depth),
        target_hi=float(h4 - b4 * P.fib_a), target_lo=float(h4 - b4 * P.fib_b),
        i_l3=l3 - base, i_h4=k - base, i_low=j - base,
    ))
    return st


def readiness(rsis: list, fs: dict, P: Params) -> int:
    """준비 점수 100점: RSI 깊이 40 + 3파 거래량 15 + RSI 다이버전스 진행 15
    + 5파 진행도 15 + 거래량 감소 15"""
    tot = 0.0
    for w, v in zip(RSI_W, rsis):
        if v is None or np.isnan(v):
            continue
        tot += w * min(1.0, max(0.0, (45 - v) / 20))
    pts = 40 * tot / sum(RSI_W)
    if fs.get("l3p") is not None:
        pts += 15 if fs.get("v3") else 0
        pts += 15 if fs.get("rsi_div") else 0
        if fs.get("below"):
            pts += 15 * min(1.0, fs.get("depth", 0) / P.fib_a)
        pts += 15 if fs.get("vol_shrink") else 0
    return int(round(pts))


# ===== 프레임 만들기 =====
def _agg(df, rule, **kw):
    g = df.resample(rule, **kw)
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
        "close": g["close"].last(), "volume": g["volume"].sum(),
    })
    return out.dropna(subset=["open", "close"])


def to_4h(h1: pd.DataFrame) -> pd.DataFrame:
    # 한국장 기준 09:00~13:00, 13:00~ 두 덩어리
    return _agg(h1, "4h", origin="start_day", offset="9h")


def to_week(d1: pd.DataFrame) -> pd.DataFrame:
    return _agg(d1, "W-MON", label="left", closed="left")


def to_month(d1: pd.DataFrame) -> pd.DataFrame:
    return _agg(d1, "MS")
