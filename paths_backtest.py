"""
경로 저장 — 바닥 신호마다 신호 전 60일 ~ 신호 후 260일의 가격·지표를 통째로 저장 (상장폐지 포함)
이후 매수·매도·비중 조합 수천 가지는 이 파일로 따로 계산 (Actions 다시 안 돌려도 됨)

저장 대상: 1일/주봉 다이버전스 중 52주 -40% 이상 + 5파 깊이 1.2 이상 (예전 A 를 넉넉히 포함하는 범위)
저장 내용 (state/paths.npz · 가격은 신호일 종가 대비 비율, float16)
  가격: 시가·고가·저가·종가·거래량
  지표: RSI14 · 5/20/60/120일선 · 주60/주120/월60/월120선(완성봉 기준) · MACD/시그널 · 기준선 · 구름 상단
        일봉/주봉 하락 다이버전스 표시
  메타 (state/paths_meta.csv): 종목·날짜·예전 A·새 점수·5파 저점·4파 고점·특징들·외국인/기관 수급(신호 직전 5/20일)
"""
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import stop_backtest as SBT
import timing_backtest as TB
import entry_backtest as ENB
import flow_backtest as FB
from ma_backtest import log

PRE, POST = 60, 260
COLS = ["open", "high", "low", "close", "volume", "rsi", "ma5", "ma20", "ma60", "ma120",
        "w60", "w120", "m60", "m120", "macd", "sig", "kijun", "ctop", "div_d", "div_w"]
PX_COLS = ["open", "high", "low", "close", "ma5", "ma20", "ma60", "ma120", "w60", "w120", "m60", "m120", "kijun", "ctop"]
ARR, META = [], []
_lock = threading.Lock()
_t = threading.local()


def prep(d1, t):
    from analysis import rsi
    C, H, L = (d1[c].to_numpy(float) for c in ("close", "high", "low"))
    Cs, Hs, Ls = pd.Series(C), pd.Series(H), pd.Series(L)
    if getattr(TB._t, "fid", None) != id(d1):
        TB.prep(d1)
    ten = (Hs.rolling(9).max() + Ls.rolling(9).min()) / 2
    kij = (Hs.rolling(26).max() + Ls.rolling(26).min()) / 2
    sa = ((ten + kij) / 2).shift(26)
    sb = ((Hs.rolling(52).max() + Ls.rolling(52).min()) / 2).shift(26)
    hi250 = Hs.rolling(250, min_periods=60).max().to_numpy()
    _t.full = np.column_stack([
        d1["open"].to_numpy(float), H, L, C, d1["volume"].to_numpy(float),
        np.asarray(rsi(C, 14), float), t.ma5, t.ma20, Cs.rolling(60).mean().to_numpy(), Cs.rolling(120).mean().to_numpy(),
        t.res[0], t.res[1], t.res[2], t.res[3], t.macd, t.sig, kij.to_numpy(), np.fmax(sa, sb).to_numpy(),
        TB._t.f["div_d"].astype(float), TB._t.f["div_w"].astype(float)]).astype(np.float32)
    _t.hi250 = hi250
    _t.fid = id(d1)


def extra(d1, i, snap, wb, ei, entry, l5, t):
    if getattr(_t, "fid", None) != id(d1):
        prep(d1, t)
    s = getattr(ENB._t, "s", {})
    out = dict(delisted=bool(s.get("_dl")), det_date=d1.index[i].strftime("%Y-%m-%d"))
    hi = _t.hi250[i]
    dd = (1 - d1["close"].iloc[i] / hi) * 100 if hi > 0 else 0
    depth = float(snap.get("depth") or 0)
    if dd < 40 or depth < 1.2:
        return out
    a, b = i - PRE, i + POST + 1
    blk = np.full((PRE + POST + 1, len(COLS)), np.nan, np.float32)
    lo = max(0, a)
    hi_ = min(len(d1), b)
    blk[lo - a:hi_ - a] = _t.full[lo:hi_]
    # 용량 줄이기: 가격류는 신호일 종가 대비 비율, 거래량은 20일 평균 대비 비율로 저장 (float16)
    base = float(_t.full[i, 3])
    vb = float(np.nanmean(_t.full[max(0, i - 19):i + 1, 4])) or 1.0
    for c in PX_COLS:
        blk[:, COLS.index(c)] /= base
    blk[:, COLS.index("volume")] /= vb
    for c in ("macd", "sig"):
        blk[:, COLS.index(c)] = blk[:, COLS.index(c)] / base * 100
    blk = blk.astype(np.float16)
    with _lock:
        k = len(ARR)
        ARR.append(blk)
    out.update(path_id=k, base_px=base, base_vol=vb, h4=float(snap.get("h4") or np.nan), l3p=float(snap["l3p"]), l5p=float(snap["l5p"]),
               l3i=int(snap["_l3i"]) - i if not wb and snap.get("_l3i") is not None else np.nan,
               l5i=int(snap["_l5i"]) - i if not wb and snap.get("_l5i") is not None else np.nan,
               last_day=hi_ - 1 - i)
    return out


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in s.items() if k not in SBT.MODES} for s in sigs])
    df = df[df.path_id.notna()].copy() if "path_id" in df else df.iloc[0:0]
    df["path_id"] = df.path_id.astype(int)
    df["year"] = pd.to_datetime(df.date).dt.year
    log(f"저장 대상 {len(df)}건 · 수급 받는 중")
    # 수급: 네이버 모바일 매매동향 (신호일 기준 25거래일)
    from concurrent.futures import ThreadPoolExecutor
    def one(r):
        try:
            rows = FB._trend_api(r.code, r.det_date) or {}
        except Exception:
            rows = {}
        if not rows:
            return r.Index, None
        f = pd.DataFrame(rows, index=["close", "ins", "frg"]).T.sort_index().loc[:pd.Timestamp(r.det_date)]
        if len(f) < 20:
            return r.Index, None
        blk = ARR[r.path_id].astype(np.float32)
        base = float(np.nanmean(blk[PRE - 19:PRE + 1, 3] * blk[PRE - 19:PRE + 1, 4])) * r.base_px * r.base_vol
        if not base:
            return r.Index, None
        fv, iv = f.frg * f.close, f.ins * f.close
        return r.Index, dict(frg5=fv.tail(5).sum() / base, frg20=fv.tail(20).sum() / base,
                             ins5=iv.tail(5).sum() / base, ins20=iv.tail(20).sum() / base)
    feats = {}
    with ThreadPoolExecutor(6) as ex:
        for idx, fe in ex.map(one, df.itertuples()):
            if fe:
                feats[idx] = fe
    for k in ("frg5", "frg20", "ins5", "ins20"):
        df[k] = [feats.get(i, {}).get(k, np.nan) for i in df.index]
    arr = np.stack(ARR) if ARR else np.zeros((0, PRE + POST + 1, len(COLS)), np.float16)
    np.savez_compressed(scan.STATE / "paths.npz", arr=arr, cols=np.array(COLS), pre=PRE, scaled=1,
                        px_cols=np.array(PX_COLS))
    df.to_csv(scan.STATE / "paths_meta.csv", index=False)
    L = ["# 경로 저장", "", f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 · 저장 {len(df)}건 "
         f"(예전 A {int(df.old_a.sum())} · 상폐 {int(df.delisted.sum())}) · 수급 {len(feats)}건",
         f"- 배열 크기 {arr.shape} · 파일 state/paths.npz, state/paths_meta.csv", "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    log("저장 끝")


def main():
    since = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * SBT.YEARS))
    dl = ENB.fetch_delisted(since)
    _fu = scan.fetch_universe

    def uni():
        u = _fu()
        have = {s["code"] for s in u}
        for code, name, mk, dt, why in dl:
            if code not in have:
                u.append({"code": code, "name": name, "market": mk, "_dl": True})
        return u
    scan.fetch_universe = uni
    SBT.TOP_N = 100000
    SBT.MODES = ["기준"]
    SBT.path = TB.path
    SBT.report = report
    SBT.EXTRA = lambda *a: {**TB.extra(*a), **extra(*a)}
    SBT.stock_sigs = ENB.stock_sigs
    SBT.main()


if __name__ == "__main__":
    main()
