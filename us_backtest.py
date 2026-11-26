"""
미국 검증 — 한국 바닥 신호(경로저장)와 똑같은 계산을 미국 S&P 1500 (대형 500 · 중형 400 · 소형 600) 종목에 돌림
- 시세: 야후 (수정주가: 액면분할·배당 반영)
- 시장 지수: S&P 500 (코스피 자리 · 120일선 아래 조건도 S&P 500 기준)
- 점수표(score_rules.json)는 한국 데이터로 배운 그대로 씀 → 미국은 처음 보는 데이터라 진짜 검증
- 수급(외국인·기관)·상장폐지는 없음 → 지금 S&P 1500에 남아 있는 종목만이라 결과가 실제보다 좋게 나올 수 있음
결과: state/paths_us.npz · state/paths_us_meta.csv (한국 paths.npz 는 그대로 둠)
"""
import io
import shutil
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import scan
import stop_backtest as SBT
import timing_backtest as TB
import entry_backtest as ENB
import flow_backtest as FB
import deep_backtest as DB
import paths_backtest as PB
from ma_backtest import log

FX = 1400.0          # 거래대금을 원화(억)로 바꿀 때 쓰는 환율 (한국 결과와 같은 단위로 비교하려고)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
WIKI = [("S&P500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
        ("S&P400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
        ("S&P600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")]
_DATA = {}
_lock = threading.Lock()


def us_universe():
    out, seen = [], set()
    for idx, url in WIKI:
        try:
            html = requests.get(url, headers=UA, timeout=30).text
            tabs = pd.read_html(io.StringIO(html))
            t = next(t for t in tabs if any(str(c).lower() in ("symbol", "ticker", "ticker symbol") for c in t.columns))
            col = next(c for c in t.columns if str(c).lower() in ("symbol", "ticker", "ticker symbol"))
            ncol = next((c for c in t.columns if str(c).lower() in ("security", "company")), col)
            n = 0
            for sym, name in zip(t[col].astype(str), t[ncol].astype(str)):
                sym = sym.strip().replace(".", "-")
                if sym and sym not in seen:
                    seen.add(sym)
                    out.append({"code": sym, "name": name, "market": "KS", "cap": 0, "_idx": idx})
                    n += 1
            log(f"{idx}: {n}종목")
        except Exception as e:
            log(f"{idx} 목록 오류: {e!r}")
    return out


def preload(uni, years):
    """야후에서 100종목씩 한꺼번에 받음 (종목별로 받으면 너무 느리고 막힘)"""
    import yfinance as yf
    start = (pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * (years + 3.5)))).strftime("%Y-%m-%d")
    syms = [s["code"] for s in uni]
    for k in range(0, len(syms), 100):
        chunk = syms[k:k + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=start, interval="1d", auto_adjust=True, group_by="ticker",
                                progress=False, threads=True)
                break
            except Exception as e:
                log(f"  다운로드 재시도 {attempt + 1}: {e!r}")
                time.sleep(10 * (attempt + 1))
        else:
            continue
        for sym in chunk:
            try:
                x = d[sym] if isinstance(d.columns, pd.MultiIndex) else d
                x = x.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                x.index = pd.to_datetime(x.index).tz_localize(None)
                x = x[x.close > 0]
                if len(x) >= 400:
                    _DATA[sym] = x.astype(float)
            except Exception:
                pass
        log(f"  시세 {min(k + 100, len(syms))}/{len(syms)} · 받은 종목 {len(_DATA)}")
        time.sleep(1)


def us_daily(code, count=1500, market="KS"):
    d = _DATA.get(code)
    if d is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return d.iloc[-count:] if count and len(d) > count else d


def spx():
    import yfinance as yf
    for sym in ("^GSPC", "SPY"):
        try:
            d = yf.download(sym, period="max", interval="1d", progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            s = d["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            if len(s) > 1000:
                return s
        except Exception:
            pass
    return None


def main():
    t0 = time.time()
    st = scan.STATE
    # 한국 경로 파일 보관 (경로저장 코드가 같은 이름으로 쓰기 때문)
    keep = {}
    for f in ("paths.npz", "paths_meta.csv", "backtest.md"):
        if (st / f).exists():
            keep[f] = st / f"_kr_{f}"
            shutil.copy(st / f, keep[f])
    uni = us_universe()
    if len(uni) < 300:
        (st / "backtest.md").write_text(f"# 미국 검증 실패\n\n종목 목록을 못 받음 ({len(uni)}개)", encoding="utf-8")
        return
    preload(uni, SBT.YEARS)
    idx_of = {s["code"]: s["_idx"] for s in uni}
    index = spx()
    log(f"S&P500 지수: {'있음' if index is not None else '없음'} · 시세 받은 종목 {len(_DATA)}/{len(uni)}")

    # 한국 전용 부분을 미국용으로 바꿔 끼움
    scan.fetch_universe = lambda: [s for s in uni if s["code"] in _DATA]
    scan.probe_sources = lambda: None
    scan.naver_daily = us_daily
    SBT.EB._orig_daily = us_daily
    DB.kospi = lambda: index
    SBT.index_series = lambda *a, **k: index
    FB._trend_api = lambda *a, **k: {}
    SBT.TOP_N = 100000
    SBT.MODES = ["기준"]
    SBT.path = TB.path
    SBT.EXTRA = lambda *a: {**TB.extra(*a), **PB.extra(*a), "idx": idx_of.get(getattr(ENB._t, "s", {}).get("code"), "")}
    SBT.stock_sigs = ENB.stock_sigs

    def report(sigs, n, t0_):
        PB.report(sigs, n, t0_)
        # 결과를 미국 이름으로 옮기고 한국 파일 되돌림
        m = pd.read_csv(st / "paths_meta.csv", dtype={"code": str})
        if "val20" in m:
            m["val20"] = m["val20"] * FX            # 달러 → 원화(억)
        m.to_csv(st / "paths_us_meta.csv", index=False)
        shutil.move(st / "paths.npz", st / "paths_us.npz")
        (st / "paths_meta.csv").unlink(missing_ok=True)
        for f, bak in keep.items():
            if f != "backtest.md":
                shutil.move(bak, st / f)
        keep.pop("paths.npz", None); keep.pop("paths_meta.csv", None)
        sub = m[m.path_id.notna()]
        y = pd.to_datetime(sub.date).dt.year
        L = ["# 미국 검증 (S&P 1500 · 경로 저장)", "",
             f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 목록 {len(uni)}종목 · 시세 {len(_DATA)}종목 · 저장 {len(sub)}건 "
             f"(예전 A {int(sub.old_a.sum())} · 강 {int(sub.strong.sum())})",
             f"- 지수별: " + " · ".join(f"{k} {v}" for k, v in sub.idx.value_counts().items()),
             f"- 연도별: " + " · ".join(f"{k} {v}" for k, v in y.value_counts().sort_index().items()),
             "- 파일: state/paths_us.npz · state/paths_us_meta.csv",
             "- 주의: 상장폐지 종목 없음(지금 지수에 남은 종목만) → 실제보다 좋게 나올 수 있음", "",
             f"_계산 시간 {time.time() - t0:.0f}초_"]
        (st / "backtest.md").write_text("\n".join(L), encoding="utf-8")

    SBT.report = report
    try:
        SBT.main()
    finally:
        for f, bak in keep.items():
            if bak.exists() and f != "backtest.md":
                shutil.move(bak, st / f)
        for bak in st.glob("_kr_*"):
            bak.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
