"""
지수저장 — 쉬는 돈 운용 실험용으로 지수·ETF 일봉을 state/index.csv 에 저장 (1분 안 걸림)
코스피 · 코스닥 · KODEX 200(069500) · KODEX 코스닥150(229200) · KODEX 단기채권(153130) · TIGER CD금리(357870)
"""
import re
import time
from datetime import datetime

import pandas as pd
import requests

import scan
from ma_backtest import log

ITEMS = [("KOSPI", "KOSPI", "^KS11"), ("KOSDAQ", "KOSDAQ", "^KQ11"), ("K200", "069500", "069500.KS"),
         ("KQ150", "229200", "229200.KS"), ("BOND", "153130", "153130.KS"), ("CD", "357870", "357870.KS")]


def naver(sym):
    r = requests.get(f"https://fchart.stock.naver.com/sise.nhn?symbol={sym}&timeframe=day&count=5000&requestType=0",
                     headers=scan.UA, timeout=20)
    rows = []
    for it in re.findall(r'data="([^"]+)"', r.content.decode("euc-kr", "ignore")):
        d, o, h, l, c, v = it.split("|")
        rows.append((pd.Timestamp(d), float(o), float(c)))
    return pd.DataFrame(rows, columns=["date", "open", "close"]).set_index("date")


def yahoo(sym):
    import yfinance as yf
    d = yf.download(sym, period="max", interval="1d", progress=False, auto_adjust=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.lower)[["open", "close"]].dropna()
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


def main():
    t0 = time.time()
    cols, info = {}, []
    for name, nsym, ysym in ITEMS:
        d = None
        for fn, s in ((naver, nsym), (yahoo, ysym)):
            try:
                d = fn(s)
                if d is not None and len(d) > 200:
                    break
            except Exception as e:
                log(f"{name} {fn.__name__} 오류 {e!r}")
                d = None
        if d is None or len(d) <= 200:
            info.append(f"{name} 없음")
            continue
        d = d[d.close > 0]
        cols[f"{name}_open"], cols[f"{name}_close"] = d["open"], d["close"]
        info.append(f"{name} {len(d)}일 ({d.index[0].date()}~)")
        log(info[-1])
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df.to_csv(scan.STATE / "index.csv")
    (scan.STATE / "backtest.md").write_text(
        "# 지수 저장\n\n- 실행: " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n- " + "\n- ".join(info) +
        f"\n- 파일: state/index.csv\n\n_계산 시간 {time.time() - t0:.0f}초_", encoding="utf-8")


if __name__ == "__main__":
    main()
