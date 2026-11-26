"""
진입 시점 · 종목 규모 · 상장폐지 포함 검증

1) 상장폐지 포함: 최근 기간에 상장폐지된 종목을 KRX 목록에서 받아 분석 대상에 추가
   (주가 데이터를 구한 종목만 들어감 → 보고서에 몇 개 구했는지 표시)
   상폐 종목을 들고 있으면 마지막 거래일 종가(정리매매 포함)에 판 것으로 계산
2) 진입 시점: 다음 날 시가 매수(지금 방식) vs 신호 당일 종가 근처 매수(저녁장)
3) 규모: 20일 평균 거래대금 · 주가 · 현재 시총 순위별 성과
매도는 수익형 그대로: 5파 저점 종가 이탈 → 위쪽 주·월봉 60/120선(없으면 +50%)에서 절반 → 최고가 -20%
"""
import re
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

import scan
import strategy as S
import stop_backtest as SBT
import score_backtest as SB
from ma_backtest import log

MODES = ["시가 진입", "종가 진입"]
BUY_COST, SELL_COST = SBT.BUY_COST, SBT.SELL_COST
DELISTED = {}
INFO = {}
_t = threading.local()


def fetch_delisted(since):
    """상장폐지 목록 [(code, name, market, 폐지일)] — FinanceDataReader → KIND 순서로 시도"""
    out = []
    try:
        import FinanceDataReader as fdr
        d = fdr.StockListing("KRX-DELISTING")
        for _, r in d.iterrows():
            code = str(r.get("Symbol", "")).zfill(6)
            dt = pd.to_datetime(r.get("DelistingDate"), errors="coerce")
            mk = str(r.get("Market", ""))
            if re.fullmatch(r"\d{6}", code) and pd.notna(dt) and dt >= since and ("KOSDAQ" in mk or "KOSPI" in mk or "코스" in mk or "유가" in mk):
                out.append((code, str(r.get("Name", "")), "KQ" if ("KOSDAQ" in mk or "코스닥" in mk) else "KS", dt,
                            str(r.get("Reason", ""))))
        INFO["src"] = f"FinanceDataReader {len(out)}건"
    except Exception as e:
        INFO["fdr_err"] = type(e).__name__
    if not out:
        for mt, mk in (("1", "KS"), ("2", "KQ")):
            try:
                r = requests.post("https://kind.krx.co.kr/investwarn/delcompany.do",
                                  data={"method": "searchDelCompanySub", "currentPageSize": "3000", "pageIndex": "1",
                                        "orderMode": "1", "orderStat": "D", "marketType": mt,
                                        "fromDate": since.strftime("%Y-%m-%d"), "toDate": datetime.now().strftime("%Y-%m-%d")},
                                  headers={**scan.UA, "Referer": "https://kind.krx.co.kr/investwarn/delcompany.do?method=searchDelCompanyMain"},
                                  timeout=30)
                html = r.content.decode("utf-8", "ignore")
                for tr in re.findall(r"<tr.*?</tr>", html, re.S):
                    txt = re.sub(r"<[^>]+>", " ", tr)
                    m_code = re.search(r"\b(\d{6})\b", txt)
                    m_dt = re.search(r"(20\d\d-\d\d-\d\d)", txt)
                    if not (m_code and m_dt):
                        continue
                    name = re.search(r'title="([^"]+)"', tr)
                    out.append((m_code.group(1), name.group(1) if name else m_code.group(1), mk, pd.Timestamp(m_dt.group(1)), ""))
            except Exception as e:
                INFO[f"kind_err_{mk}"] = type(e).__name__
        INFO["src"] = f"KIND {len(out)}건"
    seen, res = set(), []
    for c in out:
        if c[0] in seen or "스팩" in c[1]:
            continue
        seen.add(c[0])
        res.append(c)
    return res


def path(O, H, L, C, ei, stop, mode):
    t = SBT._tl
    if mode == "종가 진입":
        entry, j0 = C[ei - 1], ei
    else:
        entry, j0 = O[ei], ei
    n = len(C)
    last = min(n - 1, ei + S.HOLD_MAX)
    held, cash, tp, peak = 1.0, -BUY_COST, False, entry
    ups = [m[ei] for m in t.res if np.isfinite(m[ei]) and m[ei] >= entry * 1.10]
    rline = min(min(ups), entry * 1.5) if ups else entry * 1.5
    vals = []
    for j in range(j0, last + 1):
        peak = max(peak, H[j])
        if not tp and H[j] >= rline:
            px = max(O[j], rline) if (j > ei or mode == "종가 진입") else rline
            cash += 0.5 * px / entry * (1 - SELL_COST); held = 0.5; tp = True
        c = C[j]
        why = None
        if c < stop:
            why = "5파 이탈"
        elif tp and c < peak * 0.80:
            why = "추적 -20%"
        if why or j == last:
            cash += held * c / entry * (1 - SELL_COST)
            vals.append(cash)
            return np.array(vals), why or ("상장폐지" if j == n - 1 and getattr(_t, "dl", False) else "기간 만료"), j
        vals.append(held * c / entry + cash)
    return np.array(vals), "기간 만료", last


def extra(d1, i, snap, wb, ei, entry, l5, t):
    C, V = d1["close"].to_numpy(float), d1["volume"].to_numpy(float)
    val20 = float(np.mean(C[max(0, i - 19):i + 1] * V[max(0, i - 19):i + 1])) / 1e8
    s = getattr(_t, "s", {})
    return dict(gap=(d1["open"].iloc[ei] / C[i] - 1) * 100, val20=val20, cap_rank=s.get("_rank", np.nan),
                delisted=bool(s.get("_dl")), dl_date=s.get("_dl_date", ""), dl_reason=s.get("_dl_reason", ""))


_orig = SBT.stock_sigs


def delisted_daily(code):
    """상폐 종목 일봉: 네이버 → FinanceDataReader(KRX 상장폐지 데이터) 순서"""
    try:
        d = SBT.EB._orig_daily(code, 3200, "KS")
        if d is not None and len(d) >= 300:
            return d, "네이버"
    except Exception:
        pass
    try:
        import FinanceDataReader as fdr
        d = fdr.DataReader(f"KRX-DELISTING:{code}")
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype(float)
        d.index = pd.to_datetime(d.index)
        d = d[(d.close > 0)].sort_index()
        d["open"] = d["open"].where(d["open"] > 0, d["close"])
        d["high"] = d[["high", "open", "close"]].max(axis=1)
        d["low"] = d[["low", "open", "close"]].replace(0, np.nan).min(axis=1)
        if len(d) >= 300:
            return d, "FDR"
    except Exception:
        pass
    return None, None


def stock_sigs(s, start, kp, mk):
    _t.s = s
    _t.dl = bool(s.get("_dl"))
    if s.get("_dl"):
        d, src = delisted_daily(s["code"])
        with SBT.EB._lock:
            INFO[f"상폐 데이터 {src or '없음'}"] = INFO.get(f"상폐 데이터 {src or '없음'}", 0) + 1
        if d is None:
            return []
        with SBT.EB._lock:
            SBT.EB._cache[s["code"]] = d
    return _orig(s, start, kp, mk)


def stat(v):
    return SB.stat(v)


def row(lab, v):
    return SB.row(lab, v)


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in t.items() if k not in MODES} for t in sigs])
    for m in MODES:
        df[m] = [t[m]["ret"] for t in sigs]
        df[m + " 일수"] = [len(t[m]["vals"]) for t in sigs]
        df[m + " 사유"] = [t[m]["reason"] for t in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    Y = "시가 진입"
    sets = {"예전 A + 새 점수 강": df.old_a & df.strong, "예전 A": df.old_a, "넓은 신호 전체": df.old_a | ~df.old_a}
    n_dl = INFO.get("dl_total", 0)
    n_dl_sig = int(df.delisted.sum())
    L = ["# 진입 시점 · 규모 · 상장폐지 포함 검증", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 분석 {n}종목 (상장폐지 {n_dl}종목 포함 시도)",
         f"- 상장폐지 목록: {INFO.get('src', '못 받음')} · " + ", ".join(f"{k} {v}" for k, v in INFO.items() if k.startswith("상폐") or "err" in k),
         f"- 상장폐지 종목 중 주가 데이터가 있어 신호가 잡힌 것: {df[df.delisted].code.nunique()}종목 · 신호 {n_dl_sig}건", ""]
    L += ["## 1. 상장폐지 포함 전후 (시가 진입)", "", SB.HEAD]
    for nm, m in sets.items():
        L += [row(f"{nm} · 지금 상장 종목만", df[m & ~df.delisted][Y]), row(f"{nm} · 상장폐지 포함", df[m][Y]),
              row(f"{nm} · 상장폐지 종목만", df[m & df.delisted][Y])]
    L.append("")
    g = df[df.delisted & df.old_a]
    if len(g):
        L += ["상장폐지 종목의 예전 A 신호 (손실 큰 순)", "", "| 종목 | 매수일 | 폐지일 | 사유 | 결과 | 청산 |", "|---|---|---|---|---|---|"]
        for _, r in g.sort_values(Y).head(15).iterrows():
            L.append(f"| {r['name']} | {str(r.date)[:10]} | {r.dl_date} | {r.dl_reason[:20]} | {r[Y]:+.1f}% | {r[Y + ' 사유']} |")
        L.append("")
    L += ["## 2. 진입 시점: 다음 날 시가 vs 신호 당일 종가(저녁장)", "", SB.HEAD]
    for nm, m in sets.items():
        L += [row(f"{nm} · 다음 날 시가", df[m]["시가 진입"]), row(f"{nm} · 당일 종가", df[m]["종가 진입"])]
    L += ["", "다음 날 시초가 갭별 (예전 A, 시가 진입 기준 결과)", "", SB.HEAD]
    a = df[df.old_a]
    for lab, mm in [("갭 -2% 아래", a.gap < -2), ("갭 -2~0%", (a.gap >= -2) & (a.gap < 0)), ("갭 0~2%", (a.gap >= 0) & (a.gap < 2)),
                    ("갭 2~5%", (a.gap >= 2) & (a.gap < 5)), ("갭 5% 이상", a.gap >= 5)]:
        L.append(row(lab, a[mm]["시가 진입"]))
    L.append("")
    L += ["## 3. 규모별 (예전 A + 강 / 예전 A · 시가 진입)", ""]
    for nm in ("예전 A + 새 점수 강", "예전 A"):
        g = df[sets[nm]]
        L += [f"### {nm}", SB.HEAD]
        for lab, mm in [("거래대금 10억 미만", g.val20 < 10), ("거래대금 10~30억", (g.val20 >= 10) & (g.val20 < 30)),
                        ("거래대금 30~100억", (g.val20 >= 30) & (g.val20 < 100)), ("거래대금 100억 이상", g.val20 >= 100),
                        ("주가 2천원 미만", g.price < 2000), ("주가 2천~1만원", (g.price >= 2000) & (g.price < 10000)),
                        ("주가 1만원 이상", g.price >= 10000),
                        ("현재 시총 1~500위", g.cap_rank <= 500), ("현재 시총 500~1500위", (g.cap_rank > 500) & (g.cap_rank <= 1500)),
                        ("현재 시총 1500위 밖", g.cap_rank > 1500), ("상장폐지 (시총 순위 없음)", g.delisted)]:
            L.append(row(lab, g[mm]["시가 진입"]))
        L.append("")
    L += ["- 현재 시총 순위는 '지금' 기준이라 그동안 많이 오른 종목이 위로 올라와 있어요 (참고용). 거래대금은 신호 당시 기준이라 믿을 만해요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    keep = [c for c in df.columns]
    df[keep].to_csv(scan.STATE / "entry_trades.csv.gz", index=False, compression="gzip")
    log("보고서 저장")


def main():
    since = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * SBT.YEARS))
    dl = fetch_delisted(since)
    INFO["dl_total"] = len(dl)
    log(f"상장폐지 목록: {len(dl)}종목 ({INFO.get('src')})")
    _fu = scan.fetch_universe

    def uni():
        u = _fu()
        for r, s in enumerate(u, 1):
            s["_rank"] = r
        have = {s["code"] for s in u}
        for code, name, mk, dt, why in dl:
            if code not in have:
                u.append({"code": code, "name": name, "market": mk, "_dl": True,
                          "_dl_date": dt.strftime("%Y-%m-%d"), "_dl_reason": why})
        return u
    scan.fetch_universe = uni
    SBT.TOP_N = 100000
    SBT.MODES = MODES
    SBT.path = path
    SBT.report = report
    SBT.EXTRA = extra
    SBT.stock_sigs = stock_sigs
    SBT.main()


if __name__ == "__main__":
    main()
