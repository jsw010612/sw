"""
섹터 · 수급 검증 — 바닥 신호에 섹터 흐름과 외국인·기관 매매를 붙여 효과 비교

섹터 (네이버 업종 분류)
  sec_val    : 섹터 전체 거래대금 (신호일 / 직전 20일 평균) — 섹터에 평소보다 돈이 몰리나
  sec_ret20  : 섹터 평균 20일 수익률
  sec_above  : 섹터 종목 중 20일선 위 비율 (%)
  sec_cobot  : 최근 10일 안에 같은 섹터에서 바닥 신호가 나온 다른 종목 수 (섹터 전체 바닥)
수급 (네이버 종목별 외국인·기관 순매매량 × 종가) — 예전 A 신호만
  frg5/frg20 : 신호일까지 5일·20일 외국인 순매수 합 ÷ 20일 평균 거래대금
  ins5/ins20 : 같은 방식 기관
  frg_turn   : 20일 합은 순매도인데 최근 5일은 순매수 (외국인 매수 전환)
매도는 수익형(5파 저점 → 저항선 절반 → 최고가 -20%)
"""
import re
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import stop_backtest as SBT
import hybrid_backtest as HB
import score_backtest as SB
from ma_backtest import log

MODES = ["A 5파 | 추적20"]
Y = MODES[0]
VAL, CLOSE = {}, {}
SECTOR = {}
INFO = {}
_lock = threading.Lock()
FLOW_BUDGET = 30 * 60      # 수급 받는 데 최대 30분


def fetch_sectors():
    try:
        import FinanceDataReader as fdr
        d = fdr.StockListing("KRX-DESC")
        cands = [c for c in ("Industry", "Sector") if c in d.columns]
        # 너무 굵은 분류(10개 미만)는 섹터 비교에 쓸모없어서, 업종 수가 30~400개인 열을 고름
        cands = sorted(cands, key=lambda c: -(30 <= d[c].nunique() <= 400))
        col = cands[0] if cands else None
        INFO["FDR 열"] = ", ".join(f"{c}:{d[c].nunique()}" for c in ("Sector", "Industry") if c in d.columns)
        if col and not (30 <= d[col].nunique() <= 400):
            col = None
        cc = "Code" if "Code" in d.columns else ("Symbol" if "Symbol" in d.columns else None)
        if col and cc:
            for c, sec in zip(d[cc].astype(str).str.zfill(6), d[col]):
                if isinstance(sec, str) and sec.strip():
                    SECTOR[c] = sec.strip()
        if len(SECTOR) > 500 and len(set(SECTOR.values())) >= 30:
            INFO["섹터"] = f"KRX 업종 {len(set(SECTOR.values()))}개 · {len(SECTOR)}종목"
            return
    except Exception as e:
        INFO["섹터 FDR 오류"] = type(e).__name__
    SECTOR.clear()
    _naver_sectors()


def _naver_sectors():
    base = "https://finance.naver.com"
    r = scan._http(f"{base}/sise/sise_group.naver?type=upjong")
    if r is None or r.status_code != 200:
        INFO["섹터"] = "목록 실패"
        return
    html = r.content.decode("euc-kr", "ignore")
    groups = re.findall(r'href="/sise/sise_group_detail\.naver\?type=upjong(?:&amp;|&)no=(\d+)"[^>]*>([^<]+)</a>', html)
    for no, name in groups:
        r2 = scan._http(f"{base}/sise/sise_group_detail.naver?type=upjong&no={no}")
        if r2 is None or r2.status_code != 200:
            continue
        for code in set(re.findall(r"/item/main\.naver\?code=(\d{6})", r2.content.decode("euc-kr", "ignore"))):
            SECTOR[code] = name.strip()
        time.sleep(0.1)
    INFO["섹터"] = f"{len(groups)}개 업종 · {len(SECTOR)}종목"


_orig = SBT.stock_sigs


def stock_sigs(s, start, kp, mk):
    d = SBT.EB.cached_daily(s["code"], 2000, s["market"])
    if d is not None and len(d) > 0:
        d = d[~d.index.duplicated()].sort_index()
        with _lock:
            VAL[s["code"]] = (d["close"] * d["volume"]).astype("float32")
            CLOSE[s["code"]] = d["close"].astype("float32")
    return _orig(s, start, kp, mk)


def sector_features(df):
    for col in ("sector", "sec_val", "sec_ret20", "sec_above", "sec_n", "sec_cobot"):
        df[col] = np.nan
    if not SECTOR:
        return df
    codes = [c for c in VAL if c in SECTOR]
    val = pd.DataFrame({c: VAL[c] for c in codes}).sort_index()
    cl = pd.DataFrame({c: CLOSE[c] for c in codes}).sort_index()
    above = (cl > cl.rolling(20).mean()).astype(float).where(cl.notna())
    ret20 = cl / cl.shift(20) - 1
    sec_of = pd.Series({c: SECTOR[c] for c in codes})
    out = {}
    for sec, cs in sec_of.groupby(sec_of):
        cs = list(cs.index)
        v = val[cs].sum(axis=1)
        out[sec] = pd.DataFrame({"sec_val": v / v.rolling(20).mean().shift(1),
                                 "sec_ret20": ret20[cs].mean(axis=1) * 100,
                                 "sec_above": above[cs].mean(axis=1) * 100,
                                 "sec_n": cl[cs].notna().sum(axis=1)})
    df["sector"] = df.code.map(SECTOR)
    det = pd.to_datetime(df.det_date)
    for col in ("sec_val", "sec_ret20", "sec_above", "sec_n"):
        df[col] = [out[s].loc[:d, col].iloc[-1] if isinstance(s, str) and s in out and len(out[s].loc[:d]) else np.nan
                   for s, d in zip(df.sector, det)]
    # 같은 섹터 동시 바닥: 최근 10일 안에 같은 섹터 다른 종목 신호 수
    cob = []
    dd = pd.to_datetime(df.det_date)
    for sec_, code_, d in zip(df.sector, df.code, dd):
        if not isinstance(sec_, str):
            cob.append(np.nan); continue
        m = (df.sector == sec_) & (df.code != code_) & (dd <= d) & (dd > d - pd.Timedelta(days=14))
        cob.append(int(df.code[m].nunique()))
    df["sec_cobot"] = cob
    return df


def _num(x):
    x = re.sub(r"[^0-9\-+]", "", x)
    try:
        return float(x)
    except ValueError:
        return np.nan


def _frgn_page(code, page):
    r = scan._http(f"https://finance.naver.com/item/frgn.naver?code={code}&page={page}",
                   headers={"Referer": f"https://finance.naver.com/item/main.naver?code={code}"})
    if r is None or r.status_code != 200:
        INFO.setdefault("frgn 상태", getattr(r, "status_code", "접속실패"))
        return None
    html = r.content.decode("euc-kr", "ignore")
    INFO.setdefault("frgn 응답 길이", len(html))
    rows = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        tds = [re.sub(r"<[^>]+>", " ", t) for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        if len(tds) < 7:
            continue
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", tds[0])
        if not m:
            continue
        d = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        rows[d] = (_num(tds[1]), _num(tds[5]), _num(tds[6]))      # 종가, 기관 순매매량, 외국인 순매매량
    return rows


def _trend_api(code, det):
    """네이버 모바일 '투자자별 매매동향': 기준일(bizdate)부터 거꾸로 25거래일"""
    import requests
    url = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=25&bizdate={pd.Timestamp(det).strftime('%Y%m%d')}"
    r = requests.get(url, headers={**scan.UA, "Referer": "https://m.stock.naver.com/"}, timeout=15)
    if r.status_code != 200:
        INFO.setdefault("모바일API 상태", r.status_code)
        return None
    j = r.json()
    if isinstance(j, dict):
        j = next((v for v in j.values() if isinstance(v, list)), [])
    rows = {}
    for it in j:
        ks = {k.lower(): k for k in it}
        dk = next((ks[k] for k in ks if "bizdate" in k or k == "date"), None)
        fk = next((ks[k] for k in ks if "foreign" in k and "pure" in k), None)
        ok_ = next((ks[k] for k in ks if "organ" in k and "pure" in k), None)
        ck = next((ks[k] for k in ks if "closeprice" in k), None)
        if not (dk and fk and ok_ and ck):
            INFO.setdefault("모바일API 키", ",".join(list(it)[:12]))
            return None
        rows[pd.Timestamp(str(it[dk])[:8])] = (_num(str(it[ck])), _num(str(it[ok_])), _num(str(it[fk])))
    return rows


def flow_features(df):
    """네이버 '외국인·기관 순매매' 페이지(20일씩)에서 신호일 직전 20거래일만 받음 — 예전 A 신호만"""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    tgt = df[df.old_a == True]
    jobs = []
    for idx, r in tgt.iterrows():
        cl = CLOSE.get(r.code)
        if cl is None:
            continue
        off = len(cl.loc[pd.Timestamp(r.det_date):]) - 1 + (len(pd.bdate_range(cl.index[-1], datetime.now())) - 1)
        jobs.append((idx, r.code, r.det_date, sorted({off // 20 + 1, (off + 25) // 20 + 1})))
    cache, ok = {}, [0, 0]

    def get(code, p):
        k = (code, p)
        if k not in cache:
            try:
                cache[k] = _frgn_page(code, p)
            except Exception:
                cache[k] = None
            ok[0 if cache[k] else 1] += 1
            time.sleep(0.05)
        return cache[k]

    def one(job):
        idx, code, det, pages = job
        if time.time() - t0 > FLOW_BUDGET:
            return idx, None
        rows = {}
        try:
            rows = _trend_api(code, det) or {}
            ok[0 if rows else 1] += 1
        except Exception as e:
            INFO.setdefault("모바일API 오류", type(e).__name__)
        if not rows:
            for p in pages:
                rows.update(get(code, p) or {})
        if not rows:
            return idx, None
        f = pd.DataFrame(rows, index=["close", "ins", "frg"]).T.sort_index()
        f = f.loc[:pd.Timestamp(det)]
        v = VAL.get(code)
        if len(f) < 20 or v is None:
            return idx, None
        base = float(v.loc[:pd.Timestamp(det)].tail(20).mean())
        if not base:
            return idx, None
        fv, iv = f.frg * f.close, f.ins * f.close
        return idx, dict(frg5=fv.tail(5).sum() / base, frg20=fv.tail(20).sum() / base,
                         ins5=iv.tail(5).sum() / base, ins20=iv.tail(20).sum() / base)

    feats = {}
    with ThreadPoolExecutor(6) as ex:
        for idx, fe in ex.map(one, jobs):
            if fe:
                feats[idx] = fe
    INFO["수급"] = f"네이버 {len(feats)}/{len(jobs)}건 · 요청 성공 {ok[0]} 실패 {ok[1]}"
    for k in ("frg5", "frg20", "ins5", "ins20"):
        df[k] = [feats.get(i, {}).get(k, np.nan) for i in df.index]
    df["frg_turn"] = np.where(df.frg5.notna(), ((df.frg20 < 0) & (df.frg5 > 0)).astype(float), np.nan)
    return df


def extra(d1, i, snap, wb, ei, entry, l5, t):
    return dict(det_date=d1.index[i].strftime("%Y-%m-%d"))


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in s.items() if k not in MODES} for s in sigs])
    df[Y] = [s[Y]["ret"] for s in sigs]
    df[Y + " 일수"] = [len(s[Y]["vals"]) for s in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    log("섹터 계산")
    df = sector_features(df)
    log("수급 받기")
    df = flow_features(df)
    A = df[df.old_a == True]
    W = df
    L = ["# 섹터 · 수급 검증", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 · 신호 {len(df)}건 · 예전 A {len(A)}건",
         "- " + " · ".join(f"{k}: {v}" for k, v in INFO.items()), "",
         "## 1. 섹터 (예전 A / 넓은 신호)", ""]
    for nm, g in (("예전 A", A), ("넓은 신호", W)):
        L += [f"### {nm}", SB.HEAD]
        for lab, m in [("섹터 거래대금 평소의 0.8배 미만", g.sec_val < 0.8), ("0.8~1.2배", g.sec_val.between(0.8, 1.2)),
                       ("1.2~2배", (g.sec_val > 1.2) & (g.sec_val <= 2)), ("2배 초과", g.sec_val > 2),
                       ("섹터 20일 -10% 아래", g.sec_ret20 < -10), ("섹터 20일 -10~0%", g.sec_ret20.between(-10, 0)),
                       ("섹터 20일 플러스", g.sec_ret20 > 0),
                       ("섹터 20일선 위 30% 미만", g.sec_above < 30), ("섹터 20일선 위 30~60%", g.sec_above.between(30, 60)),
                       ("섹터 20일선 위 60% 초과", g.sec_above > 60),
                       ("같은 섹터 동시 바닥 0개", g.sec_cobot == 0), ("동시 바닥 1~2개", g.sec_cobot.between(1, 2)),
                       ("동시 바닥 3개 이상", g.sec_cobot >= 3)]:
            L.append(SB.row(lab, g[m][Y]))
        L.append("")
    if "frg5" in df:
        L += ["## 2. 외국인·기관 (예전 A)", "", SB.HEAD]
        g = A
        for lab, m in [("외국인 20일 순매도", g.frg20 < 0), ("외국인 20일 순매수", g.frg20 > 0),
                       ("외국인 5일 순매수", g.frg5 > 0), ("외국인 매수 전환 (20일 매도 → 5일 매수)", g.frg_turn == 1),
                       ("외국인 5일 강한 매수 (거래대금 50%↑)", g.frg5 > 0.5),
                       ("기관 20일 순매도", g.ins20 < 0), ("기관 20일 순매수", g.ins20 > 0), ("기관 5일 순매수", g.ins5 > 0),
                       ("외국인·기관 5일 모두 순매수", (g.frg5 > 0) & (g.ins5 > 0)),
                       ("외국인·기관 5일 모두 순매도", (g.frg5 < 0) & (g.ins5 < 0)), ("수급 데이터 없음", g.frg5.isna())]:
            L.append(SB.row(lab, g[m][Y]))
        L.append("")
    L += ["- 배운 기간/검증 기간 나눠보기와 계좌 계산은 flow_trades.csv 로 따로 해요.", "",
          f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    df.to_csv(scan.STATE / "flow_trades.csv", index=False)
    log("보고서 저장")


def main():
    fetch_sectors()
    log(f"섹터: {INFO.get('섹터')}")
    SBT.MODES = MODES
    SBT.path = HB.path
    SBT.report = report
    SBT.EXTRA = extra
    SBT.stock_sigs = stock_sigs
    SBT.main()


if __name__ == "__main__":
    main()
