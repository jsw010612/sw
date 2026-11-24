"""
재무 건강 점수 (참고용)

데이터: 네이버 금융 종목 페이지의 "기업실적분석" 표 (최근 연간 실적)
        → 안 되면 네이버 모바일 증권 API

재무 건강 점수 100점
  수익성 25 : 영업이익 흑자 10 + 순이익 흑자 5 + ROE 10
  안정성 25 : 부채비율 15 + 당좌비율 10
  성장성 20 : 매출 증가 10 + 영업이익 증가 10
  지속성 15 : 최근 3년 중 영업이익 흑자 연수 x5
  가치   15 : PBR 8 + PER 7

위험 경고: 관리종목·거래정지 등 표시, 자본잠식 의심, 3년 연속 영업적자,
          부채비율 400% 초과, 매출 아주 작음, 재무 데이터 없음
"""
import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

ROWS = {
    "매출액": "rev", "영업이익": "op", "당기순이익": "ni", "영업이익률": "opm", "순이익률": "npm",
    "ROE": "roe", "부채비율": "debt", "당좌비율": "quick", "유보율": "reserve",
    "EPS": "eps", "PER": "per", "BPS": "bps", "PBR": "pbr", "주당배당금": "dps", "시가배당률": "dy",
}
STATUS_WORDS = ["관리종목", "거래정지", "정리매매", "투자위험", "투자경고", "투자주의환기", "불성실공시", "상장적격성"]


def _num(t):
    t = re.sub(r"<[^>]+>", "", t or "").replace(",", "").replace("&nbsp;", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _key(title):
    title = re.sub(r"<[^>]+>", "", title).strip()
    for k, v in ROWS.items():
        if title.startswith(k):
            return v
    return None


def parse_main_html(html: str) -> dict:
    """네이버 금융 종목 메인 페이지 → 연간 실적(실제값만) + 상태 표시"""
    out = {"years": [], "status": []}
    # 상태 표시 (종목명 주변 아이콘/문구)
    head = html
    m = re.search(r'<div class="wrap_company">(.*?)</div>\s*</div>', html, re.S)
    if m:
        head = m.group(1)
    alts = re.findall(r'alt="([^"]+)"', html)
    for w in STATUS_WORDS:
        if any(w in a for a in alts) or w in re.sub(r"<[^>]+>", " ", head):
            out["status"].append(w)

    sec = re.search(r'class="section cop_analysis"(.*?)</table>', html, re.S)
    if not sec:
        return out
    sec = sec.group(1)
    heads = re.findall(r"(\d{4}\.\d{2})\s*(\(E\))?", re.sub(r"<[^>]+>", " ", sec.split("<tbody")[0]))
    annual = heads[:4]
    data = {}
    for tr in re.findall(r"<tr.*?</tr>", sec, re.S):
        th = re.search(r"<th[^>]*>(.*?)</th>", tr, re.S)
        if not th:
            continue
        k = _key(th.group(1))
        if not k:
            continue
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        data[k] = [_num(x) for x in tds[:len(annual)]]
    years = []
    for i, (ym, est) in enumerate(annual):
        if est:
            continue
        years.append({"y": ym, **{k: (v[i] if i < len(v) else None) for k, v in data.items()}})
    out["years"] = years[-3:]
    return out


def parse_mobile_api(js: dict) -> dict:
    """네이버 모바일 API finance/annual 응답 (형식이 다르면 빈 결과)"""
    out = {"years": [], "status": []}
    fi = js.get("financeInfo") or js
    titles = fi.get("trTitleList") or []
    rows = fi.get("rowList") or []
    cols = [(t.get("key"), t.get("title", ""), str(t.get("isConsensus", "N")).upper() == "Y") for t in titles]
    years = []
    for key, title, est in cols:
        if est or not key:
            continue
        y = {"y": title.strip(". ")}
        for r in rows:
            k = _key(r.get("title", ""))
            if k:
                c = (r.get("columns") or {}).get(key) or {}
                y[k] = _num(str(c.get("value", "")))
        years.append(y)
    out["years"] = years[-3:]
    return out


def fetch(code: str) -> dict:
    res = {"years": [], "status": [], "src": None}
    try:
        r = requests.get(f"https://finance.naver.com/item/main.naver?code={code}", headers=UA, timeout=15)
        if r.status_code == 200:
            p = parse_main_html(r.content.decode("euc-kr", "ignore"))
            res.update(p)
            if p["years"]:
                res["src"] = "html"
    except Exception:
        pass
    if not res["years"]:
        try:
            r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/finance/annual", headers=UA, timeout=15)
            if r.status_code == 200:
                p = parse_mobile_api(r.json())
                if p["years"]:
                    res["years"] = p["years"]
                    res["src"] = "api"
        except Exception:
            pass
    return res


def score(f: dict, market: str = "KS") -> dict:
    ys = f.get("years") or []
    flags = list(f.get("status") or [])
    out = {"score": None, "grade": "정보없음", "parts": {}, "flags": flags, "years": ys}
    if not ys:
        out["flags"] = flags + ["재무 데이터 없음"]
        if any(w in flags for w in ("관리종목", "거래정지", "정리매매", "상장적격성")):
            out["grade"] = "위험"
        return out
    last = ys[-1]
    prev = ys[-2] if len(ys) > 1 else {}
    g = lambda d, k: d.get(k) if d else None

    # 수익성 25
    p1 = 0
    op, ni, roe = g(last, "op"), g(last, "ni"), g(last, "roe")
    p1 += 10 if op is not None and op > 0 else 0
    p1 += 5 if ni is not None and ni > 0 else 0
    if roe is not None:
        p1 += 10 if roe >= 10 else 6 if roe >= 5 else 3 if roe > 0 else 0
    # 안정성 25
    p2 = 0
    debt, quick = g(last, "debt"), g(last, "quick")
    if debt is not None:
        p2 += 15 if debt <= 100 else 10 if debt <= 200 else 5 if debt <= 300 else 0
    if quick is not None:
        p2 += 10 if quick >= 100 else 5 if quick >= 50 else 0
    # 성장성 20
    p3 = 0
    if g(last, "rev") is not None and g(prev, "rev") is not None and last["rev"] > prev["rev"]:
        p3 += 10
    if g(last, "op") is not None and g(prev, "op") is not None and last["op"] > prev["op"]:
        p3 += 10
    # 지속성 15
    pos = sum(1 for y in ys if y.get("op") is not None and y["op"] > 0)
    p4 = pos * 5
    # 가치 15
    p5 = 0
    pbr, per = g(last, "pbr"), g(last, "per")
    if pbr is not None and pbr > 0:
        p5 += 8 if pbr < 1 else 5 if pbr < 2 else 2 if pbr < 4 else 0
    if per is not None and per > 0:
        p5 += 7 if per < 15 else 4 if per < 30 else 1
    total = p1 + p2 + p3 + p4 + p5

    # 위험 경고
    ops = [y.get("op") for y in ys if y.get("op") is not None]
    if len(ops) >= 3 and all(o < 0 for o in ops[-3:]):
        flags.append("3년 연속 영업적자")
    elif ops and ops[-1] < 0:
        flags.append("최근 영업적자")
    res_ = g(last, "reserve")
    if res_ is not None and res_ < 0:
        flags.append("자본잠식 의심")
    if debt is not None and debt > 400:
        flags.append(f"부채비율 {debt:.0f}%")
    rev = g(last, "rev")
    if rev is not None and rev < (50 if market == "KS" else 30):
        flags.append("매출 아주 작음")

    severe = any(w in " ".join(flags) for w in
                 ("관리종목", "거래정지", "정리매매", "상장적격성", "자본잠식", "3년 연속"))
    grade = "위험" if severe else "A" if total >= 75 else "B" if total >= 55 else "C" if total >= 35 else "D"
    out.update(score=int(total), grade=grade, flags=flags,
               parts={"수익성": [p1, 25], "안정성": [p2, 25], "성장성": [p3, 20], "지속성": [p4, 15], "가치": [p5, 15]})
    return out
