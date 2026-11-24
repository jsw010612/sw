"""
코스피·코스닥 단계별 RSI 다이버전스 스캐너

흐름
 1) 하루 한 번: 네이버에서 코스피·코스닥 전 종목 목록(시총순)을 받고,
    일봉 RSI가 40 이하인 종목만 "후보"로 추림 (전 종목을 매시간 다 보면 너무 오래 걸려서)
 2) 매시간: 후보 종목의 1시간봉(야후)과 일봉(네이버)으로 5개 프레임을 만들어 분석
 3) 기준 점수 이상 매수 신호가 새로 나오면 ntfy로 폰 알림
 4) 결과는 state/latest.md 에 점수순으로 저장
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from analysis import (BULL_CANDLE, FRAMES, Params, evaluate, forming_state, grade,
                      readiness, rsi, to_4h, to_month, to_week)

# ================= 설정 =================
MIN_STAGE = int(os.getenv("MIN_STAGE", "3"))        # 3 = 1일 이상이 최상위일 때만
MIN_SCORE = int(os.getenv("MIN_SCORE", "85"))       # 이 점수 이상만 푸시 알림
WATCH_SCORE = int(os.getenv("WATCH_SCORE", "70"))   # 이 점수 이상은 결과표에만 기록
PREFILTER_RSI = float(os.getenv("PREFILTER_RSI", "40"))  # 일봉 RSI가 이 값 이하인 종목만 매시간 검사
UNIVERSE = os.getenv("UNIVERSE", "ALL")             # ALL = 전 종목, 숫자(예: 300) = 시총 상위 N개
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")
SUMMARY = os.getenv("SUMMARY", "1") == "1"         # 매시간 준비 순위 TOP5 요약 알림
WORKERS = 8

P = Params()  # 분석 세부 설정은 analysis.py 의 Params 에서 바꿀 수 있음

KST = timezone(timedelta(hours=9))
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

ETF_WORDS = ["KODEX", "TIGER", "KBSTAR", "RISE", "ACE", "SOL ", "HANARO", "ARIRANG", "KOSEF",
             "PLUS", "KIWOOM", "1Q", "TIMEFOLIO", "WON ", "TREX", "마이티", "ETN", "스팩", "히어로즈",
             "에셋플러스", "VITA", "UNICORN", "BNK", "FOCUS", "파워 "]


def log(*a):
    print(datetime.now(KST).strftime("%H:%M:%S"), *a, flush=True)


# ================= 종목 목록 =================
def _http(url, **kw):
    """요청 + 실패 이유 기록"""
    try:
        r = requests.get(url, headers={**UA, **kw.pop("headers", {})}, timeout=20, **kw)
        return r
    except Exception as e:
        log("접속 실패", url.split("?")[0], type(e).__name__)
        return None


def _universe_naver_api() -> list:
    """① 네이버 모바일 증권 API (JSON, 시총순)"""
    out = []
    for mk, market in (("KOSPI", "KS"), ("KOSDAQ", "KQ")):
        for page in range(1, 40):
            r = _http(f"https://m.stock.naver.com/api/stocks/marketValue/{mk}?page={page}&pageSize=100")
            if r is None or r.status_code != 200:
                log(f"  네이버API {mk} p{page}: {getattr(r, 'status_code', '-')}")
                break
            try:
                stocks = r.json().get("stocks", [])
            except Exception:
                log(f"  네이버API {mk}: JSON 아님 ({len(r.text)}자)")
                break
            if not stocks:
                break
            for st in stocks:
                code = str(st.get("itemCode", ""))
                if not re.fullmatch(r"\d{6}", code):
                    continue
                if str(st.get("stockEndType", "stock")).lower() not in ("stock", ""):
                    continue
                cap = int(re.sub(r"[^\d]", "", str(st.get("marketValue", "0"))) or 0)
                out.append({"code": code, "name": st.get("stockName", code), "market": market, "cap": cap})
            time.sleep(0.15)
    return out


def _universe_naver_html() -> list:
    """② 네이버 금융 시가총액 페이지"""
    out, seen = [], set()
    for sosok, market in ((0, "KS"), (1, "KQ")):
        for page in range(1, 80):
            r = _http(f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}")
            if r is None or r.status_code != 200:
                log(f"  네이버HTML {market} p{page}: {getattr(r, 'status_code', '-')}")
                break
            html = r.content.decode("euc-kr", "ignore")
            new = 0
            for row in html.split("<tr")[1:]:
                m = re.search(r'code=(\d{6})"[^>]*class="tltle"[^>]*>([^<]+)</a>', row)
                if not m or m.group(1) in seen:
                    continue
                seen.add(m.group(1))
                new += 1
                nums = re.findall(r'<td class="number">([\d,]+)</td>', row)
                cap = int(nums[2].replace(",", "")) if len(nums) > 2 else 0
                out.append({"code": m.group(1), "name": m.group(2).strip(), "market": market, "cap": cap})
            if new == 0:
                if page == 1:
                    log(f"  네이버HTML {market}: 표를 못 찾음 ({len(html)}자)")
                break
            time.sleep(0.2)
    return out


def _universe_kind() -> list:
    """③ 한국거래소 KIND 상장법인 목록 (시총 정보 없음)"""
    out = []
    for mt, market in (("stockMkt", "KS"), ("kosdaqMkt", "KQ")):
        r = _http(f"https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType={mt}")
        if r is None or r.status_code != 200:
            log(f"  KIND {market}: {getattr(r, 'status_code', '-')}")
            continue
        html = r.content.decode("euc-kr", "ignore")
        for tr in re.findall(r"<tr.*?</tr>", html, re.S):
            tds = [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            code = next((t for t in tds if re.fullmatch(r"\d{6}", t)), None)
            if code and tds:
                out.append({"code": code, "name": tds[0], "market": market, "cap": 0})
    return out


def fetch_universe() -> list:
    """코스피·코스닥 종목 목록 (여러 경로 중 되는 것 사용)"""
    out = []
    for name, fn in (("네이버API", _universe_naver_api), ("네이버HTML", _universe_naver_html), ("KIND", _universe_kind)):
        try:
            out = fn()
        except Exception as e:
            log(f"  {name} 오류: {e}")
            out = []
        log(f"종목 목록 [{name}] {len(out)}개")
        if len(out) > 500:
            break
    seen, uniq = set(), []
    for s in out:
        if s["code"] in seen or any(w in s["name"] for w in ETF_WORDS):
            continue
        seen.add(s["code"])
        uniq.append(s)
    uniq.sort(key=lambda x: -x["cap"])
    log(f"종목 목록 최종 {len(uniq)}개")
    return uniq


# ================= 시세 =================
_NAVER_DAILY_OK = [True]


def naver_daily(code: str, count: int = 1500, market: str = "KS") -> pd.DataFrame:
    """일봉: 네이버 차트 → 안 되면 야후"""
    if _NAVER_DAILY_OK[0]:
        url = f"https://fchart.stock.naver.com/sise.nhn?symbol={code}&timeframe=day&count={count}&requestType=0"
        for _ in range(2):
            r = _http(url)
            if r is None or r.status_code != 200:
                time.sleep(0.5)
                continue
            txt = r.content.decode("euc-kr", "ignore")
            rows = []
            for it in re.findall(r'data="([^"]+)"', txt):
                try:
                    d, o, h, l, c, v = it.split("|")
                    rows.append((pd.Timestamp(d), float(o), float(h), float(l), float(c), float(v)))
                except Exception:
                    pass
            if rows:
                df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"]).set_index("date")
                return df[df["close"] > 0]
            break
    return yahoo_daily(code, market, count)


def yahoo_daily(code: str, market: str, count: int = 1500) -> pd.DataFrame:
    import yfinance as yf
    period = "10y" if count > 1000 else "2y"
    try:
        d = yf.download(f"{code}.{market}", period=period, interval="1d", auto_adjust=False,
                        progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        d.index = pd.to_datetime(d.index).tz_localize(None)
        return d[d["close"] > 0].tail(count)
    except Exception:
        return pd.DataFrame()


def probe_sources():
    """시작할 때 시세 경로가 살아 있는지 확인"""
    r = _http("https://fchart.stock.naver.com/sise.nhn?symbol=005930&timeframe=day&count=30&requestType=0")
    ok = r is not None and r.status_code == 200 and 'data="' in r.content.decode("euc-kr", "ignore")
    _NAVER_DAILY_OK[0] = ok
    log(f"네이버 일봉: {'정상' if ok else '안 됨 → 야후 일봉 사용'}")
    y = yahoo_daily("005930", "KS", 30)
    log(f"야후 일봉: {'정상' if len(y) else '안 됨'} ({len(y)}개)")


def yahoo_hourly(tickers: list) -> dict:
    """야후에서 1시간봉 1년치를 한꺼번에 받음"""
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        try:
            data = yf.download(chunk, period="1y", interval="60m", group_by="ticker",
                               auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            log("야후 오류", e)
            continue
        for t in chunk:
            try:
                d = data[t] if len(chunk) > 1 else data
                d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
                if d.index.tz is None:
                    d.index = d.index.tz_localize("UTC")
                d.index = d.index.tz_convert("Asia/Seoul")
                if len(d) > 50:
                    out[t] = d
            except Exception:
                pass
        time.sleep(1)
    return out


# ================= 1단계: 하루 한 번 후보 추리기 =================
def build_candidates(today: str) -> list:
    uni = fetch_universe()
    if UNIVERSE != "ALL":
        uni = uni[:int(UNIVERSE)]
    if MIN_STAGE < 3:
        log("최소 단계가 3 미만이라 일봉 사전 필터 없이 목록 전체를 검사")
        return uni

    def chk(s):
        df = naver_daily(s["code"], 200, s["market"])
        if len(df) < 30:
            return None
        r = rsi(df["close"].to_numpy(float), P.rsi_len)[-1]
        return {**s, "d_rsi": round(float(r), 1)} if r <= PREFILTER_RSI else None

    with ThreadPoolExecutor(WORKERS) as ex:
        cands = [c for c in ex.map(chk, uni) if c]
    log(f"후보 {len(cands)}개 (일봉 RSI {PREFILTER_RSI} 이하)")
    if not uni:
        return []
    json.dump({"date": today, "items": cands}, open(STATE / "candidates.json", "w"), ensure_ascii=False, indent=1)
    return cands


def load_candidates(today: str) -> list:
    f = STATE / "candidates.json"
    if f.exists():
        j = json.load(open(f))
        if j.get("date") == today and j.get("items"):
            return j["items"]
    return build_candidates(today)


# ================= 2단계: 분석 =================
def fmt_t(t) -> str:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return "-"
    t = pd.Timestamp(t)
    return t.strftime("%y/%m/%d %H:%M") if (t.hour or t.minute) else t.strftime("%y/%m/%d")


def analyze_stock(s: dict, h1: pd.DataFrame) -> dict:
    d1 = naver_daily(s["code"], 1500, s["market"])
    if len(d1) < 60 or h1 is None:
        return None
    frames = [h1, to_4h(h1), d1, to_week(d1), to_month(d1)]
    ev = evaluate(frames, 1, P)
    ev["stock"] = s
    ev["price"] = float(d1["close"].iloc[-1])
    ev["chg"] = float(d1["close"].iloc[-1] / d1["close"].iloc[-2] - 1) * 100 if len(d1) > 1 else 0.0
    # 구조를 볼 프레임: 기본 1일, 최상위가 주·월이면 그 프레임
    sf = ev["top"] if ev["top"] >= 3 else 2
    ev["sframe"] = sf
    ev["forming"] = forming_state(frames[sf], P)
    ev["ready"] = readiness(ev["rsi"], ev["forming"], P)
    return ev


def describe(ev: dict) -> str:
    f = ev["frame"]
    c3 = BULL_CANDLE[f["l3c"]] if f.get("l3c") is not None else "-"
    c5 = BULL_CANDLE[f["l5c"]] if f.get("l5c") is not None else "-"
    return (f"{FRAMES[ev['top']]} 기준 {ev['score']}점 [{grade(ev['score'])}]\n"
            f"현재가 {ev['price']:,.0f}\n"
            f"3파 저점 {fmt_t(f['l3t'])} {f['l3p']:,.0f} · {c3} · RSI {f['l3r']:.1f} · 거래량 {f['l3vr']:.1f}배\n"
            f"5파 저점 {fmt_t(f['l5t'])} {f['l5p']:,.0f} · {c5} · RSI {f['l5r']:.1f} · 거래량 {f['l5vr']:.1f}배\n"
            f"4파 고점 {f['h4']:,.0f} (1차 익절) · 5파 깊이 {f['depth']:.2f}배 · 4파 되돌림 {f['retr4']:.2f}\n"
            f"조건: 3파거래량 {'O' if f['v3'] else 'X'} / 거래량다이버전스 {'O' if f['vd'] else 'X'} / "
            f"피보구간 {'O' if f['zone'] else 'X'} / 4파되돌림 {'O' if f['ok4'] else 'X'}")


def push(title: str, msg: str, code: str = "", priority: int = 4, tags=None):
    if not NTFY_TOPIC:
        log("NTFY_TOPIC 없음, 알림 생략:", title)
        return
    body = {"topic": NTFY_TOPIC, "title": title, "message": msg, "priority": priority,
            "tags": tags or ["chart_with_upwards_trend"]}
    body["click"] = f"divscan://open?code={code}" if code else "divscan://open"
    try:
        requests.post("https://ntfy.sh/", json=body, timeout=10)
    except Exception as e:
        log("알림 실패", e)


def _iso(t):
    if t is None:
        return None
    try:
        return pd.Timestamp(t).isoformat()
    except Exception:
        return None


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return _iso(v)
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    return v


def to_item(ev: dict) -> dict:
    s = ev["stock"]
    it = {
        "code": s["code"], "name": s["name"], "market": "코스피" if s["market"] == "KS" else "코스닥",
        "price": ev["price"], "chg": ev.get("chg", 0.0), "ready": ev.get("ready", 0),
        "signal": ev["score"] if ev["stage"] >= MIN_STAGE else 0,
        "stage": ev["stage"], "top": ev["top"], "rsi": ev["rsi"],
        "sframe": ev.get("sframe", 2), "forming": ev.get("forming", {}),
    }
    if ev.get("frame") and ev["frame"].get("act"):
        f = dict(ev["frame"])
        f["l3c_name"] = BULL_CANDLE[f["l3c"]]
        f["l5c_name"] = BULL_CANDLE[f["l5c"]]
        it["sig"] = f
    fm = it["forming"]
    if fm.get("l3c") is not None:
        fm["l3c_name"] = BULL_CANDLE[fm["l3c"]]
    return _clean(it)


def write_results(now, evs, n):
    items = [to_item(e) for e in evs]
    items.sort(key=lambda x: (-(x["signal"] or 0), -(x["ready"] or 0)))
    data = {"updated": now.isoformat(timespec="minutes"), "analyzed": n,
            "min_score": MIN_SCORE, "frames": FRAMES, "items": items[:150]}
    (STATE / "results.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return items


def push_summary(now, items):
    if not SUMMARY or not items:
        return
    top = sorted(items, key=lambda x: -(x["ready"] or 0))[:5]
    lines = []
    for i, it in enumerate(top, 1):
        d = FRAMES[2] if it["sframe"] == 2 else FRAMES[it["sframe"]]
        r = it["rsi"][it["sframe"]]
        sig = f" 🔔{it['signal']}" if it["signal"] else ""
        lines.append(f"{i}. {it['name']} {it['ready']}점{sig} · {d} RSI {r:.0f}" if r is not None
                     else f"{i}. {it['name']} {it['ready']}점{sig}")
    push(f"준비 순위 TOP5 · {now.strftime('%H:%M')}", "\n".join(lines), priority=3, tags=["bar_chart"])


def main():
    now = datetime.now(KST)
    today = now.strftime("%Y-%m-%d")
    probe_sources()
    cands = load_candidates(today)
    if not cands:
        log("후보 없음")
        write_report(now, [], 0)
        write_results(now, [], 0)
        return

    tickers = [f"{c['code']}.{c['market']}" for c in cands]
    log(f"1시간봉 받는 중 {len(tickers)}개")
    hourly = yahoo_hourly(tickers)
    log(f"1시간봉 {len(hourly)}개 받음")

    def job(c):
        try:
            return analyze_stock(c, hourly.get(f"{c['code']}.{c['market']}"))
        except Exception as e:
            log("분석 오류", c["name"], e)
            return None

    with ThreadPoolExecutor(WORKERS) as ex:
        evs = [e for e in ex.map(job, cands) if e]

    sent_f = STATE / "sent.json"
    sent = json.load(open(sent_f)) if sent_f.exists() else {}
    hits = []
    for ev in evs:
        if ev["score"] < WATCH_SCORE or ev["stage"] < MIN_STAGE:
            continue
        hits.append(ev)
        if ev["score"] < MIN_SCORE:
            continue
        s = ev["stock"]
        key = f"{s['code']}|{ev['top']}|{fmt_t(ev['frame']['l5t'])}"
        if key in sent:
            continue
        push(f"[매수 {ev['score']}점] {s['name']}", describe(ev), s["code"])
        sent[key] = now.isoformat(timespec="minutes")
        log("알림:", s["name"], ev["score"])

    # 30일 지난 기록 정리
    cut = (now - timedelta(days=30)).isoformat()
    sent = {k: v for k, v in sent.items() if v >= cut}
    json.dump(sent, open(sent_f, "w"), ensure_ascii=False, indent=1)
    write_report(now, hits, len(evs))
    items = write_results(now, evs, len(evs))
    push_summary(now, items)


def write_report(now, hits, n):
    hits = sorted(hits, key=lambda e: -e["score"])
    lines = [f"# 다이버전스 스캔 결과", "",
             f"- 스캔 시각: {now.strftime('%Y-%m-%d %H:%M')} (KST)",
             f"- 분석 종목: {n}개 / 표시 기준: {WATCH_SCORE}점 이상, 알림 기준: {MIN_SCORE}점 이상, 최소 단계: {FRAMES[MIN_STAGE - 1]}", ""]
    if not hits:
        lines.append("지금은 기준에 맞는 신호가 없어요.")
    for ev in hits:
        s = ev["stock"]
        mark = "🔔 " if ev["score"] >= MIN_SCORE else ""
        lines += [f"## {mark}{s['name']} ({s['code']}) — {ev['score']}점 [{grade(ev['score'])}]",
                  "```", describe(ev), "```", ""]
    (STATE / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"결과 저장: 신호 {len(hits)}개")


if __name__ == "__main__":
    if "--test-push" in sys.argv:
        push("스캐너 연결 확인", "알림이 잘 오면 설정 완료예요.", "005930")
    else:
        main()
