"""
매수·매도 타이밍 넓게 비교 — 같은 바닥 신호에서 진입 방식·매도 방식만 바꿈 (상장폐지 포함)

기준(지금 방식): 다음 날 시가 전량 → 5파 저점 종가 이탈 손절 → 위쪽 주·월봉 60/120선(없으면 +50%)에서 절반 → 최고가 -20%

진입 방식 (매도는 기준과 같음)
  분할 5일고가   : 시가에 절반 · 20일 안에 종가가 신호 전 5일 고가를 넘으면 나머지 절반
  분할 4파고점   : 시가에 절반 · 20일 안에 종가가 4파 고점을 넘으면 나머지 절반
  3분할          : 시가 1/3 · 5일 고가 돌파 1/3 · 20일선 회복 1/3
  확인 후 전량   : 10일 안에 5일 고가 돌파하는 날 종가에 전량 (안 오면 안 삼)
  지정가 -1/-2/-3%: 신호일 종가보다 1~3% 낮은 지정가, 3일 안에 안 닿으면 안 삼
  손절 후 재진입 : 5파 저점 이탈로 손절된 뒤 20일 안에 종가가 옛 5파 저점 위로 돌아오면 다시 삼 (새 손절 = 그 사이 최저가)
매도 방식 (진입은 기준과 같음)
  추적 폭        : 최고가 -10/-15/-25/-30/-35%
  저항선 매도 비율: 0 · 1/3 · 2/3 · 전량
  1차 익절 위치  : +30/+50/+80/+100% (저항선 대신)
  하락 다이버전스: +20% 넘은 뒤 일봉(또는 주봉) 하락 다이버전스 나오면 나머지 매도 (추적 -30%는 안전장치로 유지)
  50일 신저가    : 나머지는 50일 최저가 종가 이탈 때 매도 (느슨한 추세 추종)
섞은 것: 분할 5일고가 + 추적 25% · 분할 5일고가 + 하락 다이버전스 · 재진입 + 하락 다이버전스
"""
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

import scan
import strategy as S
import stop_backtest as SBT
import score_backtest as SB
import entry_backtest as ENB
from analysis import rsi, to_week
from ma_backtest import log

BASE = "기준"
MODES = [BASE,
         "진입 분할 5일고가", "진입 분할 4파고점", "진입 3분할", "진입 확인 후 전량",
         "진입 지정가 -1%", "진입 지정가 -2%", "진입 지정가 -3%", "진입 손절 후 재진입",
         "추적 -10%", "추적 -15%", "추적 -25%", "추적 -30%", "추적 -35%",
         "저항 매도 0", "저항 매도 1/3", "저항 매도 2/3", "저항 매도 전량",
         "1차 +30%", "1차 +50%", "1차 +80%", "1차 +100%",
         "하락 다이버전스 일봉", "하락 다이버전스 주봉", "50일 신저가",
         "섞음 분할5일 + 추적25", "섞음 분할5일 + 다이버전스", "섞음 재진입 + 다이버전스"]
BUY_COST, SELL_COST = SBT.BUY_COST, SBT.SELL_COST
_t = threading.local()


def prep(d1):
    """종목마다 한 번: 고가 피벗(하락 다이버전스용), 주봉 하락 다이버전스 날짜, 50일 최저가, 20일선"""
    C, H, L = (d1[c].to_numpy(float) for c in ("close", "high", "low"))
    n = len(C)
    r = np.asarray(rsi(C, 14), float)
    div_d = np.zeros(n, bool)
    piv = []
    for k in range(3, n - 3):
        if H[k] >= H[k - 3:k].max() and H[k] > H[k + 1:k + 4].max():
            j = k + 3                              # 오른쪽 3봉 지나야 확정
            if piv and k - piv[-1] <= 60 and H[k] > H[piv[-1]] and r[k] < r[piv[-1]] - 2:
                div_d[j] = True
            piv.append(k)
    wk = to_week(d1)
    wC, wH = wk["close"].to_numpy(float), wk["high"].to_numpy(float)
    wr = np.asarray(rsi(wC, 14), float)
    div_w = np.zeros(n, bool)
    wp = []
    for k in range(2, len(wk) - 2):
        if wH[k] >= wH[k - 2:k].max() and wH[k] > wH[k + 1:k + 3].max():
            if wp and k - wp[-1] <= 26 and wH[k] > wH[wp[-1]] and wr[k] < wr[wp[-1]] - 2:
                a = d1.index.searchsorted(wk.index[k + 2] + pd.Timedelta(days=6), side="right") - 1
                if 0 <= a < n:
                    div_w[a] = True
            wp.append(k)
    Ls = pd.Series(L)
    _t.f = dict(div_d=div_d, div_w=div_w, low50=Ls.rolling(50).min().shift(1).to_numpy(),
                ma20=pd.Series(C).rolling(20).mean().to_numpy(),
                hi5=pd.Series(H).rolling(5).max().shift(1).to_numpy())
    _t.fid = id(d1)


def extra(d1, i, snap, wb, ei, entry, l5, t):
    if getattr(_t, "fid", None) != id(d1):
        prep(d1)
    _t.det, _t.h4 = i, float(snap.get("h4") or np.nan)
    s = getattr(ENB._t, "s", {})
    return dict(delisted=bool(s.get("_dl")), det_date=d1.index[i].strftime("%Y-%m-%d"))


def path(O, H, L, C, ei, stop0, mode):
    """반환: (1원 기준 가치 배열, 사유, 마지막 위치). 못 산 경우 vals=[1.0] 에 사유 '미체결'"""
    f, t = _t.f, SBT._tl
    n = len(C)
    det = _t.det
    # ── 설정 ──
    trail, frac, tp_fix, div, low50 = 0.20, 0.5, None, None, False
    entry_mode, reentry = "open", False
    m = mode
    if m.startswith("추적 -"):
        trail = int(m.split("-")[1].rstrip("%")) / 100
    elif m.startswith("저항 매도"):
        frac = {"0": 0.0, "1/3": 1 / 3, "2/3": 2 / 3, "전량": 1.0}[m.split()[-1]]
    elif m.startswith("1차 +"):
        tp_fix = int(m.split("+")[1].rstrip("%")) / 100
    elif m == "하락 다이버전스 일봉":
        div, trail = "d", 0.30
    elif m == "하락 다이버전스 주봉":
        div, trail = "w", 0.30
    elif m == "50일 신저가":
        low50, trail = True, None
    elif m.startswith("진입 "):
        entry_mode = m[3:]
    elif m.startswith("섞음"):
        entry_mode = "분할 5일고가" if "분할5일" in m else "open"
        reentry = "재진입" in m
        if "추적25" in m:
            trail = 0.25
        if "다이버전스" in m:
            div, trail = "d", 0.30
    if entry_mode == "손절 후 재진입":
        reentry, entry_mode = True, "open"
    # ── 진입 ──
    legs = []                                       # (위치, 가격, 비율)
    hi5 = f["hi5"][det] if np.isfinite(f["hi5"][det]) else H[det]
    if entry_mode == "open":
        legs = [(ei, O[ei], 1.0)]
    elif entry_mode.startswith("지정가"):
        pct = int(entry_mode.split("-")[1].rstrip("%")) / 100
        lim = C[det] * (1 - pct)
        for j in range(ei, min(n, ei + 3)):
            if L[j] <= lim:
                legs = [(j, min(O[j], lim), 1.0)]
                break
        if not legs:
            return np.array([1.0]), "미체결", ei
    elif entry_mode == "확인 후 전량":
        for j in range(ei, min(n, ei + 10)):
            if C[j] < stop0:
                break
            if C[j] > hi5:
                legs = [(j, C[j], 1.0)]
                break
        if not legs:
            return np.array([1.0]), "미체결", ei
    else:
        legs = [(ei, O[ei], 0.5 if entry_mode != "3분할" else 1 / 3)]
    first = legs[0][0]
    entry = legs[0][1]
    last = min(n - 1, first + S.HOLD_MAX)
    ups = [mm[first] for mm in t.res if np.isfinite(mm[first]) and mm[first] >= entry * 1.10]
    rline = min(min(ups), entry * 1.5) if ups else entry * 1.5
    tp_px = entry * (1 + tp_fix) if tp_fix else rline
    # 상태
    cost_total = sum(fr for _, _, fr in legs)
    shares = sum(fr / px for _, px, fr in legs)      # 1원 배분 기준 보유 수량
    cash = -sum(fr for _, _, fr in legs) * (1 + BUY_COST)
    pend = []
    if entry_mode == "분할 5일고가":
        pend = [("5일고가", 0.5)]
    elif entry_mode == "분할 4파고점":
        pend = [("4파고점", 0.5)]
    elif entry_mode == "3분할":
        pend = [("5일고가", 1 / 3), ("20일선", 1 / 3)]
    stop, peak, tp, sold_frac = stop0, entry, False, 0.0
    vals, reason = [], "기간 만료"
    re_used, stopped_at, lowest = False, None, np.inf
    j = first
    while j <= last:
        c = C[j]
        # 대기 중인 추가 매수 (20일 안)
        if pend and j > first and j - first <= 20 and not tp:
            keep = []
            for kind, fr in pend:
                lvl = hi5 if kind == "5일고가" else (_t.h4 if kind == "4파고점" else f["ma20"][j])
                if np.isfinite(lvl) and c > lvl and c > stop:
                    shares += fr / c; cash -= fr * (1 + BUY_COST); cost_total += fr
                else:
                    keep.append((kind, fr))
            pend = keep
        elif pend and j - first > 20:
            pend = []
        if shares > 0:
            peak = max(peak, H[j])
            # 1차 익절
            if not tp and frac > 0 and H[j] >= tp_px and j >= first:
                px = max(O[j], tp_px) if j > first else tp_px
                q = shares * frac
                cash += q * px * (1 - SELL_COST); shares -= q; tp = True
                pend = []
                if shares <= 1e-12:
                    vals.append(1 + cash)
                    return np.array(vals), "전량 익절", j
            elif not tp and frac == 0 and H[j] >= tp_px:
                tp = True
            why = None
            gain_ok = peak >= entry * 1.2
            if c < stop:
                why = "5파 이탈"
            elif div == "d" and gain_ok and f["div_d"][j]:
                why = "하락 다이버전스(일봉)"
            elif div == "w" and gain_ok and f["div_w"][j]:
                why = "하락 다이버전스(주봉)"
            elif trail is not None and (tp or peak >= tp_px) and c < peak * (1 - trail):
                why = f"추적 -{int(trail * 100)}%"
            elif low50 and tp and c < f["low50"][j]:
                why = "50일 신저가"
            if why:
                cash += shares * c * (1 - SELL_COST); shares = 0.0
                reason = why
                pend = []
                if why == "5파 이탈" and reentry and not re_used:
                    stopped_at, lowest = j, L[j]
                else:
                    vals.append(1 + cash)
                    return np.array(vals), reason, j
        else:
            if stopped_at is None:                   # 전량 익절 등으로 다 판 경우
                vals.append(1 + cash)
                return np.array(vals), reason if reason != "기간 만료" else "전량 익절", j
            # 재진입 대기
            lowest = min(lowest, L[j])
            if j - stopped_at > 20:
                vals.append(1 + cash)
                return np.array(vals), "5파 이탈", j
            if c > stop0:
                fr = max(0.05, 1 + cash)             # 남은 돈(첫 거래 손익 반영)으로 다시 삼
                shares = fr / c; cash -= fr * (1 + BUY_COST); cost_total += fr
                stop, peak, re_used, tp, entry = lowest * 0.99, c, True, False, c
                ups = [mm[j] for mm in t.res if np.isfinite(mm[j]) and mm[j] >= c * 1.10]
                rline = min(min(ups), c * 1.5) if ups else c * 1.5
                tp_px = rline
        vals.append(1 + cash + shares * c)
        j += 1
    cash += shares * C[last] * (1 - SELL_COST)
    if vals:
        vals[-1] = 1 + cash
    else:
        vals = [1 + cash]
    return np.array(vals), reason, last


def report(sigs, n, t0):
    df = pd.DataFrame([{k: v for k, v in s.items() if k not in MODES} for s in sigs])
    for m in MODES:
        df[m] = [(s[m]["ret"] if s[m]["reason"] != "미체결" else np.nan) for s in sigs]
        df[m + " 일수"] = [len(s[m]["vals"]) for s in sigs]
    df["year"] = pd.to_datetime(df.date).dt.year
    groups = {"예전 A + 새 점수 강": df.old_a & df.strong, "예전 A": df.old_a}
    L = ["# 매수·매도 타이밍 비교 (상장폐지 포함)", "",
         f"- 실행: {datetime.now().strftime('%Y-%m-%d %H:%M')} · {n}종목 · 신호 {len(df)}건 · 상폐 신호 {int(df.delisted.sum())}건",
         "- 분할 매수는 '처음 정한 금액 전체' 기준 수익률 (안 산 부분은 현금) · 하루당 = 수익 합 ÷ 보유일 합", ""]
    for gname, m in groups.items():
        g = df[m]
        L += [f"## {gname} ({len(g)}건)", "",
              "| 방식 | 체결 | 승률 | 평균 | 평균 이익 | 평균 손실 | 손익비 | PF | 배운 PF | 검증 PF | 보유일 | 하루당 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = []
        for md in MODES:
            v = g[md].dropna()
            s = SB.stat(v)
            if not s:
                continue
            a, b = SB.stat(g[g.year <= 2023][md].dropna()), SB.stat(g[g.year >= 2024][md].dropna())
            dd = g.loc[v.index, md + " 일수"]
            rows.append((md, f"| {md}{' ◀' if md == BASE else ''} | {len(v)} | {s['win']:.0f}% | {s['avg']:+.2f}% | {s['aw']:+.1f}% | "
                             f"{s['al']:+.1f}% | {SB.f2(s['rr'])} | {SB.f2(s['pf'])} | {SB.f2(a['pf']) if a else '-'} | "
                             f"{SB.f2(b['pf']) if b else '-'} | {dd.mean():.0f} | {v.sum() / dd.sum():.3f}% |"))
        L += [r for _, r in rows] + [""]
    L += ["- 계좌 결과(운 범위 포함)는 timing_trades.csv.gz 로 따로 계산해요.", "", f"_계산 시간 {time.time() - t0:.0f}초_"]
    (scan.STATE / "backtest.md").write_text("\n".join(L), encoding="utf-8")
    keep = ["code", "name", "date", "det_date", "year", "old_a", "strong", "new_score", "stop_dist", "delisted"] + \
           [c for c in df.columns if c in MODES or c.endswith(" 일수")]
    df[keep].to_csv(scan.STATE / "timing_trades.csv.gz", index=False, compression="gzip")
    log("보고서 저장")


def main():
    since = pd.Timestamp(datetime.now()) - pd.Timedelta(days=int(365 * SBT.YEARS))
    dl = ENB.fetch_delisted(since)
    log(f"상장폐지 목록 {len(dl)}종목")
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
    SBT.MODES = MODES
    SBT.path = path
    SBT.report = report
    SBT.EXTRA = extra
    SBT.stock_sigs = ENB.stock_sigs
    SBT.main()


if __name__ == "__main__":
    main()
