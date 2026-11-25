"""
깊은 바닥 반등 전략 (스캐너·백테스트 공통 규칙)

백테스트(5년·1,000종목)에서 큰 반등(+50%)이 많이 나온 조건만 남긴 규칙이에요.

필수 조건 (하나라도 안 맞으면 신호 없음)
  - 1일봉 또는 주봉 RSI 다이버전스 확정 (저점 확정 2봉)
  - 52주 고점 대비 -45% 이상 하락한 상태
  - 5파 깊이 1.27배 이상 (4파 반등폭 대비)
  - 진입가가 5파 저점보다 위 (저점이 이미 깨졌으면 안 삼)
  - 재무 ⚠위험 아님 (관리종목·자본잠식·3년 연속 적자 등)  ← 실시간 스캐너에서만 확인

점수 (100점)
  필수 조건 충족 50
  + 코스피가 120일선 아래 (시장 전체 약세) 20
  + 월봉 RSI 35 미만 15
  + 주봉 다이버전스 기준 10
  + 3파 거래량 4배 이상 (투매) 5

등급 · 1회 최대 손실(원금 대비) · 종목당 최대 비중
  A (85점↑) : 1.25% · 20%
  B (70점↑) : 1.0%  · 15%
  C (50점↑) : 0.5%  · 10%
  비중 = min(최대 비중, 1회 손실 ÷ 손절폭)

계좌 한도: 하루 신규 3종목 · 동시 보유 8종목 · 총 보유 70%

청산
  손절: 5파 저점 종가 이탈
  수익: 최고가가 +30%를 넘은 뒤 최고가 대비 -25% 종가 이탈 (추적 손절)
  최대 보유 250거래일
"""
import numpy as np

DD_MIN = 45.0        # 52주 고점 대비 최소 하락 %
DEPTH_MIN = 1.27     # 5파 깊이 최소
MRSI_LOW = 35.0
VOL_CLIMAX = 4.0

GRADES = {  # 등급: (최소 점수, 1회 최대 손실, 종목당 최대 비중)
    "A": (85, 0.0125, 0.20),
    "B": (70, 0.010, 0.15),
    "C": (50, 0.005, 0.10),
}
DAILY_NEW = 3
MAX_OPEN = 8
MAX_EXPOSURE = 0.70

TRAIL_ARM = 0.30     # 최고가가 이만큼 오르면 추적 손절 시작
TRAIL_GAP = 0.25     # 최고가 대비 이만큼 빠지면 매도
HOLD_MAX = 250


def qualifies(dd52, depth, entry, l5, fin_grade=None):
    if dd52 is None or np.isnan(dd52) or dd52 < DD_MIN:
        return False
    if depth is None or np.isnan(depth) or depth < DEPTH_MIN:
        return False
    if not entry > l5:
        return False
    if fin_grade == "위험":
        return False
    return True


def score(market_weak, mrsi, weekly_base, l3vr):
    s = 50
    s += 20 if market_weak else 0
    s += 15 if (mrsi is not None and not np.isnan(mrsi) and mrsi < MRSI_LOW) else 0
    s += 10 if weekly_base else 0
    s += 5 if (l3vr is not None and l3vr >= VOL_CLIMAX) else 0
    return s


def grade(sc):
    for g, (mn, _, _) in GRADES.items():
        if sc >= mn:
            return g
    return None


def weight(g, entry, stop):
    """원금 대비 비중 (0~1)"""
    if g not in GRADES or entry <= stop:
        return 0.0
    _, risk, cap = GRADES[g]
    dist = (entry - stop) / entry
    return float(min(cap, risk / max(dist, 0.01)))


def trail_exit(peak, entry, close):
    """추적 손절 조건: 최고가가 +30% 넘었고, 종가가 최고가 대비 -25% 아래"""
    return peak >= entry * (1 + TRAIL_ARM) and close < peak * (1 - TRAIL_GAP)
