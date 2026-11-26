"""백테스트 실행 (전략 선택: BT_MODE = 경로저장 / 타이밍 / 유명기법 / 수급섹터 / 진입검증 / 단기스윙 / 조합탐색 / 혼합매도 / 손절분석 / 점수분석 / 매도분석 / 통합 / 파동하모닉 / 이평선지지 / 프랙탈 / 횡보돌파 / 바닥반등)"""
import os

mode = os.getenv("BT_MODE", "이평선지지") or "이평선지지"
if mode.startswith("바닥"):
    import deep_backtest as m
elif mode.startswith("횡보"):
    import base_backtest as m
elif mode.startswith("경로"):
    import paths_backtest as m
elif mode.startswith("타이밍"):
    import timing_backtest as m
elif mode.startswith("유명"):
    import famous_backtest as m
elif mode.startswith("수급"):
    import flow_backtest as m
elif mode.startswith("진입"):
    import entry_backtest as m
elif mode.startswith("단기"):
    import swing_backtest as m
elif mode.startswith("조합"):
    import feature_backtest as m
elif mode.startswith("혼합"):
    import hybrid_backtest as m
elif mode.startswith("손절"):
    import stop_backtest as m
elif mode.startswith("점수"):
    import score_backtest as m
elif mode.startswith("매도"):
    import exit_backtest as m
elif mode.startswith("통합"):
    import combo_backtest as m
elif mode.startswith("파동"):
    import wave_backtest as m
elif mode.startswith("프랙탈"):
    import fractal_backtest as m
else:
    import ma_backtest as m

if __name__ == "__main__":
    m.main()
    # 모드별로도 따로 남겨 둠 (다른 백테스트가 backtest.md 를 덮어써도 결과가 안 사라지게)
    import shutil
    from pathlib import Path
    src = Path(__file__).resolve().parent / "state" / "backtest.md"
    if src.exists():
        shutil.copy(src, src.with_name(f"결과_{mode}.md"))
