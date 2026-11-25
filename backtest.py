"""백테스트 실행 (전략 선택: BT_MODE = 파동하모닉 / 이평선지지 / 프랙탈 / 횡보돌파 / 바닥반등)"""
import os

mode = os.getenv("BT_MODE", "이평선지지") or "이평선지지"
if mode.startswith("바닥"):
    import deep_backtest as m
elif mode.startswith("횡보"):
    import base_backtest as m
elif mode.startswith("파동"):
    import wave_backtest as m
elif mode.startswith("프랙탈"):
    import fractal_backtest as m
else:
    import ma_backtest as m

if __name__ == "__main__":
    m.main()
