"""백테스트 실행 (전략 선택: BT_MODE = 횡보돌파 / 바닥반등)"""
import os

mode = os.getenv("BT_MODE", "횡보돌파") or "횡보돌파"
if mode.startswith("바닥"):
    import deep_backtest as m
else:
    import base_backtest as m

if __name__ == "__main__":
    m.main()
