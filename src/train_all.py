"""Train all 3 models for every ticker in the universe and save predictions."""

from __future__ import annotations

from src.config import load_config
from src.train import run_pipeline


def main() -> None:
    cfg = load_config()
    tickers = cfg["universe"]["tickers"]
    print(f"Training {len(tickers)} tickers: {tickers}")
    for t in tickers:
        print(f"\n===== {t} =====")
        try:
            run_pipeline(cfg, ticker=t, mlflow_enable=False)
        except Exception as e:
            print(f"  {t} FAILED: {e}")


if __name__ == "__main__":
    main()