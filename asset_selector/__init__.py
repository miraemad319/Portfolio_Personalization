"""
EGX30 Asset Selector
====================
Standalone pipeline that classifies EGX30 stocks into three risk profiles
(conservative, balanced, aggressive) using a Reinforcement Learning agent
trained on rolling OHLCV feature windows.

Quick start
-----------
    from asset_selector import run_pipeline
    classification, benchmarking_map = run_pipeline()

Modules
-------
    download_egx_prices    Download OHLCV price CSVs from EGX
    retry_download         Retry failed ticker downloads
    asset_selector         RL risk profiling
    visualizer             Plots
    main                   Pipeline orchestration
"""

from asset_selector.main import run_pipeline
from asset_selector.asset_selector import classify_assets, get_profile_tickers

__all__ = [
    "run_pipeline",
    "classify_assets",
    "get_profile_tickers",
]
