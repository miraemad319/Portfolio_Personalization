"""
EGX30 Historical Tickers — Download Expectations
-------------------------------------------------
First run downloads ~59/63 tickers successfully.

Known issues (not code bugs):
  - EKHO, EKHOA : active stocks but provider rate-limits them consistently.
  - ARAB         : active stock (formerly PORT) but same rate-limit issue.
  - ESRS         : delisted from EGX March 2025, provider won't serve it. Skip.
  - GDWA, PRDC   : only ~1086 rows instead of 1500, limited provider history.

Rebrand merges (old ticker prepended to new for longer history):
  - GBCO  (GBAU→GBCO) : merges successfully.
  - MASR  (MNHD→MASR) : merge may fail due to timeouts, retries automatically.
  - ASPI  (PIOH→ASPI) : same as above.
  - ARAB  (PORT→ARAB)  : ARAB must download first, then merge retries on next run.

Expected final state: 59 CSVs on disk (63 minus ESRS, EKHO, EKHOA, ARAB).
Keep retrying the 3 active ones in fresh terminal sessions — they will
eventually succeed. ESRS is a permanent skip.

download_egx_prices.py
----------------------
Downloads OHLCV price data for EGX stocks.

HOW IT WORKS
------------

First run (no CSVs on disk yet):
  1. Load all 64 historical tickers from assets_egx30.json.
  2. Scrape the live EGX30 from the EGX website.
  3. Union both lists → download everything.
  4. Run rebrand merges (MNHD→MASR, GBAU→GBCO).

Every subsequent run:
  1. Scrape the live EGX30.
  2. Check which tickers from the scrape don't have a CSV yet.
  3. Download ONLY those new tickers (everything else is already on disk).
  4. This handles index rebalancing automatically — when the EGX30
     changes in June (or any other date), the new entrants get
     downloaded on the next run without touching existing data.

The rule is simple:
  CSV exists on disk  →  skip
  CSV does not exist  →  download
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.egxlytics_client import EGXDataClient
from src.ingestion.parse_egx30 import get_all_tickers

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent        # asset_selector/
PROJECT_ROOT = SCRIPT_DIR.parent                      # Portfolio_Personalization/
CONFIG_PATH  = SCRIPT_DIR / "assets_egx30.json"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "raw" / "prices"

N_BARS = 1500  # ~5 years of trading days


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def already_downloaded(ticker: str, output_dir: Path) -> bool:
    """Return True if a non-empty CSV exists for this ticker."""
    path = output_dir / f"{ticker}.csv"
    return path.exists() and path.stat().st_size > 0


def download_ticker(client: EGXDataClient, ticker: str, output_dir: Path) -> bool:
    """Download OHLCV for one ticker and save to CSV. Returns True on success."""
    df = client.fetch_stock_prices(ticker, N_BARS)
    if df is None or df.empty:
        logger.warning(f"  ✗ No data for {ticker}")
        return False

    path = output_dir / f"{ticker}.csv"
    df.to_csv(path)
    logger.info(f"  ✓ {ticker}: {len(df)} rows saved")
    return True


# ---------------------------------------------------------------------------
# Rebrand merge
# ---------------------------------------------------------------------------

def merge_rebranded_tickers(
    rebrand_map: dict,
    output_dir: Path,
    client: EGXDataClient,
) -> None:
    for old_ticker, new_ticker in rebrand_map.items():
        logger.info(f"\nRebrand merge: {old_ticker} → {new_ticker}")

        new_path = output_dir / f"{new_ticker}.csv"
        old_path = output_dir / f"{old_ticker}.csv"
        sentinel = output_dir / f".{old_ticker}_merged"  # ← hidden file, e.g. .MNHD_merged

        # ── Skip if already merged on a previous run ──────────────────────────
        if sentinel.exists():
            logger.info(f"  Already merged on a previous run — skipping.")
            continue

        if not new_path.exists():
            logger.warning(
                f"  {new_ticker}.csv not found — skipping merge. "
                f"Run the main download first."
            )
            continue

        # Use pre-downloaded CSV if available (put there by retry_download.py),
        # otherwise fetch live. This lets retry_download.py handle stubborn tickers.
        if old_path.exists():
            logger.info(f"  Using existing {old_ticker}.csv (pre-downloaded by retry)")
        else:
            logger.info(f"  Fetching old-ticker history: {old_ticker}")
            df_old_raw = client.fetch_stock_prices(old_ticker, N_BARS)
            if df_old_raw is None or df_old_raw.empty:
                logger.warning(
                    f"  No data returned for old ticker {old_ticker}. "
                    f"Merge skipped — {new_ticker}.csv stays as-is."
                )
                continue
            df_old_raw.to_csv(old_path)

        df_old = pd.read_csv(old_path, index_col=0, parse_dates=True)
        df_new = pd.read_csv(new_path, index_col=0, parse_dates=True)

        df_merged = pd.concat([df_old, df_new]).sort_index()
        df_merged = df_merged[~df_merged.index.duplicated(keep="last")]

        df_merged.to_csv(new_path)
        logger.info(
            f"  Merged: {len(df_old)} rows ({old_ticker}) + "
            f"{len(df_new)} rows ({new_ticker}) "
            f"= {len(df_merged)} unique rows → saved as {new_ticker}.csv"
        )

        old_path.unlink()
        logger.info(f"  Deleted {old_ticker}.csv")

        # ── Mark as done so future runs skip this pair ─────────────────────────
        sentinel.touch()
        logger.info(f"  Sentinel written: {sentinel.name}")

def resolve_tickers_to_download(
    historical: list,
    current_scraped: list,
    output_dir: Path,
    rebrand_map: dict,
) -> dict:
    """
    Build the download plan for this run.

    Universe = historical_tickers ∪ current_scraped_tickers
               minus old rebrand tickers (handled separately in merge step)

    For each ticker in universe:
      CSV on disk?  →  skip
      No CSV?       →  download

    On first run, no CSVs exist so everything gets downloaded.
    On subsequent runs, only tickers without a CSV get downloaded —
    which will typically be only the new index entrants from the scrape.
    """
    old_tickers = set(rebrand_map.keys())

    universe = sorted(
        (set(historical) | set(current_scraped)) - old_tickers
    )

    already_have = [t for t in universe if already_downloaded(t, output_dir)]
    to_download  = [t for t in universe if not already_downloaded(t, output_dir)]

    # Tickers from the live scrape that are not in our historical list
    # — these are new index entrants we haven't seen before
    new_from_index = sorted(
        set(current_scraped) - set(historical) - old_tickers
    )

    is_first_run = len(already_have) == 0

    return {
        "to_download":    to_download,
        "already_have":   already_have,
        "new_from_index": new_from_index,
        "is_first_run":   is_first_run,
        "universe":       universe,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_all_prices() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config      = load_config()
    client      = EGXDataClient()
    rebrand_map = config.get("rebrand_map", {})
    historical  = config.get("historical_tickers", [])

    # ── Step 1: Scrape live EGX30 ─────────────────────────────────────────────
    ticker_result = get_all_tickers(CONFIG_PATH)

    if ticker_result["scrape_ok"]:
        logger.info(
            f"Live EGX30 scrape succeeded: "
            f"{len(ticker_result['current'])} tickers found"
        )
    else:
        logger.warning(
            "Live EGX30 scrape failed — falling back to historical list only."
        )

    current_scraped = ticker_result["current"]

    # ── Step 2: Build download plan ───────────────────────────────────────────
    plan = resolve_tickers_to_download(
        historical, current_scraped, OUTPUT_DIR, rebrand_map
    )

    # ── Step 3: Log what this run will do ─────────────────────────────────────
    logger.info(f"\n{'='*60}")
    if plan["is_first_run"]:
        logger.info(
            f"FIRST RUN — no existing data found.\n"
            f"Will download all {len(plan['universe'])} tickers."
        )
    else:
        logger.info(
            f"INCREMENTAL RUN\n"
            f"  Already on disk : {len(plan['already_have'])} tickers → skipping\n"
            f"  Missing CSVs    : {len(plan['to_download'])} tickers → downloading"
        )

    if plan["new_from_index"]:
        logger.info(
            f"\n⚠ New tickers in live EGX30 not in our historical list:\n"
            f"  {plan['new_from_index']}\n"
            f"  These will be downloaded now. Add them to assets_egx30.json\n"
            f"  so they are included in future runs even if the scrape fails."
        )

    logger.info(f"{'='*60}")

    # ── Step 4: Download missing tickers ─────────────────────────────────────
    if not plan["to_download"]:
        logger.info("Nothing to download — all tickers already on disk.")
    else:
        successful, failed = [], []

        for i, ticker in enumerate(plan["to_download"], 1):
            logger.info(f"[{i}/{len(plan['to_download'])}] {ticker}")
            ok = download_ticker(client, ticker, OUTPUT_DIR)
            (successful if ok else failed).append(ticker)

        logger.info(f"\n{'='*60}")
        logger.info(f"Download summary")
        logger.info(f"  ✓ Successful : {len(successful)}")
        logger.info(f"  ✗ Failed     : {len(failed)}")
        if failed:
            logger.warning(f"  Failed tickers: {', '.join(failed)}")

    # ── Step 5: Rebrand merges ────────────────────────────────────────────────
    # Always runs — safe to re-run because merge checks if new_ticker CSV
    # already contains the old history before re-merging.
    if rebrand_map:
        logger.info(f"\n{'='*60}")
        logger.info("Running rebrand merges...")
        merge_rebranded_tickers(rebrand_map, OUTPUT_DIR, client)

    # ── Final state ───────────────────────────────────────────────────────────
    total_on_disk = sum(1 for _ in OUTPUT_DIR.glob("*.csv"))
    logger.info(f"\n{'='*60}")
    logger.info(f"Done. Total CSV files on disk: {total_on_disk}")


if __name__ == "__main__":
    download_all_prices()