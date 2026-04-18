"""
parse_egx30.py
--------------
Parses the current EGX30 index constituents from a manually downloaded file.

WHY MANUAL:
The EGX website actively blocks all automated requests including headless
Chrome (ERR_CONNECTION_RESET). Manual download takes 2 seconds and is reliable.

HOW TO UPDATE (every 6 months — January and July when EGX30 rebalances):
  1. Go to: https://www.egx.com.eg/en/CurrentIndexConstituntes.aspx?type=1&Nav=1
  2. Click the Excel icon (top right of the table)
  3. Save the file to: data/raw/egx30_constituents.xls
  4. Run download_egx_prices.py — it picks up the new file automatically.

If the file is missing, falls back to historical_tickers from assets_egx30.json
so the pipeline never hard-stops.
"""

import json
import logging
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
XLS_PATH     = PROJECT_ROOT / "data" / "raw" / "egx30_constituents.xls"
CONFIG_PATH  = PROJECT_ROOT / "src" / "config" / "assets_egx30.json"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_tickers_from_file(filepath: Path) -> list[str]:
    """
    Parse tickers from the EGX Excel export (HTML disguised as XLS).
    Reuters codes appear as ABUK.CA — strip .CA to get EGX ticker.
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        logger.warning("No table found in file.")
        return []

    tickers = []
    for row in table.find_all("tr")[1:]:  # skip header
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) >= 2:
            ticker = cells[1].replace(".CA", "").strip()
            if ticker:
                tickers.append(ticker)

    return sorted(tickers)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_egx30() -> list[str]:
    """
    Parse the manually downloaded EGX30 constituents file.
    Returns empty list if file doesn't exist or parsing fails.
    """
    if not XLS_PATH.exists():
        logger.warning(
            f"EGX30 constituents file not found at: {XLS_PATH}\n"
            f"  → Download from: https://www.egx.com.eg/en/CurrentIndexConstituntes.aspx?type=1&Nav=1\n"
            f"  → Save as: data/raw/egx30_constituents.xls\n"
            f"  → Falling back to historical tickers only."
        )
        return []

    tickers = _parse_tickers_from_file(XLS_PATH)

    if len(tickers) >= 20:
        logger.info(f"Parsed {len(tickers)} tickers from {XLS_PATH.name}")
        return tickers

    logger.warning(f"Only {len(tickers)} tickers parsed — too few, ignoring.")
    return []


def load_historical_tickers(config_path: Path = CONFIG_PATH) -> list[str]:
    """Load historical_tickers from assets_egx30.json."""
    with open(config_path, "r") as f:
        return json.load(f)["historical_tickers"]


def get_all_tickers(config_path: Path = CONFIG_PATH) -> dict:
    """
    Returns:
      - current   : live EGX30 tickers from XLS file (empty if file missing)
      - historical: all tickers from config
      - combined  : union of both, deduplicated and sorted
      - scrape_ok : True if XLS file was parsed successfully
    """
    current    = get_current_egx30()
    historical = load_historical_tickers(config_path)
    combined   = sorted(set(current) | set(historical))

    return {
        "current":    current,
        "historical": historical,
        "combined":   combined,
        "scrape_ok":  len(current) >= 20,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = get_all_tickers()

    print(f"\nFile parsed      : {result['scrape_ok']}")
    print(f"Current EGX30    : {len(result['current'])} tickers")
    print(f"Historical list  : {len(result['historical'])} tickers")
    print(f"Combined total   : {len(result['combined'])} tickers")

    if result["scrape_ok"]:
        print(f"\nLive tickers: {result['current']}")

    new_entries = set(result["current"]) - set(result["historical"])
    if new_entries:
        print(f"\n⚠ New tickers not in historical list: {sorted(new_entries)}")
        print("  Add these to assets_egx30.json for future runs.")