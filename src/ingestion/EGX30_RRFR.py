"""
CBE Market Data Fetcher - Selenium Edition
Bypasses WAF by using real Chrome browser
"""

import re
import csv
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, WebDriverException

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MARKET_DATA_FILE = DATA_DIR / "market_data.csv"
DEBUG_DIR = DATA_DIR / "debug"

DATA_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)

CONIA_URL = "https://www.cbe.org.eg/en/economic-research/statistics/conia"
INFLATION_URL = "https://www.cbe.org.eg/en/economic-research/statistics/inflation-rates"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SELENIUM SETUP
# =============================================================================

def create_driver(headless=True):
    """
    Create Chrome driver with anti-detection measures.
    """
    options = Options()
    
    if headless:
        options.add_argument("--headless=new")  # New headless mode
    
    # Stealth options
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Random user agent
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.0")
    
    # Disable automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # Additional privacy
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")  # Faster loading
    
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Remove webdriver property
        driver.execute_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """)
        
        # Set reasonable page load timeout
        driver.set_page_load_timeout(30)
        
        return driver
        
    except Exception as e:
        logger.error(f"Failed to create Chrome driver: {e}")
        raise


def safe_get(driver, url, max_retries=2):
    """
    Safely navigate to URL with retries.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Navigating to {url} (attempt {attempt + 1}/{max_retries})")
            driver.get(url)
            
            # Check if we got blocked
            if "Request Rejected" in driver.page_source or "rejected" in driver.page_source.lower():
                logger.warning("WAF block detected, waiting and retrying...")
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                continue
            
            return True
            
        except TimeoutException:
            logger.warning(f"Page load timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(3)
            continue
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
            continue
    
    return False


# =============================================================================
# DATE HELPERS
# =============================================================================

def get_last_business_date():
    """Get most recent CBE business date (Fri/Sat weekend in Egypt)."""
    today = datetime.now().date()
    days_back = 0
    
    # Egypt weekend: Friday (4), Saturday (5)
    while (today - timedelta(days=days_back)).weekday() in [4, 5]:
        days_back += 1
    
    # Early morning check (CBE publishes ~9am Cairo)
    now = datetime.now()
    if now.hour < 9:
        days_back += 1
        while (today - timedelta(days=days_back)).weekday() in [4, 5]:
            days_back += 1
    
    return today - timedelta(days=days_back)


def get_current_month():
    """Current month as YYYY-MM."""
    return datetime.now().strftime("%Y-%m")


def parse_date_from_string(date_str):
    """Parse various date formats."""
    if pd.isna(date_str) or date_str is None:
        return None
    
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%Y", "%Y-%m",
        "%b %Y", "%B %Y", "%Y-%b", "%d-%b-%Y",
        "%m/%d/%Y", "%Y/%m/%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except ValueError:
            continue
    
    # Try pandas for ambiguous dates
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        pass
    
    return None

# =============================================================================
# CACHE FUNCTIONS
# =============================================================================

def read_market_cache():
    """Read latest data from CSV cache."""
    if not MARKET_DATA_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(MARKET_DATA_FILE)
        if df.empty:
            return None
        
        latest = df.iloc[-1].to_dict()
        
        return {
            "conia": {
                "value": float(latest["conia_value"]) if pd.notna(latest.get("conia_value")) and str(latest.get("conia_value")) != "" else None,
                "date": latest.get("conia_date"),
                "fetched_at": latest.get("fetched_at")
            },
            "inflation": {
                "headline_yy": float(latest["headline_yy"]) if pd.notna(latest.get("headline_yy")) and str(latest.get("headline_yy")) != "" else None,
                "core_yy": float(latest["core_yy"]) if pd.notna(latest.get("core_yy")) and str(latest.get("core_yy")) != "" else None,
                "month": latest.get("inflation_month"),
                "published_date": latest.get("inflation_published_date"),
                "fetched_at": latest.get("fetched_at")
            }
        }
    except Exception as e:
        logger.error(f"Cache read error: {e}")
        return None


def save_market_cache(conia_value, conia_date, headline_yy, core_yy, inflation_month, inflation_published=None):
    """Append data to CSV cache."""
    file_exists = MARKET_DATA_FILE.exists()
    
    row = {
        "conia_date": conia_date or "",
        "conia_value": f"{conia_value:.6f}" if conia_value is not None else "",
        "headline_yy": f"{headline_yy:.4f}" if headline_yy is not None else "",
        "core_yy": f"{core_yy:.4f}" if core_yy is not None else "",
        "inflation_month": inflation_month or "",
        "inflation_published_date": inflation_published or "",
        "fetched_at": datetime.now().isoformat()
    }
    
    with open(MARKET_DATA_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    logger.info(f"Saved to cache: {MARKET_DATA_FILE}")

# =============================================================================
# STALENESS CHECKING
# =============================================================================

def check_conia_staleness(cached_data):
    """Check if CONIA needs refresh."""
    if cached_data is None or cached_data.get("conia", {}).get("value") is None:
        return True, "no_cache"
    
    cached_date = cached_data["conia"].get("date")
    target_date = get_last_business_date().isoformat()
    
    if cached_date == target_date:
        return False, f"fresh (cached {cached_date})"
    
    return True, f"stale (cached {cached_date}, need {target_date})"


def check_inflation_staleness(cached_data):
    """Check if inflation needs refresh."""
    if cached_data is None or cached_data.get("inflation", {}).get("headline_yy") is None:
        return True, "no_cache"
    
    cached_month = cached_data["inflation"].get("month")
    target_month = get_current_month()
    
    if cached_month == target_month:
        return False, f"fresh (cached {cached_month})"
    
    return True, f"stale (cached {cached_month}, need {target_month})"

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_conia_from_html(html):
    """
    Extract CONIA value from page HTML.
    Returns (value, success_flag)
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Strategy 1: Find tables with pandas
    tables = soup.find_all('table')
    logger.info(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        try:
            df = pd.read_html(StringIO(str(table)))[0]
            
            # Flatten column names if multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            # Search all cells
            for col in df.columns:
                for idx, val in enumerate(df[col]):
                    if pd.notna(val):
                        text = str(val).strip()
                        
                        # Look for percentage patterns
                        match = re.search(r'(\d+\.?\d*)%?', text)
                        if match:
                            num_str = match.group(1)
                            try:
                                num = float(num_str)
                                
                                # Validate CONIA range (typically 10-30%)
                                if 0.10 <= num <= 0.50:  # Decimal format
                                    logger.info(f"Found CONIA in table {i}, row {idx}: {num:.4%}")
                                    return num, True
                                elif 10 <= num <= 50:  # Percentage format
                                    logger.info(f"Found CONIA in table {i}, row {idx}: {num:.2f}%")
                                    return num / 100, True
                                    
                            except ValueError:
                                continue
                                
        except Exception as e:
            logger.debug(f"Table {i} error: {e}")
            continue
    
    # Strategy 2: Regex on full text
    text = soup.get_text()
    
    patterns = [
        r'CONIA[:\s]+(\d+\.?\d*)%',
        r'Overnight[:\s]+(\d+\.?\d*)%',
        r'Index\s+Value[:\s]+(\d+\.?\d*)%',
        r'(\d{2}\.\d{2,4})\s*%',  # Generic percentage
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                num = float(match)
                if 10 <= num <= 50:
                    logger.info(f"Found CONIA via regex: {num:.2f}%")
                    return num / 100, True
            except ValueError:
                continue
    
    return None, False

def extract_inflation_from_html(html):
    """
    Extract inflation data from page HTML.
    Returns (headline_yy, core_yy, month, success_flag)
    """
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    logger.info(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        try:
            # Parse with StringIO to avoid FutureWarning
            df = pd.read_html(StringIO(str(table)))[0]
            
            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns.values]
            
            # Log what we found for debugging
            logger.info(f"Table {i} columns: {list(df.columns)}")
            logger.info(f"Table {i} shape: {df.shape}")
            if not df.empty:
                logger.info(f"Table {i} first row: {df.iloc[0].to_dict()}")
            
            # Check if this looks like an inflation table by column names
            col_names = [str(c).lower() for c in df.columns]
            
            # Look for headline and core columns (more flexible matching)
            has_headline = any('headline' in c for c in col_names)
            has_core = any('core' in c for c in col_names)
            
            logger.info(f"Table {i}: has_headline={has_headline}, has_core={has_core}")
            
            if not (has_headline and has_core):
                logger.info(f"Table {i}: Skipping - doesn't look like inflation table")
                continue
            
            # Map columns by name matching (case-insensitive)
            headline_col = None
            core_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                
                # Headline y/y - look for headline AND y/y/year/annual
                if 'headline' in col_str and any(x in col_str for x in ['y/y', 'yoy', 'year', 'annual']):
                    headline_col = col
                    logger.info(f"Found headline_col: {col}")
                    
                # Core y/y - look for core AND y/y/year/annual  
                if 'core' in col_str and any(x in col_str for x in ['y/y', 'yoy', 'year', 'annual']):
                    core_col = col
                    logger.info(f"Found core_col: {col}")
            
            # Must have headline to proceed
            if headline_col is None:
                logger.warning(f"Table {i}: No headline y/y column found in {list(df.columns)}")
                continue
            
            # Get first data row (latest)
            if len(df) == 0:
                logger.warning(f"Table {i}: DataFrame is empty")
                continue
                
            latest = df.iloc[0]
            logger.info(f"Processing row: {latest.to_dict()}")
            
            # Extract headline y/y
            headline_val = latest[headline_col]
            if pd.isna(headline_val):
                logger.warning(f"Table {i}: Headline value is NaN")
                continue
                
            headline_text = str(headline_val).replace('%', '').replace(',', '').strip()
            logger.info(f"Headline text to parse: '{headline_text}'")
            
            try:
                headline_yy = float(headline_text)
                # Convert percentage to decimal if needed (>1 means it's in percent format like 11.9)
                if headline_yy > 1:
                    headline_yy = headline_yy / 100
                logger.info(f"Parsed headline_yy: {headline_yy:.4%}")
            except ValueError as e:
                logger.error(f"Table {i}: Could not parse headline value '{headline_val}': {e}")
                continue
            
            # Extract core y/y (optional)
            core_yy = None
            if core_col:
                core_val = latest[core_col]
                if pd.notna(core_val):
                    try:
                        core_text = str(core_val).replace('%', '').replace(',', '').strip()
                        core_yy = float(core_text)
                        if core_yy > 1:
                            core_yy = core_yy / 100
                        logger.info(f"Parsed core_yy: {core_yy:.4%}")
                    except ValueError as e:
                        logger.warning(f"Could not parse core value '{core_val}': {e}")
            
            # Determine month - use current month as default
            month = get_current_month()
            
            # Try to find a date/month column
            for col in df.columns:
                col_str = str(col).lower()
                if any(x in col_str for x in ['month', 'period', 'date']) and \
                   not any(x in col_str for x in ['monthly', 'm/m']):
                    month_val = latest[col]
                    if pd.notna(month_val):
                        parsed = parse_date_from_string(str(month_val))
                        if parsed:
                            month = parsed.strftime("%Y-%m")
                            break
            
            # FIX: Format core string separately to avoid f-string expression error
            core_str = f"{core_yy:.2%}" if core_yy is not None else 'N/A'
            logger.info(f"SUCCESS: Headline={headline_yy:.2%}, Core={core_str}, Month={month}")
            return headline_yy, core_yy, month, True
            
        except Exception as e:
            logger.error(f"Table {i} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    logger.error("Could not extract inflation from any table")
    return None, None, None, False

# =============================================================================
# MAIN FETCH FUNCTIONS
# =============================================================================

def fetch_conia():
    """
    Fetch CONIA using Selenium browser.
    """
    driver = None
    try:
        driver = create_driver(headless=True)
        
        if not safe_get(driver, CONIA_URL):
            logger.error("Failed to load CONIA page")
            return None, None
        
        # Wait for JavaScript rendering
        logger.info("Waiting for page to render...")
        time.sleep(3)
        
        # Additional wait for table
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            logger.info("Table element found")
        except TimeoutException:
            logger.warning("Table timeout, proceeding with available content")
        
        # Get rendered HTML
        html = driver.page_source
        logger.info(f"Page rendered: {len(html)} chars")

        """
        # Save debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = DEBUG_DIR / f"conia_{timestamp}.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Debug saved: {debug_file}")
        """

        # Extract data
        value, success = extract_conia_from_html(html)
        
        if success:
            return value, get_last_business_date().isoformat()
        else:
            logger.error("Could not extract CONIA from page")
            return None, None
            
    except WebDriverException as e:
        logger.error(f"Browser error: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None, None
    finally:
        if driver:
            try:
                driver.quit()
                logger.info("Browser closed")
            except:
                pass


def fetch_inflation():
    """
    Fetch inflation using Selenium browser.
    """
    driver = None
    try:
        driver = create_driver(headless=True)
        
        if not safe_get(driver, INFLATION_URL):
            logger.error("Failed to load inflation page")
            return None, None, None, None
        
        logger.info("Waiting for page to render...")
        time.sleep(3)
        
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except TimeoutException:
            pass
        
        html = driver.page_source
        logger.info(f"Page rendered: {len(html)} chars")
        
        """
        # Save debug
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = DEBUG_DIR / f"inflation_{timestamp}.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(html)
        """

        # Extract data
        headline, core, month, success = extract_inflation_from_html(html)
        
        if success:
            return headline, core, month, datetime.now().strftime("%Y-%m-%d")
        else:
            logger.error("Could not extract inflation from page")
            return None, None, None, None
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return None, None, None, None
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def get_market_data():
    """
    Main entry point. Returns complete market data with status flags.
    """
    result = {
        "conia": {"value": None, "date": None, "status": "error", "source": "none"},
        "inflation": {"headline_yy": None, "core_yy": None, "month": None, "status": "error", "source": "none"},
        "calculated": {},
        "meta": {"fetched_at": datetime.now().isoformat(), "notes": []}
    }
    
    # Read cache
    cached = read_market_cache()
    
    # CONIA
    conia_needs, conia_reason = check_conia_staleness(cached)
    if conia_needs:
        logger.info(f"CONIA: {conia_reason}")
        value, date = fetch_conia()
        
        if value is not None:
            result["conia"] = {
                "value": value,
                "date": date,
                "status": "fresh",
                "source": "web"
            }
        else:
            # Fallback to cache
            if cached and cached.get("conia", {}).get("value"):
                result["conia"] = {
                    "value": cached["conia"]["value"],
                    "date": cached["conia"]["date"],
                    "status": "stale",
                    "source": "cache"
                }
                result["meta"]["notes"].append("CONIA fetch failed, using stale cache")
            else:
                result["meta"]["notes"].append("CONIA fetch failed, no cache available")
    else:
        logger.info(f"CONIA: {conia_reason}")
        result["conia"] = {
            "value": cached["conia"]["value"],
            "date": cached["conia"]["date"],
            "status": "fresh",
            "source": "cache"
        }
    
    # Inflation
    inflation_needs, inflation_reason = check_inflation_staleness(cached)
    if inflation_needs:
        logger.info(f"Inflation: {inflation_reason}")
        headline, core, month, published = fetch_inflation()
        
        if headline is not None:
            result["inflation"] = {
                "headline_yy": headline,
                "core_yy": core,
                "month": month,
                "status": "fresh",
                "source": "web"
            }
        else:
            # Fallback to cache
            if cached and cached.get("inflation", {}).get("headline_yy"):
                result["inflation"] = {
                    "headline_yy": cached["inflation"]["headline_yy"],
                    "core_yy": cached["inflation"].get("core_yy"),
                    "month": cached["inflation"]["month"],
                    "status": "stale",
                    "source": "cache"
                }
                result["meta"]["notes"].append("Inflation fetch failed, using stale cache")
            else:
                result["meta"]["notes"].append("Inflation fetch failed, no cache available")
    else:
        logger.info(f"Inflation: {inflation_reason}")
        result["inflation"] = {
            "headline_yy": cached["inflation"]["headline_yy"],
            "core_yy": cached["inflation"].get("core_yy"),
            "month": cached["inflation"]["month"],
            "status": "fresh",
            "source": "cache"
        }
    
    # Save to cache if any fresh data
    if result["conia"]["source"] == "web" or result["inflation"]["source"] == "web":
        save_market_cache(
            conia_value=result["conia"]["value"],
            conia_date=result["conia"]["date"],
            headline_yy=result["inflation"]["headline_yy"],
            core_yy=result["inflation"]["core_yy"],
            inflation_month=result["inflation"]["month"],
            inflation_published=datetime.now().strftime("%Y-%m-%d")
        )
    
    # Calculate real rate using exact Fisher equation
    if result["conia"]["value"] and result["inflation"]["headline_yy"]:
        nominal = result["conia"]["value"]
        inflation = result["inflation"]["headline_yy"]
        
        # Exact Fisher equation: (1 + n) / (1 + i) - 1
        real_rate_exact = (1 + nominal) / (1 + inflation) - 1
        
        result["calculated"] = {
            "nominal_rate_annual": nominal,
            "inflation_used": inflation,
            "real_rate_exact": real_rate_exact,
            "real_rate_overnight": real_rate_exact
        }
    
    return result

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Suppress Selenium logging
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('webdriver_manager').setLevel(logging.WARNING)
    
    print("\n" + "="*60)
    print("CBE MARKET DATA FETCHER - SELENIUM EDITION")
    print("="*60 + "\n")
    
    data = get_market_data()
    
    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    
    c = data["conia"]
    val_str = f"{c['value']:.4%}" if c['value'] is not None else "N/A"
    print(f"CONIA:        {val_str}")
    print(f"  Date:       {c['date'] or 'N/A'}")
    print(f"  Status:     {c['status']} ({c['source']})")
    
    i = data["inflation"]
    head_str = f"{i['headline_yy']:.2%}" if i['headline_yy'] is not None else "N/A"
    core_str = f"{i['core_yy']:.2%}" if i['core_yy'] is not None else "N/A"
    print(f"\nInflation:")
    print(f"  Headline:   {head_str} (y/y)")
    print(f"  Core:       {core_str} (y/y)")
    print(f"  Month:      {i['month'] or 'N/A'}")
    print(f"  Status:     {i['status']} ({i['source']})")
    
    calc = data["calculated"]
    if calc:
        print(f"\nCalculated:")
        nom_str = f"{calc['nominal_rate_annual']:.2%}" if calc.get('nominal_rate_annual') else "N/A"
        inf_str = f"{calc['inflation_used']:.2%}" if calc.get('inflation_used') else "N/A"
        real_str = f"{calc['real_rate_overnight']:.2%}" if calc.get('real_rate_overnight') else "N/A"
        print(f"  Nominal:    {nom_str}")
        print(f"  Inflation:  {inf_str}")
        print(f"  Real rate:  {real_str}")
    
    if data["meta"]["notes"]:
        print(f"\nNotes: {', '.join(data['meta']['notes'])}")
    
    print("="*60)