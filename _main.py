import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from selenium.webdriver.common.by import By

# Set page configuration
st.set_page_config(page_title="Investment Data Scraper", page_icon="📊", layout="wide")

def get_selenium_driver(debug_mode=False):
    """Initialize and return a Selenium WebDriver."""
    if debug_mode:
        st.write("🔍 **DEBUG:** Initializing Selenium WebDriver")
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--window-size=1920x1080")
    
    # Add more robust error handling
    for attempt in range(3):
        try:
            if debug_mode:
                st.write(f"🔍 **DEBUG:** Attempt {attempt+1}/3 to initialize WebDriver")
            
            # Try a different approach to initialize the driver
            try:
                # Method 1: Using ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                if debug_mode:
                    st.write("🔍 **DEBUG:** WebDriver initialized successfully with ChromeDriverManager")
                return driver
            except Exception as e1:
                if debug_mode:
                    st.write(f"🔍 **DEBUG:** ChromeDriverManager failed: {str(e1)}")
                
                try:
                    # Method 2: Try using executable_path directly
                    driver = webdriver.Chrome(options=options)
                    if debug_mode:
                        st.write("🔍 **DEBUG:** WebDriver initialized successfully with direct Chrome")
                    return driver
                except Exception as e2:
                    if debug_mode:
                        st.write(f"🔍 **DEBUG:** Direct Chrome method failed: {str(e2)}")
                    
                    try:
                        # Method 3: Try Firefox as a fallback
                        from selenium.webdriver.firefox.options import Options as FirefoxOptions
                        from selenium.webdriver.firefox.service import Service as FirefoxService
                        from webdriver_manager.firefox import GeckoDriverManager
                        
                        firefox_options = FirefoxOptions()
                        firefox_options.add_argument("--headless")
                        firefox_service = FirefoxService(GeckoDriverManager().install())
                        driver = webdriver.Firefox(service=firefox_service, options=firefox_options)
                        if debug_mode:
                            st.write("🔍 **DEBUG:** WebDriver initialized successfully with Firefox")
                        return driver
                    except Exception as e3:
                        if debug_mode:
                            st.write(f"🔍 **DEBUG:** Firefox fallback failed: {str(e3)}")
                        raise Exception(f"All WebDriver methods failed: {str(e1)}, {str(e2)}, {str(e3)}")
                        
        except Exception as e:
            if attempt < 2:  # Try 3 times total
                if debug_mode:
                    st.write(f"🔍 **DEBUG:** Attempt {attempt+1} failed: {str(e)}. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to initialize WebDriver after 3 attempts: {str(e)}")
                raise
    
    # If we get here, all attempts failed
    raise Exception("Failed to initialize WebDriver after multiple attempts")

def debug_print(message, data=None):
    """Print debug information if debug mode is enabled."""
    # Get debug_mode from session state instead of global variable
    if st.session_state.get("debug_mode_main", False):
        st.write(f"🔍 **DEBUG:** {message}")
        if data is not None:
            # Format different types of data appropriately
            if isinstance(data, (dict, list)):
                st.write("```python")
                st.write(json.dumps(data, indent=2, default=str))
                st.write("```")
            elif isinstance(data, pd.DataFrame):
                st.write("DataFrame:")
                st.dataframe(data)
            else:
                st.write(f"```\n{data}\n```")

def scrape_with_requests(url, debug_mode=False):
    """Fallback scraper using regular requests instead of Selenium."""
    try:
        st.info(f"Attempting to scrape with basic requests: {url}")
        if debug_mode:
            st.write("🔍 **DEBUG:** Using requests as fallback method")
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        if debug_mode:
            st.write(f"🔍 **DEBUG:** Requests successful, received {len(response.text)} bytes")
            st.write(f"🔍 **DEBUG:** First 500 characters: {response.text[:500]}...")
            
        return soup, response.text
    except Exception as e:
        st.error(f"Error with requests scraping: {str(e)}")
        if debug_mode:
            import traceback
            st.write(f"🔍 **DEBUG:** Requests error details: {traceback.format_exc()}")
        return None, None

def scrape_with_selenium(url, wait_time=5, debug_mode=False):
    """Scrape content using Selenium for JavaScript-heavy pages with requests fallback."""
    
    if debug_mode:
        st.write(f"🔍 **DEBUG:** Starting Selenium scraping for URL: {url}")
    
    try:
        driver = get_selenium_driver(debug_mode=debug_mode)
        
        try:
            st.info(f"Loading page with Selenium: {url}")
            if debug_mode:
                st.write(f"🔍 **DEBUG:** Attempting to load URL with Selenium")
                
            driver.get(url)
            
            # Wait for JavaScript to load
            if debug_mode:
                st.write(f"🔍 **DEBUG:** Waiting {wait_time} seconds for JavaScript to load")
            time.sleep(wait_time)
            
            # Get the page source after JavaScript execution
            page_source = driver.page_source
            
            if debug_mode:
                st.write(f"🔍 **DEBUG:** Page source received, length: {len(page_source)} characters")
                st.write("🔍 **DEBUG:** First 500 characters of page source:")
                st.write(f"```html\n{page_source[:500]}...\n```")
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(page_source, 'html.parser')
            if debug_mode:
                st.write(f"🔍 **DEBUG:** BeautifulSoup object created")
                
            return soup, page_source
        except Exception as e:
            st.error(f"Error with Selenium: {str(e)}")
            if debug_mode:
                st.write("🔍 **DEBUG:** Full Selenium error:")
                import traceback
                st.write(f"```python\n{traceback.format_exc()}\n```")
                st.write("🔍 **DEBUG:** Falling back to requests method...")
            
            # Fallback to requests method
            return scrape_with_requests(url, debug_mode=debug_mode)
        finally:
            # Close the driver as we don't need it anymore
            if debug_mode:
                st.write("🔍 **DEBUG:** Attempting to close WebDriver")
            try:
                driver.quit()
                if debug_mode:
                    st.write("🔍 **DEBUG:** WebDriver closed successfully")
            except Exception as e:
                if debug_mode:
                    st.write(f"🔍 **DEBUG:** Error closing WebDriver: {str(e)}")
    except Exception as e:
        st.error(f"Error initializing Selenium: {str(e)}")
        if debug_mode:
            st.write("🔍 **DEBUG:** Selenium initialization failed, trying requests fallback...")
        
        # Fallback to requests if Selenium can't be initialized
        return scrape_with_requests(url, debug_mode=debug_mode)
    
def extract_financial_tables(soup):
    """Extract financial tables from a BeautifulSoup object."""
    # Access debug_mode from global scope or pass it as a parameter
    global debug_mode  # Add this line to access debug_mode from global scope
    
    financial_data = {
        "quarterly": {},
        "annual": {},
        "balance_sheet": {},
        "ratios": {},
        "cash_flow": {}
    }
    
    # Find all tables for financial data
    tables = soup.find_all('table')
    
    for i, table in enumerate(tables):
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
        
        # Filter out graph columns and other unwanted columns
        unwanted_terms = ['graph', 'created with highcharts', 'earnings_transcripts', 
                         'results_pdf', 'result_notes', 'Annual_Reports']
        
        clean_headers = []
        clean_indices = []
        
        for j, header in enumerate(headers):
            if not any(term in header.lower() for term in unwanted_terms):
                clean_headers.append(header)
                clean_indices.append(j)
        
        # Extract rows - Fix the 'td' error here by extracting all cells from each row
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all(['td', 'th'])]  # Changed 'td' to 'tr'
            
            # Skip rows with unwanted terms in the first column
            if cells and any(term in cells[0].lower() for term in 
                           ['earnings_transcripts', 'results_pdf', 'result_notes', 'Annual_Reports']):
                continue
                
            # Filter the cells to match clean headers
            clean_cells = [cells[j] for j in clean_indices if j < len(cells)]
            
            if clean_cells and len(clean_cells) > 1:  # Ensure row has content
                rows.append(clean_cells)
        
        # Skip empty tables
        if not rows:
            continue
            
        # Categorize the table based on its content
        # First, create table_text using both headers and rows (rows is now defined)
        table_text = ' '.join(clean_headers).lower() + ' ' + ' '.join([' '.join(row) for row in rows]).lower()
        
        # Check for quarterly indicators in headers
        quarter_pattern = re.compile(r'q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec')
        date_pattern = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*['`]?\d{2}", re.IGNORECASE)
        year_pattern = re.compile(r'20\d\d|19\d\d|fy\d\d')
        cash_flow_pattern = re.compile(r'cash|operating|investing|financing')
        
        quarter_matches = sum(1 for h in clean_headers if quarter_pattern.search(str(h).lower()))
        date_matches = sum(1 for h in clean_headers if date_pattern.search(str(h)))
        year_matches = sum(1 for h in clean_headers if year_pattern.search(str(h).lower()))
        
        # Check if this is a quarterly results table specifically
        is_quarterly_results = 'quarterly results' in table_text.lower() or 'quarterly revenue' in table_text.lower()
        
        # Check for multiple date columns in the format "MMM 'YY"
        has_quarterly_dates = False
        date_format_pattern = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*['\']?\d{2}", re.IGNORECASE)
        date_columns_count = sum(1 for h in clean_headers if date_format_pattern.search(str(h)))
        if date_columns_count >= 3:  # If at least 3 columns have date formats like "Dec '24"
            has_quarterly_dates = True
            
        # Special case for ICICI Bank pattern (look for specific content markers)
        icici_quarterly_pattern = False
        if any('total rev' in row[0].lower() for row in rows if row and len(row) > 0):
            if any('dec' in str(h).lower() and "'" in str(h).lower() for h in clean_headers):
                icici_quarterly_pattern = True
                if debug_mode:
                    st.write(f"🔍 **DEBUG:** Detected ICICI Bank quarterly results pattern in Table {i+1}")
        
        # Determine category based on table content and patterns
        if 'cash flow' in table_text or 'operating activities' in table_text or 'financing' in table_text:
            category = "cash_flow"
        elif 'balance sheet' in table_text or 'assets' in table_text or 'liabilities' in table_text:
            category = "balance_sheet"
        elif 'ratio' in table_text or 'roe' in table_text or 'roa' in table_text:
            category = "ratios"
        elif is_quarterly_results or has_quarterly_dates or icici_quarterly_pattern:
            category = "quarterly"
            if debug_mode:
                reason = []
                if is_quarterly_results: reason.append("contains 'quarterly results' text")
                if has_quarterly_dates: reason.append("has quarterly date columns")
                if icici_quarterly_pattern: reason.append("matches ICICI quarterly pattern")
                st.write(f"🔍 **DEBUG:** Table {i+1} categorized as quarterly because: {', '.join(reason)}")
        elif 'quarterly' in table_text or 'quarter' in table_text:
            category = "quarterly"
        elif 'annual' in table_text or 'yearly' in table_text or 'year' in table_text:
            category = "annual"
        elif cash_flow_pattern.search(table_text):
            category = "cash_flow"
        elif date_matches >= 3 or quarter_matches > year_matches:
            # Special case for tables with multiple date columns like "Dec '24"
            category = "quarterly"
            if debug_mode:
                st.write(f"🔍 **DEBUG:** Table {i+1} categorized as quarterly based on date patterns")
        elif year_matches > 0:
            category = "annual"
        else:
            # Default to quarterly for tables that look like financial data
            # but don't clearly fit other categories
            category = "quarterly"
        
        if debug_mode:
            st.write(f"🔍 **DEBUG:** Table {i+1} final category: {category}")
            st.write(f"🔍 **DEBUG:** Table {i+1} has {len(clean_headers)} headers and {len(rows)} rows")
        
        # Store the table data with clean headers
        financial_data[category][f"Table_{i+1}"] = {
            "headers": clean_headers,
            "rows": rows
        }
    
    # Add summary of categorized tables if in debug mode
    if debug_mode:
        for category, tables in financial_data.items():
            st.write(f"🔍 **DEBUG:** Category '{category}' has {len(tables)} tables")
    
    return financial_data

def extract_share_price_history(soup):
    """Extract share price history and returns data from a BeautifulSoup object."""
    price_history = {
        "daily_data": [],
        "returns_comparison": [],
        "returns_seasonality": {},
        "returns_deep_dive": {}
    }
    
    # Extract text content for regex patterns
    text = soup.get_text()
    
    # Extract daily data from tables
    tables = soup.find_all('table')
    for table in tables:
        # Skip tables without headers
        headers = [th.text.strip() for th in table.find_all('th')] if table.find('th') else []
        if not headers:
            headers = [td.text.strip() for td in table.find_all('tr')[0].find_all('td')] if table.find('tr') else []
        
        if not headers:
            continue
            
        # Check if this is a daily data table by looking for date and price columns
        is_daily_table = any('date' in h.lower() for h in headers) and any(price_type in ' '.join(headers).lower() 
                                                                           for price_type in ['open', 'high', 'low', 'close'])
        
        # Check if this is a returns comparison table
        is_returns_comparison = any('returns' in h.lower() for h in headers) and any('time' in h.lower() or 'period' in h.lower() 
                                                                                    for h in headers)
        
        # Check if this is a seasonality table
        is_seasonality = any('jan' in h.lower() or 'feb' in h.lower() or 'mar' in h.lower() for h in headers)
        
        if is_daily_table:
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [cell.text.strip() for cell in row.find_all(['td', 'th'])]
                
                # Skip rows with specific indicators or metadata
                if not cells or len(cells) < 2:
                    continue
                
                # Skip rows containing metadata indicators
                if any(indicator in cells[0] for indicator in ["POSITIVE", "NEGATIVE", "Notes", "Earnings Transcripts", "Results PDF"]):
                    continue
                
                # Create a clean row with proper data
                row_data = {}
                for i, header in enumerate(headers):
                    if i < len(cells):
                        # Clean up the header name
                        header_clean = header.replace('_', ' ').strip()
                        # Skip graph or chart columns
                        if not ('graph' in header_clean.lower() or 'chart' in header_clean.lower()):
                            row_data[header_clean] = cells[i]
                
                if row_data:  # Only add if we have data
                    # FIX: Append to price_history["daily_data"] instead of undefined rows
                    price_history["daily_data"].append(row_data)
        
        elif is_returns_comparison:
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [cell.text.strip() for cell in row.find_all(['td', 'th'])]
                if len(cells) < 2:  # Need at least period and return
                    continue
                    
                # Create a dictionary from the cells
                row_data = {}
                for i, header in enumerate(headers):
                    if i < len(cells):
                        # Clean up the header name
                        header_clean = header.replace('_', ' ').strip()
                        # Skip graph or chart columns
                        if not ('graph' in header_clean.lower() or 'chart' in header_clean.lower()):
                            row_data[header_clean] = cells[i]
                
                if row_data:  # Only add if we have data
                    # For returns comparison, we'll use 'entity' or 'time' as a key field
                    if 'Time' in row_data:
                        row_data['entity'] = row_data['Time']
                    elif 'Period' in row_data:
                        row_data['entity'] = row_data['Period']
                    
                    price_history["returns_comparison"].append(row_data)
        
        elif is_seasonality:
            seasonality_data = []
            
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [cell.text.strip() for cell in row.find_all(['td', 'th'])]
                if len(cells) < 2:  # Need at least year and one month
                    continue
                    
                # First column is usually Year
                year = cells[0]
                
                # Create a row with year and monthly data
                row_data = {"Year": year}
                
                for i, header in enumerate(headers[1:], 1):  # Skip the Year header
                    if i < len(cells):
                        # Clean up the header name
                        header_clean = header.replace('_', ' ').strip()
                        # Skip graph or chart columns
                        if not ('graph' in header_clean.lower() or 'chart' in header_clean.lower()):
                            row_data[header_clean] = cells[i]
                
                if len(row_data) > 1:  # Only add if we have year plus at least one month
                    seasonality_data.append(row_data)
            
            if seasonality_data:
                price_history["returns_seasonality"] = seasonality_data
    
    # If tables weren't structured as expected, try to extract data with regex
    if not price_history["daily_data"]:
        # Try to find a structured section with daily price data
        daily_section = re.search(r'Date\s+Open\s+High\s+Low\s+(?:Prev\.?\s+Close)?\s+(?:LTP)?\s+Close\s+Volume(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if daily_section:
            daily_text = daily_section.group(1)
            # Split by lines and parse each line
            for line in daily_text.strip().split('\n'):
                if line.strip():
                    # Parse fields: Date, Open, High, Low, Prev Close, LTP, Close, Volume
                    parts = re.findall(r"[\d',]+(?:\.\d+)?|\w+\s'\d{2}", line)
                    if len(parts) >= 7:  # At least need the basic price data
                        row_data = {
                            "Date": parts[0],
                            "Open": parts[1],
                            "High": parts[2],
                            "Low": parts[3],
                            "Close": parts[6] if len(parts) > 6 else parts[5],
                            "Volume": parts[7] if len(parts) > 7 else ""
                        }
                        price_history["daily_data"].append(row_data)
    
    # If returns comparison is empty, try to extract from structured text
    if not price_history["returns_comparison"]:
        # Look for time period and return percentage patterns
        returns_section = re.search(r'(?:Time|Period|Returns Comparison)(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if returns_section:
            returns_text = returns_section.group(1)
            # Find patterns like "1 Day", "1 Week", etc. with percentages
            period_pattern = r'(Day|Week|Month|Qtr|Half Year|1 Yr|3 Yr|5 Yr|10 Yr).*?([\d.]+%)'
            matches = re.findall(period_pattern, returns_text, re.IGNORECASE)
            for period, percentage in matches:
                price_history["returns_comparison"].append({
                    "entity": period,
                    "Returns": percentage
                })
    
    # If seasonality is empty, try to extract from structured text
    if not price_history["returns_seasonality"]:
        # Look for yearly and monthly return patterns
        seasonality_section = re.search(r'(?:Seasonality|Monthly Returns)(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if seasonality_section:
            seasonality_text = seasonality_section.group(1)
            # Find year patterns with monthly data
            year_pattern = r'(\d{4})\s+([-\d.%]+)\s+([-\d.%]+)\s+([-\d.%]+)\s+([-\d.%]+)\s+([-\d.%]+)\s+([-\d.%]+)'
            matches = re.findall(year_pattern, seasonality_text)
            
            seasonality_data = []
            for match in matches:
                year = match[0]
                months_data = match[1:]
                row_data = {"Year": year}
                
                # Add month data (assuming order: Jan, Feb, Mar, etc.)
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                for i, month_data in enumerate(months_data):
                    if i < len(month_names):
                        row_data[month_names[i]] = month_data
                
                seasonality_data.append(row_data)
            
            if seasonality_data:
                price_history["returns_seasonality"] = seasonality_data
    
    return price_history


def extract_technical_indicators(soup):
    """Extract technical indicators from a BeautifulSoup object."""
    technical_data = {
        "current_price": None,
        "indicators": {},
        "performance": {},
        "moving_averages": {},
        "oscillators": {},
        "pivot_points": {},
        "candlestick_patterns": {},
        "volume_analysis": {},
        "beta": {}
    }
    
    # Extract text content
    text = soup.get_text()
    
    # Find current price
    price_pattern = re.compile(r'(\d+\.\d+)\s*(?:\([-+]?\d+\.\d+%\))?', re.IGNORECASE)
    price_match = price_pattern.search(text)
    if price_match:
        technical_data["current_price"] = price_match.group(1)
    
    # Common technical indicators - look for specific formats in the data
    indicator_patterns = [
        (r'RSI\(14\)\s*(\d+\.\d+)', 'RSI'),
        (r'Day RSI\s*(\d+\.\d+)', 'RSI'),
        (r'Day MFI\s*(\d+\.\d+)', 'MFI'),
        (r'Day MACD\(12, 26, 9\)\s*(\d+\.\d+)', 'MACD'),
        (r'Day MACD Signal\s*(\d+\.\d+)', 'MACD Signal'),
        (r'Day ATR\s*(\d+\.\d+)', 'ATR'),
        (r'Day ADX\s*(\d+\.\d+)', 'ADX'),
        (r'Beta 1Year\s*([-+]?\d+\.\d+)', 'Beta 1Year'),
        (r'Day Trendlyne Momentum Score\s*(\d+\.\d+)', 'Momentum Score'),
        (r'Day ROC\(21\)\s*([-+]?\d+\.\d+)', 'ROC(21)'),
        (r'Day ROC\(125\)\s*([-+]?\d+\.\d+)', 'ROC(125)')
    ]
    
    for pattern, indicator in indicator_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["indicators"][indicator] = match.group(1)
    
    # Extract moving averages
    ma_patterns = [
        (r'50Day SMA\s*([\d,.]+)', 'SMA50'),
        (r'200Day SMA\s*([\d,.]+)', 'SMA200'),
        (r'5 Day SMA\s*([\d,.]+)', 'SMA5'),
        (r'10 Day SMA\s*([\d,.]+)', 'SMA10'),
        (r'20 Day SMA\s*([\d,.]+)', 'SMA20'),
        (r'30 Day SMA\s*([\d,.]+)', 'SMA30'),
        (r'100 Day SMA\s*([\d,.]+)', 'SMA100'),
        (r'150 Day SMA\s*([\d,.]+)', 'SMA150')
    ]
    
    for pattern, ma_type in ma_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["moving_averages"][ma_type] = match.group(1)
    
    # Extract performance metrics
    performance_patterns = [
        (r'1 Day Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '1 Day'),
        (r'1 Week Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '1 Week'),
        (r'1 Month Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '1 Month'),
        (r'3 Months Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '3 Months'),
        (r'6 Months Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '6 Months'),
        (r'1 Year Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '1 Year'),
        (r'3 Year Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '3 Years'),
        (r'5 Year Range\s*([-+]?\d+\.\d+)\s*\(([-+]?\d+\.\d+)%\)', '5 Years')
    ]
    
    for pattern, period in performance_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["performance"][period] = {
                "absolute": match.group(1),
                "percentage": match.group(2) + "%"
            }
    
    # Extract oscillator data
    oscillator_patterns = [
        (r'RSI\(14\)\s*(\d+\.\d+)', 'RSI'),
        (r'Stochastic Oscillator\s*(\d+\.\d+)', 'Stochastic'),
        (r'CCI 20\s*([-+]?\d+\.\d+)', 'CCI'),
        (r'Awesome Oscillator\s*([-+]?\d+\.\d+)', 'Awesome'),
        (r'Momentum Oscillator\s*([-+]?\d+\.\d+)', 'Momentum'),
        (r'MACD\(12, 26, 9\)\s*([-+]?\d+\.\d+)', 'MACD'),
        (r'Stochastic RSI\s*([-+]?\d+\.\d+)', 'Stochastic RSI'),
        (r'William\s*([-+]?\d+\.\d+)', 'Williams %R'),
        (r'Ultimate Oscillator\s*([-+]?\d+\.\d+)', 'Ultimate')
    ]
    
    for pattern, oscillator in oscillator_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["oscillators"][oscillator] = match.group(1)
    
    # Extract pivot points
    pivot_section = re.search(r'PIVOT LEVEL(.*?)(?:ICICI Bank Ltd|$)', text, re.DOTALL)
    if pivot_section:
        pivot_text = pivot_section.group(1)
        
        # Extract pivot point data
        pivot_patterns = [
            (r'Resistance 3\s*([\d,.]+)', 'R3'),
            (r'Resistance 2\s*([\d,.]+)', 'R2'),
            (r'Resistance 1\s*([\d,.]+)', 'R1'),
            (r'Pivot Point\s*([\d,.]+)', 'PP'),
            (r'Support 1\s*([\d,.]+)', 'S1'),
            (r'Support 2\s*([\d,.]+)', 'S2'),
            (r'Support 3\s*([\d,.]+)', 'S3')
        ]
        
        for pattern, label in pivot_patterns:
            match = re.search(pattern, pivot_text, re.IGNORECASE)
            if match:
                technical_data["pivot_points"][label] = match.group(1)
    
    # Extract Beta information
    beta_section = re.search(r'BETA(.*?)(?:ICICI Bank Ltd|$)', text, re.DOTALL)
    if beta_section:
        beta_text = beta_section.group(1)
        
        beta_patterns = [
            (r'1 Month\s*([-+]?\d+\.\d+)', '1 Month'),
            (r'3 Month\s*([-+]?\d+\.\d+)', '3 Months'),
            (r'1 Year\s*([-+]?\d+\.\d+)', '1 Year'),
            (r'3 Year\s*([-+]?\d+\.\d+)', '3 Years')
        ]
        
        for pattern, period in beta_patterns:
            match = re.search(pattern, beta_text, re.IGNORECASE)
            if match:
                technical_data["beta"][period] = match.group(1)
    
    # Extract volume analysis
    volume_section = re.search(r'VOLUME ANALYSIS(.*?)(?:ICICI Bank Ltd|$)', text, re.DOTALL)
    if volume_section:
        volume_text = volume_section.group(1)
        
        delivery_pattern = re.search(r'Delivery Volume %\s*Day\s*([\d,.]+)%\s*Week\s*([\d,.]+)%\s*1 Month\s*([\d,.]+)%', volume_text, re.IGNORECASE)
        if delivery_pattern:
            technical_data["volume_analysis"]["Delivery Volume %"] = {
                "Day": delivery_pattern.group(1) + "%",
                "Week": delivery_pattern.group(2) + "%",
                "Month": delivery_pattern.group(3) + "%"
            }
        
        volume_pattern = re.search(r'NSE\+BSE Traded Volume: ([\d,.]+)M', volume_text, re.IGNORECASE)
        if volume_pattern:
            technical_data["volume_analysis"]["NSE+BSE Volume"] = volume_pattern.group(1) + "M"
        
        delivery_volume_pattern = re.search(r'Combined Delivery Volume: ([\d,.]+)M', volume_text, re.IGNORECASE)
        if delivery_volume_pattern:
            technical_data["volume_analysis"]["Delivery Volume"] = delivery_volume_pattern.group(1) + "M"
    
    # Extract pivot points - generalized for any stock symbol
    pivot_section_pattern = re.compile(r'PIVOT\s+LEVEL(.*?)(?:\n\n|\Z)', re.DOTALL | re.IGNORECASE)
    pivot_section = pivot_section_pattern.search(text)

    if not pivot_section:
        # Try alternative patterns that might indicate a pivot section
        pivot_section = re.search(r'(?:PIVOT|SUPPORT.*?RESISTANCE)(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)

    if pivot_section:
        pivot_text = pivot_section.group(1)
        
        # Initialize pivot points structure
        technical_data["pivot_points"] = {}
        
        # Try to extract pivot data from HTML table structure
        pivot_table = soup.find('table', class_='pivot-table')
        
        if pivot_table:
            # Found a structured table - extract methods from header
            headers = [th.text.strip() for th in pivot_table.find('thead').find_all('th')]
            methods = headers[1:]  # Skip the first header (usually empty)
            
            # Process each row in the table
            for row in pivot_table.find('tbody').find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:  # Need at least one value
                    level_name = cells[0].text.strip()
                    
                    # Determine the pivot level code
                    level_code = None
                    if 'Resistance 3' in level_name:
                        level_code = 'R3'
                    elif 'Resistance 2' in level_name:
                        level_code = 'R2'
                    elif 'Resistance 1' in level_name:
                        level_code = 'R1'
                    elif 'Pivot Point' in level_name:
                        level_code = 'PP'
                    elif 'Support 1' in level_name:
                        level_code = 'S1'
                    elif 'Support 2' in level_name:
                        level_code = 'S2'
                    elif 'Support 3' in level_name:
                        level_code = 'S3'
                    
                    if level_code:
                        # Extract values for each method
                        for i, method in enumerate(methods):
                            if i+1 < len(cells):  # Check if cell exists
                                value = cells[i+1].text.strip()
                                
                                # Store in nested structure
                                if method not in technical_data["pivot_points"]:
                                    technical_data["pivot_points"][method] = {}
                                
                                technical_data["pivot_points"][method][level_code] = value
        else:
            # Fallback to regex extraction if table not found
            method_pattern = re.compile(r'(Classic|Woodie|Camarilla|Fibonacci|Traditional|DeMark)', re.IGNORECASE)
            method_matches = method_pattern.findall(pivot_text)
            
            # If no methods found in text, use default generic method
            methods = method_matches if method_matches else ["Method1"]
            
            # For each method, initialize empty dict
            for method in methods:
                technical_data["pivot_points"][method] = {}
            
            # Define pivot levels to extract
            pivot_levels = [
                ('Resistance 3', 'R3'),
                ('Resistance 2', 'R2'), 
                ('Resistance 1', 'R1'),
                ('Pivot Point', 'PP'),
                ('Support 1', 'S1'),
                ('Support 2', 'S2'),
                ('Support 3', 'S3')
            ]
            
            # Extract each level
            for level_name, level_code in pivot_levels:
                # Look for the level in text
                pattern = re.compile(f'{level_name}\\s*([-\\d.,]+)', re.IGNORECASE)
                match = pattern.search(pivot_text)
                
                if match:
                    value_str = match.group(1).strip()
                    
                    # Handle different possible formats
                    if '.' in value_str and value_str.count('.') > methods.count('.'):
                        # Try to split concatenated values
                        values = re.findall(r'(\d+\.\d+)', value_str)
                        
                        # Assign values to methods
                        for i, method in enumerate(methods):
                            if i < len(values):
                                technical_data["pivot_points"][method][level_code] = values[i]
                    else:
                        # If only one value, assign to first method
                        technical_data["pivot_points"][methods[0]][level_code] = value_str
    
    return technical_data

def save_data_to_file(data, filename):
    """Save scraped data to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving data to {filename}: {str(e)}")
        return False

debug_mode = False 

def main():
    global debug_mode  # Tell Python we're using the global version
    st.title("📊 Advanced Stock Data Scraper")
    st.write("Enter company details and URLs to scrape investment research data.")
    
    # Debug mode toggle should be at the beginning
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", False)
    if debug_mode:
        st.sidebar.warning("Debug mode is enabled. Detailed information will be shown.")
    
    # Create form for input
    with st.form("stock_data_form"):
        st.subheader("Enter Company Details")
        
        company_name = st.text_input("Company Name", value="ICICI Bank Ltd")
        stock_symbol = st.text_input("Stock Symbol", value="ICICIBANK.NS")
        
        st.subheader("URLs for Scraping")
        financials_url = st.text_input(
            "Financials URL", 
            value="",
            help="URL containing company financial data in tabular format"
        )
        
        technicals_url = st.text_input(
            "Technical Analysis URL (Optional)",
            value="",
            help="URL containing technical analysis data for the stock"
        )
        
        price_history_url = st.text_input(
            "Share Price History URL (Optional)",
            value="",
            help="URL containing historical share price data, returns comparison, and seasonality"
        )
        
        submit_button = st.form_submit_button("Scrape Data")
    
    if submit_button:
        # Create a container to hold all scraped data
        scraped_data = {
            "metadata": {
                "company_name": company_name,
                "stock_symbol": stock_symbol,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Create a timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{stock_symbol.replace('.', '_')}_{timestamp}"
        
        # Scrape financial data if URL provided
        if financials_url:
            with st.expander("Financial Data Scraping Results", expanded=True):
                st.subheader("Financial Data")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(financials_url)

                if soup:
                    # Extract financial tables
                    financial_data = extract_financial_tables(soup)
                    scraped_data["financial_data"] = financial_data
                        
                    # Remove unwanted fields from all tables
                    for category in financial_data:
                        if isinstance(financial_data[category], dict):
                            for table_name, table_data in financial_data[category].items():
                                # Filter rows that contain unwanted fields
                                filtered_rows = []
                                for row in table_data["rows"]:
                                    if row and not any(unwanted in row[0] for unwanted in 
                                                    ["earnings_transcripts", "results_pdf", "result_notes", "Annual_Reports"]):
                                        filtered_rows.append(row)
                                
                                table_data["rows"] = filtered_rows
                    
                    # Create tabs for different financial sections
                    fin_tabs = st.tabs(["Quarterly Results", "Annual Results", "Balance Sheet", "Financial Ratios", "Cash Flow"])
                    
                    # Display each section in its tab (implementation of each tab)
                                        # Quarterly Results
                    # In the quarterly tab display code
                    with fin_tabs[0]:
                        if financial_data["quarterly"]:
                            for table_name, table_data in financial_data["quarterly"].items():
                                st.write(f"**{table_name}**")
                                
                                # Extract and clean up headers and rows
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        # Filter out graph columns
                                        headers = []
                                        for h in table_data["headers"]:
                                            clean_h = h.strip()
                                            if not ('graph' in clean_h.lower()):
                                                headers.append(clean_h)
                                        
                                        # Process rows
                                        rows = []
                                        for row in table_data["rows"]:
                                            # Skip metadata rows
                                            if not row or len(row) < 2:
                                                continue
                                            if any(indicator in row[0] for indicator in ["POSITIVE", "NEGATIVE", "Notes", "Earnings Transcripts", "Results PDF"]):
                                                continue
                                            
                                            # Filter row to match headers length
                                            filtered_row = []
                                            for i, h in enumerate(table_data["headers"]):
                                                if not ('graph' in h.lower()) and i < len(row):
                                                    filtered_row.append(row[i])
                                            
                                            if filtered_row and len(filtered_row) > 1:
                                                rows.append(filtered_row)
                                        
                                        if headers and rows:
                                            # Create dataframe with the cleaned data
                                            df = pd.DataFrame(rows, columns=headers)
                                            st.dataframe(df)
                                        else:
                                            st.warning(f"No valid data found in {table_name} after filtering")
                                    except Exception as e:
                                        st.error(f"Error displaying table: {str(e)}")
                            else:
                                st.info("No quarterly data found in the provided URL")
                    
                    # Annual Results
                    with fin_tabs[1]:
                        if financial_data["annual"]:
                            for table_name, table_data in financial_data["annual"].items():
                                st.write(f"**{table_name}**")
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        # Filter out graph columns
                                        headers = [h for h in table_data["headers"] if 'graph' not in h.lower()]
                                        rows = []
                                        for row in table_data["rows"]:
                                            # Filter row to match headers length
                                            filtered_row = []
                                            for i, h in enumerate(table_data["headers"]):
                                                if 'graph' not in h.lower() and i < len(row):
                                                    filtered_row.append(row[i])
                                            if filtered_row:
                                                rows.append(filtered_row)
                                                
                                        if headers and rows:
                                            df = pd.DataFrame(rows, columns=headers)
                                            st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                        else:
                            st.info("No annual data found")
                    
                    # Balance Sheet
                    with fin_tabs[2]:
                        if financial_data["balance_sheet"]:
                            for table_name, table_data in financial_data["balance_sheet"].items():
                                st.write(f"**{table_name}**")
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        # Filter out graph columns
                                        headers = [h for h in table_data["headers"] if 'graph' not in h.lower()]
                                        rows = []
                                        for row in table_data["rows"]:
                                            # Filter row to match headers length
                                            filtered_row = []
                                            for i, h in enumerate(table_data["headers"]):
                                                if 'graph' not in h.lower() and i < len(row):
                                                    filtered_row.append(row[i])
                                            if filtered_row:
                                                rows.append(filtered_row)
                                                
                                        if headers and rows:
                                            df = pd.DataFrame(rows, columns=headers)
                                            st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                        else:
                            st.info("No balance sheet data found")
                    
                    # Financial Ratios
                    with fin_tabs[3]:
                        if financial_data["ratios"]:
                            for table_name, table_data in financial_data["ratios"].items():
                                st.write(f"**{table_name}**")
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        # Filter out graph columns
                                        headers = [h for h in table_data["headers"] if 'graph' not in h.lower()]
                                        rows = []
                                        for row in table_data["rows"]:
                                            # Filter row to match headers length
                                            filtered_row = []
                                            for i, h in enumerate(table_data["headers"]):
                                                if 'graph' not in h.lower() and i < len(row):
                                                    filtered_row.append(row[i])
                                            if filtered_row:
                                                rows.append(filtered_row)
                                                
                                        if headers and rows:
                                            df = pd.DataFrame(rows, columns=headers)
                                            st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                        else:
                            st.info("No financial ratios data found")
                    
                    # Cash Flow
                    with fin_tabs[4]:
                        if financial_data["cash_flow"]:
                            for table_name, table_data in financial_data["cash_flow"].items():
                                st.write(f"**{table_name}**")
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        # Filter out graph columns
                                        headers = [h for h in table_data["headers"] if 'graph' not in h.lower()]
                                        rows = []
                                        for row in table_data["rows"]:
                                            # Filter row to match headers length
                                            filtered_row = []
                                            for i, h in enumerate(table_data["headers"]):
                                                if 'graph' not in h.lower() and i < len(row):
                                                    filtered_row.append(row[i])
                                            if filtered_row:
                                                rows.append(filtered_row)
                                                
                                        if headers and rows:
                                            df = pd.DataFrame(rows, columns=headers)
                                            st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                        else:
                            st.info("No cash flow data found")
                else:
                    st.error("Failed to retrieve financial data. Please check the URL and try again.")
                    st.info("You can still proceed with other sections if available.")
        
        # Scrape technical data if URL provided
        # Scrape technical data if URL provided
        if technicals_url:
            with st.expander("Technical Analysis Results", expanded=True):
                st.subheader("Technical Analysis")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(technicals_url)
                
                if soup:
                    # Save HTML for debugging
                    # html_filename = f"debug_{base_filename}_technicals.html"
                    # with open(html_filename, 'w', encoding='utf-8') as f:
                    #     f.write(html_content)
                    # st.success(f"Saved raw HTML to {html_filename} for debugging")
                    
                    # Extract technical indicators
                    technical_data = extract_technical_indicators(soup)
                    scraped_data["technical_data"] = technical_data
                    
                    # Create tabs for different technical sections
                    tech_tabs = st.tabs(["Overview", "Indicators", "Moving Averages", "Oscillators", "Volume", "Pivot Points"])
                    
                    # Overview Tab
                    with tech_tabs[0]:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if technical_data["current_price"]:
                                st.metric("Current Price", f"₹{technical_data['current_price']}")
                            
                        with col2:
                            if technical_data["moving_averages"].get("SMA50"):
                                st.metric("50-Day SMA", f"₹{technical_data['moving_averages']['SMA50']}")
                        
                        with col3:
                            if technical_data["moving_averages"].get("SMA200"):
                                st.metric("200-Day SMA", f"₹{technical_data['moving_averages']['SMA200']}")
                        
                        # Performance summary
                        if technical_data["performance"]:
                            st.subheader("Performance Summary")
                            perf_data = []
                            
                            for period, data in technical_data["performance"].items():
                                perf_data.append({
                                    "Period": period,
                                    "Absolute Change": data.get("absolute", ""),
                                    "Percentage Change": data.get("percentage", "")
                                })
                            
                            if perf_data:
                                st.dataframe(pd.DataFrame(perf_data))
                        
                        # Beta information
                        if technical_data["beta"]:
                            st.subheader("Beta")
                            beta_data = []
                            
                            for period, value in technical_data["beta"].items():
                                beta_data.append({
                                    "Period": period,
                                    "Beta": value
                                })
                            
                            if beta_data:
                                st.dataframe(pd.DataFrame(beta_data))
                    
                    # Indicators Tab
                    with tech_tabs[1]:
                        if technical_data["indicators"]:
                            st.subheader("Technical Indicators")
                            
                            # Create multiple columns for indicators
                            cols = st.columns(3)
                            
                            # Distribute indicators across columns
                            indicators = list(technical_data["indicators"].items())
                            for i, (indicator, value) in enumerate(indicators):
                                col_idx = i % 3
                                with cols[col_idx]:
                            # Display indicators with different colors based on value
                                    try:
                                        value_float = float(value)
                                        # Determine color based on indicator type and value
                                        color = "black"
                                        if indicator == "RSI":
                                            if value_float > 70:
                                                color = "red"  # Overbought
                                            elif value_float < 30:
                                                color = "green"  # Oversold
                                        elif indicator == "MFI":
                                            if value_float > 80:
                                                color = "red"  # Overbought
                                            elif value_float < 20:
                                                color = "green"  # Oversold
                                        
                                        st.markdown(f"<p style='color:{color}'><b>{indicator}:</b> {value}</p>", unsafe_allow_html=True)
                                    except ValueError:
                                        st.markdown(f"**{indicator}:** {value}")
                        else:
                            st.info("No technical indicators found")
                    
                    # Moving Averages Tab
                    with tech_tabs[2]:
                        if technical_data["moving_averages"]:
                            st.subheader("Moving Averages")
                            
                            # Create a dataframe for moving averages
                            ma_data = []
                            for ma_type, value in technical_data["moving_averages"].items():
                                ma_data.append({
                                    "Type": ma_type,
                                    "Value": value
                                })
                            
                            if ma_data:
                                # Sort by MA period (5, 10, 20, etc.)
                                def extract_period(ma_type):
                                    # Extract numeric part from MA type (e.g., SMA50 -> 50)
                                    match = re.search(r'(\d+)', ma_type)
                                    return int(match.group(1)) if match else 999
                                
                                ma_df = pd.DataFrame(ma_data)
                                ma_df['Sort'] = ma_df['Type'].apply(extract_period)
                                ma_df = ma_df.sort_values('Sort').drop('Sort', axis=1)
                                
                                # Display the dataframe
                                st.dataframe(ma_df)
                                
                                # Display price vs major MAs
                                if technical_data["current_price"]:
                                    current_price = float(technical_data["current_price"].replace(',', ''))
                                    
                                    st.subheader("Price vs Major Moving Averages")
                                    
                                    ma_comparison = []
                                    for ma_type, value in technical_data["moving_averages"].items():
                                        ma_value = float(value.replace(',', ''))
                                        status = "Above" if current_price > ma_value else "Below"
                                        diff_percent = abs(current_price - ma_value) / ma_value * 100
                                        
                                        ma_comparison.append({
                                            "Moving Average": ma_type,
                                            "Value": value,
                                            "Status": status,
                                            "Difference (%)": f"{diff_percent:.2f}%"
                                        })
                                    
                                    if ma_comparison:
                                        st.dataframe(pd.DataFrame(ma_comparison))
                        else:
                            st.info("No moving average data found")
                    
                    # Oscillators Tab
                    with tech_tabs[3]:
                        if technical_data["oscillators"]:
                            st.subheader("Oscillators")
                            
                            # Create multiple columns for oscillators
                            cols = st.columns(2)
                            
                            # Distribute oscillators across columns
                            oscillators = list(technical_data["oscillators"].items())
                            for i, (oscillator, value) in enumerate(oscillators):
                                col_idx = i % 2
                                with cols[col_idx]:
                                    # Determine color and interpretation based on oscillator type and value
                                    try:
                                        value_float = float(value)
                                        color = "black"
                                        interpretation = ""
                                        
                                        if oscillator == "RSI":
                                            if value_float > 70:
                                                color = "red"
                                                interpretation = "Overbought"
                                            elif value_float < 30:
                                                color = "green"
                                                interpretation = "Oversold"
                                            else:
                                                interpretation = "Neutral"
                                        elif oscillator == "Stochastic":
                                            if value_float > 80:
                                                color = "red"
                                                interpretation = "Overbought"
                                            elif value_float < 20:
                                                color = "green"
                                                interpretation = "Oversold"
                                            else:
                                                interpretation = "Neutral"
                                        elif oscillator == "CCI":
                                            if value_float > 100:
                                                color = "red"
                                                interpretation = "Overbought"
                                            elif value_float < -100:
                                                color = "green"
                                                interpretation = "Oversold"
                                            else:
                                                interpretation = "Neutral"
                                        elif oscillator == "Williams %R":
                                            if value_float > -20:
                                                color = "red"
                                                interpretation = "Overbought"
                                            elif value_float < -80:
                                                color = "green"
                                                interpretation = "Oversold"
                                            else:
                                                interpretation = "Neutral"
                                        
                                        if interpretation:
                                            st.markdown(f"<p style='color:{color}'><b>{oscillator}:</b> {value} ({interpretation})</p>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<p style='color:{color}'><b>{oscillator}:</b> {value}</p>", unsafe_allow_html=True)
                                    except ValueError:
                                        st.markdown(f"**{oscillator}:** {value}")
                        else:
                            st.info("No oscillator data found")
                    
                    # Volume Tab
                    with tech_tabs[4]:
                        if technical_data["volume_analysis"]:
                            st.subheader("Volume Analysis")
                            
                            # Display delivery percentage
                            if "Delivery Volume %" in technical_data["volume_analysis"]:
                                delivery_data = technical_data["volume_analysis"]["Delivery Volume %"]
                                
                                cols = st.columns(3)
                                with cols[0]:
                                    st.metric("Day Delivery %", delivery_data.get("Day", "N/A"))
                                with cols[1]:
                                    st.metric("Week Delivery %", delivery_data.get("Week", "N/A"))
                                with cols[2]:
                                    st.metric("Month Delivery %", delivery_data.get("Month", "N/A"))
                            
                            # Display volume info
                            col1, col2 = st.columns(2)
                            with col1:
                                if "NSE+BSE Volume" in technical_data["volume_analysis"]:
                                    st.metric("Total Volume", technical_data["volume_analysis"]["NSE+BSE Volume"])
                            with col2:
                                if "Delivery Volume" in technical_data["volume_analysis"]:
                                    st.metric("Delivery Volume", technical_data["volume_analysis"]["Delivery Volume"])
                        else:
                            st.info("No volume analysis data found")
                    
                    # Pivot Points Tab
                    # Pivot Points Tab
                    with tech_tabs[5]:
                        if technical_data["pivot_points"]:
                            st.subheader("Pivot Points")
                            
                            # Check if we have a nested structure with methods
                            methods = list(technical_data["pivot_points"].keys())
                            
                            # Create tabs for each method if we have multiple methods
                            if len(methods) > 1:
                                pivot_tabs = st.tabs(methods)
                                
                                # Display each method's pivot points in its own tab
                                for i, method in enumerate(methods):
                                    with pivot_tabs[i]:
                                        method_data = technical_data["pivot_points"][method]
                                        
                                        # Display pivots in columns
                                        res_col, pivot_col, supp_col = st.columns(3)
                                        
                                        with res_col:
                                            st.subheader("Resistance Levels")
                                            for level in ["R3", "R2", "R1"]:
                                                if level in method_data:
                                                    st.markdown(f"**{level}:** ₹{method_data[level]}")
                                        
                                        with pivot_col:
                                            st.subheader("Pivot Point")
                                            if "PP" in method_data:
                                                st.markdown(f"**PP:** ₹{method_data['PP']}")
                                        
                                        with supp_col:
                                            st.subheader("Support Levels")
                                            for level in ["S1", "S2", "S3"]:
                                                if level in method_data:
                                                    st.markdown(f"**{level}:** ₹{method_data[level]}")
                                        
                                        # Display price comparison for this method
                                        if technical_data["current_price"]:
                                            st.subheader("Price Location")
                                            
                                            try:
                                                # Convert current price to float
                                                current_price = float(technical_data["current_price"].replace(',', ''))
                                                
                                                # Process pivot levels for this method
                                                pivot_levels = {}
                                                for level, value in method_data.items():
                                                    try:
                                                        pivot_levels[level] = float(value.replace(',', ''))
                                                    except ValueError:
                                                        st.warning(f"Could not parse pivot value: {value}")
                                                
                                                # Determine price location
                                                if pivot_levels:
                                                    above_levels = []
                                                    below_levels = []
                                                    at_levels = []
                                                    
                                                    # Define a small tolerance (0.1% of price)
                                                    tolerance = current_price * 0.001
                                                    
                                                    for level, value in pivot_levels.items():
                                                        if current_price > value + tolerance:
                                                            above_levels.append(level)
                                                        elif current_price < value - tolerance:
                                                            below_levels.append(level)
                                                        else:
                                                            at_levels.append(level)
                                                    
                                                    if above_levels:
                                                        st.markdown(f"**Price is above:** {', '.join(above_levels)}")
                                                    if at_levels:
                                                        st.markdown(f"**Price is at/near:** {', '.join(at_levels)}")
                                                    if below_levels:
                                                        st.markdown(f"**Price is below:** {', '.join(below_levels)}")
                                                else:
                                                    st.warning("No valid pivot levels found for comparison")
                                                    
                                            except Exception as e:
                                                st.error(f"Error processing pivot points: {str(e)}")
                                                if debug_mode:
                                                    import traceback
                                                    st.error(f"Traceback: {traceback.format_exc()}")
                            else:
                                # For single method or flat structure, show directly
                                # Determine what data structure we have
                                if methods and isinstance(technical_data["pivot_points"][methods[0]], dict):
                                    # We have a nested structure with one method
                                    method_data = technical_data["pivot_points"][methods[0]]
                                    
                                    # Display columns
                                    res_col, pivot_col, supp_col = st.columns(3)
                                    
                                    with res_col:
                                        st.subheader("Resistance Levels")
                                        for level in ["R3", "R2", "R1"]:
                                            if level in method_data:
                                                st.markdown(f"**{level}:** ₹{method_data[level]}")
                                    
                                    with pivot_col:
                                        st.subheader("Pivot Point")
                                        if "PP" in method_data:
                                            st.markdown(f"**PP:** ₹{method_data['PP']}")
                                    
                                    with supp_col:
                                        st.subheader("Support Levels")
                                        for level in ["S1", "S2", "S3"]:
                                            if level in method_data:
                                                st.markdown(f"**{level}:** ₹{method_data[level]}")
                                else:
                                    # We have a flat structure (old format)
                                    res_col, pivot_col, supp_col = st.columns(3)
                                    
                                    with res_col:
                                        st.subheader("Resistance Levels")
                                        for level in ["R3", "R2", "R1"]:
                                            if level in technical_data["pivot_points"]:
                                                st.markdown(f"**{level}:** ₹{technical_data['pivot_points'][level]}")
                                    
                                    with pivot_col:
                                        st.subheader("Pivot Point")
                                        if "PP" in technical_data["pivot_points"]:
                                            st.markdown(f"**PP:** ₹{technical_data['pivot_points']['PP']}")
                                    
                                    with supp_col:
                                        st.subheader("Support Levels")
                                        for level in ["S1", "S2", "S3"]:
                                            if level in technical_data["pivot_points"]:
                                                st.markdown(f"**{level}:** ₹{technical_data['pivot_points'][level]}")
                                
                                # Display price comparison
                                if technical_data["current_price"]:
                                    st.subheader("Price Location")
                                    
                                    try:
                                        # Convert current price to float
                                        current_price = float(technical_data["current_price"].replace(',', ''))
                                        
                                        # Determine the structure of pivot_points
                                        pivot_levels = {}
                                        
                                        if methods and isinstance(technical_data["pivot_points"][methods[0]], dict):
                                            # Nested structure
                                            method_data = technical_data["pivot_points"][methods[0]]
                                            for level, value in method_data.items():
                                                try:
                                                    pivot_levels[level] = float(value.replace(',', ''))
                                                except ValueError:
                                                    st.warning(f"Could not parse pivot value: {value}")
                                        else:
                                            # Flat structure
                                            for level, value in technical_data["pivot_points"].items():
                                                if isinstance(value, str):
                                                    # Check if the value might be concatenated
                                                    if value.count('.') > 1:
                                                        # Extract first decimal number
                                                        decimal_matches = re.findall(r'(\d+\.\d+)', value)
                                                        if decimal_matches:
                                                            try:
                                                                pivot_levels[level] = float(decimal_matches[0])
                                                            except ValueError:
                                                                st.warning(f"Could not parse pivot value: {value}")
                                                    else:
                                                        try:
                                                            pivot_levels[level] = float(value.replace(',', ''))
                                                        except ValueError:
                                                            st.warning(f"Could not parse pivot value: {value}")
                                        
                                        # Determine price location
                                        if pivot_levels:
                                            above_levels = []
                                            below_levels = []
                                            at_levels = []
                                            
                                            # Define a small tolerance (0.1% of price)
                                            tolerance = current_price * 0.001
                                            
                                            for level, value in pivot_levels.items():
                                                if current_price > value + tolerance:
                                                    above_levels.append(level)
                                                elif current_price < value - tolerance:
                                                    below_levels.append(level)
                                                else:
                                                    at_levels.append(level)
                                            
                                            if above_levels:
                                                st.markdown(f"**Price is above:** {', '.join(above_levels)}")
                                            if at_levels:
                                                st.markdown(f"**Price is at/near:** {', '.join(at_levels)}")
                                            if below_levels:
                                                st.markdown(f"**Price is below:** {', '.join(below_levels)}")
                                        else:
                                            st.warning("No valid pivot levels found for comparison")
                                            
                                    except Exception as e:
                                        st.error(f"Error processing pivot points: {str(e)}")
                                        if debug_mode:
                                            st.error(f"Debug info - pivot_points data: {technical_data['pivot_points']}")
                                            import traceback
                                            st.error(f"Traceback: {traceback.format_exc()}")
                                else:
                                    st.info("Current price information not available for comparison")
                        else:
                            st.info("No pivot point data found")
        
        # Scrape share price history data if URL provided
        # Scrape share price history data if URL provided
        if price_history_url:
            with st.expander("Share Price History Results", expanded=True):
                st.subheader("Share Price History Analysis")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(price_history_url)
                
                if soup:
                    # Save HTML for debugging
                    html_filename = f"debug_{base_filename}_price_history.html"
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    st.success(f"Saved raw HTML to {html_filename} for debugging")
                    
                    # Extract share price history
                    price_history_data = extract_share_price_history(soup)
                    scraped_data["price_history_data"] = price_history_data
                    
                    # Create tabs for different price history sections
                    history_tabs = st.tabs(["Daily Data", "Returns Comparison", "Seasonality", "Chart"])
                    
                    # Daily Data
                    with history_tabs[0]:
                        if price_history_data["daily_data"]:
                            st.write("**Daily Trading Data**")
                            
                            # Convert to DataFrame
                            df_daily = pd.DataFrame(price_history_data["daily_data"])
                            
                            # Display table
                            st.dataframe(df_daily)
                            
                            # Download CSV button for daily data
                            csv = df_daily.to_csv(index=False)
                            st.download_button(
                                label="Download Daily Data CSV",
                                data=csv,
                                file_name=f"{stock_symbol.replace('.', '_')}_daily_data.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No daily share price data found")
                    
                    # Returns Comparison
                    with history_tabs[1]:
                        if price_history_data["returns_comparison"]:
                            st.write("**Returns Comparison**")
                            
                            # Convert to DataFrame
                            df_comparison = pd.DataFrame(price_history_data["returns_comparison"])
                            
                            # Display table
                            st.dataframe(df_comparison)
                            
                            # If there are percentage returns, create a bar chart
                            if 'Returns' in df_comparison.columns:
                                try:
                                    # Create a copy to avoid modifying the original
                                    df_chart = df_comparison.copy()
                                    
                                    # Convert percentage strings to float values
                                    df_chart['Returns_numeric'] = df_chart['Returns'].str.rstrip('%').astype('float')
                                    
                                    # Sort by entity for better visualization
                                    entity_order = ['Day', '1 Day', 'Week', '1 Week', 'Month', '1 Month', 
                                                '3 Months', 'Qtr', 'Half Year', '6 Months', '1 Yr', '1 Year', 
                                                '3 Yr', '3 Years', '5 Yr', '5 Years', '10 Yr', '10 Years']
                                    
                                    # Create a custom sort key
                                    def custom_sort(entity):
                                        try:
                                            return entity_order.index(entity)
                                        except ValueError:
                                            return 999  # Put unknown entities at the end
                                    
                                    if 'entity' in df_chart.columns:
                                        df_chart['sort_key'] = df_chart['entity'].apply(custom_sort)
                                        df_chart = df_chart.sort_values('sort_key')
                                        
                                        # Create the chart
                                        st.subheader("Returns Comparison Chart")
                                        
                                        chart_data = pd.DataFrame({
                                            'Period': df_chart['entity'],
                                            'Returns (%)': df_chart['Returns_numeric']
                                        })
                                        
                                        st.bar_chart(chart_data.set_index('Period'))
                                except Exception as e:
                                    st.error(f"Error creating comparison chart: {e}")
                        else:
                            st.info("No returns comparison data found")
                    
                    # Seasonality
                    with history_tabs[2]:
                        if price_history_data["returns_seasonality"]:
                            st.write("**Returns Seasonality**")
                            
                            # Convert to DataFrame
                            df_seasonality = pd.DataFrame(price_history_data["returns_seasonality"])
                            
                            # Display table
                            st.dataframe(df_seasonality)
                            
                            # Try to create a heatmap-like visualization
                            if not df_seasonality.empty and 'Year' in df_seasonality.columns:
                                try:
                                    # Set Year as index
                                    df_heat = df_seasonality.set_index('Year')
                                    
                                    # Convert percentage strings to numeric
                                    for col in df_heat.columns:
                                        if isinstance(df_heat[col].iloc[0], str) and '%' in df_heat[col].iloc[0]:
                                            df_heat[col] = df_heat[col].str.rstrip('%').astype('float')
                                    
                                    # Display with conditional formatting
                                    st.subheader("Monthly Returns Heatmap")
                                    
                                    def color_negative_red(val):
                                        """
                                        Takes a scalar and returns a string with
                                        the css property `'color: red'` for negative
                                        values, green for positive values
                                        """
                                        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                                        return f'color: {color}'
                                    
                                    # Apply the style function to the dataframe
                                    styled_df = df_heat.style.applymap(color_negative_red)
                                    st.dataframe(styled_df)
                                except Exception as e:
                                    st.error(f"Error creating seasonality heatmap: {e}")
                        else:
                            st.info("No seasonality data found")
                    
                    # Chart
                    with history_tabs[3]:
                        if price_history_data["daily_data"]:
                            st.write("**Price Chart**")
                            
                            try:
                                df = pd.DataFrame(price_history_data["daily_data"])
                                
                                # Find date and close columns
                                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                                close_col = next((col for col in df.columns if 'close' in col.lower()), None)
                                
                                if date_col and close_col:
                                    # Convert data types
                                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                                    df[close_col] = pd.to_numeric(df[close_col].str.replace(',', ''), errors='coerce')
                                    
                                    # Sort by date
                                    df = df.sort_values(by=date_col)
                                    
                                    # Create the chart
                                    st.line_chart(df.set_index(date_col)[close_col])
                                    
                                    # Add volume chart if available
                                    volume_col = next((col for col in df.columns if 'volume' in col.lower()), None)
                                    if volume_col:
                                        df[volume_col] = pd.to_numeric(df[volume_col].str.replace(',', ''), errors='coerce')
                                        st.subheader("Volume Chart")
                                        st.bar_chart(df.set_index(date_col)[volume_col])
                            except Exception as e:
                                st.error(f"Error creating chart: {e}")
                        else:
                            st.info("No data available for charting")
            # After processing is complete
            if debug_mode:
                st.write("🔍 **DEBUG:** Data scraping complete")
                st.write(f"🔍 **DEBUG:** Data saved to file: {json_filename}")
        
        # Save all scraped data to JSON file
        json_filename = f"{base_filename}_all_data.json"
        if save_data_to_file(scraped_data, json_filename):
            st.success(f"All scraped data saved to {json_filename}")
            
            # Provide download button for the JSON data
            with open(json_filename, 'r', encoding='utf-8') as f:
                json_data = f.read()
                
            st.download_button(
                label="Download Scraped Data (JSON)",
                data=json_data,
                file_name=json_filename,
                mime="application/json"
            )

if __name__ == "__main__":
    main()