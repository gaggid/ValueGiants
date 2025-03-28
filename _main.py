import streamlit as st
import pandas as pd
import yfinance as yf
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

# Set page configuration
st.set_page_config(page_title="Investment Data Scraper", page_icon="📊", layout="wide")

@st.cache_resource
def get_selenium_driver():
    """Initialize and return a Selenium WebDriver."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-features=VizDisplayCompositor")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_with_selenium(url, wait_time=5):
    """Scrape content using Selenium for JavaScript-heavy pages."""
    driver = get_selenium_driver()
    
    try:
        st.info(f"Loading page with Selenium: {url}")
        driver.get(url)
        # Wait for JavaScript to load
        time.sleep(wait_time)
        
        # Get the page source after JavaScript execution
        page_source = driver.page_source
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        return soup, page_source
    except Exception as e:
        st.error(f"Error with Selenium: {str(e)}")
        return None, None
    finally:
        # Don't close the driver as it's cached
        pass

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
                if len(cells) < len(headers):
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

def get_stock_info(symbol):
    """Get basic stock information using yfinance."""
    try:
        st.info(f"Fetching stock information for: {symbol}")
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Extract relevant information
        relevant_info = {
            "name": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currentPrice": info.get("currentPrice", "N/A"),
            "marketCap": info.get("marketCap", "N/A"),
            "trailingPE": info.get("trailingPE", "N/A"),
            "dividendYield": info.get("dividendYield", "N/A") * 100 if info.get("dividendYield") else "N/A",
            "52WeekHigh": info.get("fiftyTwoWeekHigh", "N/A"),
            "52WeekLow": info.get("fiftyTwoWeekLow", "N/A"),
        }
        
        return relevant_info
    except Exception as e:
        st.error(f"Error fetching stock information: {str(e)}")
        return {}

def extract_financial_tables(soup):
    """Extract financial tables from a BeautifulSoup object."""
    financial_data = {
        "quarterly": {},
        "annual": {},
        "balance_sheet": {},
        "ratios": {},
        "cash_flow": {}
    }
    
    # Find all tables
    tables = soup.find_all('table')
    st.write(f"Found {len(tables)} tables on the page")
    
    for i, table in enumerate(tables):
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
        
        # Extract rows
        rows = []
        for tr in table.find_all('tr')[1:] if headers else table.find_all('tr'):
            row = [td.text.strip() for td in tr.find_all(['td', 'th'])]
            if row and len(row) > 1:  # Ensure row has content
                rows.append(row)
        
        # Skip empty tables
        if not rows:
            continue
            
        # Categorize the table based on its content
        table_text = ' '.join(headers) + ' ' + ' '.join([' '.join(row) for row in rows])
        table_text = table_text.lower()
        
        category = None
        
        # Determine category based on table content
        if 'quarterly' in table_text or 'quarter' in table_text:
            category = "quarterly"
        elif 'annual' in table_text or 'yearly' in table_text or 'year' in table_text:
            category = "annual"
        elif 'balance sheet' in table_text or 'assets' in table_text or 'liabilities' in table_text:
            category = "balance_sheet"
        elif 'ratio' in table_text or 'roe' in table_text or 'roa' in table_text:
            category = "ratios"
        elif 'cash flow' in table_text or 'operating activities' in table_text:
            category = "cash_flow"
        else:
            # Try to infer from structure if not explicitly stated
            # Quarter columns often have Q1, Q2, Q3, Q4 or month names
            quarter_pattern = re.compile(r'q[1-4]|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec')
            year_pattern = re.compile(r'20\d\d|19\d\d|fy\d\d')
            
            quarter_matches = sum(1 for h in headers if quarter_pattern.search(h.lower()))
            year_matches = sum(1 for h in headers if year_pattern.search(h.lower()))
            
            if quarter_matches > year_matches:
                category = "quarterly"
            elif year_matches > 0:
                category = "annual"
            else:
                # Default to quarterly as it's most common
                category = "quarterly"
        
        # Store the table data
        financial_data[category][f"Table_{i+1}"] = {
            "headers": headers,
            "rows": rows
        }
    
    return financial_data

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
    
    # Extract support and resistance
    support_resistance = {}
    
    # Support levels
    # Support levels
    support_patterns = [
        (r'Support 1\s*([\d,.]+)', 'S1'),
        (r'Support 2\s*([\d,.]+)', 'S2'),
        (r'Support 3\s*([\d,.]+)', 'S3')
    ]

    support_list = []
    for pattern, label in support_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                # Make sure we're only getting valid number strings
                if re.match(r'^[\d,.]+$', match):
                    support_list.append(match)

    # Resistance levels
    resistance_patterns = [
        (r'Resistance 1\s*([\d,.]+)', 'R1'),
        (r'Resistance 2\s*([\d,.]+)', 'R2'),
        (r'Resistance 3\s*([\d,.]+)', 'R3')
    ]

    resistance_list = []
    for pattern, label in resistance_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                # Make sure we're only getting valid number strings
                if re.match(r'^[\d,.]+$', match):
                    resistance_list.append(match)
    
    if resistance_list:
        support_resistance["resistance"] = resistance_list
    
    technical_data["support_resistance"] = support_resistance
    
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

def main():
    st.title("📊 Advanced Stock Data Scraper")
    st.write("Enter company details and URLs to scrape investment research data.")
    
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
        
        # Add new field for share price history
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
        
        # Try to get basic stock information from yfinance
        stock_info = get_stock_info(stock_symbol)
        if stock_info:
            scraped_data["stock_info"] = stock_info
            
            # Display basic stock information
            st.subheader("Basic Stock Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.metric("Current Price", f"₹{stock_info.get('currentPrice', 'N/A')}")
            
            with info_col2:
                market_cap = stock_info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap = f"₹{market_cap/10000000:.2f} Cr" if market_cap > 10000000 else f"₹{market_cap:,}"
                st.metric("Market Cap", market_cap)
            
            with info_col3:
                st.metric("P/E Ratio", stock_info.get('trailingPE', 'N/A'))
        
        # Scrape financial data if URL provided
        if financials_url:
            with st.expander("Financial Data Scraping Results", expanded=True):
                st.subheader("Financial Data")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(financials_url)
                
                if soup:
                    # Save HTML for debugging
                    html_filename = f"debug_{base_filename}_financials.html"
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    st.success(f"Saved raw HTML to {html_filename} for debugging")
                    
                    # Extract financial tables
                    financial_data = extract_financial_tables(soup)
                    scraped_data["financial_data"] = financial_data
                    
                    # Create tabs for different financial sections
                    fin_tabs = st.tabs(["Quarterly Results", "Annual Results", "Balance Sheet", "Financial Ratios", "Cash Flow"])
                    
                    # Quarterly Results
                    with fin_tabs[0]:
                        if financial_data["quarterly"]:
                            for table_name, table_data in financial_data["quarterly"].items():
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
                            st.info("No quarterly data found")
                    
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
        
        # Scrape technical data if URL provided
        # Scrape technical data if URL provided
        if technicals_url:
            with st.expander("Technical Analysis Results", expanded=True):
                st.subheader("Technical Analysis")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(technicals_url)
                
                if soup:
                    # Save HTML for debugging
                    html_filename = f"debug_{base_filename}_technicals.html"
                    with open(html_filename, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    st.success(f"Saved raw HTML to {html_filename} for debugging")
                    
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
                    with tech_tabs[5]:
                        if technical_data["pivot_points"]:
                            st.subheader("Pivot Points")
                            
                            # Create columns for resistances and supports
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
                            
                            # Display price vs pivot levels if current price is available
                            # Display price vs pivot levels if current price is available
                            if technical_data["current_price"]:
                                st.subheader("Price Location")
                                current_price = float(technical_data["current_price"].replace(',', ''))
                                
                                pivot_levels = {}
                                for level, value in technical_data["pivot_points"].items():
                                    try:
                                        pivot_levels[level] = float(value.replace(',', ''))
                                    except ValueError:
                                        # Skip values that can't be converted to float
                                        st.warning(f"Could not parse pivot value: {value}")
                                        continue
                                
                                # Only proceed if we have valid pivot levels
                                if pivot_levels:
                                    # Determine where price is relative to pivot levels
                                    above_levels = []
                                    below_levels = []
                                    
                                    for level, value in pivot_levels.items():
                                        if current_price > value:
                                            above_levels.append(level)
                                        else:
                                            below_levels.append(level)
                                    
                                    if above_levels:
                                        st.markdown(f"**Price is above:** {', '.join(above_levels)}")
                                    if below_levels:
                                        st.markdown(f"**Price is below:** {', '.join(below_levels)}")
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