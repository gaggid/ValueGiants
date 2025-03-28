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

def extract_news(soup):
    """Extract news articles from a BeautifulSoup object."""
    news_items = []
    
    # Look for news containers
    news_articles = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in x.lower() 
                                                                             for term in ['news', 'article', 'story']))
    
    # If no articles found, try looking for news lists
    if not news_articles:
        news_lists = soup.find_all(['ul', 'ol'], class_=lambda x: x and ('news' in x.lower() or 'list' in x.lower()))
        for news_list in news_lists:
            news_articles.extend(news_list.find_all('li'))
    
    # Process each news item
    for i, article in enumerate(news_articles[:20]):  # Limit to 20 items
        # Extract title
        title_elem = article.find(['h1', 'h2', 'h3', 'h4', 'a'])
        title = title_elem.text.strip() if title_elem else article.text.strip()[:100]
        
        # Extract date
        date_elem = article.find(['time', 'span'], class_=lambda x: x and ('date' in x.lower() or 'time' in x.lower()))
        date = date_elem.text.strip() if date_elem else "Date not found"
        
        # Extract link
        link = ""
        if title_elem and title_elem.name == 'a':
            link = title_elem.get('href', '')
        else:
            link_elem = article.find('a')
            if link_elem:
                link = link_elem.get('href', '')
        
        # Extract summary
        summary_elem = article.find('p')
        summary = summary_elem.text.strip() if summary_elem else ""
        
        if title and (summary or link):
            news_items.append({
                "id": i+1,
                "title": title,
                "date": date,
                "link": link,
                "summary": summary
            })
    
    return news_items

def extract_technical_indicators(soup):
    """Extract technical indicators from a BeautifulSoup object."""
    technical_data = {
        "current_price": None,
        "indicators": {},
        "support_resistance": {},
        "moving_averages": {}
    }
    
    # Extract text content
    text = soup.get_text()
    
    # Find current price
    price_pattern = re.compile(r'current\s+price[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', re.IGNORECASE)
    price_match = price_pattern.search(text)
    if price_match:
        technical_data["current_price"] = price_match.group(2)
    
    # Common technical indicators
    indicator_patterns = [
        (r'RSI[:\s]*(\d[\d,.]*(?:\.\d+)?)', 'RSI'),
        (r'MACD[:\s]*([+-]?\d[\d,.]*(?:\.\d+)?)', 'MACD'),
        (r'Stochastic[:\s]*(\d[\d,.]*(?:\.\d+)?)', 'Stochastic'),
        (r'ADX[:\s]*(\d[\d,.]*(?:\.\d+)?)', 'ADX'),
        (r'Volume[:\s]*([\d,.]*(?:\.\d+)?)', 'Volume'),
    ]
    
    for pattern, indicator in indicator_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["indicators"][indicator] = match.group(1)
    
    # Support and resistance levels
    support_pattern = re.compile(r'support[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', re.IGNORECASE)
    resistance_pattern = re.compile(r'resistance[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', re.IGNORECASE)
    
    support_matches = support_pattern.findall(text)
    resistance_matches = resistance_pattern.findall(text)
    
    if support_matches:
        technical_data["support_resistance"]["support"] = [match[-1] for match in support_matches[:3]]
    
    if resistance_matches:
        technical_data["support_resistance"]["resistance"] = [match[-1] for match in resistance_matches[:3]]
    
    # Moving averages
    ma_patterns = [
        (r'50[- ]day MA[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', 'MA50'),
        (r'200[- ]day MA[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', 'MA200'),
        (r'20[- ]day MA[:\s]*(₹|Rs\.?|INR)?\s*(\d[\d,.]*(?:\.\d+)?)', 'MA20'),
    ]
    
    for pattern, ma_type in ma_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            technical_data["moving_averages"][ma_type] = match.group(2)
    
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
        
        news_url = st.text_input(
            "News URL (Optional)",
            value="",
            help="URL containing recent news about the company"
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
                                        df = pd.DataFrame(table_data["rows"])
                                        if len(df.columns) == len(table_data["headers"]):
                                            df.columns = table_data["headers"]
                                        st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                                        st.write("Headers:", table_data["headers"])
                                        st.write("Sample row:", table_data["rows"][0] if table_data["rows"] else "No rows")
                        else:
                            st.info("No quarterly data found")
                    
                    # Annual Results
                    with fin_tabs[1]:
                        if financial_data["annual"]:
                            for table_name, table_data in financial_data["annual"].items():
                                st.write(f"**{table_name}**")
                                if table_data["headers"] and table_data["rows"]:
                                    try:
                                        df = pd.DataFrame(table_data["rows"])
                                        if len(df.columns) == len(table_data["headers"]):
                                            df.columns = table_data["headers"]
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
                                        df = pd.DataFrame(table_data["rows"])
                                        if len(df.columns) == len(table_data["headers"]):
                                            df.columns = table_data["headers"]
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
                                        df = pd.DataFrame(table_data["rows"])
                                        if len(df.columns) == len(table_data["headers"]):
                                            df.columns = table_data["headers"]
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
                                        df = pd.DataFrame(table_data["rows"])
                                        if len(df.columns) == len(table_data["headers"]):
                                            df.columns = table_data["headers"]
                                        st.dataframe(df)
                                    except Exception as e:
                                        st.error(f"Error displaying table: {e}")
                        else:
                            st.info("No cash flow data found")
        
        # Scrape technical data if URL provided
        if technicals_url:
            with st.expander("Technical Analysis Results", expanded=True):
                st.subheader("Technical Analysis")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(technicals_url)
                
                if soup:
                    # Extract technical indicators
                    technical_data = extract_technical_indicators(soup)
                    scraped_data["technical_data"] = technical_data
                    
                    # Display technical data
                    if technical_data["current_price"]:
                        st.metric("Current Price", f"₹{technical_data['current_price']}")
                    
                    # Display technical indicators
                    if technical_data["indicators"]:
                        st.subheader("Technical Indicators")
                        indicators_col1, indicators_col2 = st.columns(2)
                        
                        # Split indicators between columns
                        indicators = list(technical_data["indicators"].items())
                        half = len(indicators) // 2
                        
                        with indicators_col1:
                            for indicator, value in indicators[:half]:
                                st.metric(indicator, value)
                        
                        with indicators_col2:
                            for indicator, value in indicators[half:]:
                                st.metric(indicator, value)
                    
                    # Display support and resistance
                    if technical_data["support_resistance"]:
                        st.subheader("Support and Resistance Levels")
                        support_col, resistance_col = st.columns(2)
                        
                        with support_col:
                            st.write("**Support Levels:**")
                            if "support" in technical_data["support_resistance"]:
                                for i, level in enumerate(technical_data["support_resistance"]["support"]):
                                    st.write(f"S{i+1}: ₹{level}")
                        
                        with resistance_col:
                            st.write("**Resistance Levels:**")
                            if "resistance" in technical_data["support_resistance"]:
                                for i, level in enumerate(technical_data["support_resistance"]["resistance"]):
                                    st.write(f"R{i+1}: ₹{level}")
                    
                    # Display moving averages
                    if technical_data["moving_averages"]:
                        st.subheader("Moving Averages")
                        for ma_type, value in technical_data["moving_averages"].items():
                            st.metric(ma_type, f"₹{value}")
        
        # Scrape news data if URL provided
        if news_url:
            with st.expander("News Scraping Results", expanded=True):
                st.subheader("Latest News")
                
                # Use Selenium to handle JavaScript-rendered content
                soup, html_content = scrape_with_selenium(news_url)
                
                if soup:
                    # Extract news
                    news_items = extract_news(soup)
                    scraped_data["news_data"] = {"news_items": news_items}
                    
                    # Display news items
                    if news_items:
                        for item in news_items:
                            st.write(f"**{item['title']}**")
                            st.write(f"*{item['date']}*")
                            if item["summary"]:
                                st.write(item["summary"])
                            if item["link"]:
                                st.write(f"[Read more]({item['link']})")
                            st.write("---")
                    else:
                        st.info("No news items found")
        
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