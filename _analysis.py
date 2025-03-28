import streamlit as st
import json
import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

st.set_page_config(page_title="Stock Analysis Report Generator", page_icon="📊", layout="wide")

def load_data_from_json(file_path):
    """Load previously scraped data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def run_phased_analysis(data, model_name="phi3:medium-128k"):
    """Run analysis in distinct phases with improved chunking and validation."""
    
    company_name = data['metadata']['company_name']
    stock_symbol = data['metadata']['stock_symbol']
    current_price = data.get('technical_data', {}).get('current_price', 'N/A')
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Extract industry type if available
    industry_type = get_industry_type(data)
    
    # Phase 1: Extract and analyze key data points
    st.info("Phase 1/5: Extracting Key Financial Data...")
    key_data_points = extract_key_data_points(data)
    progress_bar.progress(10)
    
    # Phase 2: Fundamental Analysis
    st.info("Phase 2/5: Running Fundamental Analysis...")
    fundamental_analysis = run_fundamental_analysis(key_data_points, company_name, stock_symbol, industry_type, model_name)
    progress_bar.progress(30)
    
    # Phase 3: Technical Analysis
    st.info("Phase 3/5: Running Technical Analysis...")
    technical_analysis = run_technical_analysis(key_data_points, company_name, stock_symbol, current_price, model_name)
    progress_bar.progress(50)
    
    # Phase 4: Price History Analysis
    st.info("Phase 4/5: Analyzing Price History and Performance...")
    price_history_analysis = analyze_price_history(key_data_points, company_name, stock_symbol, model_name)
    progress_bar.progress(70)
    
    # Phase 5: Generate final assessment and recommendation
    st.info("Phase 5/5: Preparing Final Assessment and Recommendation...")
    final_assessment = generate_recommendation(key_data_points, company_name, stock_symbol, current_price, 
                                              fundamental_analysis, technical_analysis, price_history_analysis, model_name)
    progress_bar.progress(100)
    
    # Construct the full report
    full_analysis = f"""
    # Investment Analysis Report: {company_name} ({stock_symbol})

    <div style="background-color:#f5f5f5; padding:15px; border-left:5px solid #4CAF50; margin-bottom:20px;">
    <strong>Date:</strong> {datetime.now().strftime("%B %d, %Y")}<br>
    <strong>Current Price:</strong> {current_price}
    </div>

    ## Executive Summary

    {final_assessment}

    <hr style="border-top:2px solid #ccc; margin:30px 0;">

    ## Detailed Analysis

    <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; margin-bottom:20px;">
    <h3>1. Fundamental Analysis</h3>
    {fundamental_analysis}
    </div>

    <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; margin-bottom:20px;">
    <h3>2. Technical Analysis</h3>
    {technical_analysis}
    </div>

    <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; margin-bottom:20px;">
    <h3>3. Historical Performance Analysis</h3>
    {price_history_analysis}
    </div>

    ---

    <small><em>This report was generated using automated analysis with human-like reasoning capabilities.</em></small>
    """
    
    return full_analysis

def get_industry_type(data):
    """Identify the industry type from the data if possible"""
    # First check if explicitly stated
    if "stock_info" in data and "industry" in data["stock_info"]:
        return data["stock_info"]["industry"]
    
    # If not, try to infer from company name or data
    company_name = data['metadata']['company_name'].lower()
    
    # Basic industry detection - expand as needed
    if any(term in company_name for term in ["bank", "financial", "insurance", "finance"]):
        return "Financial Services"
    elif any(term in company_name for term in ["tech", "software", "digital", "electronic"]):
        return "Technology"
    elif any(term in company_name for term in ["pharma", "health", "drug", "medical"]):
        return "Healthcare"
    elif any(term in company_name for term in ["oil", "gas", "petro", "energy", "power"]):
        return "Energy"
    else:
        # Look for clues in the financial data
        if 'financial_data' in data:
            if 'ratios' in data['financial_data']:
                for table_name, table_data in data['financial_data']['ratios'].items():
                    for row in table_data.get('rows', []):
                        if len(row) > 0 and 'Net Interest Margin' in row[0]:
                            return "Financial Services"
        
        return "General"

def extract_key_data_points(data):
    """Extract and structure the most important data points to reduce model load"""
    key_data = {
        "company_info": {
            "name": data['metadata']['company_name'],
            "symbol": data['metadata']['stock_symbol'],
            "current_price": data.get('technical_data', {}).get('current_price', 'N/A'),
        },
        "financial_summary": {},
        "technical_summary": {},
        "performance_summary": {}
    }
    
    # Extract key financial metrics
    if 'financial_data' in data:
        financial_data = data['financial_data']
        
        # Quarterly Revenue and Profit trends
        if 'quarterly' in financial_data:
            quarterly = {}
            for table_name, table_data in financial_data['quarterly'].items():
                if not table_data.get('rows'):
                    continue
                
                # Extract revenue, profit, and margins
                for row in table_data['rows']:
                    if not row or len(row) < 3:
                        continue
                    
                    if 'Total Rev' in row[0]:
                        quarterly['revenue'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[2:7] if val],
                            'headers': table_data['headers'][2:7] if len(table_data['headers']) >= 7 else table_data['headers'][2:]
                        }
                    elif 'Net Profit' in row[0]:
                        quarterly['net_profit'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[2:7] if val],
                            'headers': table_data['headers'][2:7] if len(table_data['headers']) >= 7 else table_data['headers'][2:]
                        }
                    elif 'Operating Profit Margin' in row[0]:
                        quarterly['operating_margin'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[2:7] if val],
                            'headers': table_data['headers'][2:7] if len(table_data['headers']) >= 7 else table_data['headers'][2:]
                        }
            
            key_data['financial_summary']['quarterly'] = quarterly
        
        # Annual data
        if 'annual' in financial_data:
            annual = {}
            for table_name, table_data in financial_data['annual'].items():
                if not table_data.get('rows'):
                    continue
                
                # Extract key annual metrics
                for row in table_data['rows']:
                    if not row or len(row) < 5:
                        continue
                    
                    if any(metric in row[0] for metric in ['Total Rev', 'Total Revenue']):
                        annual['revenue'] = {
                            'label': row[0],
                            'cagr_3yr': row[2] if len(row) > 2 else 'N/A',
                            'cagr_5yr': row[3] if len(row) > 3 else 'N/A',
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
                    elif 'Net Profit' in row[0]:
                        annual['net_profit'] = {
                            'label': row[0],
                            'cagr_3yr': row[2] if len(row) > 2 else 'N/A',
                            'cagr_5yr': row[3] if len(row) > 3 else 'N/A',
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
                    elif 'NPM' in row[0] or 'Net Profit Margin' in row[0] or 'NETPCT' in row[0]:
                        annual['net_margin'] = {
                            'label': row[0],
                            'cagr_3yr': row[2] if len(row) > 2 else 'N/A',
                            'cagr_5yr': row[3] if len(row) > 3 else 'N/A',
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
            
            key_data['financial_summary']['annual'] = annual
        
        # Key ratios
        if 'ratios' in financial_data:
            ratios = {}
            for table_name, table_data in financial_data['ratios'].items():
                if not table_data.get('rows'):
                    continue
                
                # Extract important ratios
                for row in table_data['rows']:
                    if not row or len(row) < 5:
                        continue
                    
                    if 'ROE' in row[0]:
                        ratios['roe'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
                    elif 'ROA' in row[0]:
                        ratios['roa'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
                    elif any(d in row[0] for d in ['Debt to Equity', 'DEBT_CE']):
                        ratios['debt_equity'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
                    elif 'Price' in row[0] and 'BV' in row[0]:
                        ratios['price_to_book'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[4:9] if val],
                            'headers': table_data['headers'][4:9] if len(table_data['headers']) >= 9 else table_data['headers'][4:]
                        }
            
            key_data['financial_summary']['ratios'] = ratios
    
    # Extract key technical indicators
    if 'technical_data' in data:
        tech_data = data['technical_data']
        
        # Basic technical indicators
        if 'indicators' in tech_data:
            key_data['technical_summary']['indicators'] = {
                k: v for k, v in tech_data['indicators'].items() 
                if k in ['RSI', 'MACD', 'MFI', 'ATR', 'ADX', 'Momentum Score']
            }
        
        # Moving averages
        if 'moving_averages' in tech_data:
            key_data['technical_summary']['moving_averages'] = {
                k: v for k, v in tech_data['moving_averages'].items()
                if k in ['SMA20', 'SMA50', 'SMA100', 'SMA200'] or (k.startswith('SMA') and len(tech_data['moving_averages']) <= 4)
            }
        
        # Support and resistance levels
        if 'support_resistance' in tech_data:
            support = tech_data['support_resistance'].get('support', [])
            resistance = tech_data['support_resistance'].get('resistance', [])
            
            if support or resistance:
                key_data['technical_summary']['support_resistance'] = {
                    'support': support[:3] if support else [],  # Top 3 support levels
                    'resistance': resistance[:3] if resistance else []  # Top 3 resistance levels
                }
        
        # Performance metrics
        if 'performance' in tech_data:
            key_data['technical_summary']['performance'] = {
                k: v for k, v in tech_data['performance'].items()
                if k in ['1 Day', '1 Week', '1 Month', '3 Months', '1 Year']
            }
    
    # Extract performance comparison data
    if 'price_history_data' in data:
        # Returns comparison
        if 'returns_comparison' in data['price_history_data']:
            comparison_data = []
            for item in data['price_history_data']['returns_comparison']:
                if 'Time' in item and item['Time'] in ['1 Month', '3 Months', '1 Year', '3 Year', '5 Years']:
                    comparison_item = {'period': item['Time']}
                    for k, v in item.items():
                        if k not in ['Time', 'entity'] and 'Returns' in k:
                            comparison_item[k] = v
                    comparison_data.append(comparison_item)
            
            if comparison_data:
                key_data['performance_summary']['returns_comparison'] = comparison_data
        
        # Seasonality
        if 'returns_seasonality' in data['price_history_data']:
            seasonality = data['price_history_data']['returns_seasonality']
            
            # Get the most recent 3 years
            recent_years = seasonality[:3] if len(seasonality) >= 3 else seasonality
            
            if recent_years:
                key_data['performance_summary']['seasonality'] = []
                
                for year_data in recent_years:
                    if 'Year' in year_data and 'Annual Returns' in year_data:
                        year_item = {
                            'year': year_data['Year'],
                            'annual_return': year_data['Annual Returns']
                        }
                        
                        # Find best and worst months
                        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        month_returns = []
                        
                        for month in months:
                            if month in year_data and year_data[month] != "-":
                                try:
                                    value = float(year_data[month].strip('%'))
                                    month_returns.append((month, value))
                                except (ValueError, TypeError):
                                    pass
                        
                        if month_returns:
                            month_returns.sort(key=lambda x: x[1])
                            year_item['worst_month'] = f"{month_returns[0][0]} ({month_returns[0][1]}%)"
                            year_item['best_month'] = f"{month_returns[-1][0]} ({month_returns[-1][1]}%)"
                        
                        key_data['performance_summary']['seasonality'].append(year_item)
    
    return key_data

def run_fundamental_analysis(key_data, company_name, stock_symbol, industry_type, model_name):
    """Run fundamental analysis using the extracted key data points"""
    
    # Prepare the data for the fundamental analysis
    quarterly_data = format_quarterly_data(key_data)
    annual_data = format_annual_data(key_data)
    ratio_data = format_ratio_data(key_data)
    
    # Create a focused prompt for fundamental analysis
    fundamental_prompt = f"""
    You are a seasoned financial analyst specializing in equity research. Analyze the fundamental financial performance of {company_name} ({stock_symbol}) in the {industry_type} sector.
    
    FOCUS ONLY on these data points:
    
    QUARTERLY TRENDS:
    {quarterly_data}
    
    ANNUAL PERFORMANCE:
    {annual_data}
    
    KEY FINANCIAL RATIOS:
    {ratio_data}
    
    Provide a concise, professional fundamental analysis that covers:
    1. Revenue and profit growth trends, highlighting YoY and sequential growth rates
    2. Margin trends and their implications for profitability
    3. Return metrics (ROE, ROA) and capital efficiency
    4. Financial health based on debt/equity and other relevant metrics
    5. Valuation assessment based on available metrics
    
    Format your analysis in 3-4 clear paragraphs. Focus only on facts evident in the data provided.
    Use specific numbers to support your points. Identify clear strengths and weaknesses.
    """
    
    # Run the fundamental analysis
    fundamental_analysis = run_ollama_analysis(fundamental_prompt, model_name)
    
    # Validate output
    if not validate_analysis(fundamental_analysis, company_name):
        # Try a simplified prompt if the first attempt fails
        simplified_prompt = f"""
        As a financial analyst, provide a brief factual analysis of {company_name} ({stock_symbol}).
        
        Use only these quarterly metrics: {quarterly_data}
        Annual performance: {annual_data}
        Key ratios: {ratio_data}
        
        Write a concise fundamental analysis in 2-3 paragraphs. Use specific numbers from the data.
        Focus only on {company_name} and avoid speculating beyond the data.
        """
        
        fundamental_analysis = run_ollama_analysis(simplified_prompt, model_name)
    
    return fundamental_analysis

def run_technical_analysis(key_data, company_name, stock_symbol, current_price, model_name):
    """Run technical analysis using the extracted key data points"""
    
    # Extract technical data
    indicators_data = format_technical_indicators(key_data)
    moving_averages_data = format_moving_averages(key_data)
    support_resistance_data = format_support_resistance(key_data)
    performance_data = format_performance_data(key_data, company_name)
    
    # Create a focused prompt for technical analysis
    technical_prompt = f"""
    You are a technical analyst with expertise in equity markets. Analyze the technical indicators and price patterns for {company_name} ({stock_symbol}) currently trading at {current_price}.
    
    FOCUS ONLY on these data points:
    
    TECHNICAL INDICATORS:
    {indicators_data}
    
    MOVING AVERAGES:
    {moving_averages_data}
    
    SUPPORT/RESISTANCE LEVELS:
    {support_resistance_data}
    
    PRICE PERFORMANCE:
    {performance_data}
    
    Provide a concise, professional technical analysis that covers:
    1. Current trend direction and strength based on indicators and price action
    2. Key support and resistance levels and their significance
    3. Momentum signals (bullish/bearish) from key indicators
    4. Potential price targets based on technical factors
    5. Overall technical outlook (bullish, bearish, or neutral)
    
    Format your analysis in 3-4 clear paragraphs. Focus only on technical factors in the data provided.
    Be specific about price levels when discussing support/resistance and targets.
    """
    
    # Run the technical analysis
    technical_analysis = run_ollama_analysis(technical_prompt, model_name)
    
    # Validate output
    if not validate_analysis(technical_analysis, company_name):
        # Try a simplified prompt if the first attempt fails
        simplified_prompt = f"""
        As a technical analyst, provide a brief technical assessment of {company_name} ({stock_symbol}) at price {current_price}.
        
        Key indicators: {indicators_data}
        Moving averages: {moving_averages_data}
        Support/Resistance: {support_resistance_data}
        
        Write a concise 2-paragraph technical analysis focusing only on current trend and key levels.
        Be specific about price targets and avoid speculation beyond the data provided.
        """
        
        technical_analysis = run_ollama_analysis(simplified_prompt, model_name)
    
    return technical_analysis

def generate_recommendation(key_data, company_name, stock_symbol, current_price, 
                           fundamental_analysis, technical_analysis, price_history_analysis, model_name):
    """Generate a final investment recommendation with improved formatting"""
    
    # Create a focused prompt for the final recommendation with formatting instructions
    recommendation_prompt = f"""
    You are the head of research at a top investment firm. Create a clear, actionable investment recommendation for {company_name} ({stock_symbol}) at current price {current_price}.
    
    BASED ON THIS ANALYSIS:
    
    FUNDAMENTAL ANALYSIS:
    {extract_key_points(fundamental_analysis)}
    
    TECHNICAL ANALYSIS:
    {extract_key_points(technical_analysis)}
    
    HISTORICAL PERFORMANCE:
    {extract_key_points(price_history_analysis)}
    
    Provide a professional investment recommendation that includes:
    
    1. A clear investment rating using EXACTLY one of these terms:
       - STRONG BUY
       - BUY
       - HOLD
       - SELL
       - STRONG SELL
    
    2. Specific price targets:
       - Bearish target (worst case scenario): [specific price]
       - Base target (most likely scenario): [specific price]
       - Bullish target (best case scenario): [specific price]
    
    3. THREE key factors supporting your recommendation
    
    4. TWO main risk factors investors should consider
    
    5. Expected timeframe for price targets (3 months, 6 months, 1 year, etc.)
    
    IMPORTANT FORMATTING INSTRUCTIONS:
    - Start with a prominent heading: ## INVESTMENT RECOMMENDATION: [YOUR RATING]
    - Format the price targets as a table with three columns (Scenario, Target Price, Timeframe)
    - Make supporting factors and risk factors stand out using bold text and bullet points
    - Use the <div style="background-color:#e8f4f8; padding:10px; border-radius:5px; margin:15px 0;"> tag to highlight important sections
    - Present key metrics in bold where appropriate
    
    Format your recommendation in a clear, structured manner with specific price targets and timeframes.
    """
    
    # Run the recommendation analysis
    recommendation = run_ollama_analysis(recommendation_prompt, model_name)
    
    # Validate output
    if not validate_recommendation(recommendation):
        # Try a simplified prompt if the first attempt fails
        simplified_prompt = f"""
        As an investment analyst, provide a clear BUY/HOLD/SELL recommendation for {company_name} ({stock_symbol}) at {current_price}.
        
        Based on the fundamental and technical analysis, state:
        1. Your EXACT recommendation (STRONG BUY, BUY, HOLD, SELL, or STRONG SELL)
        2. THREE specific price targets (bearish, base, bullish)
        3. TWO key supporting factors
        4. ONE major risk
        5. Timeframe (3 months, 6 months, or 1 year)
        
        Start with "INVESTMENT RECOMMENDATION:" followed by your rating.
        Keep your response concise and focused on actionable advice.
        """
        
        recommendation = run_ollama_analysis(simplified_prompt, model_name)
    
    return recommendation

def format_quarterly_data(key_data):
    """Format quarterly data into a clear, concise string"""
    result = []
    
    if 'financial_summary' in key_data and 'quarterly' in key_data['financial_summary']:
        quarterly = key_data['financial_summary']['quarterly']
        
        # Revenue
        if 'revenue' in quarterly:
            headers = quarterly['revenue'].get('headers', [])
            values = quarterly['revenue'].get('recent_values', [])
            if headers and values:
                result.append("Revenue (Quarterly):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Net Profit
        if 'net_profit' in quarterly:
            headers = quarterly['net_profit'].get('headers', [])
            values = quarterly['net_profit'].get('recent_values', [])
            if headers and values:
                result.append("\nNet Profit (Quarterly):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Operating Margin
        if 'operating_margin' in quarterly:
            headers = quarterly['operating_margin'].get('headers', [])
            values = quarterly['operating_margin'].get('recent_values', [])
            if headers and values:
                result.append("\nOperating Margin (Quarterly):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
    
    return "\n".join(result) if result else "Quarterly data not available"

def format_annual_data(key_data):
    """Format annual data into a clear, concise string"""
    result = []
    
    if 'financial_summary' in key_data and 'annual' in key_data['financial_summary']:
        annual = key_data['financial_summary']['annual']
        
        # Revenue
        if 'revenue' in annual:
            result.append(f"Revenue Growth: 3-Year CAGR: {annual['revenue'].get('cagr_3yr', 'N/A')}, 5-Year CAGR: {annual['revenue'].get('cagr_5yr', 'N/A')}")
            headers = annual['revenue'].get('headers', [])
            values = annual['revenue'].get('recent_values', [])
            if headers and values:
                result.append("Annual Revenue:")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Net Profit
        if 'net_profit' in annual:
            result.append(f"\nNet Profit Growth: 3-Year CAGR: {annual['net_profit'].get('cagr_3yr', 'N/A')}, 5-Year CAGR: {annual['net_profit'].get('cagr_5yr', 'N/A')}")
            headers = annual['net_profit'].get('headers', [])
            values = annual['net_profit'].get('recent_values', [])
            if headers and values:
                result.append("Annual Net Profit:")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Net Margin
        if 'net_margin' in annual:
            headers = annual['net_margin'].get('headers', [])
            values = annual['net_margin'].get('recent_values', [])
            if headers and values:
                result.append("\nNet Profit Margin (Annual):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
    
    return "\n".join(result) if result else "Annual data not available"

def format_ratio_data(key_data):
    """Format ratio data into a clear, concise string"""
    result = []
    
    if 'financial_summary' in key_data and 'ratios' in key_data['financial_summary']:
        ratios = key_data['financial_summary']['ratios']
        
        # ROE
        if 'roe' in ratios:
            headers = ratios['roe'].get('headers', [])
            values = ratios['roe'].get('recent_values', [])
            if headers and values:
                result.append("Return on Equity (ROE):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # ROA
        if 'roa' in ratios:
            headers = ratios['roa'].get('headers', [])
            values = ratios['roa'].get('recent_values', [])
            if headers and values:
                result.append("\nReturn on Assets (ROA):")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Debt/Equity
        if 'debt_equity' in ratios:
            headers = ratios['debt_equity'].get('headers', [])
            values = ratios['debt_equity'].get('recent_values', [])
            if headers and values:
                result.append("\nDebt to Equity Ratio:")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
        
        # Price to Book
        if 'price_to_book' in ratios:
            headers = ratios['price_to_book'].get('headers', [])
            values = ratios['price_to_book'].get('recent_values', [])
            if headers and values:
                result.append("\nPrice to Book Value:")
                for i in range(min(len(headers), len(values))):
                    result.append(f"  {headers[i]}: {values[i]}")
    
    return "\n".join(result) if result else "Ratio data not available"

def format_technical_indicators(key_data):
    """Format technical indicators into a clear, concise string"""
    result = []
    
    if 'technical_summary' in key_data and 'indicators' in key_data['technical_summary']:
        indicators = key_data['technical_summary']['indicators']
        
        for key, value in indicators.items():
            result.append(f"{key}: {value}")
    
    return "\n".join(result) if result else "Technical indicators not available"

def format_moving_averages(key_data):
    """Format moving averages into a clear, concise string"""
    result = []
    
    if 'technical_summary' in key_data and 'moving_averages' in key_data['technical_summary']:
        moving_averages = key_data['technical_summary']['moving_averages']
        
        for key, value in moving_averages.items():
            result.append(f"{key}: {value}")
    
    return "\n".join(result) if result else "Moving averages not available"

def format_support_resistance(key_data):
    """Format support and resistance levels into a clear, concise string"""
    result = []
    
    if 'technical_summary' in key_data and 'support_resistance' in key_data['technical_summary']:
        sr_data = key_data['technical_summary']['support_resistance']
        
        if 'support' in sr_data and sr_data['support']:
            result.append("Support Levels:")
            for level in sr_data['support']:
                result.append(f"  {level}")
        
        if 'resistance' in sr_data and sr_data['resistance']:
            result.append("\nResistance Levels:")
            for level in sr_data['resistance']:
                result.append(f"  {level}")
    
    return "\n".join(result) if result else "Support/resistance data not available"

def format_performance_data(key_data, company_name):
    """Format performance data into a clear, concise string"""
    result = []
    
    if 'technical_summary' in key_data and 'performance' in key_data['technical_summary']:
        performance = key_data['technical_summary']['performance']
        
        for period, data in performance.items():
            if isinstance(data, dict) and 'percentage' in data:
                result.append(f"{period}: {data['percentage']}")
            else:
                result.append(f"{period}: {data}")
    
    # Add relative performance if available
    if 'performance_summary' in key_data and 'returns_comparison' in key_data['performance_summary']:
        result.append("\nRelative Performance:")
        for item in key_data['performance_summary']['returns_comparison']:
            period = item.get('period', '')
            company_return = item.get(next((k for k in item.keys() if 'ICICI' in k or company_name in k), ''), '')
            index_return = item.get(next((k for k in item.keys() if 'Nifty' in k or 'Sensex' in k or 'Index' in k), ''), '')
            
            if period and company_return:
                comparison_text = f"  {period}: {company_return}"
                if index_return:
                    comparison_text += f" vs Index: {index_return}"
                result.append(comparison_text)
    
    return "\n".join(result) if result else "Performance data not available"

def extract_key_points(analysis_text, max_points=10):
    """Extract the most important points from a longer analysis text"""
    # Use the parameter name consistently
    lines = [line.strip() for line in analysis_text.split('\n') if line.strip()]
    
    # Filter out lines that are likely titles or headers
    content_lines = [line for line in lines if len(line) > 15 and not line.isupper() and not line.endswith(':')]
    
    # Take the most important lines (first few lines of each paragraph)
    key_points = []
    current_paragraph = []
    
    for line in content_lines:
        if not line:
            if current_paragraph:
                key_points.append(current_paragraph[0])  # Take first line of paragraph
                current_paragraph = []
        else:
            current_paragraph.append(line)
    
    # Add the last paragraph if exists
    if current_paragraph:
        key_points.append(current_paragraph[0])
    
    # Limit to max points
    key_points = key_points[:max_points]
    
    return "\n".join(key_points)

def validate_analysis(text, company_name):
    """Validate if the analysis meets quality standards"""
    if not text or len(text) < 100:
        return False
    
    # Check if the company name is mentioned
    if company_name not in text:
        return False
    
    # Check for nonsensical content
    nonsense_indicators = [
        "MPT Industries", "XYZ", "ABC Corp", "Example Company",
        "BlueChip Inc", "John Doe", "Lorem ipsum"
    ]
    
    if any(indicator in text for indicator in nonsense_indicators):
        return False
    
    # Check if it has some financial terms
    financial_terms = [
        "revenue", "profit", "margin", "growth", "ratio", 
        "performance", "trend", "increase", "decrease"
    ]
    
    if not any(term in text.lower() for term in financial_terms):
        return False
    
    return True

def validate_recommendation(text):
    """Validate if the recommendation meets quality standards"""
    if not text or len(text) < 100:
        return False
    
    # Check for recommendation keywords
    recommendation_terms = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    
    if not any(term in text for term in recommendation_terms):
        return False
    
    # Check for price targets
    if not any(term in text.lower() for term in ["target", "price target", "bearish", "bullish"]):
        return False
    
    # Check for timeframe
    timeframe_terms = ["month", "year", "quarter", "short term", "long term", "mid term"]
    
    if not any(term in text.lower() for term in timeframe_terms):
        return False
    
    return True

def analyze_price_history(key_data, company_name, stock_symbol, model_name):
    """Analyze price history and comparative performance data"""
    
    # Format the returns comparison and seasonality data
    returns_comparison = format_returns_comparison(key_data)
    seasonality_data = format_seasonality_data(key_data)
    
    # Create a focused prompt for price history analysis
    price_history_prompt = f"""
    You are a market strategist specializing in comparative analysis. Analyze the price history and relative performance of {company_name} ({stock_symbol}).
    
    FOCUS ONLY on these data points:
    
    PERFORMANCE VS BENCHMARKS:
    {returns_comparison}
    
    SEASONALITY PATTERNS:
    {seasonality_data}
    
    Provide a concise, professional analysis that covers:
    1. How the stock has performed compared to relevant indices
    2. Any recurring seasonal patterns or cyclical behavior
    3. Long-term performance trends and key inflection points
    4. Relative strength versus the market in different conditions
    
    Format your analysis in 2-3 clear paragraphs. Focus only on observable patterns in the data provided.
    Use specific percentages and time periods to support your observations.
    """
    
    # Run the price history analysis
    price_history_analysis = run_ollama_analysis(price_history_prompt, model_name)
    
    # Validate output - FIXED: using price_history_analysis instead of analysis_text
    if not validate_analysis(price_history_analysis, company_name):
        # Try a simplified prompt if the first attempt fails
        simplified_prompt = f"""
        As a market analyst, provide a brief analysis of {company_name}'s ({stock_symbol}) historical performance.
        
        Performance vs benchmarks: {returns_comparison}
        Seasonality: {seasonality_data}
        
        Write 1-2 paragraphs focusing only on how the stock has performed relative to the market and any
        clear seasonal patterns. Use specific numbers from the data provided.
        """
        
        price_history_analysis = run_ollama_analysis(simplified_prompt, model_name)
    
    return price_history_analysis

def format_returns_comparison(key_data):
    """Format returns comparison data into a clear, concise string"""
    result = []
    
    if 'performance_summary' in key_data and 'returns_comparison' in key_data['performance_summary']:
        comparison_data = key_data['performance_summary']['returns_comparison']
        
        for item in comparison_data:
            period = item.get('period', '')
            company_keys = [k for k in item.keys() if k not in ['period', 'entity']]
            
            if period:
                period_line = f"{period}:"
                for key in company_keys:
                    period_line += f" {key}: {item.get(key, 'N/A')},"
                result.append(period_line.rstrip(','))
    
    return "\n".join(result) if result else "Returns comparison data not available"

def format_seasonality_data(key_data):
    """Format seasonality data into a clear, concise string"""
    result = []
    
    if 'performance_summary' in key_data and 'seasonality' in key_data['performance_summary']:
        seasonality = key_data['performance_summary']['seasonality']
        
        for year_item in seasonality:
            year_line = f"{year_item.get('year', '')}: Annual Return: {year_item.get('annual_return', 'N/A')}"
            
            if 'best_month' in year_item:
                year_line += f", Best Month: {year_item['best_month']}"
                
            if 'worst_month' in year_item:
                year_line += f", Worst Month: {year_item['worst_month']}"
                
            result.append(year_line)
    
    return "\n".join(result) if result else "Seasonality data not available"

def run_ollama_analysis(prompt, model_name):
    """Run a single phase of analysis using Ollama with better error handling"""
    try:
        # Create a more focused system prompt
        system_prompt = "You are a professional financial analyst producing clear, factual analysis based only on provided data. Avoid speculation and focus on specific metrics."
        
        # Combine system prompt and user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Run the ollama command with the prompt as input
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send the prompt to the model
        stdout, stderr = process.communicate(input=full_prompt)
        
        if process.returncode != 0:
            st.error(f"Error running LLM: {stderr}")
            return f"Analysis failed: Process returned code {process.returncode}"
        
        # Clean up the response by removing any potential preamble
        lines = stdout.split('\n')
        cleaned_lines = []
        started_content = False
        
        for line in lines:
            # Skip lines until we see actual content
            if not started_content and (not line.strip() or line.startswith('I') or line.startswith('As')):
                continue
            started_content = True
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    except Exception as e:
        st.error(f"Error running LLM analysis: {str(e)}")
        return f"Analysis failed: {str(e)}"

def create_visualizations(data):
    """Create visualizations from the financial and technical data."""
    visualizations = {}
    
    # Financial data visualizations
    if "financial_data" in data:
        # Try to extract quarterly revenue and profit
        if "quarterly" in data["financial_data"] and data["financial_data"]["quarterly"]:
            for table_name, table_data in data["financial_data"]["quarterly"].items():
                if "Net Profit" in " ".join([row[0] for row in table_data["rows"] if len(row) > 0]):
                    # Extract dates and profit values
                    try:
                        dates = [col for col in table_data["headers"] if "'" in col]
                        net_profit_row = next(row for row in table_data["rows"] if "Net Profit" in row[0])
                        profit_values = [float(val.replace(",", "")) if val else 0 for val in net_profit_row[2:len(dates)+2]]
                        
                        # Create dataframe
                        profit_df = pd.DataFrame({
                            'Date': dates,
                            'Net Profit': profit_values
                        })
                        
                        visualizations['quarterly_profit'] = profit_df
                    except (StopIteration, ValueError, IndexError) as e:
                        st.warning(f"Could not create quarterly profit visualization: {e}")
    
    # Price history visualizations
    if "price_history_data" in data and "daily_data" in data["price_history_data"]:
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data["price_history_data"]["daily_data"])
            
            # Find date and close columns
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            close_col = next((col for col in df.columns if 'close' in col.lower()), None)
            
            if date_col and close_col:
                # Convert data types
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[close_col] = pd.to_numeric(df[close_col].str.replace(',', ''), errors='coerce')
                
                # Sort by date
                df = df.sort_values(by=date_col)
                
                visualizations['price_history'] = df[[date_col, close_col]]
        except Exception as e:
            st.warning(f"Could not create price history visualization: {e}")
    
    # Returns comparison visualization
    if "price_history_data" in data and "returns_comparison" in data["price_history_data"]:
        try:
            returns_data = data["price_history_data"]["returns_comparison"]
            
            # Find entries with ICICI Bank and benchmark returns
            comparison_rows = [row for row in returns_data if "ICICI Bank" in str(row) and "Returns" in str(row)]
            
            if comparison_rows:
                # Create dataframe for visualization
                returns_df = pd.DataFrame(comparison_rows)
                visualizations['returns_comparison'] = returns_df
        except Exception as e:
            st.warning(f"Could not create returns comparison visualization: {e}")
    
    return visualizations

def create_price_projection_chart(data, final_assessment, company_name, stock_symbol):
    """Create a price projection chart based on the model's price targets"""
    
    # Extract current price
    current_price = float(data.get('technical_data', {}).get('current_price', 0))
    current_date = datetime.now()
    
    # Extract price targets and timeframes from the recommendation
    targets = extract_price_targets(final_assessment)
    
    if not targets or not current_price:
        return None
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical prices if available
    if "price_history_data" in data and "daily_data" in data["price_history_data"]:
        try:
            # Convert historical data to DataFrame
            df = pd.DataFrame(data["price_history_data"]["daily_data"])
            
            # Find date and close columns
            date_col = next((col for col in df.columns if 'date' in col.lower()), None)
            close_col = next((col for col in df.columns if 'close' in col.lower()), None)
            
            if date_col and close_col:
                # Process data
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[close_col] = pd.to_numeric(df[close_col].str.replace(',', ''), errors='coerce')
                df = df.sort_values(by=date_col)
                
                # Plot historical prices
                ax.plot(df[date_col], df[close_col], label='Historical Price', color='blue', linewidth=2)
                
                # Use the last historical date as the starting point
                if not df.empty:
                    current_date = df[date_col].iloc[-1]
        except Exception as e:
            print(f"Error plotting historical data: {e}")
    
    # Plot the current price point
    ax.scatter([current_date], [current_price], color='black', s=100, zorder=5, label='Current Price')
    
    # Plot price projections
    colors = {'bearish': 'red', 'base': 'green', 'bullish': 'purple'}
    
    for scenario, target_info in targets.items():
        if 'price' in target_info and 'months' in target_info:
            target_price = target_info['price']
            target_date = current_date + pd.DateOffset(months=target_info['months'])
            
            # Plot the target point
            ax.scatter([target_date], [target_price], color=colors[scenario], s=100, zorder=5)
            
            # Draw a line from current price to target
            ax.plot([current_date, target_date], [current_price, target_price], 
                    color=colors[scenario], linestyle='--', alpha=0.7, linewidth=2,
                    label=f"{scenario.capitalize()} Target: {target_price}")
            
            # Add an annotation for the target price
            ax.annotate(f"{target_price}", (target_date, target_price), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add labels and title
    ax.set_title(f"{company_name} ({stock_symbol}) - Price Projection", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Format the date axis
    plt.gcf().autofmt_xdate()
    
    return fig

def extract_price_targets(recommendation_text):
    """Extract price targets and timeframes from the recommendation text"""
    targets = {}
    
    # Look for bearish target
    bearish_match = re.search(r"bearish\s+target.*?[rR]s\.?\s*(\d+\.?\d*)|bearish\s+target.*?(\d+\.?\d*)", recommendation_text, re.IGNORECASE)
    if bearish_match:
        bearish_price = float(bearish_match.group(1) or bearish_match.group(2))
        targets['bearish'] = {'price': bearish_price}
    
    # Look for base target
    base_match = re.search(r"base\s+target.*?[rR]s\.?\s*(\d+\.?\d*)|base\s+target.*?(\d+\.?\d*)", recommendation_text, re.IGNORECASE)
    if base_match:
        base_price = float(base_match.group(1) or base_match.group(2))
        targets['base'] = {'price': base_price}
    
    # Look for bullish target
    bullish_match = re.search(r"bullish\s+target.*?[rR]s\.?\s*(\d+\.?\d*)|bullish\s+target.*?(\d+\.?\d*)", recommendation_text, re.IGNORECASE)
    if bullish_match:
        bullish_price = float(bullish_match.group(1) or bullish_match.group(2))
        targets['bullish'] = {'price': bullish_price}
    
    # Look for timeframes
    time_patterns = {
        'bearish': r"bearish\D+(\d+)\s*(month|year)",
        'base': r"base\D+(\d+)\s*(month|year)",
        'bullish': r"bullish\D+(\d+)\s*(month|year)"
    }
    
    for scenario, pattern in time_patterns.items():
        if scenario in targets:
            time_match = re.search(pattern, recommendation_text, re.IGNORECASE)
            if time_match:
                value = int(time_match.group(1))
                unit = time_match.group(2).lower()
                
                # Convert to months
                if unit.startswith('year'):
                    value *= 12
                
                targets[scenario]['months'] = value
            else:
                # Default timeframes if not found
                defaults = {'bearish': 6, 'base': 12, 'bullish': 24}
                targets[scenario]['months'] = defaults[scenario]
    
    return targets

def extract_recommendation_section(report_text):
    """Extract just the recommendation/executive summary from the full report"""
    
    # Look for the Executive Summary section
    match = re.search(r"## Executive Summary\s+(.+?)(?=##|\Z)", report_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return report_text  # Return full text if section not found

def main():
    st.title("📈 Stock Analysis Report Generator")
    st.write("Generate a detailed stock analysis report using local LLM")
    
    # File selection
    st.subheader("Select Data Source")
    
    # Find all JSON files in the current directory
    json_files = [f for f in os.listdir('.') if f.endswith('_all_data.json')]
    if not json_files:
        st.error("No JSON data files found in the current directory.")
        return
    
    selected_file = st.selectbox("Select a stock data file", json_files)
    
    # Model selection
    models = ["phi3:medium-128k", "llama3:8b", "mistral:7b", "gemma:7b"]
    selected_model = st.selectbox("Select LLM model", models)
    
    # Load the data
    data = load_data_from_json(selected_file)
    
    if data:
        st.success(f"Successfully loaded data for {data['metadata']['company_name']} ({data['metadata']['stock_symbol']})")
        
        # Display company info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"₹{data.get('technical_data', {}).get('current_price', 'N/A')}")
        with col2:
            if "stock_info" in data and "marketCap" in data["stock_info"]:
                market_cap = data["stock_info"]["marketCap"]
                if market_cap != "N/A":
                    market_cap_display = f"₹{market_cap/10000000:.2f} Cr" if market_cap > 10000000 else f"₹{market_cap:,}"
                    st.metric("Market Cap", market_cap_display)
        with col3:
            if "stock_info" in data and "trailingPE" in data["stock_info"]:
                st.metric("P/E Ratio", data["stock_info"]["trailingPE"])
        
        # Create tabs for different sections
        tabs = st.tabs(["Data Overview", "Generate Analysis", "Visualizations"])
        
        # Data Overview tab
        with tabs[0]:
            st.subheader("Data Overview")
            
            data_tabs = st.tabs(["Financial Data", "Technical Indicators", "Price History"])
            
            # Financial Data tab
            with data_tabs[0]:
                if "financial_data" in data:
                    fin_tabs = st.tabs(["Quarterly", "Annual", "Balance Sheet", "Ratios"])
                    
                    # Quarterly tab
                    with fin_tabs[0]:
                        if "quarterly" in data["financial_data"] and data["financial_data"]["quarterly"]:
                            for table_name, table_data in data["financial_data"]["quarterly"].items():
                                if table_data["headers"] and table_data["rows"]:
                                    st.write(f"#### {table_name}")
                                    
                                    # Filter out Graph column and create DataFrame
                                    headers = [h for h in table_data["headers"] if "Graph" not in h]
                                    filtered_rows = []
                                    
                                    for row in table_data["rows"]:
                                        filtered_row = []
                                        for i, h in enumerate(table_data["headers"]):
                                            if "Graph" not in h and i < len(row):
                                                filtered_row.append(row[i])
                                        if filtered_row:
                                            filtered_rows.append(filtered_row)
                                    
                                    if headers and filtered_rows:
                                        df = pd.DataFrame(filtered_rows, columns=headers)
                                        st.dataframe(df)
                    
                    # Annual tab
                    with fin_tabs[1]:
                        if "annual" in data["financial_data"] and data["financial_data"]["annual"]:
                            for table_name, table_data in data["financial_data"]["annual"].items():
                                if table_data["headers"] and table_data["rows"]:
                                    st.write(f"#### {table_name}")
                                    
                                    # Filter out Graph column and create DataFrame
                                    headers = [h for h in table_data["headers"] if "Graph" not in h]
                                    filtered_rows = []
                                    
                                    for row in table_data["rows"]:
                                        filtered_row = []
                                        for i, h in enumerate(table_data["headers"]):
                                            if "Graph" not in h and i < len(row):
                                                filtered_row.append(row[i])
                                        if filtered_row:
                                            filtered_rows.append(filtered_row)
                                    
                                    if headers and filtered_rows:
                                        df = pd.DataFrame(filtered_rows, columns=headers)
                                        st.dataframe(df)
                    
                    # Balance Sheet tab
                    with fin_tabs[2]:
                        if "balance_sheet" in data["financial_data"] and data["financial_data"]["balance_sheet"]:
                            for table_name, table_data in data["financial_data"]["balance_sheet"].items():
                                if table_data["headers"] and table_data["rows"]:
                                    st.write(f"#### {table_name}")
                                    
                                    # Filter out Graph column and create DataFrame
                                    headers = [h for h in table_data["headers"] if "Graph" not in h]
                                    filtered_rows = []
                                    
                                    for row in table_data["rows"]:
                                        filtered_row = []
                                        for i, h in enumerate(table_data["headers"]):
                                            if "Graph" not in h and i < len(row):
                                                filtered_row.append(row[i])
                                        if filtered_row:
                                            filtered_rows.append(filtered_row)
                                    
                                    if headers and filtered_rows:
                                        df = pd.DataFrame(filtered_rows, columns=headers)
                                        st.dataframe(df)
                    
                    # Ratios tab
                    with fin_tabs[3]:
                        if "ratios" in data["financial_data"] and data["financial_data"]["ratios"]:
                            for table_name, table_data in data["financial_data"]["ratios"].items():
                                if table_data["headers"] and table_data["rows"]:
                                    st.write(f"#### {table_name}")
                                    
                                    # Filter out Graph column and create DataFrame
                                    headers = [h for h in table_data["headers"] if "Graph" not in h]
                                    filtered_rows = []
                                    
                                    for row in table_data["rows"]:
                                        filtered_row = []
                                        for i, h in enumerate(table_data["headers"]):
                                            if "Graph" not in h and i < len(row):
                                                filtered_row.append(row[i])
                                        if filtered_row:
                                            filtered_rows.append(filtered_row)
                                    
                                    if headers and filtered_rows:
                                        df = pd.DataFrame(filtered_rows, columns=headers)
                                        st.dataframe(df)
            
            # Technical Indicators tab
            with data_tabs[1]:
                if "technical_data" in data:
                    tech_data = data["technical_data"]
                    
                    # Display key indicators
                    st.subheader("Key Technical Indicators")
                    
                    if "indicators" in tech_data:
                        indicators_df = pd.DataFrame(tech_data["indicators"].items(), columns=["Indicator", "Value"])
                        st.dataframe(indicators_df)
                    
                    # Display moving averages
                    if "moving_averages" in tech_data:
                        st.subheader("Moving Averages")
                        ma_df = pd.DataFrame(tech_data["moving_averages"].items(), columns=["MA Type", "Value"])
                        st.dataframe(ma_df)
                    
                    # Display pivot points
                    if "pivot_points" in tech_data:
                        st.subheader("Pivot Points")
                        pivot_df = pd.DataFrame(tech_data["pivot_points"].items(), columns=["Level", "Value"])
                        st.dataframe(pivot_df)
            
            # Price History tab
            with data_tabs[2]:
                if "price_history_data" in data:
                    price_tabs = st.tabs(["Daily Data", "Returns Comparison", "Seasonality"])
                    
                    # Daily Data tab
                    with price_tabs[0]:
                        if "daily_data" in data["price_history_data"]:
                            st.subheader("Daily Price Data")
                            daily_df = pd.DataFrame(data["price_history_data"]["daily_data"])
                            st.dataframe(daily_df)
                    
                    # Returns Comparison tab
                    with price_tabs[1]:
                        if "returns_comparison" in data["price_history_data"]:
                            st.subheader("Returns Comparison")
                            returns_df = pd.DataFrame(data["price_history_data"]["returns_comparison"])
                            st.dataframe(returns_df)
                    
                    # Seasonality tab
                    with price_tabs[2]:
                        if "returns_seasonality" in data["price_history_data"]:
                            st.subheader("Seasonality Analysis")
                            seasonality_df = pd.DataFrame(data["price_history_data"]["returns_seasonality"])
                            st.dataframe(seasonality_df)
        
        # Generate Analysis tab
        with tabs[1]:
            st.subheader("Generate Analysis Report")
            
            if st.button(f"Generate Analysis using {selected_model}"):
                with st.spinner("Analyzing data... This may take several minutes depending on your hardware."):
                    analysis_result = run_phased_analysis(data, selected_model)

                    # Extract the recommendation section
                    final_assessment = extract_recommendation_section(analysis_result)

                    # Create and display the price projection chart
                    st.subheader("Price Projection Chart")
                    projection_chart = create_price_projection_chart(data, final_assessment, 
                                                                    data['metadata']['company_name'], 
                                                                    data['metadata']['stock_symbol'])
                    if projection_chart:
                        st.pyplot(projection_chart)
                    else:
                        st.warning("Couldn't generate price projection chart due to missing target information")
                    
                    # Display the analysis result
                    # Display the analysis result
                    st.subheader("Analysis Report")
                    st.markdown(analysis_result, unsafe_allow_html=True)

                    # Extract the recommendation rating
                    if "STRONG BUY" in analysis_result:
                        recommendation = "STRONG BUY"
                        color = "green"
                    elif "BUY" in analysis_result:
                        recommendation = "BUY"
                        color = "lightgreen"
                    elif "HOLD" in analysis_result:
                        recommendation = "HOLD"
                        color = "orange"
                    elif "SELL" in analysis_result:
                        recommendation = "SELL"
                        color = "red"
                    elif "STRONG SELL" in analysis_result:
                        recommendation = "STRONG SELL"
                        color = "darkred"
                    else:
                        recommendation = "No clear recommendation"
                        color = "gray"

                    # Display recommendation prominently
                    st.markdown(f"<h2 style='color:{color};'>Recommendation: {recommendation}</h2>", unsafe_allow_html=True)
                    
                    # Save the report to a file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"{data['metadata']['stock_symbol'].replace('.', '_')}_analysis_{timestamp}.txt"
                    
                    with open(report_filename, "w", encoding="utf-8") as f:
                        f.write(analysis_result)
                    
                    st.success(f"Analysis report saved to {report_filename}")
                    
                    # Provide download button
                    st.download_button(
                        label="Download Analysis Report",
                        data=analysis_result,
                        file_name=report_filename,
                        mime="text/plain"
                    )
        
        # Visualizations tab
        with tabs[2]:
            st.subheader("Data Visualizations")
            
            # Create visualizations
            visualizations = create_visualizations(data)
            
            if visualizations:
                viz_tabs = st.tabs(["Price Trends", "Financial Performance", "Returns Comparison"])
                
                # Price Trends tab
                with viz_tabs[0]:
                    if 'price_history' in visualizations:
                        st.subheader("Historical Price Trend")
                        price_df = visualizations['price_history']
                        
                        # Get column names
                        date_col = price_df.columns[0]
                        close_col = price_df.columns[1]
                        
                        # Create the chart
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(price_df[date_col], price_df[close_col])
                        ax.set_title(f"{data['metadata']['company_name']} - Price Trend")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Price (₹)")
                        ax.grid(True, alpha=0.3)
                        
                        # Add moving averages if available
                        if "technical_data" in data and "moving_averages" in data["technical_data"]:
                            for ma_type, value in data["technical_data"]["moving_averages"].items():
                                try:
                                    ma_value = float(value.replace(',', ''))
                                    ax.axhline(y=ma_value, color='r' if 'SMA50' in ma_type else 'g', 
                                               linestyle='--', alpha=0.7, 
                                               label=f"{ma_type}: ₹{ma_value:.2f}")
                                except (ValueError, AttributeError):
                                    pass
                        
                        ax.legend()
                        st.pyplot(fig)
                
                # Financial Performance tab
                with viz_tabs[1]:
                    if 'quarterly_profit' in visualizations:
                        st.subheader("Quarterly Net Profit Trend")
                        profit_df = visualizations['quarterly_profit']
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.bar(profit_df['Date'], profit_df['Net Profit'])
                        ax.set_title(f"{data['metadata']['company_name']} - Quarterly Net Profit")
                        ax.set_xlabel("Quarter")
                        ax.set_ylabel("Net Profit (₹ Cr)")
                        ax.grid(True, alpha=0.3)
                        
                        for i, v in enumerate(profit_df['Net Profit']):
                            ax.text(i, v + 0.1, f"{v:.1f}", ha='center')
                        
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                
                # Returns Comparison tab
                with viz_tabs[2]:
                    if 'returns_comparison' in visualizations:
                        st.subheader("Returns Comparison")
                        returns_df = visualizations['returns_comparison']
                        
                        # Try to find appropriate columns
                        entity_col = next((col for col in returns_df.columns if col.lower() in ['entity', 'time', 'period']), None)
                        company_col = next((col for col in returns_df.columns if 'icici' in col.lower()), None)
                        nifty_col = next((col for col in returns_df.columns if 'nifty' in col.lower()), None)
                        
                        if entity_col and (company_col or nifty_col):
                            # Prepare data for plotting
                            plot_data = {}
                            plot_data['Period'] = returns_df[entity_col]
                            
                            if company_col:
                                plot_data['Company'] = pd.to_numeric(returns_df[company_col].str.rstrip('%'), errors='coerce')
                            
                            if nifty_col:
                                plot_data['Nifty'] = pd.to_numeric(returns_df[nifty_col].str.rstrip('%'), errors='coerce')
                            
                            plot_df = pd.DataFrame(plot_data)
                            plot_df = plot_df.dropna()
                            
                            # Create the chart
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Bar positions
                            x = range(len(plot_df['Period']))
                            width = 0.35
                            
                            # Plot bars
                            if 'Company' in plot_df.columns:
                                ax.bar([pos - width/2 for pos in x], plot_df['Company'], width, label='ICICI Bank')
                            
                            if 'Nifty' in plot_df.columns:
                                ax.bar([pos + width/2 for pos in x], plot_df['Nifty'], width, label='Nifty')
                            
                            # Add labels and title
                            ax.set_xlabel('Period')
                            ax.set_ylabel('Returns (%)')
                            ax.set_title('Returns Comparison')
                            ax.set_xticks(x)
                            ax.set_xticklabels(plot_df['Period'], rotation=45)
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            st.pyplot(fig)
            else:
                st.info("No visualizations could be created from the available data.")

if __name__ == "__main__":
    main()