# Import necessary modules
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from langchain.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import re
import streamlit as st
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# Set page configuration
st.set_page_config(page_title="AI Stock Researcher and Analyst", page_icon="📊", layout="wide")

def get_output_format_reference(format_type):
    """
    Centralized reference for output formats to reduce prompt verbosity
    
    Args:
        format_type: The type of format reference to retrieve
        
    Returns:
        String with format specifications
    """
    format_references = {
        "fundamental_analysis": """
            {
                "revenue_trend": "Analysis of revenue growth trend",
                "profit_trend": "Analysis of profit margin trend",
                "key_ratios": [
                    {
                        "name": "ROE",
                        "value": "18.5%",
                        "interpretation": "Return on Equity interpretation"
                    }
                ],
                "financial_health": "Assessment of overall financial health",
                "full_analysis": "Complete fundamental analysis"
            }
        """,
        
        "technical_analysis": """
            {
                "trend_direction": "bullish|bearish|neutral",
                "key_levels": {
                    "support": [142.50, 138.75],
                    "resistance": [150.25, 157.80]
                },
                "momentum_indicators": [
                    {
                        "name": "RSI",
                        "value": "67.8",
                        "interpretation": "RSI interpretation"
                    }
                ],
                "volume_analysis": "Volume analysis text",
                "full_analysis": "Complete technical analysis"
            }
        """,
        
        "cash_flow_analysis": """
            {
                "operating_cf_trend": "Analysis of operating cash flow trend",
                "free_cash_flow": "Analysis of free cash flow",
                "cash_flow_metrics": {
                    "Operating CF to Net Income": "1.76",
                    "FCF Yield": "2.4%"
                },
                "investing_activities": "Analysis of investing activities",
                "financing_activities": "Analysis of financing activities",
                "full_analysis": "Complete cash flow analysis"
            }
        """,
        
        "recommendation": """
            {
                "rating": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL",
                "rationale": "Brief rationale for recommendation",
                "price_targets": [
                    {
                        "scenario": "Bearish",
                        "price": 142.50,
                        "timeframe": "6 months"
                    },
                    {
                        "scenario": "Base",
                        "price": 165.75,
                        "timeframe": "12 months"
                    },
                    {
                        "scenario": "Bullish",
                        "price": 185.25,
                        "timeframe": "12 months"
                    }
                ],
                "supporting_factors": [
                    "Key factor supporting the recommendation",
                    "Another supporting factor",
                    "Third supporting factor"
                ],
                "risk_factors": [
                    "Key risk factor to consider",
                    "Another risk factor"
                ]
            }
        """
    }
    
    return format_references.get(format_type, "Format reference not found.")

def get_context_window_limit(model_name):
    """
    Get the approximate token limit for different model context windows
    
    Args:
        model_name: Name of the LLM model
        
    Returns:
        Approximate context window size in tokens
    """
    # Define context window sizes for different models
    context_windows = {
        "phi3:medium-128k": 128000,
        "phi3:mini": 16000,
        "mistral:7b-instruct-q6_K": 32000,
        "qwen2.5:7b": 32000,
        "mixtral-offloaded:latest": 32000,
        "qwen2.5-32b-Stock-Research:latest": 32000,
        "hf.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF:Q6_K": 16000
    }
    
    # Default to a conservative limit if model not found
    return context_windows.get(model_name.lower(), 16000)

def estimate_token_count(text):
    """
    Estimate the number of tokens in a text.
    This is a rough approximation: ~4 characters per token for English text.
    
    Args:
        text: Text to estimate token count for
        
    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 characters per token for English text
    return len(text) // 4

def extract_key_metrics_to_context(key_data_points, context):
    """
    Extract key metrics from data and store in the context dictionary
    
    Args:
        key_data_points: Dictionary with extracted key data
        context: The global context dictionary
    """
    # Store current price
    if 'company_info' in key_data_points and 'current_price' in key_data_points['company_info']:
        try:
            context['key_metrics']['current_price'] = float(str(key_data_points['company_info']['current_price']).replace('₹', '').replace(',', ''))
        except (ValueError, TypeError):
            pass
    
    # Store fundamental metrics if available
    if 'financial_summary' in key_data_points and 'ratios' in key_data_points['financial_summary']:
        ratios = key_data_points['financial_summary']['ratios']
        
        # Extract ROE
        if 'roe' in ratios and 'recent_values' in ratios['roe'] and ratios['roe']['recent_values']:
            context['key_metrics']['roe'] = ratios['roe']['recent_values'][0]
        
        # Extract debt/equity
        if 'debt_equity' in ratios and 'recent_values' in ratios['debt_equity'] and ratios['debt_equity']['recent_values']:
            context['key_metrics']['debt_equity'] = ratios['debt_equity']['recent_values'][0]
        
        # Extract P/E ratio if available
        if 'pe_ratio' in ratios and 'recent_values' in ratios['pe_ratio'] and ratios['pe_ratio']['recent_values']:
            context['key_metrics']['pe_ratio'] = ratios['pe_ratio']['recent_values'][0]
    
    # Store technical metrics if available
    if 'technical_summary' in key_data_points and 'indicators' in key_data_points['technical_summary']:
        indicators = key_data_points['technical_summary']['indicators']
        
        # Extract RSI
        if 'RSI' in indicators:
            context['key_metrics']['rsi'] = indicators['RSI']

# Define structured data models for different components of the analysis
class PriceTarget(BaseModel):
    """Price target for different scenarios"""
    scenario: str = Field(description="Scenario name (Bearish, Base, Bullish)")
    price: float = Field(description="Target price value")
    timeframe: str = Field(description="Timeframe for the target (e.g., '6 months')")
    
    @field_validator('price')
    @classmethod
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return round(v, 2)  # Round to 2 decimal places for consistency

class KeyMetric(BaseModel):
    """Individual financial or technical metric"""
    name: str = Field(description="Name of the metric")
    value: str = Field(description="Value of the metric")
    interpretation: str = Field(description="Brief interpretation of the metric")

class FundamentalAnalysis(BaseModel):
    """Structured fundamental analysis output"""
    revenue_trend: str = Field(description="Analysis of revenue growth trend")
    profit_trend: str = Field(description="Analysis of profit margin trend")
    key_ratios: List[KeyMetric] = Field(description="List of key financial ratios with interpretation")
    financial_health: str = Field(description="Assessment of overall financial health")
    full_analysis: str = Field(description="Complete fundamental analysis in text form")

class TechnicalAnalysis(BaseModel):
    """Structured technical analysis output"""
    trend_direction: str = Field(description="Current price trend direction (bullish, bearish, neutral)")
    key_levels: Dict[str, List[float]] = Field(description="Support and resistance levels")
    momentum_indicators: List[KeyMetric] = Field(description="Key technical indicators with interpretation")
    volume_analysis: str = Field(description="Analysis of trading volume patterns")
    full_analysis: str = Field(description="Complete technical analysis in text form")

class StockRecommendation(BaseModel):
    """Structured stock recommendation"""
    rating: str = Field(description="Investment rating (STRONG BUY, BUY, HOLD, SELL, STRONG SELL)")
    rationale: str = Field(description="Brief rationale for the recommendation")
    price_targets: List[PriceTarget] = Field(description="Price targets for different scenarios")
    supporting_factors: List[str] = Field(description="Key factors supporting the recommendation")
    risk_factors: List[str] = Field(description="Key risk factors to consider")
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        valid_ratings = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
        if v not in valid_ratings:
            # Default to HOLD if invalid rating is provided
            return "HOLD"
        return v

class CashFlowAnalysis(BaseModel):
    """Structured cash flow analysis output"""
    operating_cf_trend: str = Field(description="Analysis of operating cash flow trend")
    free_cash_flow: str = Field(description="Analysis of free cash flow")
    cash_flow_metrics: Dict[str, str] = Field(description="Key cash flow metrics with values")
    investing_activities: str = Field(description="Analysis of investing activities")
    financing_activities: str = Field(description="Analysis of financing activities")
    full_analysis: str = Field(description="Complete cash flow analysis in text form")

class StockAnalysisReport(BaseModel):
    """Complete stock analysis report"""
    company_name: str = Field(description="Name of the company")
    stock_symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price")
    date: str = Field(description="Date of analysis")
    fundamental_analysis: FundamentalAnalysis = Field(description="Fundamental analysis component")
    technical_analysis: TechnicalAnalysis = Field(description="Technical analysis component")
    cash_flow_analysis: Optional[CashFlowAnalysis] = Field(None, description="Cash flow analysis component")
    recommendation: StockRecommendation = Field(description="Investment recommendation")
    price_history_analysis: str = Field(description="Analysis of historical price performance")
    moving_averages: Optional[Dict[str, str]] = Field(None, description="Moving average values")
    pivot_points: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Pivot point levels")
    support_resistance: Optional[Dict[str, List[str]]] = Field(None, description="Support and resistance levels")
    price_history: Optional[List[Dict[str, Any]]] = Field(None, description="Historical price data")

def create_analysis_context():
    """
    Create a global context dictionary to store and share key metrics and insights
    across different analysis steps.
    
    Returns:
        Dictionary with structure for storing analysis insights
    """
    return {
        "fundamental": {
            "revenue_insights": "",
            "profit_insights": "",
            "financial_health": "",
            "key_metrics": {},
            "summary": ""
        },
        "technical": {
            "trend_insights": "",
            "support_resistance_insights": "",
            "momentum_insights": "",
            "volume_insights": "",
            "summary": ""
        },
        "cash_flow": {
            "operating_cf_insights": "",
            "free_cash_flow_insights": "",
            "investing_insights": "",
            "financing_insights": "",
            "summary": ""
        },
        "key_metrics": {
            "current_price": 0.0,
            "pe_ratio": None,
            "market_cap": None,
            "debt_equity": None,
            "roe": None,
            "rsi": None,
            "trend_direction": "neutral"
        },
        "insights_summary": ""
    }

def generate_structured_analysis(data, model_name="phi3:medium-128k"):
    """
    Generate a complete structured stock analysis report with context maintenance,
    prompt engineering improvements and memory/hardware efficiency optimizations
    
    Args:
        data: Dictionary containing the stock data
        model_name: Name of the LLM model to use
        
    Returns:
        StockAnalysisReport object with the complete analysis
    """
    # Extract basic information
    company_name = data['metadata']['company_name']
    stock_symbol = data['metadata']['stock_symbol']
    current_price = data.get('technical_data', {}).get('current_price', '0.0')
    
    # Convert current_price to float and handle potential formatting issues
    try:
        current_price = float(str(current_price).replace('₹', '').replace(',', ''))
    except (ValueError, TypeError):
        current_price = 0.0
    
    # Get context window size for the model
    context_limit = get_context_window_limit(model_name)
    
    # Determine data inclusion parameters based on model size
    if context_limit <= 16000:  # Smaller models
        max_quarters = 3
        max_years = 2
        max_indicators = 3
    elif context_limit <= 32000:  # Medium-sized models
        max_quarters = 4
        max_years = 3
        max_indicators = 4
    else:  # Large models
        max_quarters = 5
        max_years = 5
        max_indicators = 5
        
    # Create a progress bar
    progress_bar = st.progress(0)
    st.info(f"Starting analysis with {model_name} (context window ~{context_limit} tokens)")
    
    # Initialize LangChain LLM
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model=model_name)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None
    
    # Create analysis context for sharing insights between steps
    analysis_context = create_analysis_context()
    
    # Extract and format key data points for analysis - with memory-efficient data selection
    st.info("Phase 1/5: Extracting key data points...")
    key_data_points = extract_key_data_points(data)
    # Extract key metrics to context
    extract_key_metrics_to_context(key_data_points, analysis_context)
    progress_bar.progress(15)
    
    # Process fundamental analysis - Using our context-aware function
    st.info("Phase 2/5: Generating fundamental analysis with memory optimization...")
    # Modify parameters but keep the prompt engineering improvements
    fundamental_analysis = generate_fundamental_analysis_with_limits(
        key_data_points, company_name, stock_symbol, llm, 
        max_quarters=max_quarters, max_years=max_years,
        context=analysis_context
    )
    progress_bar.progress(35)
    
    # Process technical analysis - Using our context-aware function
    st.info("Phase 3/5: Generating technical analysis with context window management...")
    technical_analysis = generate_technical_analysis_with_limits(
        key_data_points, company_name, stock_symbol, current_price, llm,
        max_indicators=max_indicators,
        context=analysis_context
    )
    progress_bar.progress(55)
    
    # Process cash flow analysis - With context-aware analysis
    st.info("Phase 4/5: Generating cash flow analysis with selective data inclusion...")
    cash_flow_analysis = generate_cash_flow_analysis_with_limits(
        key_data_points, company_name, stock_symbol, llm,
        max_years=max_years,
        context=analysis_context
    )
    progress_bar.progress(75)
    
    # Generate final recommendation with consolidated context
    st.info("Phase 5/5: Preparing final recommendation and report...")
    recommendation = generate_recommendation(
        key_data_points, company_name, stock_symbol, current_price,
        fundamental_analysis, technical_analysis, llm,
        context=analysis_context
    )
    
    # Generate price history analysis using accumulated context
    price_history_analysis = generate_price_history_analysis(
        key_data_points, company_name, stock_symbol, llm,
        context=analysis_context
    )
    
    # Create the complete report
    analysis_report = StockAnalysisReport(
        company_name=company_name,
        stock_symbol=stock_symbol,
        current_price=current_price,
        date=datetime.now().strftime("%B %d, %Y"),
        fundamental_analysis=fundamental_analysis,
        technical_analysis=technical_analysis,
        cash_flow_analysis=cash_flow_analysis,
        recommendation=recommendation,
        price_history_analysis=price_history_analysis
    )
    
    # Add additional technical data for charts if available
    if 'technical_data' in data:
        if 'moving_averages' in data['technical_data']:
            analysis_report.moving_averages = data['technical_data']['moving_averages']
        
        # Add pivot points
        if 'pivot_points' in data['technical_data']:
            # Clean up pivot point values for consistency
            pivot_points = {}
            for method, levels in data['technical_data']['pivot_points'].items():
                pivot_points[method] = {}
                for level, value in levels.items():
                    try:
                        if isinstance(value, str):
                            value = float(value.replace(',', ''))
                        pivot_points[method][level] = value
                    except (ValueError, TypeError):
                        pivot_points[method][level] = value
            
            analysis_report.pivot_points = pivot_points
        
        # Process support/resistance levels if available
        if 'support_resistance' in data['technical_data']:
            support_resistance = {
                'support': [],
                'resistance': []
            }
            
            if 'support' in data['technical_data']['support_resistance']:
                support_data = data['technical_data']['support_resistance']['support']
                for level in support_data:
                    try:
                        # Handle complex support level strings
                        cleaned_levels = extract_price_levels(level)
                        support_resistance['support'].extend(cleaned_levels)
                    except Exception:
                        # If parsing fails, just include as is
                        support_resistance['support'].append(level)
            
            # Process resistance levels if available
            if 'resistance' in data['technical_data']['support_resistance']:
                resistance_data = data['technical_data']['support_resistance']['resistance']
                for level in resistance_data:
                    try:
                        # Handle complex resistance level strings
                        cleaned_levels = extract_price_levels(level)
                        support_resistance['resistance'].extend(cleaned_levels)
                    except Exception:
                        # If parsing fails, just include as is
                        support_resistance['resistance'].append(level)
            
            analysis_report.support_resistance = support_resistance
    
    # Add price history data if available
    if 'price_history_data' in data and 'daily_data' in data['price_history_data']:
        analysis_report.price_history = data['price_history_data']['daily_data']
    
    progress_bar.progress(100)
    
    return analysis_report

def extract_json_from_text(text):
    """
    Extract valid JSON from text with improved handling for "extra data" errors
    
    Args:
        text: The text that might contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
    
    import json
    
    # Strategy 1: Find code blocks with JSON (common LLM output format)
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
        # Try to find the first valid JSON object within this text
        try:
            # Find the indices of all opening and closing braces
            open_indices = [i for i, char in enumerate(json_text) if char == '{']
            close_indices = [i for i, char in enumerate(json_text) if char == '}']
            
            # Try each potential combination of opening and closing braces
            for start in open_indices:
                for end in close_indices:
                    if end > start:  # Ensure closing brace comes after opening
                        try:
                            potential_json = json_text[start:end+1]
                            parsed = json.loads(potential_json)
                            return potential_json  # Return first valid JSON
                        except json.JSONDecodeError:
                            continue  # Try next combination
        except Exception:
            pass  # If brute force approach fails, continue to next strategy
    
    # Strategy 2: Find anything that looks like a complete JSON object
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        json_text = json_match.group(1).strip()
        
        # Handle the "Extra data" error by carefully extracting only up to the matching closing brace
        try:
            # Parse char by char to find exactly one valid JSON object
            brace_count = 0
            for i, char in enumerate(json_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                if brace_count == 0:
                    # We found a complete, balanced JSON object
                    potential_json = json_text[:i+1]
                    # Validate by parsing
                    parsed = json.loads(potential_json)
                    return potential_json
        except Exception:
            pass
    
    # Strategy 3: Look for JSON arrays (for cases where we expect an array)
    array_match = re.search(r'(\[.*\])', text, re.DOTALL)
    if array_match:
        array_text = array_match.group(1).strip()
        try:
            # Validate by parsing
            parsed = json.loads(array_text)
            return array_text
        except Exception:
            pass
            
    # No valid JSON found
    return None

def process_fundamental_json(json_str, company_name, stock_symbol):
    """
    Process fundamental analysis JSON directly
    
    Args:
        json_str: JSON string to process
        company_name: Company name for fallback
        stock_symbol: Stock symbol for fallback
        
    Returns:
        FundamentalAnalysis object
    """
    try:
        import json
        
        # Clean up the JSON string
        json_str = json_str.replace('\n', ' ').replace('\t', ' ')
        # Handle any single quotes that should be double quotes
        json_str = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', json_str)
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Extract the key components
        revenue_trend = data.get('revenue_trend', f"Analysis of {company_name}'s revenue trends could not be generated.")
        profit_trend = data.get('profit_trend', f"Analysis of {company_name}'s profit trends could not be generated.")
        
        # Process key ratios
        key_ratios = []
        if 'key_ratios' in data and isinstance(data['key_ratios'], list):
            for ratio in data['key_ratios']:
                if isinstance(ratio, dict) and 'name' in ratio and 'value' in ratio:
                    key_ratios.append(KeyMetric(
                        name=ratio['name'],
                        value=str(ratio['value']),
                        interpretation=ratio.get('interpretation', f"Interpretation of {ratio['name']} not available.")
                    ))
        
        # Add at least one key ratio if none found
        if not key_ratios:
            key_ratios.append(KeyMetric(
                name="ROE", 
                value="N/A", 
                interpretation="Return on Equity data could not be processed."
            ))
        
        # Get financial health and full analysis
        financial_health = data.get('financial_health', "Financial health assessment could not be generated.")
        full_analysis = data.get('full_analysis', f"Complete fundamental analysis for {company_name} could not be generated.")
        
        # Create and return the object
        return FundamentalAnalysis(
            revenue_trend=revenue_trend,
            profit_trend=profit_trend,
            key_ratios=key_ratios,
            financial_health=financial_health,
            full_analysis=full_analysis
        )
    
    except Exception as e:
        st.error(f"Error processing fundamental JSON: {str(e)}")
        return None


def process_technical_json(json_str, company_name, stock_symbol, current_price):
    """
    Process technical analysis JSON directly
    
    Args:
        json_str: JSON string to process
        company_name: Company name for fallback
        stock_symbol: Stock symbol for fallback
        current_price: Current stock price
        
    Returns:
        TechnicalAnalysis object
    """
    try:
        import json
        
        # Clean up the JSON string
        json_str = json_str.replace('\n', ' ').replace('\t', ' ')
        # Handle any single quotes that should be double quotes
        json_str = re.sub(r"(?<!\w)'([^']*)'(?!\w)", r'"\1"', json_str)
        
        # Parse the JSON
        data = json.loads(json_str)
        
        # Extract the key components
        trend_direction = data.get('trend_direction', "neutral")
        
        # Process key levels
        key_levels = {"support": [], "resistance": []}
        if 'key_levels' in data and isinstance(data['key_levels'], dict):
            # Handle support levels
            if 'support' in data['key_levels'] and isinstance(data['key_levels']['support'], list):
                for level in data['key_levels']['support']:
                    try:
                        key_levels["support"].append(float(level))
                    except:
                        pass
            
            # Handle resistance levels
            if 'resistance' in data['key_levels'] and isinstance(data['key_levels']['resistance'], list):
                for level in data['key_levels']['resistance']:
                    try:
                        key_levels["resistance"].append(float(level))
                    except:
                        pass
        
        # Create default levels if none found
        if not key_levels["support"]:
            key_levels["support"] = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
        
        if not key_levels["resistance"]:
            key_levels["resistance"] = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
        
        # Process momentum indicators
        momentum_indicators = []
        if 'momentum_indicators' in data and isinstance(data['momentum_indicators'], list):
            for indicator in data['momentum_indicators']:
                if isinstance(indicator, dict) and 'name' in indicator and 'value' in indicator:
                    momentum_indicators.append(KeyMetric(
                        name=indicator['name'],
                        value=str(indicator['value']),
                        interpretation=indicator.get('interpretation', f"Interpretation of {indicator['name']} not available.")
                    ))
        
        # Add at least one indicator if none found
        if not momentum_indicators:
            momentum_indicators.append(KeyMetric(
                name="RSI", 
                value="N/A", 
                interpretation="RSI data could not be processed."
            ))
        
        # Get volume analysis and full analysis
        volume_analysis = data.get('volume_analysis', "Volume analysis could not be generated.")
        full_analysis = data.get('full_analysis', f"Complete technical analysis for {company_name} could not be generated.")
        
        # Create and return the object
        return TechnicalAnalysis(
            trend_direction=trend_direction,
            key_levels=key_levels,
            momentum_indicators=momentum_indicators,
            volume_analysis=volume_analysis,
            full_analysis=full_analysis
        )
    
    except Exception as e:
        st.error(f"Error processing technical JSON: {str(e)}")
        return None

def generate_fundamental_analysis(key_data_points, company_name, stock_symbol, llm):
    """
    Generate structured fundamental analysis using LLM with improved prompt engineering
    """
    # Format financial data for the prompt
    financial_data_str = format_financial_data_for_prompt(key_data_points)
    
    # Step 1: First prompt for revenue and profit analysis (simpler component)
    revenue_profit_prompt = f"""
    You are a financial analyst specializing in equity research. Analyze {company_name}'s ({stock_symbol}) revenue and profit trends.
    
    Focus on these financial data points:
    {financial_data_str}
    
    First, analyze the revenue trends:
    1. Is revenue growing, stable, or declining?
    2. What is the growth rate compared to industry averages?
    3. Are there any seasonal patterns or one-time events affecting revenue?
    
    Then, analyze the profit trends:
    1. Are profit margins expanding, stable, or contracting?
    2. How does profitability compare to competitors?
    3. What factors are driving profitability changes?
    
    Respond with a JSON object containing:
    {{
        "revenue_trend": "Your detailed analysis of revenue trends here...",
        "profit_trend": "Your detailed analysis of profit trends here..."
    }}
    
    IMPORTANT: Respond ONLY with valid JSON. No preamble or explanation outside the JSON.
    """
    
    try:
        # Get revenue and profit analysis
        revenue_profit_response = llm.invoke(revenue_profit_prompt)
        revenue_profit_json = extract_json_from_text(revenue_profit_response)
        
        if not revenue_profit_json:
            st.warning("Could not extract valid JSON from revenue/profit analysis")
            revenue_profit_data = {
                "revenue_trend": f"Analysis of {company_name}'s revenue trends could not be generated.",
                "profit_trend": f"Analysis of {company_name}'s profit trends could not be generated."
            }
        else:
            import json
            revenue_profit_data = json.loads(revenue_profit_json)
        
        # Step 2: Next prompt for key ratios (another manageable component)
        key_ratios_prompt = f"""
        You are a financial analyst specializing in equity research. Analyze {company_name}'s ({stock_symbol}) key financial ratios.
        
        Focus on these financial data points:
        {financial_data_str}
        
        Analyze these key financial ratios:
        1. ROE (Return on Equity)
        2. ROA (Return on Assets)
        3. Debt-to-Equity
        4. Current Ratio
        5. P/E Ratio
        
        For each ratio:
        - Identify the current value
        - Assess if it's improving or deteriorating
        - Compare to industry benchmarks
        - Explain what it means for investors
        
        Respond with a JSON array of key ratio objects:
        [
            {{
                "name": "ROE",
                "value": "15.2%",
                "interpretation": "Return on Equity interpretation here..."
            }},
            {{
                "name": "Debt/Equity",
                "value": "0.75",
                "interpretation": "Debt to Equity interpretation here..."
            }}
            // Additional ratios...
        ]
        
        IMPORTANT: Respond ONLY with valid JSON. No preamble or explanation outside the JSON.
        """
        
        # Get key ratios analysis
        key_ratios_response = llm.invoke(key_ratios_prompt)
        key_ratios_json = extract_json_from_text(key_ratios_response)
        
        if not key_ratios_json:
            st.warning("Could not extract valid JSON from key ratios analysis")
            key_ratios_data = [
                {
                    "name": "ROE", 
                    "value": "N/A", 
                    "interpretation": "Return on Equity data could not be processed."
                }
            ]
        else:
            import json
            key_ratios_data = json.loads(key_ratios_json)
            
            # Ensure key_ratios_data is a list
            if not isinstance(key_ratios_data, list):
                key_ratios_data = [
                    {
                        "name": "ROE", 
                        "value": "N/A", 
                        "interpretation": "Return on Equity data could not be processed."
                    }
                ]
        
        # Step 3: Finally, prompt for financial health and full analysis
        financial_health_prompt = f"""
        You are a financial analyst specializing in equity research. Provide an overall assessment of {company_name}'s ({stock_symbol}) financial health.
        
        Focus on these financial data points:
        {financial_data_str}
        
        Consider:
        1. Balance sheet strength and liquidity
        2. Cash flow sustainability
        3. Debt levels and interest coverage
        4. Working capital management
        5. Overall financial stability
        
        Respond with a JSON object containing:
        {{
            "financial_health": "Your detailed assessment of overall financial health here...",
            "full_analysis": "Your comprehensive fundamental analysis, bringing together all aspects including revenue trends, profit margins, ratios, and financial health here..."
        }}
        
        IMPORTANT: Respond ONLY with valid JSON. No preamble or explanation outside the JSON.
        """
        
        # Get financial health analysis
        financial_health_response = llm.invoke(financial_health_prompt)
        financial_health_json = extract_json_from_text(financial_health_response)
        
        if not financial_health_json:
            st.warning("Could not extract valid JSON from financial health analysis")
            financial_health_data = {
                "financial_health": "Financial health assessment could not be generated.",
                "full_analysis": f"Complete fundamental analysis for {company_name} could not be generated."
            }
        else:
            import json
            financial_health_data = json.loads(financial_health_json)
        
        # Create the consolidated FundamentalAnalysis object from the three components
        return FundamentalAnalysis(
            revenue_trend=revenue_profit_data.get("revenue_trend", f"Analysis of {company_name}'s revenue trends could not be generated."),
            profit_trend=revenue_profit_data.get("profit_trend", f"Analysis of {company_name}'s profit trends could not be generated."),
            key_ratios=[
                KeyMetric(
                    name=ratio.get("name", "Unknown"),
                    value=ratio.get("value", "N/A"),
                    interpretation=ratio.get("interpretation", "No interpretation available")
                )
                for ratio in key_ratios_data
            ],
            financial_health=financial_health_data.get("financial_health", "Financial health assessment could not be generated."),
            full_analysis=financial_health_data.get("full_analysis", f"Complete fundamental analysis for {company_name} could not be generated.")
        )
        
    except Exception as e:
        st.warning(f"Fundamental analysis generation failed: {str(e)}")
        return fallback_fundamental_analysis(key_data_points, company_name, stock_symbol, llm)

def generate_fundamental_analysis_with_limits(key_data_points, company_name, stock_symbol, llm, max_quarters=4, max_years=3, context=None):
    """
    Generate fundamental analysis with memory optimization and context sharing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        max_quarters: Maximum number of quarters to include
        max_years: Maximum number of years to include
        context: The global context dictionary
        
    Returns:
        FundamentalAnalysis object with the analysis
    """
    # Format financial data with limits
    financial_data_str = format_financial_data_for_prompt(key_data_points, max_quarters=max_quarters, max_years=max_years)
    
    # Extract key metrics to reference in the prompt
    key_metrics_str = ""
    if context and 'key_metrics' in context:
        metrics = []
        for name, value in context['key_metrics'].items():
            if value is not None:
                metrics.append(f"{name}: {value}")
        if metrics:
            key_metrics_str = "Key metrics identified:\n" + "\n".join(metrics)
    
    # Build on our previous prompt engineering with more efficient token usage
    # and adding chain-of-thought reasoning
    fundamental_prompt = f"""
    You are a financial analyst specializing in equity research. Analyze {company_name} ({stock_symbol}).
    
    Focus on these financial data points:
    {financial_data_str}
    
    {key_metrics_str}
    
    First, I want you to think step by step about:
    1. Revenue trends and what they indicate about the company's growth trajectory
    2. Profit margin evolution and operational efficiency
    3. The company's financial health and stability based on key ratios
    4. Overall financial outlook based on these factors
    
    Once you've analyzed these aspects carefully, provide a JSON object with your complete fundamental analysis:
    {{
        "revenue_trend": "Your analysis of revenue trends here...",
        "profit_trend": "Your analysis of profit margins here...",
        "key_ratios": [
            {{
                "name": "ROE",
                "value": "15.2%",
                "interpretation": "Return on Equity interpretation here..."
            }},
            {{
                "name": "Debt/Equity",
                "value": "0.75",
                "interpretation": "Debt to Equity interpretation here..."
            }}
        ],
        "financial_health": "Your assessment of overall financial health here...",
        "full_analysis": "Your complete fundamental analysis here...",
        "summary": "A brief 2-3 sentence summary of the most important fundamental insights..."
    }}
    
    IMPORTANT: Respond with your thinking process first, then provide ONLY the JSON object.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(fundamental_prompt)
        
        # Extract JSON from the response - using our enhanced extraction function
        json_str = extract_json_from_text(response)
        
        if json_str:
            import json
            parsed_data = json.loads(json_str)
            
            # Update context with fundamental insights if context exists
            if context is not None:
                context['fundamental']['revenue_insights'] = parsed_data.get("revenue_trend", "")
                context['fundamental']['profit_insights'] = parsed_data.get("profit_trend", "")
                context['fundamental']['financial_health'] = parsed_data.get("financial_health", "")
                context['fundamental']['summary'] = parsed_data.get("summary", "")
                
                # Store key metrics interpretations
                for ratio in parsed_data.get("key_ratios", []):
                    metric_name = ratio.get("name", "").lower()
                    context['fundamental']['key_metrics'][metric_name] = {
                        "value": ratio.get("value", "N/A"),
                        "interpretation": ratio.get("interpretation", "")
                    }
                
            # Create the FundamentalAnalysis object
            return FundamentalAnalysis(
                revenue_trend=parsed_data.get("revenue_trend", f"No revenue trend analysis available for {company_name}"),
                profit_trend=parsed_data.get("profit_trend", f"No profit trend analysis available for {company_name}"),
                key_ratios=[
                    KeyMetric(
                        name=ratio.get("name", "Unknown"),
                        value=ratio.get("value", "N/A"),
                        interpretation=ratio.get("interpretation", "No interpretation available")
                    )
                    for ratio in parsed_data.get("key_ratios", [])
                ],
                financial_health=parsed_data.get("financial_health", "No financial health assessment available"),
                full_analysis=parsed_data.get("full_analysis", f"No detailed analysis available for {company_name}")
            )
        else:
            st.warning("Could not extract valid JSON from fundamental analysis response")
            return fallback_fundamental_analysis(key_data_points, company_name, stock_symbol, llm, context)
    
    except Exception as e:
        st.warning(f"Fundamental analysis generation failed: {str(e)}")
        return fallback_fundamental_analysis(key_data_points, company_name, stock_symbol, llm, context)

def generate_technical_analysis_with_limits(key_data_points, company_name, stock_symbol, current_price, llm, max_indicators=5, context=None):
    """
    Generate technical analysis with memory optimization and context sharing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        llm: LangChain LLM object
        max_indicators: Maximum number of indicators to include
        context: The global context dictionary
        
    Returns:
        TechnicalAnalysis object with the analysis
    """
    # Format technical data with limits
    technical_data_str = format_technical_data_for_prompt(key_data_points, max_indicators=max_indicators)
    
    # Include fundamental insights from context if available
    fundamental_context = ""
    if context and 'fundamental' in context and context['fundamental']['summary']:
        fundamental_context = f"""
        Fundamental Analysis Context:
        {context['fundamental']['summary']}
        
        Revenue: {context['fundamental']['revenue_insights'][:100]}...
        Financial Health: {context['fundamental']['financial_health'][:100]}...
        """
    
    # Build on our previous few-shot examples approach but with more concise examples
    # and including fundamental context
    technical_prompt = f"""
    You are a technical analyst with expertise in equity markets. Analyze {company_name} ({stock_symbol}) at {current_price}.
    
    Focus on these technical data points:
    {technical_data_str}
    
    {fundamental_context}
    
    First, think step-by-step about:
    1. The primary trend direction based on price action and moving averages
    2. Key support and resistance levels and their significance
    3. What momentum indicators reveal about buying/selling pressure
    4. Volume patterns and what they suggest about price movements
    
    After your careful analysis, provide a JSON object following this format:
    {{
        "trend_direction": "bullish",
        "key_levels": {{
            "support": [142.50, 138.75],
            "resistance": [150.25, 157.80]
        }},
        "momentum_indicators": [
            {{
                "name": "RSI",
                "value": "67.8",
                "interpretation": "RSI interpretation here..."
            }}
        ],
        "volume_analysis": "Volume analysis text here...",
        "full_analysis": "Complete technical analysis here...",
        "summary": "A brief 2-3 sentence summary of the most important technical insights..."
    }}
    
    IMPORTANT: Provide your thinking process first, then respond with ONLY the JSON object.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(technical_prompt)
        
        # Extract JSON from the response - using our enhanced extraction function
        json_str = extract_json_from_text(response)
        
        if json_str:
            import json
            parsed_data = json.loads(json_str)
            
            # Update context with technical insights if context exists
            if context is not None:
                context['technical']['trend_insights'] = f"Trend direction: {parsed_data.get('trend_direction', 'neutral')}"
                context['technical']['support_resistance_insights'] = f"Key support levels: {parsed_data.get('key_levels', {}).get('support', [])} | Key resistance levels: {parsed_data.get('key_levels', {}).get('resistance', [])}"
                context['technical']['volume_insights'] = parsed_data.get("volume_analysis", "")
                context['technical']['summary'] = parsed_data.get("summary", "")
                context['key_metrics']['trend_direction'] = parsed_data.get("trend_direction", "neutral")
                
                # Store momentum indicator insights
                momentum_insights = []
                for indicator in parsed_data.get("momentum_indicators", []):
                    if 'name' in indicator and 'interpretation' in indicator:
                        momentum_insights.append(f"{indicator['name']}: {indicator['interpretation']}")
                
                context['technical']['momentum_insights'] = " | ".join(momentum_insights)
            
            # Create the TechnicalAnalysis object
            return TechnicalAnalysis(
                trend_direction=parsed_data.get("trend_direction", "neutral"),
                key_levels=parsed_data.get("key_levels", {"support": [], "resistance": []}),
                momentum_indicators=[
                    KeyMetric(
                        name=indicator.get("name", "Unknown"),
                        value=indicator.get("value", "N/A"),
                        interpretation=indicator.get("interpretation", "No interpretation available")
                    )
                    for indicator in parsed_data.get("momentum_indicators", [])
                ],
                volume_analysis=parsed_data.get("volume_analysis", "No volume analysis available"),
                full_analysis=parsed_data.get("full_analysis", f"No detailed analysis available for {company_name}")
            )
        else:
            # If we can't find a valid JSON, fall back
            st.warning("Could not extract valid JSON from technical analysis response")
            return fallback_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm, context)
    
    except Exception as e:
        st.warning(f"Technical analysis generation failed: {str(e)}")
        return fallback_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm, context)

def generate_cash_flow_analysis_with_limits(key_data_points, company_name, stock_symbol, llm, max_years=3, context=None):
    """
    Generate cash flow analysis with memory optimization and context sharing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        max_years: Maximum number of years to include
        context: The global context dictionary
        
    Returns:
        CashFlowAnalysis object with the analysis
    """
    # Check if cash flow data exists
    if not (
        'financial_summary' in key_data_points and 
        'cash_flow' in key_data_points['financial_summary'] and 
        key_data_points['financial_summary']['cash_flow']
    ):
        # Return minimal analysis if no data available
        return CashFlowAnalysis(
            operating_cf_trend="Cash flow trend data not available.",
            free_cash_flow="Free cash flow data not available.",
            cash_flow_metrics={},
            investing_activities="Investing activities data not available.",
            financing_activities="Financing activities data not available.",
            full_analysis=f"Comprehensive cash flow analysis for {company_name} could not be generated due to missing data."
        )
    
    # Format cash flow data for the prompt with selective inclusion
    cash_flow_data = key_data_points['financial_summary']['cash_flow']
    cash_flow_str = []
    
    # Select only the most important cash flow metrics for efficiency
    key_metrics = ['operating_cf', 'investing_cf', 'financing_cf', 'net_cf']
    
    for cf_type, data in cash_flow_data.items():
        # Only include key metrics
        if not any(key in cf_type.lower() for key in key_metrics):
            continue
            
        if 'label' in data and 'recent_values' in data and 'headers' in data:
            cash_flow_str.append(f"{data['label']}:")
            
            # Limit years to max_years
            headers = data['headers'][:max_years] if len(data['headers']) > max_years else data['headers']
            values = data['recent_values'][:max_years] if len(data['recent_values']) > max_years else data['recent_values']
            
            for i, header in enumerate(headers):
                if i < len(values):
                    cash_flow_str.append(f"  {header}: {values[i]}")
    
    cash_flow_data_str = "\n".join(cash_flow_str)
    
    # Include insights from previous analyses
    previous_insights = ""
    if context:
        fundamental_summary = context.get('fundamental', {}).get('summary', '')
        technical_summary = context.get('technical', {}).get('summary', '')
        
        if fundamental_summary or technical_summary:
            previous_insights = f"""
            CONTEXT FROM PREVIOUS ANALYSES:
            
            Fundamental Analysis: {fundamental_summary}
            
            Technical Analysis: {technical_summary}
            
            Use this context to ensure your cash flow analysis is consistent with these insights.
            """
    
    # Build on our chain-of-thought prompting with more focus and previous context
    cash_flow_prompt = f"""
    You are a financial analyst specializing in cash flow analysis. Analyze {company_name} ({stock_symbol}).
    
    Focus on these cash flow data points:
    {cash_flow_data_str}
    
    {previous_insights}
    
    Think step-by-step:
    1. First analyze operating cash flow trends and quality
    2. Then calculate and interpret free cash flow generation
    3. Next evaluate investing activities and capital allocation 
    4. Then assess financing activities and capital structure decisions
    5. Finally, determine overall cash flow health and sustainability
    
    After your careful analysis, provide a JSON object with your complete cash flow analysis:
    {{
        "operating_cf_trend": "Your analysis of operating cash flow trends here...",
        "free_cash_flow": "Your analysis of free cash flow here...",
        "cash_flow_metrics": {{
            "Operating CF to Net Income": "1.76",
            "FCF Yield": "2.4%"
        }},
        "investing_activities": "Your analysis of investing activities here...",
        "financing_activities": "Your analysis of financing activities here...",
        "full_analysis": "Your complete cash flow analysis here...",
        "summary": "A brief 2-3 sentence summary of the most important cash flow insights..."
    }}
    
    IMPORTANT: Provide your thinking process first, then respond with ONLY the JSON. No preamble or explanation outside the JSON.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(cash_flow_prompt)
        
        # Extract JSON from the response - using our enhanced extraction function
        json_str = extract_json_from_text(response)
        
        if json_str:
            import json
            parsed_data = json.loads(json_str)
            
            # Update context with cash flow insights if context exists
            if context is not None:
                context['cash_flow']['operating_cf_insights'] = parsed_data.get("operating_cf_trend", "")
                context['cash_flow']['free_cash_flow_insights'] = parsed_data.get("free_cash_flow", "")
                context['cash_flow']['investing_insights'] = parsed_data.get("investing_activities", "")
                context['cash_flow']['financing_insights'] = parsed_data.get("financing_activities", "")
                context['cash_flow']['summary'] = parsed_data.get("summary", "")
            
            # Create the CashFlowAnalysis object
            return CashFlowAnalysis(
                operating_cf_trend=parsed_data.get("operating_cf_trend", "Operating cash flow trend analysis not available."),
                free_cash_flow=parsed_data.get("free_cash_flow", "Free cash flow analysis not available."),
                cash_flow_metrics=parsed_data.get("cash_flow_metrics", {}),
                investing_activities=parsed_data.get("investing_activities", "Investing activities analysis not available."),
                financing_activities=parsed_data.get("financing_activities", "Financing activities analysis not available."),
                full_analysis=parsed_data.get("full_analysis", "Comprehensive cash flow analysis not available.")
            )
        else:
            # If we can't find a valid JSON, fall back to a simple analysis
            st.warning("Could not extract valid JSON from cash flow analysis response")
            return fallback_cash_flow_analysis(key_data_points, company_name, stock_symbol, context)
    
    except Exception as e:
        st.warning(f"Cash flow analysis generation failed: {str(e)}")
        return fallback_cash_flow_analysis(key_data_points, company_name, stock_symbol, context)

def generate_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm):
    """
    Generate structured technical analysis using LLM with improved prompt engineering
    """
    # Format technical data for the prompt
    technical_data_str = format_technical_data_for_prompt(key_data_points)
    
    # Create the prompt template with few-shot examples
    technical_prompt = f"""
    You are a technical analyst with expertise in equity markets. Analyze {company_name} ({stock_symbol}) currently trading at {current_price}.
    
    Focus on these technical data points:
    {technical_data_str}
    
    Follow this step-by-step approach:
    1. First, determine the primary trend direction (bullish, bearish, or neutral)
    2. Next, identify key support and resistance levels
    3. Then, evaluate momentum indicators (RSI, MACD, etc.)
    4. Finally, analyze volume patterns and what they suggest
    
    EXAMPLE 1 - Bullish trend analysis:
    ```json
    {{
        "trend_direction": "bullish",
        "key_levels": {{
            "support": [142.50, 138.75],
            "resistance": [150.25, 157.80]
        }},
        "momentum_indicators": [
            {{
                "name": "RSI",
                "value": "67.8",
                "interpretation": "RSI approaching overbought territory (70) but still indicates strong bullish momentum."
            }},
            {{
                "name": "MACD",
                "value": "2.45",
                "interpretation": "MACD is positive and above signal line, confirming bullish momentum."
            }}
        ],
        "volume_analysis": "Trading volume has been increasing on up days and decreasing on down days, confirming the bullish trend. The 20-day average volume is 15% higher than the 50-day average.",
        "full_analysis": "The stock is in a strong bullish trend with higher highs and higher lows formed over the past month. The uptrend is supported by positive momentum indicators and healthy volume patterns. Key support exists at 142.50 with immediate resistance at 150.25. The technical setup suggests continuation of the bullish trend as long as price remains above the key support level."
    }}
    ```
    
    EXAMPLE 2 - Bearish trend analysis:
    ```json
    {{
        "trend_direction": "bearish",
        "key_levels": {{
            "support": [85.30, 82.10],
            "resistance": [90.75, 94.20]
        }},
        "momentum_indicators": [
            {{
                "name": "RSI",
                "value": "32.5",
                "interpretation": "RSI approaching oversold territory (30) but still indicates bearish momentum."
            }},
            {{
                "name": "MACD",
                "value": "-1.75",
                "interpretation": "MACD is negative and below signal line, confirming bearish momentum."
            }}
        ],
        "volume_analysis": "Trading volume has been increasing on down days and decreasing on up days, confirming the bearish trend. Recent selling pressure is evident with above-average volume on red candles.",
        "full_analysis": "The stock is in a defined bearish trend with lower highs and lower lows. Multiple resistance levels have rejected price advances. Momentum indicators confirm the bearish sentiment, with RSI at 32.5 and MACD negative. Volume patterns support the bearish outlook with increased selling pressure. The path of least resistance appears to be downward with initial support at 85.30."
    }}
    ```
    
    Now analyze {company_name} ({stock_symbol}) and respond with a JSON object following the exact same format as the examples.
    IMPORTANT: Your response must be ONLY a JSON object. Format as a code block with ```json and ``` tags.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(technical_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            
            # Parse JSON and create TechnicalAnalysis object
            import json
            parsed_data = json.loads(json_str)
            
            # Create the TechnicalAnalysis object
            return TechnicalAnalysis(
                trend_direction=parsed_data.get("trend_direction", "neutral"),
                key_levels=parsed_data.get("key_levels", {"support": [], "resistance": []}),
                momentum_indicators=[
                    KeyMetric(
                        name=indicator.get("name", "Unknown"),
                        value=indicator.get("value", "N/A"),
                        interpretation=indicator.get("interpretation", "No interpretation available")
                    )
                    for indicator in parsed_data.get("momentum_indicators", [])
                ],
                volume_analysis=parsed_data.get("volume_analysis", "No volume analysis available"),
                full_analysis=parsed_data.get("full_analysis", f"No detailed analysis available for {company_name}")
            )
            
        else:
            # Try raw JSON format
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                
                # Parse JSON and create TechnicalAnalysis object
                import json
                parsed_data = json.loads(json_str)
                
                # Create the TechnicalAnalysis object
                return TechnicalAnalysis(
                    trend_direction=parsed_data.get("trend_direction", "neutral"),
                    key_levels=parsed_data.get("key_levels", {"support": [], "resistance": []}),
                    momentum_indicators=[
                        KeyMetric(
                            name=indicator.get("name", "Unknown"),
                            value=indicator.get("value", "N/A"),
                            interpretation=indicator.get("interpretation", "No interpretation available")
                        )
                        for indicator in parsed_data.get("momentum_indicators", [])
                    ],
                    volume_analysis=parsed_data.get("volume_analysis", "No volume analysis available"),
                    full_analysis=parsed_data.get("full_analysis", f"No detailed analysis available for {company_name}")
                )
            
            # If we still can't find a valid JSON, fall back
            st.warning("Could not extract valid JSON from technical analysis response")
            return fallback_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm)
    
    except Exception as e:
        st.warning(f"Technical analysis generation failed: {str(e)}")
        return fallback_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm)

def generate_final_report(key_data_points, company_name, stock_symbol, current_price, 
                        fundamental_analysis, technical_analysis, llm, original_data,
                        cash_flow_analysis=None):
    """
    Generate the final complete report with recommendation
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        fundamental_analysis: FundamentalAnalysis object
        technical_analysis: TechnicalAnalysis object
        llm: LangChain LLM object
        original_data: Original data dictionary
        cash_flow_analysis: Optional CashFlowAnalysis object
        
    Returns:
        StockAnalysisReport object with the complete analysis
    """
    # Generate the recommendation using our improved function
    recommendation = generate_recommendation(
        key_data_points, company_name, stock_symbol, current_price,
        fundamental_analysis, technical_analysis, llm
    )
    
    # Generate price history analysis using chain-of-thought
    price_history_analysis = generate_price_history_analysis(
        key_data_points, company_name, stock_symbol, llm
    )
    
    # Use the passed cash_flow_analysis or generate it if not provided
    if not cash_flow_analysis:
        cash_flow_analysis = generate_cash_flow_analysis(
            key_data_points, company_name, stock_symbol, llm
        )
    
    # Create the complete report
    report = StockAnalysisReport(
        company_name=company_name,
        stock_symbol=stock_symbol,
        current_price=current_price,
        date=datetime.now().strftime("%B %d, %Y"),
        fundamental_analysis=fundamental_analysis,
        technical_analysis=technical_analysis,
        cash_flow_analysis=cash_flow_analysis,
        recommendation=recommendation,
        price_history_analysis=price_history_analysis
    )
    
    # Add additional technical data for charts if available
    if 'technical_data' in original_data:
        if 'moving_averages' in original_data['technical_data']:
            report.moving_averages = original_data['technical_data']['moving_averages']
        
        # Add pivot points
        if 'pivot_points' in original_data['technical_data']:
            # Clean up pivot point values for consistency
            pivot_points = {}
            for method, levels in original_data['technical_data']['pivot_points'].items():
                pivot_points[method] = {}
                for level, value in levels.items():
                    try:
                        if isinstance(value, str):
                            value = float(value.replace(',', ''))
                        pivot_points[method][level] = value
                    except (ValueError, TypeError):
                        pivot_points[method][level] = value
            
            report.pivot_points = pivot_points
        
        # Process support/resistance levels if available
        if 'support_resistance' in original_data['technical_data']:
            support_resistance = {
                'support': [],
                'resistance': []
            }
            
            if 'support' in original_data['technical_data']['support_resistance']:
                support_data = original_data['technical_data']['support_resistance']['support']
                for level in support_data:
                    try:
                        # Handle complex support level strings
                        cleaned_levels = extract_price_levels(level)
                        support_resistance['support'].extend(cleaned_levels)
                    except Exception:
                        # If parsing fails, just include as is
                        support_resistance['support'].append(level)
            
            # Process resistance levels if available
            if 'resistance' in original_data['technical_data']['support_resistance']:
                resistance_data = original_data['technical_data']['support_resistance']['resistance']
                for level in resistance_data:
                    try:
                        # Handle complex resistance level strings
                        cleaned_levels = extract_price_levels(level)
                        support_resistance['resistance'].extend(cleaned_levels)
                    except Exception:
                        # If parsing fails, just include as is
                        support_resistance['resistance'].append(level)
            
            report.support_resistance = support_resistance
    
    # Add price history data if available
    if 'price_history_data' in original_data and 'daily_data' in original_data['price_history_data']:
        report.price_history = original_data['price_history_data']['daily_data']
    
    return report

def generate_recommendation(key_data_points, company_name, stock_symbol, current_price,
                         fundamental_analysis, technical_analysis, llm, context=None):
    """
    Generate investment recommendation using cumulative context from all analyses
    
    Args:
        key_data_points: Dictionary with key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        fundamental_analysis: FundamentalAnalysis object
        technical_analysis: TechnicalAnalysis object
        llm: LangChain LLM object
        context: Global context dictionary with insights from all analyses
        
    Returns:
        StockRecommendation object
    """
    # Extract key insights from fundamental and technical analysis
    fundamental_insights = fundamental_analysis.full_analysis
    technical_insights = technical_analysis.full_analysis
    
    # Prepare comprehensive context from all analyses
    consolidated_insights = ""
    if context:
        # Create a concise summary of all key insights
        context['insights_summary'] = f"""
        FUNDAMENTAL INSIGHTS:
        - Revenue: {context['fundamental'].get('revenue_insights', '')[:100]}...
        - Profit: {context['fundamental'].get('profit_insights', '')[:100]}...
        - Financial Health: {context['fundamental'].get('financial_health', '')[:100]}...
        
        TECHNICAL INSIGHTS:
        - Trend: {context['technical'].get('trend_insights', '')}
        - Support/Resistance: {context['technical'].get('support_resistance_insights', '')}
        - Momentum: {context['technical'].get('momentum_insights', '')}
        - Volume: {context['technical'].get('volume_insights', '')[:100]}...
        
        CASH FLOW INSIGHTS:
        - Operating CF: {context['cash_flow'].get('operating_cf_insights', '')[:100]}...
        - Free Cash Flow: {context['cash_flow'].get('free_cash_flow_insights', '')[:100]}...
        - Financing: {context['cash_flow'].get('financing_insights', '')[:100]}...
        """
        
        consolidated_insights = context['insights_summary']
    
    # Get the format reference for recommendations
    recommendation_format = get_output_format_reference("recommendation")
    
    # Create a more concise prompt template with chain-of-thought and comprehensive context
    recommendation_prompt = f"""
    As an equity research analyst, provide an investment recommendation for {company_name} ({stock_symbol}) at {current_price}.
    
    CONSOLIDATED ANALYSIS INSIGHTS:
    {consolidated_insights if consolidated_insights else "No consolidated insights available."}
    
    First, think step by step about the recommendation:
    
    STEP 1: Evaluate fundamental strength
    - Analyze revenue growth, profit trends, and financial health
    - Consider competitive position and business model sustainability
    - Assess management effectiveness and capital allocation
    
    STEP 2: Assess technical setup
    - Evaluate current trend direction, support/resistance, and momentum
    - Consider volume patterns and potential price catalysts
    - Identify key price levels that could change the outlook
    
    STEP 3: Weigh cash flow quality
    - Consider operating cash flow sustainability
    - Evaluate free cash flow generation and uses
    - Assess capital allocation decisions
    
    STEP 4: Determine risk/reward balance
    - Identify potential upside scenarios and catalysts
    - Consider key downside risks and mitigating factors
    - Evaluate probability of different outcomes
    
    STEP 5: Formulate final recommendation
    - STRONG BUY: Exceptional opportunity, high conviction (>25% upside)
    - BUY: Positive outlook, reasonable valuation (10-25% upside)
    - HOLD: Balanced risk/reward, fair valuation (±10% potential)
    - SELL: Negative outlook, better alternatives exist (10-25% downside)
    - STRONG SELL: Significant downside risk, avoid (>25% downside)
    
    Based on your careful analysis, provide your recommendation in this exact JSON format:
    {recommendation_format}
    
    IMPORTANT: Provide your thinking process first, then respond with ONLY the valid JSON object.
    """
    
    # Try to generate structured output with error handling
    try:
        # Invoke the LLM
        response = llm.invoke(recommendation_prompt)
        
        # Extract JSON and create the recommendation object
        json_str = extract_json_from_text(response)
        if json_str:
            try:
                import json
                recommendation_data = json.loads(json_str)
                
                # Process price targets
                price_targets = []
                for target in recommendation_data.get("price_targets", []):
                    try:
                        price_targets.append(
                            PriceTarget(
                                scenario=target.get("scenario", "Unknown"),
                                price=float(target.get("price", current_price)),
                                timeframe=target.get("timeframe", "12 months")
                            )
                        )
                    except (ValueError, TypeError):
                        # Handle cases where price might be a string with non-numeric characters
                        try:
                            price_str = str(target.get("price", current_price))
                            price_str = re.sub(r'[^\d.]', '', price_str)
                            price_targets.append(
                                PriceTarget(
                                    scenario=target.get("scenario", "Unknown"),
                                    price=float(price_str) if price_str else current_price,
                                    timeframe=target.get("timeframe", "12 months")
                                )
                            )
                        except:
                            pass
                
                # If no price targets were found, create default ones
                if not price_targets:
                    price_targets = [
                        PriceTarget(scenario="Bearish", price=round(current_price * 0.9, 2), timeframe="6 months"),
                        PriceTarget(scenario="Base", price=round(current_price * 1.1, 2), timeframe="12 months"),
                        PriceTarget(scenario="Bullish", price=round(current_price * 1.3, 2), timeframe="12 months")
                    ]
                
                # Get supporting and risk factors with fallbacks
                supporting_factors = recommendation_data.get("supporting_factors", [])
                if not supporting_factors or len(supporting_factors) < 3:
                    default_supports = [
                        f"{company_name} shows potential for future growth",
                        f"Technical indicators suggest a favorable entry point",
                        f"The stock appears reasonably valued"
                    ]
                    # Add default factors as needed to reach 3
                    supporting_factors.extend(default_supports[:(3-len(supporting_factors))])
                
                risk_factors = recommendation_data.get("risk_factors", [])
                if not risk_factors or len(risk_factors) < 2:
                    default_risks = [
                        f"Market volatility could impact {company_name}'s performance",
                        f"Competitive pressures may affect profit margins"
                    ]
                    # Add default factors as needed to reach 2
                    risk_factors.extend(default_risks[:(2-len(risk_factors))])
                
                return StockRecommendation(
                    rating=recommendation_data.get("rating", "HOLD"),
                    rationale=recommendation_data.get("rationale", f"Assessment based on analysis of {company_name}'s fundamentals and technicals."),
                    price_targets=price_targets,
                    supporting_factors=supporting_factors[:3],  # Ensure we have exactly 3
                    risk_factors=risk_factors[:2]  # Ensure we have exactly 2
                )
            except Exception as parse_error:
                st.warning(f"JSON parsing failed: {str(parse_error)}")
        
        # Fall back to simple extraction
        return fallback_recommendation(key_data_points, company_name, stock_symbol, 
                                     current_price, fundamental_analysis, technical_analysis, llm, context)
    
    except Exception as e:
        st.warning(f"Structured recommendation failed: {str(e)}")
        return fallback_recommendation(key_data_points, company_name, stock_symbol, 
                                     current_price, fundamental_analysis, technical_analysis, llm, context)

def generate_price_history_analysis(key_data_points, company_name, stock_symbol, llm, context=None):
    """
    Generate analysis of price history with chain-of-thought prompting and context sharing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        context: Global context dictionary with insights from all analyses
        
    Returns:
        String with price history analysis
    """
    # Format price history data for the prompt
    price_history_data = format_price_history_for_prompt(key_data_points)
    
    # Include insights from previous analyses
    previous_insights = ""
    if context and context.get('insights_summary'):
        previous_insights = f"""
        CONTEXT FROM PREVIOUS ANALYSES:
        {context['insights_summary']}
        
        Consider this context when analyzing price history to ensure consistency.
        """
    
    # Create a prompt with chain-of-thought reasoning and previous context
    price_history_prompt = f"""
    You are a market strategist specializing in historical price analysis. Analyze {company_name} ({stock_symbol}).
    
    PRICE HISTORY DATA:
    {price_history_data}
    
    {previous_insights}
    
    Think step-by-step about this analysis:
    
    STEP 1: TREND ANALYSIS
    - Identify the primary long-term trend (uptrend, downtrend, sideways)
    - Note any significant breakouts or breakdowns
    - Observe cyclical or seasonal patterns
    
    STEP 2: RELATIVE PERFORMANCE
    - Compare performance to relevant market indices
    - Identify periods of outperformance/underperformance
    - Calculate relative strength vs. the market
    
    STEP 3: VOLATILITY ASSESSMENT
    - Evaluate historical volatility compared to the market
    - Note periods of extreme price movement
    - Assess if volatility is increasing or decreasing
    
    STEP 4: VOLUME ANALYSIS
    - Examine volume patterns during price moves
    - Note any volume divergences from price
    - Identify accumulation or distribution patterns
    
    STEP 5: KEY PRICE LEVELS
    - Identify historically significant support/resistance levels
    - Note price reaction to these levels
    - Determine which levels are most relevant now
    
    STEP 6: CONSISTENCY CHECK
    - Ensure your price history analysis is consistent with the fundamental and technical insights provided
    - Address any discrepancies between price action and company fundamentals
    
    Based on this step-by-step analysis, provide a comprehensive but concise price history analysis in 2-3 paragraphs. Use specific data points and percentages where available.
       
       Your analysis should be clear, professional, and focus on actionable insights for investors.
       """
   
    # Try to generate the analysis with error handling
    try:
        # Invoke the LLM
        response = llm.invoke(price_history_prompt)
        
        # Clean up the response - remove any additional notes or formatting
        cleaned_response = re.sub(r'^(Step \d+:|STEP \d+:).*?$', '', response, flags=re.MULTILINE)
        cleaned_response = re.sub(r'^Analysis:|^Summary:', '', cleaned_response, flags=re.MULTILINE)
        
        # Update context with price history insights if context exists
        if context is not None:
            # Extract a brief summary from the response (first 200 chars)
            summary = cleaned_response.strip()[:200] + "..."
            context['price_history_summary'] = summary
        
        # Return the cleaned response
        return cleaned_response.strip()
    
    except Exception as e:
        st.warning(f"Price history analysis failed: {str(e)}")
        # Return a simple placeholder
        return f"Analysis of {company_name}'s historical price performance could not be generated. Please check the data and try again."


def parse_structured_output(response, parser, model_class):
    """
    Parse structured output from LLM response with robust error handling
    
    Args:
        response: Text response from LLM
        parser: PydanticOutputParser object
        model_class: The Pydantic model class to instantiate
        
    Returns:
        Parsed Pydantic object
    """
    # First try direct parsing with the parser
    try:
        return parser.parse(response)
    except Exception as parser_error:
        st.warning(f"Standard parser failed: {str(parser_error)}")
        
        # Look for JSON in code blocks (technical analysis response format)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # Parse with standard json module first
                import json
                parsed_data = json.loads(json_str)
                # Then create the Pydantic model
                return model_class(**parsed_data)
            except Exception as json_error:
                st.warning(f"JSON code block parsing failed: {str(json_error)}")
        
        # Look for raw JSON objects (fundamental analysis response format)
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # Parse with standard json module first
                import json
                parsed_data = json.loads(json_str)
                # Then create the Pydantic model
                return model_class(**parsed_data)
            except Exception as json_error:
                st.warning(f"Raw JSON parsing failed: {str(json_error)}")
        
        # If all parsing attempts fail, raise the original error
        raise ValueError(f"Could not parse response as {model_class.__name__}")

def extract_price_levels(level_string):
    """
    Extract multiple price levels from a potentially complex string
    
    Args:
        level_string: String containing one or more price levels
        
    Returns:
        List of extracted price levels as strings
    """
    if not isinstance(level_string, str):
        return [str(level_string)]
    
    # For strings that might contain multiple concatenated numbers
    if len(level_string) > 8:
        # Extract all floating point numbers
        matches = re.findall(r'(\d+\.\d+)', level_string)
        if matches:
            return matches
    
    # Simple case - just a single number
    try:
        clean_value = level_string.replace(',', '').replace('₹', '')
        float_value = float(clean_value)
        return [f"{float_value:.2f}"]
    except (ValueError, TypeError):
        # Return as is if we can't parse it
        return [level_string]


def format_financial_data_for_prompt(key_data, max_quarters=4, max_years=3):
    """
    Format financial data into a clear string for prompts, limiting to most relevant data points
    
    Args:
        key_data: Dictionary containing extracted key data
        max_quarters: Maximum number of recent quarters to include
        max_years: Maximum number of recent years to include
        
    Returns:
        Formatted string with financial data
    """
    result = []
    
    # Add quarterly data (limited to max_quarters)
    if 'financial_summary' in key_data and 'quarterly' in key_data['financial_summary']:
        result.append("QUARTERLY TRENDS:")
        quarterly = key_data['financial_summary']['quarterly']
        
        for metric, data in quarterly.items():
            if 'headers' in data and 'recent_values' in data:
                result.append(f"  {data.get('label', metric.upper())}:")
                
                # Limit to max_quarters most recent quarters
                headers = data['headers'][:max_quarters] if len(data['headers']) > max_quarters else data['headers']
                values = data['recent_values'][:max_quarters] if len(data['recent_values']) > max_quarters else data['recent_values']
                
                for i, header in enumerate(headers):
                    if i < len(values):
                        result.append(f"    {header}: {values[i]}")
    
    # Add annual data (limited to max_years)
    if 'financial_summary' in key_data and 'annual' in key_data['financial_summary']:
        result.append("\nANNUAL PERFORMANCE:")
        annual = key_data['financial_summary']['annual']
        
        for metric, data in annual.items():
            if 'headers' in data and 'recent_values' in data:
                result.append(f"  {data.get('label', metric.upper())}:")
                if 'cagr_3yr' in data:
                    result.append(f"    3-Year CAGR: {data['cagr_3yr']}")
                if 'cagr_5yr' in data and max_years >= 5:  # Only include 5yr CAGR if we're looking at 5+ years
                    result.append(f"    5-Year CAGR: {data['cagr_5yr']}")
                
                # Limit to max_years most recent years
                headers = data['headers'][:max_years] if len(data['headers']) > max_years else data['headers']
                values = data['recent_values'][:max_years] if len(data['recent_values']) > max_years else data['recent_values']
                
                for i, header in enumerate(headers):
                    if i < len(values):
                        result.append(f"    {header}: {values[i]}")
    
    # Add cash flow data (limited to max_years)
    if 'financial_summary' in key_data and 'cash_flow' in key_data['financial_summary']:
        result.append("\nCASH FLOW ANALYSIS:")
        cash_flow = key_data['financial_summary']['cash_flow']
        
        # Priority metrics for cash flow - focus on the most important ones
        priority_metrics = ['operating_cf', 'investing_cf', 'financing_cf', 'net_cf']
        
        # First add priority metrics
        for metric_name in priority_metrics:
            for cf_key, data in cash_flow.items():
                if metric_name in cf_key.lower() and 'headers' in data and 'recent_values' in data:
                    result.append(f"  {data.get('label', cf_key.upper())}:")
                    if 'cagr_3yr' in data and data['cagr_3yr'] != '-':
                        result.append(f"    3-Year CAGR: {data['cagr_3yr']}")
                    
                    # Limit to max_years most recent years
                    headers = data['headers'][:max_years] if len(data['headers']) > max_years else data['headers']
                    values = data['recent_values'][:max_years] if len(data['recent_values']) > max_years else data['recent_values']
                    
                    for i, header in enumerate(headers):
                        if i < len(values):
                            result.append(f"    {header}: {values[i]}")
    
    # Add key ratios (focus on most important ones)
    if 'financial_summary' in key_data and 'ratios' in key_data['financial_summary']:
        result.append("\nKEY FINANCIAL RATIOS:")
        ratios = key_data['financial_summary']['ratios']
        
        # Define important ratios to prioritize
        priority_ratios = ['roe', 'roa', 'debt_equity', 'price_to_book', 'current_ratio']
        
        # First add priority ratios
        for ratio_name in priority_ratios:
            if ratio_name in ratios and 'headers' in ratios[ratio_name] and 'recent_values' in ratios[ratio_name]:
                data = ratios[ratio_name]
                result.append(f"  {data.get('label', ratio_name.upper())}:")
                
                # Limit to max_years most recent years
                headers = data['headers'][:max_years] if len(data['headers']) > max_years else data['headers']
                values = data['recent_values'][:max_years] if len(data['recent_values']) > max_years else data['recent_values']
                
                for i, header in enumerate(headers):
                    if i < len(values):
                        result.append(f"    {header}: {values[i]}")
    
    return "\n".join(result) if result else "Financial data not available."

def format_technical_data_for_prompt(key_data, max_indicators=5):
    """
    Format technical data into a clear string for prompts, limiting to most relevant indicators
    
    Args:
        key_data: Dictionary containing extracted key data
        max_indicators: Maximum number of technical indicators to include
        
    Returns:
        Formatted string with technical data
    """
    result = []
    
    # Add technical indicators (limited to max_indicators)
    if 'technical_summary' in key_data and 'indicators' in key_data['technical_summary']:
        result.append("TECHNICAL INDICATORS:")
        
        # Define priority indicators to ensure we get the most important ones
        priority_indicators = ['RSI', 'ADX', 'Momentum Score']
        indicators = key_data['technical_summary']['indicators']
        
        # Add priority indicators first
        indicators_added = 0
        for indicator in priority_indicators:
            if indicator in indicators and indicators_added < max_indicators:
                result.append(f"  {indicator}: {indicators[indicator]}")
                indicators_added += 1
        
        # Add any remaining indicators up to max_indicators
        for indicator, value in indicators.items():
            if indicator not in priority_indicators and indicators_added < max_indicators:
                result.append(f"  {indicator}: {value}")
                indicators_added += 1
    
    # Add oscillators - New addition (selectively)
    if 'technical_summary' in key_data and 'oscillators' in key_data['technical_summary']:
        result.append("\nOSCILLATORS:")
        
        # Priority oscillators
        priority_oscillators = ['Stochastic', 'MACD', 'CCI']
        oscillators = key_data['technical_summary']['oscillators']
        
        # Add only top oscillators (max 3)
        oscillators_added = 0
        for oscillator in priority_oscillators:
            if oscillator in oscillators and oscillators_added < 3:
                result.append(f"  {oscillator}: {oscillators[oscillator]}")
                oscillators_added += 1
    
    # Add moving averages (most relevant ones only)
    if 'technical_summary' in key_data and 'moving_averages' in key_data['technical_summary']:
        result.append("\nMOVING AVERAGES:")
        
        # Priority moving averages (short, medium, long-term)
        priority_mas = ['SMA20', 'SMA50', 'SMA100']
        ma_data = key_data['technical_summary']['moving_averages']
        
        # Add only top moving averages
        for ma_type in priority_mas:
            if ma_type in ma_data:
                result.append(f"  {ma_type}: {ma_data[ma_type]}")
    
    # Add support/resistance levels
    if 'technical_summary' in key_data and 'support_resistance' in key_data['technical_summary']:
        sr_data = key_data['technical_summary']['support_resistance']
        
        if 'support' in sr_data and sr_data['support']:
            result.append("\nSUPPORT LEVELS:")
            # Only include top 2 support levels
            for level in sr_data['support'][:2]:
                result.append(f"  {level}")
        
        if 'resistance' in sr_data and sr_data['resistance']:
            result.append("\nRESISTANCE LEVELS:")
            # Only include top 2 resistance levels
            for level in sr_data['resistance'][:2]:
                result.append(f"  {level}")
    
    # Add performance data (focus on key timeframes)
    if 'technical_summary' in key_data and 'performance' in key_data['technical_summary']:
        result.append("\nPRICE PERFORMANCE:")
        
        # Focus on most relevant timeframes
        key_timeframes = ['1 Month', '3 Months', '1 Year']
        performance = key_data['technical_summary']['performance']
        
        for period in key_timeframes:
            if period in performance:
                value = performance[period]
                if isinstance(value, dict) and 'percentage' in value:
                    result.append(f"  {period}: {value['percentage']}")
                else:
                    result.append(f"  {period}: {value}")
    
    return "\n".join(result) if result else "Technical data not available."

def generate_cash_flow_analysis(key_data_points, company_name, stock_symbol, llm):
    """
    Generate structured cash flow analysis using LLM with chain-of-thought prompting
    """
    # Check if cash flow data exists
    if not (
        'financial_summary' in key_data_points and 
        'cash_flow' in key_data_points['financial_summary'] and 
        key_data_points['financial_summary']['cash_flow']
    ):
        # Return minimal analysis if no data available
        return CashFlowAnalysis(
            operating_cf_trend="Cash flow trend data not available.",
            free_cash_flow="Free cash flow data not available.",
            cash_flow_metrics={},
            investing_activities="Investing activities data not available.",
            financing_activities="Financing activities data not available.",
            full_analysis=f"Comprehensive cash flow analysis for {company_name} could not be generated due to missing data."
        )
    
    # Format cash flow data for the prompt
    cash_flow_data = key_data_points['financial_summary']['cash_flow']
    cash_flow_str = []
    
    for cf_type, data in cash_flow_data.items():
        if 'label' in data and 'recent_values' in data and 'headers' in data:
            cash_flow_str.append(f"{data['label']}:")
            for i, header in enumerate(data['headers']):
                if i < len(data['recent_values']):
                    cash_flow_str.append(f"  {header}: {data['recent_values'][i]}")
    
    cash_flow_data_str = "\n".join(cash_flow_str)
    
    # Create the prompt template with chain-of-thought prompting
    cash_flow_prompt = f"""
    You are a financial analyst specializing in equity research. Analyze the cash flow performance of {company_name} ({stock_symbol}).
    
    Focus on these cash flow data points:
    {cash_flow_data_str}
    
    Let's analyze the cash flows step-by-step:
    
    Step 1: Operating Cash Flow Analysis
    - Examine the operating cash flow trend over time
    - Compare operating cash flow to net income (quality of earnings)
    - Identify any concerning fluctuations or positive developments
    - Based on this analysis, what can we conclude about operating cash flow trends?
    
    Step 2: Free Cash Flow Analysis
    - Calculate or examine free cash flow (Operating CF - Capital Expenditures)
    - Assess free cash flow sustainability and growth trends
    - Determine if free cash flow covers dividend payments, if applicable
    - Based on this analysis, what can we conclude about free cash flow generation?
    
    Step 3: Investing Activities Analysis
    - Analyze capital expenditure trends and major investments
    - Assess if investments appear strategic or necessary for growth
    - Determine if capital allocation decisions appear sound
    - Based on this analysis, what can we conclude about investing activities?
    
    Step 4: Financing Activities Analysis
    - Examine debt issuance/repayment patterns
    - Analyze share repurchases or issuances
    - Assess dividend payment trends, if applicable
    - Based on this analysis, what can we conclude about financing activities?
    
    Step 5: Overall Cash Flow Health
    - Integrate all the above insights
    - Assess overall cash flow sustainability and quality
    - Identify key cash flow metrics and their values
    - Provide a comprehensive assessment of the company's cash flow situation
    
    Now, based on this step-by-step analysis, provide a JSON object with the following structure:
    {{
        "operating_cf_trend": "Your detailed analysis of operating cash flow trends here...",
        "free_cash_flow": "Your detailed analysis of free cash flow here...",
        "cash_flow_metrics": {{
            "Operating CF to Net Income": "1.76",
            "FCF Yield": "2.4%",
            "Cash Flow Growth Rate": "12.3%"
        }},
        "investing_activities": "Your analysis of investing activities here...",
        "financing_activities": "Your analysis of financing activities here...",
        "full_analysis": "Your complete cash flow analysis integrating all the above insights here..."
    }}
    
    IMPORTANT: Respond ONLY with the JSON object. No preamble or explanation.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(cash_flow_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            
            # Parse JSON and create CashFlowAnalysis object
            import json
            parsed_data = json.loads(json_str)
            
            # Create the CashFlowAnalysis object
            return CashFlowAnalysis(
                operating_cf_trend=parsed_data.get("operating_cf_trend", "Operating cash flow trend analysis not available."),
                free_cash_flow=parsed_data.get("free_cash_flow", "Free cash flow analysis not available."),
                cash_flow_metrics=parsed_data.get("cash_flow_metrics", {}),
                investing_activities=parsed_data.get("investing_activities", "Investing activities analysis not available."),
                financing_activities=parsed_data.get("financing_activities", "Financing activities analysis not available."),
                full_analysis=parsed_data.get("full_analysis", "Comprehensive cash flow analysis not available.")
            )
        else:
            # If we can't find a valid JSON, fall back to a simple analysis
            st.warning("Could not extract valid JSON from cash flow analysis response")
            return fallback_cash_flow_analysis(key_data_points, company_name, stock_symbol)
    
    except Exception as e:
        st.warning(f"Cash flow analysis generation failed: {str(e)}")
        return fallback_cash_flow_analysis(key_data_points, company_name, stock_symbol)

def fallback_cash_flow_analysis(key_data_points, company_name, stock_symbol, context=None):
  """
  Generate a fallback cash flow analysis when structured parsing fails
  
  Args:
      key_data_points: Dictionary containing extracted key data points
      company_name: Name of the company
      stock_symbol: Stock ticker symbol
      context: Global context dictionary
      
  Returns:
      CashFlowAnalysis object with basic analysis
  """
  # Extract basic cash flow metrics for display
  cash_flow_metrics = {}
  
  if ('financial_summary' in key_data_points and 
      'cash_flow' in key_data_points['financial_summary']):
      
      cash_flow_data = key_data_points['financial_summary']['cash_flow']
      
      # Extract operating cash flow trend
      operating_cf_trend = "Operating cash flow data not available."
      if 'operating_cf' in cash_flow_data:
          cf_data = cash_flow_data['operating_cf']
          if 'recent_values' in cf_data and cf_data['recent_values']:
              # Get the most recent value
              latest_cf = cf_data['recent_values'][0]
              cash_flow_metrics["Latest Operating CF"] = latest_cf
              
              # Try to determine trend
              if len(cf_data['recent_values']) > 1:
                  trend_direction = "stable"
                  try:
                      latest = float(latest_cf.replace(',', ''))
                      previous = float(cf_data['recent_values'][1].replace(',', ''))
                      percent_change = ((latest - previous) / previous) * 100
                      
                      if percent_change > 10:
                          trend_direction = "strongly increasing"
                      elif percent_change > 0:
                          trend_direction = "increasing"
                      elif percent_change < -10:
                          trend_direction = "strongly decreasing"
                      elif percent_change < 0:
                          trend_direction = "decreasing"
                          
                      operating_cf_trend = f"{company_name}'s operating cash flow is {trend_direction} with the most recent value at {latest_cf}."
                  except (ValueError, TypeError, ZeroDivisionError):
                      operating_cf_trend = f"{company_name}'s operating cash flow was {latest_cf} in the most recent period."
              else:
                  operating_cf_trend = f"{company_name}'s operating cash flow was {latest_cf} in the most recent period."
      
      # Extract investing cash flow information
      investing_activities = "Investing activities data not available."
      if 'investing_cf' in cash_flow_data:
          cf_data = cash_flow_data['investing_cf']
          if 'recent_values' in cf_data and cf_data['recent_values']:
              latest_cf = cf_data['recent_values'][0]
              cash_flow_metrics["Latest Investing CF"] = latest_cf
              
              try:
                  value = float(latest_cf.replace(',', ''))
                  if value < 0:
                      investing_activities = f"{company_name} is investing significantly with a cash outflow of {latest_cf} in the most recent period."
                  else:
                      investing_activities = f"{company_name} has a positive cash flow from investing activities of {latest_cf} in the most recent period."
              except (ValueError, TypeError):
                  investing_activities = f"{company_name}'s cash flow from investing activities was {latest_cf} in the most recent period."
      
      # Extract financing cash flow information
      financing_activities = "Financing activities data not available."
      if 'financing_cf' in cash_flow_data:
          cf_data = cash_flow_data['financing_cf']
          if 'recent_values' in cf_data and cf_data['recent_values']:
              latest_cf = cf_data['recent_values'][0]
              cash_flow_metrics["Latest Financing CF"] = latest_cf
              
              try:
                  value = float(latest_cf.replace(',', ''))
                  if value < 0:
                      financing_activities = f"{company_name} is reducing debt or returning capital to shareholders with a financing cash outflow of {latest_cf}."
                  else:
                      financing_activities = f"{company_name} has raised capital with a financing cash inflow of {latest_cf}."
              except (ValueError, TypeError):
                  financing_activities = f"{company_name}'s cash flow from financing activities was {latest_cf} in the most recent period."
      
      # Calculate simple free cash flow approximation
      free_cash_flow = "Free cash flow data not available."
      if ('operating_cf' in cash_flow_data and 'investing_cf' in cash_flow_data and
          'recent_values' in cash_flow_data['operating_cf'] and 
          'recent_values' in cash_flow_data['investing_cf'] and
          cash_flow_data['operating_cf']['recent_values'] and
          cash_flow_data['investing_cf']['recent_values']):
          
          try:
              op_cf = float(cash_flow_data['operating_cf']['recent_values'][0].replace(',', ''))
              inv_cf = float(cash_flow_data['investing_cf']['recent_values'][0].replace(',', ''))
              
              # Free cash flow = Operating CF + Investing CF (typically negative)
              fcf = op_cf + inv_cf
              
              cash_flow_metrics["Free Cash Flow"] = f"{fcf:,.2f}"
              
              if fcf > 0:
                  free_cash_flow = f"{company_name} generated positive free cash flow of approximately {fcf:,.2f}, indicating sufficient cash generation to fund operations and investments."
              else:
                  free_cash_flow = f"{company_name} has negative free cash flow of approximately {fcf:,.2f}, indicating the company is spending more than it generates from operations."
          except (ValueError, TypeError):
              free_cash_flow = f"Free cash flow calculation not possible due to data format issues."
  
  # Create a basic full analysis
  full_analysis = f"""
  Cash Flow Analysis for {company_name} ({stock_symbol}):
  
  Operating Cash Flow: {operating_cf_trend}
  
  Free Cash Flow: {free_cash_flow}
  
  Investing Activities: {investing_activities}
  
  Financing Activities: {financing_activities}
  
  This is a simplified analysis based on the available cash flow data. A more detailed analysis would require additional context about the company's business model, growth stage, and industry dynamics.
  """
  
  # Update context if available
  if context is not None:
      context['cash_flow']['operating_cf_insights'] = operating_cf_trend
      context['cash_flow']['free_cash_flow_insights'] = free_cash_flow
      context['cash_flow']['investing_insights'] = investing_activities
      context['cash_flow']['financing_insights'] = financing_activities
      context['cash_flow']['summary'] = f"{company_name}'s cash flow data was processed using fallback analysis."
  
  # Create the fallback analysis
  return CashFlowAnalysis(
      operating_cf_trend=operating_cf_trend,
      free_cash_flow=free_cash_flow,
      cash_flow_metrics=cash_flow_metrics,
      investing_activities=investing_activities,
      financing_activities=financing_activities,
      full_analysis=full_analysis.strip()
  )

def format_price_history_for_prompt(key_data):
    """
    Format price history data into a clear string for prompts
    
    Args:
        key_data: Dictionary containing extracted key data
        
    Returns:
        Formatted string with price history data
    """
    result = []
    
    # Add returns comparison
    if 'performance_summary' in key_data and 'returns_comparison' in key_data['performance_summary']:
        result.append("RETURNS COMPARISON:")
        for item in key_data['performance_summary']['returns_comparison']:
            period = item.get('period', '')
            if period:
                period_data = [f"{period}:"]
                for key, value in item.items():
                    if key != 'period' and key != 'entity':
                        period_data.append(f"{key}: {value}")
                result.append("  " + ", ".join(period_data))
    
    # Add seasonality data
    if 'performance_summary' in key_data and 'seasonality' in key_data['performance_summary']:
        result.append("\nSEASONALITY PATTERNS:")
        for year_item in key_data['performance_summary']['seasonality']:
            year_data = []
            for key, value in year_item.items():
                if key not in ['Pattern']:  # Skip chart pattern
                    year_data.append(f"{key}: {value}")
            result.append("  " + ", ".join(year_data))
    
    return "\n".join(result) if result else "Price history data not available."

def fallback_fundamental_analysis(key_data, company_name, stock_symbol, llm, context=None):
   """
   Generate a fallback fundamental analysis when structured parsing fails
   
   Args:
       key_data: Dictionary containing extracted key data points
       company_name: Name of the company
       stock_symbol: Stock ticker symbol
       llm: LangChain LLM object
       context: Global context dictionary
       
   Returns:
       FundamentalAnalysis object with basic analysis
   """
   # Simpler prompt for basic text generation
   simple_prompt = f"""
   As a financial analyst, provide a brief analysis of {company_name}'s ({stock_symbol}) financial performance.
   Focus on revenue trends, profit margins, and key financial ratios.
   Keep your response concise and factual, focusing only on the data provided.
   """
   
   try:
       # Get a simple text response
       response = llm.invoke(simple_prompt)
       
       # Extract basic information from the response
       revenue_trend = extract_section(response, "revenue", 100)
       profit_trend = extract_section(response, "profit", 100)
       financial_health = extract_section(response, "financial health", 100)
       
       # Update context if available
       if context is not None:
           context['fundamental']['revenue_insights'] = revenue_trend or f"Revenue trends for {company_name} could not be analyzed."
           context['fundamental']['profit_insights'] = profit_trend or f"Profit margins for {company_name} could not be analyzed."
           context['fundamental']['financial_health'] = financial_health or "Financial health could not be assessed."
           context['fundamental']['summary'] = f"{company_name}'s financial data was processed using fallback analysis."
       
       # Create basic key metrics
       key_ratios = []
       
       # Try to extract ROE
       if 'financial_summary' in key_data and 'ratios' in key_data['financial_summary']:
           ratios = key_data['financial_summary']['ratios']
           if 'roe' in ratios and 'recent_values' in ratios['roe'] and ratios['roe']['recent_values']:
               key_ratios.append(KeyMetric(
                   name="ROE",
                   value=ratios['roe']['recent_values'][0],
                   interpretation="Return on Equity measures profitability relative to shareholders' equity."
               ))
           
           if 'debt_equity' in ratios and 'recent_values' in ratios['debt_equity'] and ratios['debt_equity']['recent_values']:
               key_ratios.append(KeyMetric(
                   name="Debt/Equity",
                   value=ratios['debt_equity']['recent_values'][0],
                   interpretation="Debt to Equity ratio indicates financial leverage and risk."
               ))
       
       # Add a generic metric if none found
       if not key_ratios:
           key_ratios.append(KeyMetric(
               name="Profit Margin",
               value="Varies",
               interpretation="Measures the company's profitability as a percentage of revenue."
           ))
       
       # Create the fallback analysis
       return FundamentalAnalysis(
           revenue_trend=revenue_trend or f"Analysis of {company_name}'s revenue trends could not be generated.",
           profit_trend=profit_trend or f"Analysis of {company_name}'s profit trends could not be generated.",
           key_ratios=key_ratios,
           financial_health=financial_health or "Financial health assessment could not be generated.",
           full_analysis=response
       )
   
   except Exception as e:
       st.error(f"Fallback fundamental analysis failed: {str(e)}")
       
       # Create a minimal valid object with default values
       return FundamentalAnalysis(
           revenue_trend=f"Analysis of {company_name}'s revenue trends could not be generated.",
           profit_trend=f"Analysis of {company_name}'s profit trends could not be generated.",
           key_ratios=[KeyMetric(
               name="N/A", 
               value="N/A", 
               interpretation="Financial metrics could not be analyzed."
           )],
           financial_health="Financial health assessment could not be generated.",
           full_analysis=f"Comprehensive fundamental analysis for {company_name} could not be generated due to data processing issues. Please check the data and try again."
       )

def debug_llm_response(response, message):
    """
    Debug and display raw LLM responses
    
    Args:
        response: The raw response from the LLM
        message: A descriptive message for the debug output
    """
    st.expander(f"Debug: {message}", expanded=False).code(response, language="json")
    
    # Also print to console for backup
    print(f"\n\n--- DEBUG: {message} ---\n")
    print(response)
    print(f"\n--- END DEBUG ---\n\n")
    
    # Try to analyze JSON structure
    try:
        # Look for JSON block in markdown format
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            st.info("Found JSON in code block")
            return json_match.group(1)
            
        # Try to find anything that looks like a JSON object
        json_match = re.search(r'({.*})', response, re.DOTALL)
        if json_match:
            st.info("Found JSON-like structure")
            return json_match.group(1)
            
        return response
    except Exception as e:
        st.error(f"Error analyzing JSON structure: {str(e)}")
        return response

def fallback_technical_analysis(key_data, company_name, stock_symbol, current_price, llm, context=None):
   """
   Generate a fallback technical analysis when structured parsing fails
   
   Args:
       key_data: Dictionary containing extracted key data points
       company_name: Name of the company
       stock_symbol: Stock ticker symbol
       current_price: Current stock price
       llm: LangChain LLM object
       context: Global context dictionary
       
   Returns:
       TechnicalAnalysis object with basic analysis
   """
   # Simpler prompt for basic text generation
   simple_prompt = f"""
   As a technical analyst, provide a brief technical assessment of {company_name} ({stock_symbol}) at price {current_price}.
   Focus on current trend, support/resistance levels, and key technical indicators.
   Keep your response concise and factual.
   """
   
   try:
       # Get a simple text response
       response = llm.invoke(simple_prompt)
       
       # Extract basic trend direction
       trend_direction = "neutral"  # Default
       if "bullish" in response.lower():
           trend_direction = "bullish"
       elif "bearish" in response.lower():
           trend_direction = "bearish"
       
       # Update context if available
       if context is not None:
           context['technical']['trend_insights'] = f"Trend direction: {trend_direction}"
           context['technical']['summary'] = f"{company_name}'s technical data was processed using fallback analysis."
           context['key_metrics']['trend_direction'] = trend_direction
       
       # Extract or create support/resistance levels
       key_levels = {"support": [], "resistance": []}
       
       # Try to extract from original data
       if 'technical_summary' in key_data and 'support_resistance' in key_data['technical_summary']:
           sr_data = key_data['technical_summary']['support_resistance']
           
           if 'support' in sr_data and sr_data['support']:
               for level in sr_data['support']:
                   try:
                       clean_levels = extract_price_levels(level)
                       for clean_level in clean_levels:
                           key_levels["support"].append(float(clean_level))
                   except:
                       pass
           
           if 'resistance' in sr_data and sr_data['resistance']:
               for level in sr_data['resistance']:
                   try:
                       clean_levels = extract_price_levels(level)
                       for clean_level in clean_levels:
                           key_levels["resistance"].append(float(clean_level))
                   except:
                       pass
       
       # If no levels found, create some based on current price
       if not key_levels["support"]:
           key_levels["support"] = [round(current_price * 0.95, 2), round(current_price * 0.90, 2)]
       
       if not key_levels["resistance"]:
           key_levels["resistance"] = [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]
       
       # Update context with support/resistance
       if context is not None:
           context['technical']['support_resistance_insights'] = f"Key support levels: {key_levels['support']} | Key resistance levels: {key_levels['resistance']}"
       
       # Extract or create momentum indicators
       momentum_indicators = []
       
       # Try to extract from original data
       if 'technical_summary' in key_data and 'indicators' in key_data['technical_summary']:
           indicators = key_data['technical_summary']['indicators']
           
           if 'RSI' in indicators:
               rsi_value = indicators['RSI']
               interpretation = "Neutral"
               if float(rsi_value) > 70:
                   interpretation = "Overbought"
               elif float(rsi_value) < 30:
                   interpretation = "Oversold"
               
               momentum_indicators.append(KeyMetric(
                   name="RSI",
                   value=rsi_value,
                   interpretation=f"RSI at {rsi_value} indicates {interpretation} conditions."
               ))
           
           if 'MACD' in indicators:
               momentum_indicators.append(KeyMetric(
                   name="MACD",
                   value=indicators['MACD'],
                   interpretation="MACD is a trend-following momentum indicator."
               ))
       
       # Add generic indicator if none found
       if not momentum_indicators:
           momentum_indicators.append(KeyMetric(
               name="Momentum",
               value="N/A",
               interpretation="Momentum indicators measure the rate of price changes."
           ))
       
       # Extract volume analysis or create generic one
       volume_analysis = extract_section(response, "volume", 100)
       if not volume_analysis:
           volume_analysis = "Volume analysis could not be generated from the available data."
       
       # Update context with volume analysis
       if context is not None:
           context['technical']['volume_insights'] = volume_analysis
       
       # Create the fallback analysis
       return TechnicalAnalysis(
           trend_direction=trend_direction,
           key_levels=key_levels,
           momentum_indicators=momentum_indicators,
           volume_analysis=volume_analysis,
           full_analysis=response
       )
   
   except Exception as e:
       st.error(f"Fallback technical analysis failed: {str(e)}")
       
       # Create a minimal valid object
       return TechnicalAnalysis(
           trend_direction="neutral",
           key_levels={"support": [round(current_price * 0.95, 2), round(current_price * 0.90, 2)],
                     "resistance": [round(current_price * 1.05, 2), round(current_price * 1.10, 2)]},
           momentum_indicators=[KeyMetric(
               name="N/A", 
               value="N/A", 
               interpretation="Technical indicators could not be analyzed."
           )],
           volume_analysis="Volume analysis could not be generated.",
           full_analysis=f"Comprehensive technical analysis for {company_name} could not be generated due to data processing issues. Please check the data and try again."
       )

def fallback_recommendation(key_data, company_name, stock_symbol, current_price, 
                        fundamental_analysis, technical_analysis, llm, context=None):
   """
   Generate a fallback recommendation when structured parsing fails
   
   Args:
       key_data: Dictionary containing extracted key data points
       company_name: Name of the company
       stock_symbol: Stock ticker symbol
       current_price: Current stock price
       fundamental_analysis: FundamentalAnalysis object
       technical_analysis: TechnicalAnalysis object
       llm: LangChain LLM object
       context: Global context dictionary
       
   Returns:
       StockRecommendation object with basic recommendation
   """
   # Default to HOLD rating as the safe option
   rating = "HOLD"
   
   # Try to determine a better rating from the analysis text and context
   combined_analysis = fundamental_analysis.full_analysis + " " + technical_analysis.full_analysis
   combined_analysis = combined_analysis.upper()
   
   # Check context for trend direction if available
   if context and context.get('key_metrics', {}).get('trend_direction') == "bullish":
       # If context shows bullish trend, consider upgrading the rating
       if "STRONG BUY" in combined_analysis or "BUY" in combined_analysis:
           rating = "BUY"
   elif context and context.get('key_metrics', {}).get('trend_direction') == "bearish":
       # If context shows bearish trend, consider downgrading the rating
       if "SELL" in combined_analysis or "STRONG SELL" in combined_analysis:
           rating = "SELL"
   else:
       # Use the text analysis approach if no context is available
       if "STRONG BUY" in combined_analysis:
           rating = "STRONG BUY"
       elif "BUY" in combined_analysis:
           rating = "BUY"
       elif "SELL" in combined_analysis and "STRONG SELL" not in combined_analysis:
           rating = "SELL"
       elif "STRONG SELL" in combined_analysis:
           rating = "STRONG SELL"
   
   # Create basic price targets
   bearish_target = round(current_price * 0.9, 2)  # 10% down
   base_target = round(current_price * 1.1, 2)     # 10% up
   bullish_target = round(current_price * 1.3, 2)  # 30% up
   
   # Create price targets
   price_targets = [
       PriceTarget(scenario="Bearish (Worst Case)", price=bearish_target, timeframe="6 months"),
       PriceTarget(scenario="Base (Most Likely)", price=base_target, timeframe="6 months"),
       PriceTarget(scenario="Bullish (Best Case)", price=bullish_target, timeframe="6 months")
   ]
   
   # Extract supporting and risk factors
   supporting_factors = []
   risk_factors = []
   
   # Try to extract factors from context first if available
   if context:
       # Get insights from context for supporting factors
       if context.get('fundamental', {}).get('revenue_insights'):
           if 'growth' in context['fundamental']['revenue_insights'] or 'increase' in context['fundamental']['revenue_insights']:
               supporting_factors.append(f"Revenue growth: {context['fundamental']['revenue_insights'][:75]}...")
       
       if context.get('technical', {}).get('trend_insights'):
           if 'bullish' in context['technical']['trend_insights']:
               supporting_factors.append(f"Positive technical trend: {context['technical']['trend_insights']}")
       
       if context.get('cash_flow', {}).get('free_cash_flow_insights'):
           if 'positive' in context['cash_flow']['free_cash_flow_insights'] or 'sufficient' in context['cash_flow']['free_cash_flow_insights']:
               supporting_factors.append(f"Strong cash flow: {context['cash_flow']['free_cash_flow_insights'][:75]}...")
       
       # Get insights from context for risk factors
       if context.get('technical', {}).get('momentum_insights'):
           if 'overbought' in context['technical']['momentum_insights']:
               risk_factors.append(f"Overbought conditions: {context['technical']['momentum_insights']}")
       
       if context.get('fundamental', {}).get('financial_health'):
           if 'debt' in context['fundamental']['financial_health'] or 'weak' in context['fundamental']['financial_health']:
               risk_factors.append(f"Financial concerns: {context['fundamental']['financial_health'][:75]}...")
   
   # If we couldn't get enough from context, fall back to text analysis
   if len(supporting_factors) < 3 or len(risk_factors) < 2:
       # Try to extract factors from the analysis texts
       try:
           # For supporting factors
           for text in [fundamental_analysis.full_analysis, technical_analysis.full_analysis]:
               if "strength" in text.lower() or "positive" in text.lower() or "growth" in text.lower():
                   sentences = text.split(".")
                   for sentence in sentences:
                       if ("strength" in sentence.lower() or "positive" in sentence.lower() or 
                           "growth" in sentence.lower()) and len(sentence.strip()) > 15:
                           supporting_factors.append(sentence.strip())
                           if len(supporting_factors) >= 3:
                               break
               if len(supporting_factors) >= 3:
                   break
           
           # For risk factors
           for text in [fundamental_analysis.full_analysis, technical_analysis.full_analysis]:
               if "risk" in text.lower() or "concern" in text.lower() or "negative" in text.lower():
                   sentences = text.split(".")
                   for sentence in sentences:
                       if ("risk" in sentence.lower() or "concern" in sentence.lower() or 
                           "negative" in sentence.lower()) and len(sentence.strip()) > 15:
                           risk_factors.append(sentence.strip())
                           if len(risk_factors) >= 2:
                               break
               if len(risk_factors) >= 2:
                   break
       except:
           # Ignore errors in factor extraction
           pass
   
   # Add generic factors if needed
   if len(supporting_factors) < 3:
       generic_supports = [
           f"{company_name} shows potential for future growth based on market trends",
           f"Technical indicators suggest a favorable entry point at current levels",
           f"The stock is trading at reasonable valuations compared to its sector"
       ]
       supporting_factors.extend(generic_supports[:(3-len(supporting_factors))])
   
   if len(risk_factors) < 2:
       generic_risks = [
           f"Market volatility could impact {company_name}'s short-term performance",
           f"Competitive pressures in the industry may affect profit margins"
       ]
       risk_factors.extend(generic_risks[:(2-len(risk_factors))])
   
   # Create a basic rationale
   rationale = f"Based on balance of fundamental and technical factors, {company_name} is rated {rating}."
   
   # Create the fallback recommendation
   return StockRecommendation(
       rating=rating,
       rationale=rationale,
       price_targets=price_targets,
       supporting_factors=supporting_factors[:3],  # Ensure we only have 3
       risk_factors=risk_factors[:2]  # Ensure we only have 2
   )

def extract_section(text, keyword, max_length=100):
    """
    Extract a section from text that contains a specific keyword
    
    Args:
        text: Text to search in
        keyword: Keyword to search for
        max_length: Maximum length of the extracted section
        
    Returns:
        Extracted section or None if not found
    """
    if not text:
        return None
    
    # Split into sentences
    sentences = text.split('.')
    
    # Look for the keyword in sentences
    for i, sentence in enumerate(sentences):
        if keyword.lower() in sentence.lower():
            # Take this sentence and potentially the next one
            result = sentence.strip()
            if i + 1 < len(sentences):
                result += ". " + sentences[i + 1].strip()
            
            # Truncate if too long
            if len(result) > max_length:
                result = result[:max_length] + "..."
            
            return result
    
    return None

def extract_key_data_points(data):
    """
    Extract and structure the most important data points from the raw data
    
    Args:
        data: Dictionary containing raw stock data
        
    Returns:
        Dictionary with structured key data points
    """
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
                            'recent_values': [val for val in row[1:6] if val],  # Get 5 most recent values
                            'headers': table_data['headers'][1:6] if len(table_data['headers']) >= 6 else table_data['headers'][1:]
                        }
                    elif 'Net Profit' in row[0]:
                        quarterly['net_profit'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[1:6] if val],
                            'headers': table_data['headers'][1:6] if len(table_data['headers']) >= 6 else table_data['headers'][1:]
                        }
                    elif 'Operating Profit Margin' in row[0]:
                        quarterly['operating_margin'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[1:6] if val],
                            'headers': table_data['headers'][1:6] if len(table_data['headers']) >= 6 else table_data['headers'][1:]
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
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    elif 'Net Profit' in row[0]:
                        annual['net_profit'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    elif 'NPM' in row[0] or 'Net Profit Margin' in row[0] or 'NETPCT' in row[0]:
                        annual['net_margin'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
            
            key_data['financial_summary']['annual'] = annual
        
        # Cash Flow data - New addition
        if 'cash_flow' in financial_data:
            cash_flow = {}
            for table_name, table_data in financial_data['cash_flow'].items():
                if not table_data.get('rows'):
                    continue
                
                # Extract key cash flow metrics
                for row in table_data['rows']:
                    if not row or len(row) < 5:
                        continue
                    
                    # Operating Cash Flow
                    if any(cf in row[0] for cf in ['CFO', 'Cash from Operating']):
                        cash_flow['operating_cf'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    # Investing Cash Flow
                    elif any(cf in row[0] for cf in ['CFI', 'Cash from Investing']):
                        cash_flow['investing_cf'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    # Financing Cash Flow
                    elif any(cf in row[0] for cf in ['CFA', 'Cash from Financing']):
                        cash_flow['financing_cf'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    # Net Cash Flow
                    elif any(cf in row[0] for cf in ['NCF', 'Net Cash Flow']):
                        cash_flow['net_cf'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    # Ending Cash Balance
                    elif any(cf in row[0] for cf in ['Cash And Cash Equivalent End', 'Cash Plus Cash Eqv']):
                        cash_flow['ending_cash'] = {
                            'label': row[0],
                            'cagr_3yr': row[1] if len(row) > 1 else 'N/A',
                            'cagr_5yr': row[2] if len(row) > 2 else 'N/A',
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
            
            # Add cash flow data to financial summary
            if cash_flow:
                key_data['financial_summary']['cash_flow'] = cash_flow
        
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
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    elif 'ROA' in row[0]:
                        ratios['roa'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    elif any(d in row[0] for d in ['Debt to Equity', 'DEBT_CE']):
                        ratios['debt_equity'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
                        }
                    elif 'Price' in row[0] and 'BV' in row[0]:
                        ratios['price_to_book'] = {
                            'label': row[0],
                            'recent_values': [val for val in row[3:8] if val],
                            'headers': table_data['headers'][3:8] if len(table_data['headers']) >= 8 else table_data['headers'][3:]
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
        
        # Oscillators - New addition
        if 'oscillators' in tech_data:
            key_data['technical_summary']['oscillators'] = tech_data['oscillators']
        
        # Moving averages
        if 'moving_averages' in tech_data:
            key_data['technical_summary']['moving_averages'] = tech_data['moving_averages']
        
        # Extract Pivot Points - New addition
        if 'pivot_points' in tech_data:
            pivot_points = {}
            
            # Process each pivot point methodology
            for method, levels in tech_data['pivot_points'].items():
                pivot_points[method] = {}
                for level_type, value in levels.items():
                    # Clean up the values for consistency
                    try:
                        if isinstance(value, str):
                            value = float(value.replace(',', ''))
                        pivot_points[method][level_type] = value
                    except (ValueError, TypeError):
                        pivot_points[method][level_type] = value
            
            if pivot_points:
                key_data['technical_summary']['pivot_points'] = pivot_points
        
        # Support and resistance levels
        if 'support_resistance' in tech_data:
            support = tech_data['support_resistance'].get('support', [])
            resistance = tech_data['support_resistance'].get('resistance', [])
            
            key_data['technical_summary']['support_resistance'] = {
                'support': support,
                'resistance': resistance
            }
        
        # Performance metrics
        if 'performance' in tech_data:
            key_data['technical_summary']['performance'] = {}
            for period, data in tech_data['performance'].items():
                # Check if data is a dictionary with percentage key
                if isinstance(data, dict) and 'percentage' in data:
                    key_data['technical_summary']['performance'][period] = data['percentage']
                else:
                    key_data['technical_summary']['performance'][period] = data
        
        # Beta values - New addition
        if 'beta' in tech_data:
            key_data['technical_summary']['beta'] = tech_data['beta']
    
    # Extract price history data
    if 'price_history_data' in data:
        # Daily price data - New addition
        if 'daily_data' in data['price_history_data']:
            key_data['price_history'] = {
                'daily_data': data['price_history_data']['daily_data']
            }
        
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

def display_stock_analysis_report(report):
    """
    Display the stock analysis report using Streamlit components
    
    Args:
        report: StockAnalysisReport object containing the analysis
    """
    if report is None:
        st.error("Could not generate analysis report")
        return
    
    # Display company info
    st.title(f"Investment Analysis Report: {report.company_name} ({report.stock_symbol})")
    
    # Use columns for metrics to save vertical space
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"₹{report.current_price}")
    with col2:
        st.metric("Analysis Date", report.date)
    
    # Display recommendation prominently
    rating = report.recommendation.rating
    rating_colors = {
        "STRONG BUY": "darkgreen",
        "BUY": "green",
        "HOLD": "orange",
        "SELL": "red",
        "STRONG SELL": "darkred"
    }
    rating_color = rating_colors.get(rating, "gray")
    
    st.markdown(f"## <span style='color:{rating_color};'>RECOMMENDATION: {rating}</span>", unsafe_allow_html=True)
    st.markdown(f"**Rationale:** {report.recommendation.rationale}")
    
    # Put targets and factors in two columns to save space
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Targets")
        targets_data = []
        for target in report.recommendation.price_targets:
            targets_data.append({
                "Scenario": target.scenario,
                "Target Price": f"₹{target.price}",
                "Timeframe": target.timeframe
            })
        st.table(pd.DataFrame(targets_data))
    
    with col2:
        # Supporting factors
        st.subheader("Key Supporting Factors")
        for factor in report.recommendation.supporting_factors:
            st.markdown(f"* **{factor}**")
        
        # Risk factors
        st.subheader("Risk Factors")
        for factor in report.recommendation.risk_factors:
            st.markdown(f"* **{factor}**")
    
    # Create and display price projection chart
    st.subheader("Price Projection Chart")
    projection_chart = create_price_projection_chart(report)
    if projection_chart:
        st.pyplot(projection_chart)
    else:
        st.warning("Couldn't generate price projection chart. Check data format.")
    
    # Detailed analysis sections using tabs for better organization
    tabs = st.tabs(["Fundamental Analysis", "Technical Analysis", "Cash Flow Analysis", "Historical Performance"])
    
    with tabs[0]:
        # Display key metrics in a table
        st.subheader("Key Financial Metrics")
        metrics_data = [{
            "Metric": metric.name,
            "Value": metric.value,
            "Interpretation": metric.interpretation
        } for metric in report.fundamental_analysis.key_ratios]
        st.table(pd.DataFrame(metrics_data))
        
        # Display revenue and profit trends
        st.subheader("Revenue Trend")
        st.markdown(report.fundamental_analysis.revenue_trend)
        
        st.subheader("Profit Trend")
        st.markdown(report.fundamental_analysis.profit_trend)
        
        st.subheader("Financial Health")
        st.markdown(report.fundamental_analysis.financial_health)
        
        st.subheader("Complete Fundamental Analysis")
        st.markdown(report.fundamental_analysis.full_analysis)
    
    with tabs[1]:
        # Display technical trend and levels
        st.subheader("Current Trend")
        st.markdown(f"**Direction:** {report.technical_analysis.trend_direction.capitalize()}")
        
        # Display support and resistance levels
        st.subheader("Key Price Levels")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Support Levels:**")
            for level in report.technical_analysis.key_levels["support"]:
                st.markdown(f"* ₹{level}")
        
        with col2:
            st.markdown("**Resistance Levels:**")
            for level in report.technical_analysis.key_levels["resistance"]:
                st.markdown(f"* ₹{level}")
        
        # Display pivot points if available - New section
        if hasattr(report, 'pivot_points') and report.pivot_points:
            st.subheader("Pivot Points")
            
            # Get list of available pivot point methods
            pivot_methods = list(report.pivot_points.keys())
            # Create tabs for each pivot point method
            pivot_tabs = st.tabs(pivot_methods)
            
            # Display each method in its own tab
            for i, method in enumerate(pivot_methods):
                with pivot_tabs[i]:
                    pivot_data = []
                    for level, value in report.pivot_points[method].items():
                        pivot_data.append({
                            "Level": level,
                            "Value": f"₹{value}"
                        })
                    # Sort pivot data to show in logical order (resistances, pivot, supports)
                    pivot_data_sorted = sorted(pivot_data, key=lambda x: x["Level"])
                    st.table(pd.DataFrame(pivot_data_sorted))
        
        # Display momentum indicators
        st.subheader("Momentum Indicators")
        indicators_data = [{
            "Indicator": indicator.name,
            "Value": indicator.value,
            "Interpretation": indicator.interpretation
        } for indicator in report.technical_analysis.momentum_indicators]
        st.table(pd.DataFrame(indicators_data))
        
        st.subheader("Volume Analysis")
        st.markdown(report.technical_analysis.volume_analysis)
        
        st.subheader("Complete Technical Analysis")
        st.markdown(report.technical_analysis.full_analysis)
    
    with tabs[2]:
        # Cash Flow Analysis Tab - New tab
        if hasattr(report, 'cash_flow_analysis') and report.cash_flow_analysis:
            # Display cash flow metrics in a table if available
            if hasattr(report.cash_flow_analysis, 'cash_flow_metrics') and report.cash_flow_analysis.cash_flow_metrics:
                st.subheader("Key Cash Flow Metrics")
                cf_metrics_data = []
                for metric, value in report.cash_flow_analysis.cash_flow_metrics.items():
                    cf_metrics_data.append({
                        "Metric": metric,
                        "Value": value
                    })
                st.table(pd.DataFrame(cf_metrics_data))
            
            # Display operating cash flow trend
            st.subheader("Operating Cash Flow Trend")
            st.markdown(report.cash_flow_analysis.operating_cf_trend)
            
            # Display free cash flow
            st.subheader("Free Cash Flow")
            st.markdown(report.cash_flow_analysis.free_cash_flow)
            
            # Display investing activities
            st.subheader("Investing Activities")
            st.markdown(report.cash_flow_analysis.investing_activities)
            
            # Display financing activities
            st.subheader("Financing Activities")
            st.markdown(report.cash_flow_analysis.financing_activities)
            
            # Display complete cash flow analysis
            st.subheader("Complete Cash Flow Analysis")
            st.markdown(report.cash_flow_analysis.full_analysis)
        else:
            st.info("Cash flow analysis not available for this report.")
    
    with tabs[3]:
        if report.price_history_analysis:
            st.markdown(report.price_history_analysis)
        else:
            st.info("Price history analysis data not available.")
    
    # Add download options at the bottom of the page
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{report.stock_symbol.replace('.', '_')}_analysis_{timestamp}"
        
        # Create JSON report
        import json
        
        # Convert Pydantic model to dict for JSON serialization
        def pydantic_model_to_dict(model):
            """Convert a Pydantic model to a dictionary for JSON serialization"""
            if hasattr(model, 'dict'):
                return model.dict()
            elif hasattr(model, 'model_dump'):  # For newer Pydantic versions
                return model.model_dump()
            else:
                # Fallback to manual conversion
                result = {}
                for key, value in model.__dict__.items():
                    if not key.startswith('_'):
                        if hasattr(value, 'dict') or hasattr(value, 'model_dump') or hasattr(value, '__dict__'):
                            result[key] = pydantic_model_to_dict(value)
                        elif isinstance(value, list):
                            result[key] = [
                                pydantic_model_to_dict(item) if hasattr(item, 'dict') or hasattr(item, 'model_dump') else item 
                                for item in value
                            ]
                        else:
                            result[key] = value
                return result
        
        # Convert to dictionary and then to JSON
        report_dict = pydantic_model_to_dict(report)
        json_report = json.dumps(report_dict, indent=2, default=str)  # Use default=str to handle dates
        
        # Create a formatted text report (markdown format)
        text_report = f"""
        # Investment Analysis Report: {report.company_name} ({report.stock_symbol})
        
        Date: {report.date}
        Current Price: ₹{report.current_price}
        
        ## RECOMMENDATION: {report.recommendation.rating}
        
        {report.recommendation.rationale}
        
        ### Price Targets:
        {chr(10).join([f"- {t.scenario}: ₹{t.price} ({t.timeframe})" for t in report.recommendation.price_targets])}
        
        ### Supporting Factors:
        {chr(10).join([f"- {factor}" for factor in report.recommendation.supporting_factors])}
        
        ### Risk Factors:
        {chr(10).join([f"- {factor}" for factor in report.recommendation.risk_factors])}
        
        ## Fundamental Analysis
        
        ### Revenue Trend
        {report.fundamental_analysis.revenue_trend}
        
        ### Profit Trend
        {report.fundamental_analysis.profit_trend}
        
        ### Financial Health
        {report.fundamental_analysis.financial_health}
        
        ### Complete Fundamental Analysis
        {report.fundamental_analysis.full_analysis}
        
        ## Technical Analysis
        
        ### Current Trend
        {report.technical_analysis.trend_direction.capitalize()}
        
        ### Key Support Levels
        {chr(10).join([f"- ₹{level}" for level in report.technical_analysis.key_levels["support"]])}
        
        ### Key Resistance Levels
        {chr(10).join([f"- ₹{level}" for level in report.technical_analysis.key_levels["resistance"]])}
        
        ### Volume Analysis
        {report.technical_analysis.volume_analysis}
        
        ### Complete Technical Analysis
        {report.technical_analysis.full_analysis}
        
        ## Cash Flow Analysis
        
        ### Operating Cash Flow Trend
        {report.cash_flow_analysis.operating_cf_trend if hasattr(report, 'cash_flow_analysis') and report.cash_flow_analysis else "Not available"}
        
        ### Free Cash Flow
        {report.cash_flow_analysis.free_cash_flow if hasattr(report, 'cash_flow_analysis') and report.cash_flow_analysis else "Not available"}
        
        ### Complete Cash Flow Analysis
        {report.cash_flow_analysis.full_analysis if hasattr(report, 'cash_flow_analysis') and report.cash_flow_analysis else "Not available"}
        
        ## Historical Performance Analysis
        {report.price_history_analysis}
        """
        
        # Add download buttons in a more compact layout
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="Download JSON Report",
                data=json_report,
                file_name=f"{report_filename}.json",
                mime="application/json"
            )
        with dl_col2:
            st.download_button(
                label="Download Text Report",
                data=text_report,
                file_name=f"{report_filename}.md",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"Error preparing download options: {str(e)}")

def create_price_projection_chart(report):
    """
    Create a simplified price projection chart showing only historical prices, 
    current price, and projected price targets.
    
    Args:
        report: StockAnalysisReport object containing analysis data
        
    Returns:
        Matplotlib figure object with the chart
    """
    # Extract current price
    try:
        current_price = float(report.current_price)
    except (ValueError, TypeError):
        current_price = 0
        
    current_date = datetime.now()
    
    # Create targets dictionary from price targets
    targets = {}
    for target in report.recommendation.price_targets:
        # Extract first word of scenario (bearish, base, bullish)
        scenario = target.scenario.lower().split()[0]
        
        # Extract timeframe in months
        timeframe_match = re.search(r"(\d+)\s*month", target.timeframe, re.IGNORECASE)
        if timeframe_match:
            months = int(timeframe_match.group(1))
        else:
            # Try to find years and convert
            timeframe_match = re.search(r"(\d+)\s*year", target.timeframe, re.IGNORECASE)
            if timeframe_match:
                months = int(timeframe_match.group(1)) * 12
            else:
                # Default values based on scenario
                defaults = {'bearish': 6, 'base': 12, 'bullish': 24}
                months = defaults.get(scenario, 12)
        
        # Validate target price
        try:
            target_price = float(target.price)
            # Apply sanity check - target shouldn't be more than 2x or less than 0.5x current price
            if target_price > current_price * 2:
                target_price = current_price * (1.3 if scenario == 'bullish' else 
                                              1.1 if scenario == 'base' else 0.9)
            if target_price < current_price * 0.5:
                target_price = current_price * (0.9 if scenario == 'bearish' else 
                                              0.95 if scenario == 'base' else 1.1)
        except (ValueError, TypeError):
            # Fallback values if price can't be parsed
            target_price = current_price * (0.9 if scenario == 'bearish' else 
                                          1.1 if scenario == 'base' else 1.3)
            
        targets[scenario] = {
            'price': target_price,
            'months': months
        }
    
    # Create the figure with a clean, simple design
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    
    # Plot historical prices if available
    try:
        if hasattr(report, 'price_history') and report.price_history:
            history_df = pd.DataFrame(report.price_history)
            
            # Find date and close columns with flexible column name matching
            date_col = next((col for col in history_df.columns if 'date' in col.lower()), None)
            close_col = next((col for col in history_df.columns if 'close' in col.lower()), None)
            
            if not close_col:  # Try alternative column names
                close_col = next((col for col in history_df.columns 
                                if col.lower() in ['price', 'ltp', 'last']), None)
            
            if date_col and close_col:
                # Convert date strings to datetime objects (handle various formats)
                try:
                    history_df[date_col] = pd.to_datetime(history_df[date_col], errors='coerce')
                except:
                    pass  # If conversion fails, pandas will handle it
                
                # Convert price strings to float values
                if isinstance(history_df[close_col].iloc[0], str):
                    history_df[close_col] = history_df[close_col].str.replace(',', '').astype(float)
                else:
                    history_df[close_col] = pd.to_numeric(history_df[close_col], errors='coerce')
                
                # Sort by date and drop any rows with missing values
                history_df = history_df.sort_values(by=date_col).dropna(subset=[date_col, close_col])
                
                # Downsample for cleaner visualization if too many points
                if len(history_df) > 40:
                    downsample_factor = len(history_df) // 40 + 1
                    history_df = history_df.iloc[::downsample_factor].copy()
                
                # Plot historical prices
                ax.plot(history_df[date_col], history_df[close_col], 
                        label='Historical Price', color='blue', linewidth=1.5)
                
                # Use the last historical date as the starting point if available
                if not history_df.empty:
                    current_date = history_df[date_col].iloc[-1]
    except Exception as e:
        print(f"Error plotting historical data: {e}")
    
    # Plot the current price point
    ax.scatter([current_date], [current_price], color='black', s=70, zorder=5, label='Current Price')
    
    # Plot price projections
    colors = {'bearish': 'red', 'base': 'green', 'bullish': 'purple'}
    
    for scenario, target_info in targets.items():
        if 'price' in target_info and 'months' in target_info:
            target_price = target_info['price']
            target_date = current_date + pd.DateOffset(months=target_info['months'])
            
            # Plot the target point
            ax.scatter([target_date], [target_price], 
                       color=colors.get(scenario, 'blue'), s=70, zorder=5)
            
            # Draw a line from current price to target
            ax.plot([current_date, target_date], [current_price, target_price], 
                    color=colors.get(scenario, 'blue'), linestyle='--', 
                    alpha=0.7, linewidth=1.5,
                    label=f"{scenario.capitalize()} Target: ₹{target_price:.2f}")
            
            # Add an annotation for the target price
            ax.annotate(f"₹{target_price:.2f}", (target_date, target_price), 
                        textcoords="offset points", xytext=(0,7), ha='center',
                        fontsize=10)
    
    # Add labels and title
    ax.set_title(f"{report.company_name} ({report.stock_symbol}) - Price Projections", 
                fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (₹)", fontsize=12)
    ax.grid(True, alpha=0.3)  # Light grid for better readability
    
    # Format the date axis
    plt.gcf().autofmt_xdate()
    
    # Set reasonable y-axis limits
    y_values = [current_price] + [t['price'] for t in targets.values()]
    y_min = min(y_values) * 0.9  # 10% below the lowest point
    y_max = max(y_values) * 1.1  # 10% above the highest point
    ax.set_ylim(y_min, y_max)
    
    # Add a simple legend
    ax.legend(loc='best', fontsize=10, framealpha=0.7)
    
    return fig

def main():
    st.title("📈 Advanced Stock Analysis Report Generator")
    st.write("Generate a detailed stock analysis report using structured AI analysis")
    
    # File uploader widget
    st.subheader("Upload Data Source")
    
    # Create a file uploader that accepts only JSON files
    uploaded_file = st.file_uploader("Upload a stock data JSON file", type=['json'])
    
    # Model selection
    models = ["phi3:medium-128k", "mistral:7b-instruct-q6_K", "qwen2.5:7b", "mixtral-offloaded:latest","hf.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF:Q6_K","qwen2.5-32b-Stock-Research:latest"]
    selected_model = st.selectbox("Select LLM model", models)
    
    # Display memory efficiency settings based on model
    context_window = get_context_window_limit(selected_model)
    if context_window <= 16000:
        memory_settings = "Low Memory Mode: 3 quarters, 2 years of data"
    elif context_window <= 32000:
        memory_settings = "Medium Memory Mode: 4 quarters, 3 years of data"
    else:
        memory_settings = "High Memory Mode: 5 quarters, 5 years of data"
        
    st.info(f"Model context window: ~{context_window:,} tokens - {memory_settings}")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Load JSON data from uploaded file
            import json
            data = json.load(uploaded_file)
            
            # Display success message with company info
            st.success(f"Successfully loaded data for {data['metadata']['company_name']} ({data['metadata']['stock_symbol']})")
            
            # Display company info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"₹{data.get('technical_data', {}).get('current_price', 'N/A')}")
            with col2:
                if "stock_info" in data and "marketCap" in data["stock_info"]:
                    market_cap = data["stock_info"].get("marketCap", "N/A")
                    if market_cap != "N/A":
                        try:
                            market_cap = float(market_cap)
                            market_cap_display = f"₹{market_cap/10000000:.2f} Cr" if market_cap > 10000000 else f"₹{market_cap:,}"
                            st.metric("Market Cap", market_cap_display)
                        except:
                            st.metric("Market Cap", "N/A")
            with col3:
                if "technical_data" in data and "indicators" in data["technical_data"]:
                    rsi = data["technical_data"]["indicators"].get("RSI", "N/A")
                    st.metric("RSI", rsi)
            
            # Add memory usage estimator
            file_size_kb = len(str(data)) / 1024
            st.caption(f"Input data size: {file_size_kb:.1f} KB")
            
            # Generate analysis button
            if st.button(f"Generate Analysis using {selected_model}"):
                with st.spinner("Analyzing data... This may take several minutes depending on your hardware."):
                    # Use our memory-optimized structured analysis function
                    analysis_report = generate_structured_analysis(data, selected_model)
                    
                    if analysis_report:
                        # Display the analysis using our display function
                        display_stock_analysis_report(analysis_report)
                    else:
                        st.error("Failed to generate analysis. Please try a different model or check if your LLM service is running.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            st.info("Please ensure the uploaded file is a valid JSON file with the expected structure.")
    else:
        # Show instructions when no file is uploaded
        st.info("Please upload a JSON file with stock data to begin analysis.")
        
        # Optionally, show sample JSON structure
        with st.expander("Expected JSON Structure"):
            st.code("""
            {
                "metadata": {
                    "company_name": "Company Name",
                    "stock_symbol": "SYMBOL.NS",
                    "timestamp": "YYYY-MM-DD HH:MM:SS"
                },
                "financial_data": {
                    "quarterly": { ... },
                    "annual": { ... },
                    "balance_sheet": { ... },
                    "ratios": { ... },
                    "cash_flow": { ... }
                },
                "technical_data": {
                    "current_price": "1234.56",
                    "indicators": { ... },
                    "performance": { ... },
                    "moving_averages": { ... },
                    "oscillators": { ... },
                    "pivot_points": { ... }
                },
                "price_history_data": {
                    "daily_data": [ ... ],
                    "returns_comparison": [ ... ],
                    "returns_seasonality": [ ... ]
                }
            }
            """, language="json")

if __name__ == "__main__":
    main()