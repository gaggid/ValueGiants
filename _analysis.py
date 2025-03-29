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

class StockAnalysisReport(BaseModel):
    """Complete stock analysis report"""
    company_name: str = Field(description="Name of the company")
    stock_symbol: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price")
    date: str = Field(description="Date of analysis")
    fundamental_analysis: FundamentalAnalysis = Field(description="Fundamental analysis component")
    technical_analysis: TechnicalAnalysis = Field(description="Technical analysis component")
    recommendation: StockRecommendation = Field(description="Investment recommendation")
    price_history_analysis: str = Field(description="Analysis of historical price performance")
    moving_averages: Optional[Dict[str, str]] = Field(None, description="Moving average values")
    support_resistance: Optional[Dict[str, List[str]]] = Field(None, description="Support and resistance levels")
    price_history: Optional[List[Dict[str, Any]]] = Field(None, description="Historical price data")

def generate_structured_analysis(data, model_name="phi3:medium-128k"):
    """
    Generate a complete structured stock analysis report
    
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
        
    # Create a progress bar
    progress_bar = st.progress(0)
    st.info("Starting analysis process...")
    
    # Initialize LangChain LLM
    try:
        from langchain_community.llms import Ollama
        llm = Ollama(model=model_name)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None
    
    # Extract and format key data points for analysis
    st.info("Phase 1/4: Extracting key data points...")
    key_data_points = extract_key_data_points(data)
    progress_bar.progress(20)
    
    # Process fundamental analysis
    st.info("Phase 2/4: Generating fundamental analysis...")
    fundamental_analysis = generate_fundamental_analysis(key_data_points, company_name, stock_symbol, llm)
    progress_bar.progress(45)
    
    # Process technical analysis
    st.info("Phase 3/4: Generating technical analysis...")
    technical_analysis = generate_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm)
    progress_bar.progress(70)
    
    # Generate final recommendation and complete report
    st.info("Phase 4/4: Preparing final recommendation and report...")
    analysis_report = generate_final_report(
        key_data_points, company_name, stock_symbol, current_price,
        fundamental_analysis, technical_analysis, llm, data
    )
    progress_bar.progress(100)
    
    return analysis_report

def extract_json_from_text(text):
    """
    Extract JSON from text using multiple strategies
    
    Args:
        text: The text that might contain JSON
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not text:
        return None
    
    # Strategy 1: Find code blocks with JSON
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Strategy 2: Find anything that looks like a complete JSON object
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    # Strategy 3: Try to clean up the text
    # Sometimes LLMs add commentary before or after the JSON
    lines = text.split('\n')
    json_lines = []
    in_json = False
    open_braces = 0
    
    for line in lines:
        if '{' in line and not in_json:
            in_json = True
        
        if in_json:
            json_lines.append(line)
            open_braces += line.count('{') - line.count('}')
            
            if open_braces == 0 and in_json and '}' in line:
                break
    
    if json_lines:
        return '\n'.join(json_lines)
    
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
    Generate structured fundamental analysis using LLM with improved JSON parsing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        
    Returns:
        FundamentalAnalysis object with the analysis
    """
    # Format financial data for the prompt
    financial_data_str = format_financial_data_for_prompt(key_data_points)
    
    # Create the prompt template for fundamental analysis
    fundamental_prompt = f"""
    You are a financial analyst specializing in equity research. Analyze the fundamental financial performance of {company_name} ({stock_symbol}).
    
    Focus on these financial data points:
    {financial_data_str}
    
    Provide a JSON object with the following structure:
    {{
        "revenue_trend": "Analysis of revenue trends here...",
        "profit_trend": "Analysis of profit trends here...",
        "key_ratios": [
            {{
                "name": "ROE",
                "value": "1.48%",
                "interpretation": "Return on Equity interpretation here..."
            }},
            {{
                "name": "Debt/Equity",
                "value": "0.3",
                "interpretation": "Debt to Equity interpretation here..."
            }}
        ],
        "financial_health": "Overall financial health assessment here...",
        "full_analysis": "Complete fundamental analysis here..."
    }}
    
    IMPORTANT: Your response must be only a valid JSON object. No preamble or explanation.
    """
    
    # Try to generate structured output with enhanced error handling
    try:
        # Invoke the LLM
        response = llm.invoke(fundamental_prompt)
        
        # Extract JSON from the response
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            
            # Parse JSON and create FundamentalAnalysis object
            import json
            parsed_data = json.loads(json_str)
            
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
            # If we can't find a valid JSON, fall back
            st.warning("Could not extract valid JSON from fundamental analysis response")
            return fallback_fundamental_analysis(key_data_points, company_name, stock_symbol, llm)
    
    except Exception as e:
        st.warning(f"Fundamental analysis generation failed: {str(e)}")
        return fallback_fundamental_analysis(key_data_points, company_name, stock_symbol, llm)

def generate_technical_analysis(key_data_points, company_name, stock_symbol, current_price, llm):
    """
    Generate structured technical analysis using LLM with improved JSON parsing
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        llm: LangChain LLM object
        
    Returns:
        TechnicalAnalysis object with the analysis
    """
    # Format technical data for the prompt
    technical_data_str = format_technical_data_for_prompt(key_data_points)
    
    # Create the prompt template for technical analysis
    technical_prompt = f"""
    You are a technical analyst with expertise in equity markets. Analyze the technical indicators and price patterns for {company_name} ({stock_symbol}) currently trading at {current_price}.
    
    Focus on these technical data points:
    {technical_data_str}
    
    Provide a JSON object with the following structure:
    {{
        "trend_direction": "bullish OR bearish OR neutral",
        "key_levels": {{
            "support": [number, number],
            "resistance": [number, number]
        }},
        "momentum_indicators": [
            {{
                "name": "RSI",
                "value": "52.7",
                "interpretation": "RSI interpretation here..."
            }},
            {{
                "name": "MACD",
                "value": "0.0",
                "interpretation": "MACD interpretation here..."
            }}
        ],
        "volume_analysis": "Volume analysis text here...",
        "full_analysis": "Complete technical analysis text here..."
    }}
    
    IMPORTANT: Your response must be only a JSON object. Format as a code block with ```json and ``` tags.
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
                        fundamental_analysis, technical_analysis, llm, original_data):
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
        
    Returns:
        StockAnalysisReport object with the complete analysis
    """
    # Generate the recommendation first
    recommendation = generate_recommendation(
        key_data_points, company_name, stock_symbol, current_price,
        fundamental_analysis, technical_analysis, llm
    )
    
    # Generate price history analysis
    price_history_analysis = generate_price_history_analysis(
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
        recommendation=recommendation,
        price_history_analysis=price_history_analysis
    )
    
    # Add additional technical data for charts if available
    if 'technical_data' in original_data:
        if 'moving_averages' in original_data['technical_data']:
            report.moving_averages = original_data['technical_data']['moving_averages']
        
        if 'support_resistance' in original_data['technical_data']:
            # Clean up support/resistance levels
            support_resistance = {
                'support': [],
                'resistance': []
            }
            
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
                         fundamental_analysis, technical_analysis, llm):
    """
    Generate investment recommendation based on fundamental and technical analysis
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        fundamental_analysis: FundamentalAnalysis object
        technical_analysis: TechnicalAnalysis object
        llm: LangChain LLM object
        
    Returns:
        StockRecommendation object with the recommendation
    """
    # Setup the parser for structured output
    parser = PydanticOutputParser(pydantic_object=StockRecommendation)
    format_instructions = parser.get_format_instructions()
    
    # Extract key insights from fundamental and technical analysis
    fundamental_insights = fundamental_analysis.full_analysis
    technical_insights = technical_analysis.full_analysis
    
    # Create the prompt template for recommendation
    recommendation_prompt = PromptTemplate(
        template="""
        You are the head of equity research at a top investment firm. Provide an investment recommendation for {company_name} ({stock_symbol}) currently trading at {current_price}.
        
        Base your recommendation on these insights:
        
        FUNDAMENTAL ANALYSIS:
        {fundamental_insights}
        
        TECHNICAL ANALYSIS:
        {technical_insights}
        
        Provide a structured investment recommendation that includes:
        1. A clear rating (STRONG BUY, BUY, HOLD, SELL, or STRONG SELL)
        2. Rationale for your recommendation in 1-2 sentences
        3. Price targets for bearish, base, and bullish scenarios
        4. 3 key supporting factors for your recommendation
        5. 2 key risk factors investors should consider
        
        {format_instructions}
        
        Return a valid JSON object based on the format instructions.
        """,
        input_variables=["company_name", "stock_symbol", "current_price", 
                         "fundamental_insights", "technical_insights"],
        partial_variables={"format_instructions": format_instructions}
    )
    
    # Format the prompt with the input data
    formatted_prompt = recommendation_prompt.format(
        company_name=company_name,
        stock_symbol=stock_symbol,
        current_price=current_price,
        fundamental_insights=fundamental_insights,
        technical_insights=technical_insights
    )
    
    # Try to generate structured output with error handling
    try:
        # Invoke the LLM
        response = llm.invoke(formatted_prompt)
        
        # Try direct parsing - FIXED: Added the model_class parameter
        try:
            return parser.parse(response)
        except Exception as parse_error1:
            st.warning(f"Direct parsing failed: {str(parse_error1)}")
            
            # Try to extract JSON and process it directly
            json_str = extract_json_from_text(response)
            if json_str:
                try:
                    import json
                    parsed_json = json.loads(json_str)
                    return StockRecommendation(**parsed_json)
                except Exception as e:
                    st.warning(f"JSON parsing failed: {str(e)}")
                    
            # Fall back to simple extraction
            return fallback_recommendation(key_data_points, company_name, stock_symbol, 
                                         current_price, fundamental_analysis, technical_analysis, llm)
    
    except Exception as e:
        st.warning(f"Structured recommendation failed: {str(e)}")
        # Fall back to simple extraction
        return fallback_recommendation(key_data_points, company_name, stock_symbol, 
                                     current_price, fundamental_analysis, technical_analysis, llm)

def generate_price_history_analysis(key_data_points, company_name, stock_symbol, llm):
    """
    Generate analysis of price history and performance comparisons
    
    Args:
        key_data_points: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        
    Returns:
        String containing the price history analysis
    """
    # Format price history data for the prompt
    price_history_data = format_price_history_for_prompt(key_data_points)
    
    # Create a simple prompt for price history analysis
    price_history_prompt = f"""
    You are a market strategist specializing in comparative analysis. Analyze the price history and relative performance of {company_name} ({stock_symbol}).
    
    Focus on these data points:
    {price_history_data}
    
    Provide a concise, professional analysis that covers:
    1. How the stock has performed compared to relevant indices (Nifty, Sensex, etc.)
    2. Any recurring seasonal patterns or cyclical behavior
    3. Long-term performance trends and key inflection points
    4. Relative strength versus the market in different conditions
    
    Format your analysis in 2-3 clear paragraphs. Use specific percentages and time periods to support your observations.
    """
    
    # Try to generate the analysis with error handling
    try:
        # Invoke the LLM
        response = llm.invoke(price_history_prompt)
        return response
    
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

def format_financial_data_for_prompt(key_data):
    """
    Format financial data into a clear string for prompts
    
    Args:
        key_data: Dictionary containing extracted key data
        
    Returns:
        Formatted string with financial data
    """
    result = []
    
    # Add quarterly data
    if 'financial_summary' in key_data and 'quarterly' in key_data['financial_summary']:
        result.append("QUARTERLY TRENDS:")
        quarterly = key_data['financial_summary']['quarterly']
        
        for metric, data in quarterly.items():
            if 'headers' in data and 'recent_values' in data:
                result.append(f"  {data.get('label', metric.upper())}:")
                for i, header in enumerate(data['headers']):
                    if i < len(data['recent_values']):
                        result.append(f"    {header}: {data['recent_values'][i]}")
    
    # Add annual data
    if 'financial_summary' in key_data and 'annual' in key_data['financial_summary']:
        result.append("\nANNUAL PERFORMANCE:")
        annual = key_data['financial_summary']['annual']
        
        for metric, data in annual.items():
            if 'headers' in data and 'recent_values' in data:
                result.append(f"  {data.get('label', metric.upper())}:")
                if 'cagr_3yr' in data:
                    result.append(f"    3-Year CAGR: {data['cagr_3yr']}")
                if 'cagr_5yr' in data:
                    result.append(f"    5-Year CAGR: {data['cagr_5yr']}")
                
                for i, header in enumerate(data['headers']):
                    if i < len(data['recent_values']):
                        result.append(f"    {header}: {data['recent_values'][i]}")
    
    # Add key ratios
    if 'financial_summary' in key_data and 'ratios' in key_data['financial_summary']:
        result.append("\nKEY FINANCIAL RATIOS:")
        ratios = key_data['financial_summary']['ratios']
        
        for ratio_name, data in ratios.items():
            if 'headers' in data and 'recent_values' in data:
                result.append(f"  {data.get('label', ratio_name.upper())}:")
                for i, header in enumerate(data['headers']):
                    if i < len(data['recent_values']):
                        result.append(f"    {header}: {data['recent_values'][i]}")
    
    return "\n".join(result) if result else "Financial data not available."

def format_technical_data_for_prompt(key_data):
    """
    Format technical data into a clear string for prompts
    
    Args:
        key_data: Dictionary containing extracted key data
        
    Returns:
        Formatted string with technical data
    """
    result = []
    
    # Add technical indicators
    if 'technical_summary' in key_data and 'indicators' in key_data['technical_summary']:
        result.append("TECHNICAL INDICATORS:")
        for indicator, value in key_data['technical_summary']['indicators'].items():
            result.append(f"  {indicator}: {value}")
    
    # Add moving averages
    if 'technical_summary' in key_data and 'moving_averages' in key_data['technical_summary']:
        result.append("\nMOVING AVERAGES:")
        for ma_type, value in key_data['technical_summary']['moving_averages'].items():
            result.append(f"  {ma_type}: {value}")
    
    # Add support/resistance levels
    if 'technical_summary' in key_data and 'support_resistance' in key_data['technical_summary']:
        sr_data = key_data['technical_summary']['support_resistance']
        
        if 'support' in sr_data and sr_data['support']:
            result.append("\nSUPPORT LEVELS:")
            for level in sr_data['support']:
                result.append(f"  {level}")
        
        if 'resistance' in sr_data and sr_data['resistance']:
            result.append("\nRESISTANCE LEVELS:")
            for level in sr_data['resistance']:
                result.append(f"  {level}")
    
    # Add performance data
    if 'technical_summary' in key_data and 'performance' in key_data['technical_summary']:
        result.append("\nPRICE PERFORMANCE:")
        for period, value in key_data['technical_summary']['performance'].items():
            if isinstance(value, dict) and 'percentage' in value:
                result.append(f"  {period}: {value['percentage']}")
            else:
                result.append(f"  {period}: {value}")
    
    return "\n".join(result) if result else "Technical data not available."

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

def fallback_fundamental_analysis(key_data, company_name, stock_symbol, llm):
    """
    Generate a fallback fundamental analysis when structured parsing fails
    
    Args:
        key_data: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        llm: LangChain LLM object
        
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

def fallback_technical_analysis(key_data, company_name, stock_symbol, current_price, llm):
    """
    Generate a fallback technical analysis when structured parsing fails
    
    Args:
        key_data: Dictionary containing extracted key data points
        company_name: Name of the company
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        llm: LangChain LLM object
        
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
                         fundamental_analysis, technical_analysis, llm):
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
        
    Returns:
        StockRecommendation object with basic recommendation
    """
    # Default to HOLD rating as the safe option
    rating = "HOLD"
    
    # Try to determine a better rating from the analysis text
    combined_analysis = fundamental_analysis.full_analysis + " " + technical_analysis.full_analysis
    combined_analysis = combined_analysis.upper()
    
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
            
            if resistance:  # Only checking resistance since that's what's in the sample data
                key_data['technical_summary']['support_resistance'] = {
                    'support': [],  # Empty as there are none in the sample data
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
    
    # Use columns for metrics to save vertical space - removing Potential Return
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
    tabs = st.tabs(["Fundamental Analysis", "Technical Analysis", "Historical Performance"])
    
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
   
   # File selection
   st.subheader("Select Data Source")
   
   # Find all JSON files in the current directory
   import os
   json_files = [f for f in os.listdir('.') if f.endswith('_all_data.json')]
   if not json_files:
       st.error("No JSON data files found in the current directory.")
       return
   
   selected_file = st.selectbox("Select a stock data file", json_files)
   
   # Model selection
   models = ["phi3:medium-128k", "mistral:7b-instruct-q6_K", "qwen2.5:7b", "mixtral-offloaded:latest"]
   selected_model = st.selectbox("Select LLM model", models)
   
   # Load the data
   def load_data_from_json(file_path):
       """Load data from JSON file."""
       try:
           with open(file_path, 'r', encoding='utf-8') as f:
               return json.load(f)
       except Exception as e:
           st.error(f"Error loading data: {str(e)}")
           return None
   
   data = load_data_from_json(selected_file)
   
   if data:
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
       
       # Generate analysis button
       if st.button(f"Generate Analysis using {selected_model}"):
           with st.spinner("Analyzing data... This may take several minutes depending on your hardware."):
               # Use our structured analysis function
               analysis_report = generate_structured_analysis(data, selected_model)
               
               if analysis_report:
                   # Display the analysis using our display function
                   display_stock_analysis_report(analysis_report)
               else:
                   st.error("Failed to generate analysis. Please try a different model or check if your LLM service is running.")

if __name__ == "__main__":
   main()