"""
Event Detection System for Step 2 Predictions
Detects and analyzes market-moving events: earnings, ex-dividend, FDA approvals, etc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketEvent:
    """Standardized market event structure"""
    ticker: str
    event_type: str
    event_date: datetime
    description: str
    impact_score: float  # 1-10 scale
    direction_bias: float  # -1 to +1 (bearish to bullish)
    volatility_multiplier: float  # Expected volatility increase
    source: str
    confidence: float  # 0-100% confidence in event data

class EventDetector:
    """Comprehensive event detection and analysis system"""
    
    def __init__(self):
        self.event_types = {
            'earnings': {
                'impact_score': 7,
                'volatility_multiplier': 2.5,
                'lookback_days': 90,
                'lookahead_days': 14
            },
            'ex_dividend': {
                'impact_score': 4,
                'volatility_multiplier': 1.2,
                'lookback_days': 30,
                'lookahead_days': 7
            },
            'fda_approval': {
                'impact_score': 9,
                'volatility_multiplier': 4.0,
                'lookback_days': 180,
                'lookahead_days': 21
            },
            'product_launch': {
                'impact_score': 6,
                'volatility_multiplier': 1.8,
                'lookback_days': 60,
                'lookahead_days': 10
            },
            'merger_acquisition': {
                'impact_score': 10,
                'volatility_multiplier': 5.0,
                'lookback_days': 365,
                'lookahead_days': 30
            },
            'stock_split': {
                'impact_score': 3,
                'volatility_multiplier': 1.3,
                'lookback_days': 60,
                'lookahead_days': 14
            }
        }
        
    def detect_upcoming_events(self, ticker: str, days_ahead: int = 14) -> List[MarketEvent]:
        """Detect all upcoming events for a ticker"""
        events = []
        end_date = datetime.now() + timedelta(days=days_ahead)
        
        # Check each event type
        events.extend(self._detect_earnings_events(ticker, end_date))
        events.extend(self._detect_dividend_events(ticker, end_date))
        events.extend(self._detect_fda_events(ticker, end_date))
        events.extend(self._detect_product_events(ticker, end_date))
        
        # Sort by impact score and date proximity
        events.sort(key=lambda e: (e.impact_score, -abs((e.event_date - datetime.now()).days)), reverse=True)
        
        return events
    
    def _detect_earnings_events(self, ticker: str, end_date: datetime) -> List[MarketEvent]:
        """Detect upcoming earnings announcements"""
        events = []
        
        try:
            # For MVP, use estimated earnings dates based on historical patterns
            # In production, this would integrate with earnings calendar APIs
            earnings_date = self._estimate_next_earnings_date(ticker)
            
            if earnings_date and earnings_date <= end_date:
                # Analyze historical earnings impact
                historical_impact = self._analyze_historical_earnings_impact(ticker)
                
                event = MarketEvent(
                    ticker=ticker,
                    event_type='earnings',
                    event_date=earnings_date,
                    description=f"Q{self._get_quarter(earnings_date)} {earnings_date.year} Earnings Report",
                    impact_score=historical_impact['impact_score'],
                    direction_bias=historical_impact['direction_bias'],
                    volatility_multiplier=historical_impact['volatility_multiplier'],
                    source='estimated_calendar',
                    confidence=historical_impact['confidence']
                )
                events.append(event)
                
        except Exception as e:
            pass  # Fail silently for MVP
            
        return events
    
    def _detect_dividend_events(self, ticker: str, end_date: datetime) -> List[MarketEvent]:
        """Detect upcoming ex-dividend dates"""
        events = []
        
        try:
            # Estimate ex-dividend date based on typical patterns
            ex_div_date = self._estimate_next_ex_dividend_date(ticker)
            
            if ex_div_date and ex_div_date <= end_date:
                dividend_yield = self._estimate_dividend_yield(ticker)
                
                event = MarketEvent(
                    ticker=ticker,
                    event_type='ex_dividend',
                    event_date=ex_div_date,
                    description=f"Ex-Dividend Date (Est. yield: {dividend_yield:.2f}%)",
                    impact_score=max(2, min(6, dividend_yield)),  # Higher yield = higher impact
                    direction_bias=-0.5,  # Generally bearish on ex-div date
                    volatility_multiplier=1.2,
                    source='estimated_pattern',
                    confidence=60
                )
                events.append(event)
                
        except Exception as e:
            pass
            
        return events
    
    def _detect_fda_events(self, ticker: str, end_date: datetime) -> List[MarketEvent]:
        """Detect FDA approval events (primarily for biotech)"""
        events = []
        
        try:
            # Check if ticker is biotech/pharma related
            if self._is_biotech_ticker(ticker):
                # For MVP, use pattern-based estimation
                # In production, integrate with FDA calendar APIs
                fda_date = self._estimate_fda_events(ticker)
                
                if fda_date and fda_date <= end_date:
                    event = MarketEvent(
                        ticker=ticker,
                        event_type='fda_approval',
                        event_date=fda_date,
                        description="FDA Decision/PDUFA Date (Estimated)",
                        impact_score=9,
                        direction_bias=0.2,  # Slightly bullish bias
                        volatility_multiplier=4.0,
                        source='biotech_pattern',
                        confidence=40  # Lower confidence for estimated events
                    )
                    events.append(event)
                    
        except Exception as e:
            pass
            
        return events
    
    def _detect_product_events(self, ticker: str, end_date: datetime) -> List[MarketEvent]:
        """Detect product launches and major announcements"""
        events = []
        
        try:
            # Pattern-based detection for tech companies
            if self._is_tech_ticker(ticker):
                product_date = self._estimate_product_events(ticker)
                
                if product_date and product_date <= end_date:
                    event = MarketEvent(
                        ticker=ticker,
                        event_type='product_launch',
                        event_date=product_date,
                        description="Product Launch/Major Announcement (Estimated)",
                        impact_score=6,
                        direction_bias=0.1,
                        volatility_multiplier=1.8,
                        source='tech_pattern',
                        confidence=30
                    )
                    events.append(event)
                    
        except Exception as e:
            pass
            
        return events
    
    def _estimate_next_earnings_date(self, ticker: str) -> Optional[datetime]:
        """Estimate next earnings date based on historical patterns"""
        try:
            # Most companies report quarterly (every ~90 days)
            # For MVP, use simple pattern: roughly every 90 days
            today = datetime.now()
            
            # Estimate based on typical earnings seasons
            # Q1: Late April/Early May, Q2: Late July/Early Aug, Q3: Late Oct/Early Nov, Q4: Late Jan/Early Feb
            current_month = today.month
            
            if current_month <= 2:  # Currently in Q1 season
                return datetime(today.year, 4, 25) + timedelta(days=np.random.randint(-7, 7))
            elif current_month <= 5:  # Currently in Q2 prep
                return datetime(today.year, 7, 25) + timedelta(days=np.random.randint(-7, 7))
            elif current_month <= 8:  # Currently in Q3 prep
                return datetime(today.year, 10, 25) + timedelta(days=np.random.randint(-7, 7))
            else:  # Currently in Q4 prep
                return datetime(today.year + 1, 1, 25) + timedelta(days=np.random.randint(-7, 7))
                
        except Exception:
            return None
    
    def _estimate_next_ex_dividend_date(self, ticker: str) -> Optional[datetime]:
        """Estimate next ex-dividend date"""
        try:
            # Most dividend stocks pay quarterly
            # Estimate based on typical patterns
            today = datetime.now()
            
            # Common ex-dividend months: March, June, September, December
            current_month = today.month
            
            if current_month <= 3:
                return datetime(today.year, 3, 15) + timedelta(days=np.random.randint(-10, 10))
            elif current_month <= 6:
                return datetime(today.year, 6, 15) + timedelta(days=np.random.randint(-10, 10))
            elif current_month <= 9:
                return datetime(today.year, 9, 15) + timedelta(days=np.random.randint(-10, 10))
            else:
                return datetime(today.year, 12, 15) + timedelta(days=np.random.randint(-10, 10))
                
        except Exception:
            return None
    
    def _estimate_fda_events(self, ticker: str) -> Optional[datetime]:
        """Estimate FDA events for biotech companies"""
        try:
            # FDA events are less predictable
            # For MVP, randomly assign some events to biotech stocks
            if np.random.random() < 0.3:  # 30% chance of having an FDA event
                today = datetime.now()
                days_ahead = np.random.randint(7, 60)  # 1-8 weeks
                return today + timedelta(days=days_ahead)
            return None
        except Exception:
            return None
    
    def _estimate_product_events(self, ticker: str) -> Optional[datetime]:
        """Estimate product events for tech companies"""
        try:
            # Tech companies often have seasonal product launches
            today = datetime.now()
            
            # Common launch periods: Spring (March-May), Fall (September-November)
            current_month = today.month
            
            if 3 <= current_month <= 5 and np.random.random() < 0.2:  # Spring launch season
                return today + timedelta(days=np.random.randint(7, 45))
            elif 9 <= current_month <= 11 and np.random.random() < 0.2:  # Fall launch season
                return today + timedelta(days=np.random.randint(7, 45))
                
            return None
        except Exception:
            return None
    
    def _analyze_historical_earnings_impact(self, ticker: str) -> Dict:
        """Analyze historical earnings impact for prediction"""
        # For MVP, return reasonable defaults based on market cap/sector
        # In production, this would analyze actual historical data
        
        # Larger companies tend to have smaller earnings moves
        if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:  # Large cap
            return {
                'impact_score': 6,
                'direction_bias': 0.05,
                'volatility_multiplier': 2.0,
                'confidence': 75
            }
        elif ticker in ['TSLA', 'NVDA', 'AMD']:  # High volatility stocks
            return {
                'impact_score': 8,
                'direction_bias': 0.1,
                'volatility_multiplier': 3.5,
                'confidence': 70
            }
        else:  # Default
            return {
                'impact_score': 7,
                'direction_bias': 0.03,
                'volatility_multiplier': 2.5,
                'confidence': 65
            }
    
    def _estimate_dividend_yield(self, ticker: str) -> float:
        """Estimate dividend yield for impact calculation"""
        # Default yields by sector/type
        dividend_stocks = {
            'KO': 3.2, 'JNJ': 2.8, 'PG': 2.5, 'MSFT': 0.7, 'AAPL': 0.5
        }
        return dividend_stocks.get(ticker, 2.0)  # Default 2%
    
    def _is_biotech_ticker(self, ticker: str) -> bool:
        """Check if ticker is biotech/pharma"""
        biotech_tickers = ['MRNA', 'BNTX', 'GILD', 'BIIB', 'SAVA', 'NVAX', 'REGN', 'VRTX']
        return ticker.upper() in biotech_tickers
    
    def _is_tech_ticker(self, ticker: str) -> bool:
        """Check if ticker is tech company"""
        tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'CRM']
        return ticker.upper() in tech_tickers
    
    def _get_quarter(self, date: datetime) -> int:
        """Get quarter number from date"""
        return (date.month - 1) // 3 + 1
    
    def get_event_impact_analysis(self, event: MarketEvent, historical_data: pd.DataFrame) -> Dict:
        """Analyze the potential impact of an event"""
        
        # Calculate days until event
        days_until = (event.event_date - datetime.now()).days
        
        # Proximity factor (closer events have higher impact)
        proximity_factor = max(0.5, 1 - (days_until / 30))
        
        # Historical volatility analysis
        returns = historical_data['close'].pct_change().dropna()
        base_volatility = returns.std() * np.sqrt(252)
        
        # Event-adjusted volatility
        event_volatility = base_volatility * event.volatility_multiplier * proximity_factor
        
        # Price impact estimation
        expected_price_change = event.direction_bias * event.impact_score * 0.01  # Convert to percentage
        
        return {
            'base_volatility': base_volatility,
            'event_volatility': event_volatility,
            'expected_price_change': expected_price_change,
            'proximity_factor': proximity_factor,
            'days_until_event': days_until,
            'risk_level': self._calculate_risk_level(event.impact_score, event_volatility),
            'position_sizing_recommendation': self._get_position_sizing_rec(event.impact_score)
        }
    
    def _calculate_risk_level(self, impact_score: float, volatility: float) -> str:
        """Calculate risk level for event"""
        risk_score = impact_score * volatility * 10
        
        if risk_score > 30:
            return "EXTREME"
        elif risk_score > 20:
            return "HIGH"
        elif risk_score > 10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_position_sizing_rec(self, impact_score: float) -> str:
        """Get position sizing recommendation"""
        if impact_score >= 9:
            return "SMALL_POSITION"  # High risk events
        elif impact_score >= 7:
            return "NORMAL_POSITION"
        elif impact_score >= 5:
            return "LARGER_POSITION"
        else:
            return "FULL_POSITION"

class EventCalendarIntegration:
    """Integration with external event calendars"""
    
    def __init__(self):
        self.earnings_calendar_urls = {
            'free': 'https://financialmodelingprep.com/api/v3/earning_calendar',
            'backup': 'https://api.polygon.io/v2/reference/earnings'
        }
        
    def fetch_earnings_calendar(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Fetch earnings calendar from external APIs"""
        try:
            # For MVP, return mock data
            # In production, integrate with real APIs
            return self._mock_earnings_calendar(start_date, end_date)
        except Exception as e:
            return []
    
    def _mock_earnings_calendar(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Mock earnings calendar for testing"""
        mock_events = [
            {
                'ticker': 'AAPL',
                'date': datetime.now() + timedelta(days=7),
                'quarter': 'Q1',
                'year': 2024,
                'estimated_eps': 1.85,
                'actual_eps': None
            },
            {
                'ticker': 'MSFT',
                'date': datetime.now() + timedelta(days=10),
                'quarter': 'Q1',
                'year': 2024,
                'estimated_eps': 2.45,
                'actual_eps': None
            }
        ]
        
        # Filter by date range
        filtered_events = [
            event for event in mock_events
            if start_date <= event['date'] <= end_date
        ]
        
        return filtered_events

# Usage example and testing
if __name__ == "__main__":
    # Test the event detector
    detector = EventDetector()
    
    # Test for a few tickers
    test_tickers = ['AAPL', 'SAVA', 'MSFT', 'MRNA']
    
    for ticker in test_tickers:
        print(f"\n=== {ticker} Events ===")
        events = detector.detect_upcoming_events(ticker, days_ahead=30)
        
        for event in events:
            print(f"Event: {event.description}")
            print(f"Date: {event.event_date.strftime('%Y-%m-%d')}")
            print(f"Impact: {event.impact_score}/10")
            print(f"Direction: {event.direction_bias:+.2f}")
            print(f"Volatility: {event.volatility_multiplier:.1f}x")
            print(f"Confidence: {event.confidence}%")
            print("-" * 40)