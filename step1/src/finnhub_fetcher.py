"""
Finnhub Data Fetcher
Free tier with real-time data + after hours support
"""

import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from pathlib import Path


class FinnhubDataFetcher:
    """
    Finnhub data fetcher for free real-time market data
    Supports after-hours data and global stock exchanges
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize Finnhub data fetcher
        
        Args:
            api_key: Finnhub API key
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        # Finnhub API endpoint
        self.base_url = "https://finnhub.io/api/v1"
        
        # Load SP500 symbols
        symbols_file = Path(__file__).parent.parent / "data" / "sp500_symbols.json"
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            self.symbols = data['symbols']
        
        # Cache for session
        self._cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Rate limiting (60 calls/minute for free tier)
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def _rate_limit(self):
        """Apply rate limiting for free tier (60 calls/minute)"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make API request with rate limiting
        
        Args:
            endpoint: API endpoint
            params: Request parameters
        
        Returns:
            API response data
        """
        self._rate_limit()
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {endpoint}: {e}")
            return {}
    
    def get_quote(self, symbol: str) -> dict:
        """
        Get real-time quote for symbol (includes after-hours)
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Quote data dictionary
        """
        data = self._make_request('quote', {'symbol': symbol})
        return data
    
    def get_candles(self, symbol: str, resolution: str = 'D', 
                   days_back: int = 30) -> pd.DataFrame:
        """
        Get candlestick data for symbol
        
        Args:
            symbol: Stock symbol
            resolution: Resolution (1, 5, 15, 30, 60, D, W, M)
            days_back: Number of days back to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # Calculate timestamps
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        data = self._make_request('stock/candle', {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_time,
            'to': end_time
        })
        
        if data.get('s') != 'ok' or not data.get('c'):
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = {
            'Open': data['o'],
            'High': data['h'],
            'Low': data['l'],
            'Close': data['c'],
            'Volume': data['v']
        }
        
        # Create timestamps
        timestamps = [datetime.fromtimestamp(t) for t in data['t']]
        
        df = pd.DataFrame(df_data, index=timestamps)
        return df
    
    def get_company_profile(self, symbol: str) -> dict:
        """
        Get company profile data
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Company profile dictionary
        """
        return self._make_request('stock/profile2', {'symbol': symbol})
    
    def get_market_status(self, exchange: str = 'US') -> dict:
        """
        Get market status (open/closed)
        
        Args:
            exchange: Exchange code (US, UK, etc.)
        
        Returns:
            Market status dictionary
        """
        return self._make_request('stock/market-status', {'exchange': exchange})
    
    def fetch_batch(self, symbols: List[str] = None, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch market data batch compatible with existing scanner
        
        Args:
            symbols: List of symbols to fetch
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.symbols[:50]  # Limit for free tier rate limiting
        
        # Convert period to days
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365
        }.get(period, 30)
        
        results = {}
        
        print(f"Fetching Finnhub data for {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.get_candles(symbol, resolution='D', days_back=period_days)
                if not df.empty:
                    results[symbol] = df
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  Fetched {i + 1}/{len(symbols)} symbols")
                    
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        return results
    
    def get_real_time_data(self, symbol: str) -> Optional[dict]:
        """
        Get real-time data for mobile app (includes after-hours)
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Real-time data dictionary
        """
        # Get current quote
        quote = self.get_quote(symbol)
        
        if not quote or 'c' not in quote:
            return None
        
        current_price = quote['c']  # Current price
        prev_close = quote['pc']    # Previous close
        
        # Calculate change
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        
        return {
            "symbol": symbol,
            "price": float(current_price),
            "change": float(change),
            "change_pct": float(change_pct),
            "volume": 0,  # Volume not provided in quote endpoint
            "timestamp": datetime.now().isoformat(),
            "high": quote.get('h', current_price),
            "low": quote.get('l', current_price),
            "open": quote.get('o', current_price),
            "prev_close": float(prev_close)
        }
    
    def get_market_news(self, category: str = 'general', limit: int = 10) -> List[dict]:
        """
        Get market news
        
        Args:
            category: News category (general, forex, crypto, merger)
            limit: Number of articles (max 100)
        
        Returns:
            List of news articles
        """
        data = self._make_request('news', {
            'category': category,
            'minId': 0
        })
        
        return data[:limit] if isinstance(data, list) else []
    
    def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> dict:
        """
        Get earnings calendar
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            Earnings calendar data
        """
        if not from_date:
            from_date = datetime.now().strftime('%Y-%m-%d')
        if not to_date:
            to_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        return self._make_request('calendar/earnings', {
            'from': from_date,
            'to': to_date
        })
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful
        """
        try:
            quote = self.get_quote('AAPL')
            return 'c' in quote  # Check if current price exists
        except:
            return False


def get_finnhub_data(symbol: str, api_key: str = None) -> Optional[dict]:
    """
    Convenience function to get Finnhub real-time data for a symbol
    
    Args:
        symbol: Stock symbol
        api_key: Finnhub API key (optional if in environment)
    
    Returns:
        Real-time data dictionary
    """
    try:
        fetcher = FinnhubDataFetcher(api_key)
        return fetcher.get_real_time_data(symbol)
    except Exception as e:
        print(f"Error fetching Finnhub data for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test the Finnhub fetcher
    print("Testing Finnhub Data Fetcher...")
    
    # You'll need to set this environment variable or pass it directly
    # export FINNHUB_API_KEY="your_api_key"
    
    try:
        fetcher = FinnhubDataFetcher()
        
        # Test connection
        if fetcher.test_connection():
            print("✅ Finnhub connection successful")
            
            # Test real-time data
            data = fetcher.get_real_time_data('AAPL')
            if data:
                print(f"✅ Real-time data: {data}")
            
            # Test market status
            status = fetcher.get_market_status()
            print(f"✅ Market status: {status}")
            
            # Test batch fetch (small sample for rate limiting)
            batch_data = fetcher.fetch_batch(['AAPL', 'MSFT'], period='5d')
            print(f"✅ Batch data fetched for {len(batch_data)} symbols")
            
            # Test market news
            news = fetcher.get_market_news(limit=3)
            print(f"✅ Market news: {len(news)} articles")
            
        else:
            print("❌ Finnhub connection failed")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Set FINNHUB_API_KEY environment variable")
        print("Get free API key at: https://finnhub.io/")
    except Exception as e:
        print(f"❌ Error: {e}")