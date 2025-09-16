"""
Alpaca Markets Data Fetcher
Free tier with IEX data + after hours support
"""

import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from pathlib import Path


class AlpacaDataFetcher:
    """
    Alpaca Markets data fetcher for free IEX data
    Supports after-hours data with extended market hours
    """
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        """
        Initialize Alpaca data fetcher
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret are required")
        
        # Alpaca API endpoints
        self.base_url = "https://data.alpaca.markets/v2"
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        
        # Load SP500 symbols
        symbols_file = Path(__file__).parent.parent / "data" / "sp500_symbols.json"
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            self.symbols = data['symbols']
        
        # Cache for session
        self._cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_latest_quotes(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Get latest quotes for symbols (includes after-hours data)
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbol to quote data
        """
        if not symbols:
            return {}
        
        # Convert symbols to comma-separated string
        symbols_str = ','.join(symbols[:100])  # Max 100 symbols per request
        
        url = f"{self.base_url}/stocks/quotes/latest"
        params = {
            'symbols': symbols_str,
            'feed': 'iex'  # Free tier uses IEX data
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('quotes', {})
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching quotes: {e}")
            return {}
    
    def get_bars(self, symbols: List[str], timeframe: str = "1Day", 
                 start: str = None, end: str = None, 
                 extended_hours: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for symbols
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            extended_hours: Include extended hours data
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if not symbols:
            return {}
        
        # Set default date range (last 30 days)
        if not start:
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
        
        results = {}
        
        # Process symbols in batches (API limit)
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i+batch_size]
            symbols_str = ','.join(batch_symbols)
            
            url = f"{self.base_url}/stocks/bars"
            params = {
                'symbols': symbols_str,
                'timeframe': timeframe,
                'start': start,
                'end': end,
                'adjustment': 'all',
                'feed': 'iex',
                'asof': end,
                'page_token': None
            }
            
            if extended_hours:
                params['extended_hours'] = 'true'
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                bars_data = data.get('bars', {})
                
                # Convert to pandas DataFrames
                for symbol, bars in bars_data.items():
                    if bars:
                        df_data = []
                        for bar in bars:
                            df_data.append({
                                'timestamp': pd.to_datetime(bar['t']),
                                'Open': bar['o'],
                                'High': bar['h'],
                                'Low': bar['l'],
                                'Close': bar['c'],
                                'Volume': bar['v']
                            })
                        
                        if df_data:
                            df = pd.DataFrame(df_data)
                            df.set_index('timestamp', inplace=True)
                            results[symbol] = df
                
                # Rate limiting for free tier (200 requests/minute)
                time.sleep(0.3)  # Wait 300ms between requests
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching bars for batch {batch_symbols}: {e}")
                continue
        
        return results
    
    def get_latest_trades(self, symbols: List[str]) -> Dict[str, dict]:
        """
        Get latest trades for symbols (real-time after-hours data)
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbol to trade data
        """
        if not symbols:
            return {}
        
        symbols_str = ','.join(symbols[:100])
        
        url = f"{self.base_url}/stocks/trades/latest"
        params = {
            'symbols': symbols_str,
            'feed': 'iex'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get('trades', {})
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trades: {e}")
            return {}
    
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
            symbols = self.symbols[:100]  # Limit for free tier
        
        # Convert period to days
        period_days = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365
        }.get(period, 30)
        
        start_date = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        
        # Get daily bars with extended hours
        return self.get_bars(
            symbols=symbols,
            timeframe="1Day",
            start=start_date,
            extended_hours=True
        )
    
    def get_real_time_data(self, symbol: str) -> Optional[dict]:
        """
        Get real-time data for mobile app (includes after-hours)
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Real-time data dictionary
        """
        # Get latest quote and trade
        quotes = self.get_latest_quotes([symbol])
        trades = self.get_latest_trades([symbol])
        
        if symbol not in quotes and symbol not in trades:
            return None
        
        quote = quotes.get(symbol, {})
        trade = trades.get(symbol, {})
        
        # Get previous day's close for change calculation
        bars = self.get_bars([symbol], timeframe="1Day", 
                           start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
        
        prev_close = None
        if symbol in bars and len(bars[symbol]) > 1:
            prev_close = bars[symbol]['Close'].iloc[-2]
        
        # Current price (latest trade or quote)
        current_price = trade.get('p') or quote.get('ap') or quote.get('bp')
        
        if not current_price:
            return None
        
        # Calculate change
        change = 0
        change_pct = 0
        if prev_close:
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
        
        return {
            "symbol": symbol,
            "price": float(current_price),
            "change": float(change),
            "change_pct": float(change_pct),
            "volume": trade.get('s', 0),
            "timestamp": trade.get('t') or quote.get('t') or datetime.now().isoformat(),
            "bid": quote.get('bp'),
            "ask": quote.get('ap'),
            "bid_size": quote.get('bs'),
            "ask_size": quote.get('as')
        }
    
    def test_connection(self) -> bool:
        """
        Test API connection and authentication
        
        Returns:
            True if connection successful
        """
        try:
            quotes = self.get_latest_quotes(['AAPL'])
            return 'AAPL' in quotes
        except:
            return False


def get_alpaca_data(symbol: str, api_key: str = None, secret_key: str = None) -> Optional[dict]:
    """
    Convenience function to get Alpaca real-time data for a symbol
    
    Args:
        symbol: Stock symbol
        api_key: Alpaca API key (optional if in environment)
        secret_key: Alpaca secret key (optional if in environment)
    
    Returns:
        Real-time data dictionary
    """
    try:
        fetcher = AlpacaDataFetcher(api_key, secret_key)
        return fetcher.get_real_time_data(symbol)
    except Exception as e:
        print(f"Error fetching Alpaca data for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test the Alpaca fetcher
    print("Testing Alpaca Data Fetcher...")
    
    # You'll need to set these environment variables or pass them directly
    # export ALPACA_API_KEY="your_api_key"
    # export ALPACA_SECRET_KEY="your_secret_key"
    
    try:
        fetcher = AlpacaDataFetcher()
        
        # Test connection
        if fetcher.test_connection():
            print("✅ Alpaca connection successful")
            
            # Test real-time data
            data = fetcher.get_real_time_data('AAPL')
            if data:
                print(f"✅ Real-time data: {data}")
            
            # Test batch fetch
            batch_data = fetcher.fetch_batch(['AAPL', 'MSFT'], period='5d')
            print(f"✅ Batch data fetched for {len(batch_data)} symbols")
        else:
            print("❌ Alpaca connection failed")
            
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    except Exception as e:
        print(f"❌ Error: {e}")