"""
Step 1: Data Fetcher - Ultra-simple free market data fetching
Optimized for speed and reliability with yfinance
"""

import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path

class DataFetcher:
    """Minimal data fetcher for SP500 stocks using free yfinance API"""
    
    def __init__(self, symbols_file: str = None):
        """Initialize with SP500 symbols"""
        if symbols_file is None:
            symbols_file = Path(__file__).parent.parent / "data" / "sp500_symbols.json"
        
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            self.symbols = data['symbols']
        
        # Cache for current session to minimize API calls
        self._cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = 300  # 5 minutes cache
    
    def fetch_batch(self, symbols: List[str] = None, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for multiple symbols in batch (faster)
        
        Args:
            symbols: List of symbols to fetch (default: all SP500)
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y)
        
        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        if symbols is None:
            symbols = self.symbols
        
        # Check cache first
        now = time.time()
        result = {}
        symbols_to_fetch = []
        
        for symbol in symbols:
            cache_key = f"{symbol}_{period}"
            if cache_key in self._cache:
                if now - self._cache_timestamp.get(cache_key, 0) < self.cache_ttl:
                    result[symbol] = self._cache[cache_key]
                else:
                    symbols_to_fetch.append(symbol)
            else:
                symbols_to_fetch.append(symbol)
        
        # Fetch missing data in batch
        if symbols_to_fetch:
            try:
                # yfinance batch download is much faster
                data = yf.download(
                    symbols_to_fetch,
                    period=period,
                    interval="1d",
                    group_by='ticker',
                    auto_adjust=True,
                    prepost=False,
                    threads=True,
                    progress=False
                )
                
                # Handle single vs multiple symbols
                if len(symbols_to_fetch) == 1:
                    symbol = symbols_to_fetch[0]
                    if not data.empty:
                        result[symbol] = data
                        cache_key = f"{symbol}_{period}"
                        self._cache[cache_key] = data
                        self._cache_timestamp[cache_key] = now
                else:
                    for symbol in symbols_to_fetch:
                        if symbol in data.columns.levels[1]:
                            symbol_data = data[symbol].dropna()
                            if not symbol_data.empty:
                                result[symbol] = symbol_data
                                cache_key = f"{symbol}_{period}"
                                self._cache[cache_key] = symbol_data
                                self._cache_timestamp[cache_key] = now
                
            except Exception as e:
                print(f"Error fetching batch data: {e}")
        
        return result
    
    def fetch_single(self, symbol: str, period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol
        
        Args:
            symbol: Stock symbol
            period: Time period
        
        Returns:
            DataFrame with OHLCV data or None if error
        """
        result = self.fetch_batch([symbol], period)
        return result.get(symbol)
    
    def fetch_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest closing price for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Latest closing price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return info.get('lastPrice', info.get('previousClose'))
        except:
            # Fallback to 1-day data
            data = self.fetch_single(symbol, period="5d")
            if data is not None and not data.empty:
                return float(data['Close'].iloc[-1])
        return None
    
    def get_data_quality_score(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate data quality score (% of successfully fetched symbols)
        
        Args:
            data: Fetched market data dictionary
        
        Returns:
            Quality score between 0 and 1
        """
        if not self.symbols:
            return 0.0
        
        successful = len([s for s in self.symbols if s in data and not data[s].empty])
        return successful / len(self.symbols)
    
    def clear_cache(self):
        """Clear the data cache"""
        self._cache.clear()
        self._cache_timestamp.clear()


# Utility functions for mobile app integration
def get_minimal_data(symbol: str) -> Dict:
    """
    Get minimal data structure for mobile app
    
    Returns:
        {
            "symbol": "AAPL",
            "price": 150.25,
            "change": 2.50,
            "change_pct": 1.69,
            "volume": 50000000,
            "timestamp": "2025-01-15T10:30:00"
        }
    """
    fetcher = DataFetcher()
    data = fetcher.fetch_single(symbol, period="5d")
    
    if data is None or data.empty:
        return None
    
    latest = data.iloc[-1]
    previous = data.iloc[-2] if len(data) > 1 else latest
    
    return {
        "symbol": symbol,
        "price": float(latest['Close']),
        "change": float(latest['Close'] - previous['Close']),
        "change_pct": float((latest['Close'] - previous['Close']) / previous['Close'] * 100),
        "volume": int(latest['Volume']),
        "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
    }