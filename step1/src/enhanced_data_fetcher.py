"""
Enhanced Data Fetcher with Multiple API Support
Supports yfinance, Alpaca, and Finnhub with automatic fallback
"""

import os
import json
import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Import individual fetchers
from data_fetcher import DataFetcher
try:
    from alpaca_fetcher import AlpacaDataFetcher
except ImportError:
    AlpacaDataFetcher = None

try:
    from finnhub_fetcher import FinnhubDataFetcher
except ImportError:
    FinnhubDataFetcher = None


class EnhancedDataFetcher:
    """
    Enhanced data fetcher with multiple API support and automatic fallback
    Prioritizes APIs based on availability and market hours
    """
    
    def __init__(self, 
                 alpaca_api_key: str = None, 
                 alpaca_secret_key: str = None,
                 finnhub_api_key: str = None,
                 prefer_after_hours: bool = True):
        """
        Initialize enhanced data fetcher
        
        Args:
            alpaca_api_key: Alpaca API key
            alpaca_secret_key: Alpaca secret key
            finnhub_api_key: Finnhub API key
            prefer_after_hours: Prefer APIs with after-hours data
        """
        self.prefer_after_hours = prefer_after_hours
        
        # Initialize available fetchers
        self.fetchers = {}
        
        # Always available: yfinance (fallback)
        self.fetchers['yfinance'] = DataFetcher()
        
        # Alpaca (if credentials provided)
        if alpaca_api_key and alpaca_secret_key and AlpacaDataFetcher:
            try:
                alpaca = AlpacaDataFetcher(alpaca_api_key, alpaca_secret_key)
                if alpaca.test_connection():
                    self.fetchers['alpaca'] = alpaca
                    print("✅ Alpaca API connected")
                else:
                    print("❌ Alpaca API connection failed")
            except Exception as e:
                print(f"❌ Alpaca initialization error: {e}")
        
        # Finnhub (if API key provided)
        if finnhub_api_key and FinnhubDataFetcher:
            try:
                finnhub = FinnhubDataFetcher(finnhub_api_key)
                if finnhub.test_connection():
                    self.fetchers['finnhub'] = finnhub
                    print("✅ Finnhub API connected")
                else:
                    print("❌ Finnhub API connection failed")
            except Exception as e:
                print(f"❌ Finnhub initialization error: {e}")
        
        # Environment variable fallbacks
        if not alpaca_api_key and not alpaca_secret_key:
            alpaca_key = os.getenv('ALPACA_API_KEY')
            alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
            if alpaca_key and alpaca_secret and AlpacaDataFetcher:
                try:
                    alpaca = AlpacaDataFetcher(alpaca_key, alpaca_secret)
                    if alpaca.test_connection():
                        self.fetchers['alpaca'] = alpaca
                        print("✅ Alpaca API connected (from environment)")
                except:
                    pass
        
        if not finnhub_api_key:
            finnhub_key = os.getenv('FINNHUB_API_KEY')
            if finnhub_key and FinnhubDataFetcher:
                try:
                    finnhub = FinnhubDataFetcher(finnhub_key)
                    if finnhub.test_connection():
                        self.fetchers['finnhub'] = finnhub
                        print("✅ Finnhub API connected (from environment)")
                except:
                    pass
        
        # Set priority order based on after-hours preference
        if prefer_after_hours:
            self.priority_order = ['alpaca', 'finnhub', 'yfinance']
        else:
            self.priority_order = ['finnhub', 'alpaca', 'yfinance']
        
        print(f"📊 Available data sources: {list(self.fetchers.keys())}")
    
    def is_market_hours(self) -> bool:
        """Check if currently in market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now()
        # Simplified check - in production, consider timezone and holidays
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close and now.weekday() < 5
    
    def get_best_fetcher(self, for_real_time: bool = False) -> str:
        """
        Get the best available fetcher based on current conditions
        
        Args:
            for_real_time: Whether this is for real-time data
        
        Returns:
            Best fetcher name
        """
        market_hours = self.is_market_hours()
        
        # During market hours, any fetcher is good
        # After hours, prefer APIs with extended hours data
        if not market_hours and for_real_time and self.prefer_after_hours:
            # Prefer Alpaca/Finnhub for after-hours real-time data
            for fetcher in ['alpaca', 'finnhub']:
                if fetcher in self.fetchers:
                    return fetcher
        
        # Default priority order
        for fetcher in self.priority_order:
            if fetcher in self.fetchers:
                return fetcher
        
        return 'yfinance'  # Always available fallback
    
    def fetch_batch(self, symbols: List[str] = None, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Fetch market data batch with automatic fallback
        
        Args:
            symbols: List of symbols to fetch
            period: Time period
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        if symbols is None:
            symbols = self.fetchers['yfinance'].symbols[:100]
        
        best_fetcher = self.get_best_fetcher(for_real_time=False)
        
        print(f"📊 Using {best_fetcher} for batch data...")
        
        try:
            return self.fetchers[best_fetcher].fetch_batch(symbols, period)
        except Exception as e:
            print(f"❌ {best_fetcher} failed: {e}")
            
            # Try fallback fetchers
            for fetcher_name in self.priority_order:
                if fetcher_name != best_fetcher and fetcher_name in self.fetchers:
                    print(f"🔄 Falling back to {fetcher_name}...")
                    try:
                        return self.fetchers[fetcher_name].fetch_batch(symbols, period)
                    except Exception as e2:
                        print(f"❌ {fetcher_name} also failed: {e2}")
                        continue
            
            print("❌ All fetchers failed")
            return {}
    
    def get_real_time_data(self, symbol: str) -> Optional[dict]:
        """
        Get real-time data with best available API
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Real-time data dictionary
        """
        best_fetcher = self.get_best_fetcher(for_real_time=True)
        
        # Try the best fetcher first
        if best_fetcher in self.fetchers:
            try:
                fetcher = self.fetchers[best_fetcher]
                if hasattr(fetcher, 'get_real_time_data'):
                    return fetcher.get_real_time_data(symbol)
                elif hasattr(fetcher, 'get_minimal_data'):
                    return fetcher.get_minimal_data(symbol)
            except Exception as e:
                print(f"❌ {best_fetcher} real-time failed: {e}")
        
        # Try fallback fetchers
        for fetcher_name in self.priority_order:
            if fetcher_name != best_fetcher and fetcher_name in self.fetchers:
                try:
                    fetcher = self.fetchers[fetcher_name]
                    if hasattr(fetcher, 'get_real_time_data'):
                        return fetcher.get_real_time_data(symbol)
                    elif hasattr(fetcher, 'get_minimal_data'):
                        return fetcher.get_minimal_data(symbol)
                except Exception as e:
                    print(f"❌ {fetcher_name} real-time failed: {e}")
                    continue
        
        return None
    
    def get_data_quality_score(self, data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate data quality score
        
        Args:
            data: Fetched market data
        
        Returns:
            Quality score between 0 and 1
        """
        if hasattr(self.fetchers['yfinance'], 'get_data_quality_score'):
            return self.fetchers['yfinance'].get_data_quality_score(data)
        
        # Simple calculation
        if not self.fetchers['yfinance'].symbols:
            return 0.0
        
        successful = len([s for s in self.fetchers['yfinance'].symbols if s in data and not data[s].empty])
        return successful / len(self.fetchers['yfinance'].symbols)
    
    def get_market_status(self) -> dict:
        """
        Get market status from best available API
        
        Returns:
            Market status dictionary
        """
        # Try Finnhub first (has dedicated market status endpoint)
        if 'finnhub' in self.fetchers:
            try:
                return self.fetchers['finnhub'].get_market_status()
            except:
                pass
        
        # Fallback to simple calculation
        is_open = self.is_market_hours()
        return {
            'isOpen': is_open,
            'session': 'regular' if is_open else 'closed',
            'timezone': 'US/Eastern'
        }
    
    def get_capabilities(self) -> dict:
        """
        Get capabilities of available APIs
        
        Returns:
            Capabilities dictionary
        """
        capabilities = {
            'after_hours': [],
            'real_time': [],
            'historical': [],
            'news': [],
            'fundamentals': []
        }
        
        for name, fetcher in self.fetchers.items():
            capabilities['historical'].append(name)
            
            if name in ['alpaca', 'finnhub']:
                capabilities['after_hours'].append(name)
                capabilities['real_time'].append(name)
            
            if name == 'finnhub':
                capabilities['news'].append(name)
                capabilities['fundamentals'].append(name)
        
        return capabilities
    
    def clear_cache(self):
        """Clear cache for all fetchers"""
        for fetcher in self.fetchers.values():
            if hasattr(fetcher, 'clear_cache'):
                fetcher.clear_cache()


def get_enhanced_data(symbol: str, **kwargs) -> Optional[dict]:
    """
    Convenience function to get real-time data with automatic API selection
    
    Args:
        symbol: Stock symbol
        **kwargs: API credentials
    
    Returns:
        Real-time data dictionary
    """
    try:
        fetcher = EnhancedDataFetcher(**kwargs)
        return fetcher.get_real_time_data(symbol)
    except Exception as e:
        print(f"Error fetching enhanced data for {symbol}: {e}")
        return None


if __name__ == "__main__":
    # Test the enhanced fetcher
    print("Testing Enhanced Data Fetcher...")
    
    # Test with whatever APIs are available
    fetcher = EnhancedDataFetcher()
    
    print(f"\n📊 Capabilities: {fetcher.get_capabilities()}")
    print(f"🕐 Market hours: {fetcher.is_market_hours()}")
    print(f"📈 Market status: {fetcher.get_market_status()}")
    
    # Test real-time data
    print(f"\n🔍 Testing real-time data for AAPL...")
    data = fetcher.get_real_time_data('AAPL')
    if data:
        print(f"✅ Real-time data: {data}")
    else:
        print("❌ No real-time data available")
    
    # Test batch fetch with small sample
    print(f"\n📊 Testing batch fetch...")
    batch_data = fetcher.fetch_batch(['AAPL', 'MSFT'], period='5d')
    print(f"✅ Batch data fetched for {len(batch_data)} symbols")
    
    print(f"\n🎉 Enhanced data fetcher test complete!")