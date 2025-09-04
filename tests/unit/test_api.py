"""
Unit tests for market data API
Phase 1 MVP tests for free data sources
"""

import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class TestMarketDataAPI(unittest.TestCase):
    """Test suite for market data API endpoints"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.mock_data = {
            'AAPL': {
                'price': 150.25,
                'change': 2.5,
                'volume': 75000000,
                'history': []
            }
        }
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful stock data retrieval"""
        # Mock yfinance response
        mock_instance = MagicMock()
        mock_instance.info = {
            'currentPrice': 150.25,
            'regularMarketChangePercent': 2.5,
            'volume': 75000000
        }
        mock_instance.history.return_value.to_json.return_value = '[]'
        mock_ticker.return_value = mock_instance
        
        # Import after mocking
        from api.market_data import get_stock_data
        
        # Test
        result = get_stock_data(['AAPL'])
        
        # Assertions
        self.assertIn('AAPL', result)
        self.assertEqual(result['AAPL']['price'], 150.25)
        self.assertEqual(result['AAPL']['change'], 2.5)
        self.assertEqual(result['AAPL']['volume'], 75000000)
    
    @patch('yfinance.Ticker')
    def test_get_stock_data_error_handling(self, mock_ticker):
        """Test error handling in stock data retrieval"""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception('API Error')
        
        from api.market_data import get_stock_data
        
        # Test
        result = get_stock_data(['INVALID'])
        
        # Assertions
        self.assertIn('INVALID', result)
        self.assertIn('error', result['INVALID'])
        self.assertIn('API Error', result['INVALID']['error'])
    
    def test_data_validation(self):
        """Test data validation for API responses"""
        # Test empty symbol list
        from api.market_data import get_stock_data
        result = get_stock_data([])
        self.assertEqual(result, {})
        
        # Test None input
        result = get_stock_data(None)
        self.assertEqual(result, {})
    
    @patch('yfinance.Ticker')
    def test_multiple_symbols(self, mock_ticker):
        """Test fetching multiple symbols"""
        # Mock yfinance response
        mock_instance = MagicMock()
        mock_instance.info = {
            'currentPrice': 100,
            'regularMarketChangePercent': 1.0,
            'volume': 50000000
        }
        mock_instance.history.return_value.to_json.return_value = '[]'
        mock_ticker.return_value = mock_instance
        
        from api.market_data import get_stock_data
        
        # Test
        result = get_stock_data(self.test_symbols)
        
        # Assertions
        self.assertEqual(len(result), len(self.test_symbols))
        for symbol in self.test_symbols:
            self.assertIn(symbol, result)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # This would test rate limiting if implemented
        pass


class TestTechnicalIndicators(unittest.TestCase):
    """Test suite for technical indicators"""
    
    def setUp(self):
        """Set up test data"""
        self.test_prices = [
            100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
            111, 110, 112, 114, 113, 115, 117, 116, 118, 120
        ]
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        from services.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        sma = indicators.calculateSMA(self.test_prices, 5)
        
        # Check length
        self.assertEqual(len(sma), len(self.test_prices) - 4)
        
        # Check first SMA value
        expected_first = sum(self.test_prices[:5]) / 5
        self.assertAlmostEqual(sma[0], expected_first, places=2)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        from services.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        rsi = indicators.calculateRSI(self.test_prices)
        
        # RSI should be between 0 and 100
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
    
    def test_signal_generation(self):
        """Test signal generation logic"""
        from services.indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        
        # Mock stock data
        stock_data = {
            'prices': self.test_prices,
            'volume': [1000000] * len(self.test_prices)
        }
        
        signal = indicators.generateSignal(stock_data)
        
        # Check signal structure
        self.assertIn('action', signal)
        self.assertIn('strength', signal)
        self.assertIn(signal['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertGreaterEqual(signal['strength'], 1)
        self.assertLessEqual(signal['strength'], 5)


class TestPersonalization(unittest.TestCase):
    """Test suite for personalization engine"""
    
    def setUp(self):
        """Set up test data"""
        self.user_profile = {
            'riskProfile': 'moderate',
            'sectors': ['technology', 'healthcare'],
            'marketCap': 'large',
            'tradingStyle': 'swing'
        }
        
        self.test_signals = [
            {
                'symbol': 'AAPL',
                'action': 'BUY',
                'strength': 4,
                'sector': 'technology'
            },
            {
                'symbol': 'JNJ',
                'action': 'HOLD',
                'strength': 2,
                'sector': 'healthcare'
            },
            {
                'symbol': 'XOM',
                'action': 'SELL',
                'strength': 3,
                'sector': 'energy'
            }
        ]
    
    def test_signal_filtering(self):
        """Test signal filtering based on user profile"""
        from services.personalization import PersonalizationEngine
        
        engine = PersonalizationEngine()
        filtered = engine.filterSignals(self.test_signals, self.user_profile)
        
        # Should filter out energy sector
        self.assertEqual(len(filtered), 2)
        
        # Check remaining signals are in user's sectors
        for signal in filtered:
            self.assertIn(signal['sector'], self.user_profile['sectors'])
    
    def test_signal_ranking(self):
        """Test signal ranking based on user profile"""
        from services.personalization import PersonalizationEngine
        
        engine = PersonalizationEngine()
        ranked = engine.rankSignals(self.test_signals, self.user_profile)
        
        # Check signals are sorted by score
        scores = [engine.calculateScore(s, self.user_profile) for s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)