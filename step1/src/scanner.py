"""
Step 1: Scanner Engine - Core signal generation with confidence scoring
Designed for speed and mobile app integration
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd

from data_fetcher import DataFetcher
from indicators import TechnicalIndicators, quick_signals


class Signal:
    """Lightweight signal model for mobile app"""
    
    def __init__(self, symbol: str, action: str, confidence: float, 
                 price: float, indicators: dict, timestamp: str = None):
        self.symbol = symbol
        self.action = action  # BUY, SELL, HOLD
        self.confidence = confidence  # 0-100
        self.price = price
        self.indicators = indicators
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'confidence': round(self.confidence, 2),
            'price': round(self.price, 2),
            'indicators': self.indicators,
            'timestamp': self.timestamp
        }


class MinimalScanner:
    """Fast SP500 scanner with signal generation"""
    
    def __init__(self, max_symbols: int = 100):
        """
        Initialize scanner with top SP500 symbols
        
        Args:
            max_symbols: Maximum number of symbols to scan (default 100)
        """
        self.fetcher = DataFetcher()
        self.symbols = self.fetcher.symbols[:max_symbols]
        self.last_scan_time = None
        self.last_scan_results = None
    
    def calculate_signal_confidence(self, features: dict) -> float:
        """
        Calculate confidence score (0-100) for a signal
        
        Args:
            features: Dictionary of technical indicator features
        
        Returns:
            Confidence score between 0 and 100
        """
        if not features:
            return 0.0
        
        confidence = 50.0  # Base confidence
        
        # Trend alignment
        if features.get('price_above_sma20') and features.get('price_above_sma50'):
            confidence += 10
        elif not features.get('price_above_sma20') and not features.get('price_above_sma50'):
            confidence += 10
        
        # Moving average crossover
        if features.get('sma20_above_sma50'):
            confidence += 5
        
        # RSI extremes
        rsi = features.get('rsi')
        if rsi:
            if rsi < 30:  # Oversold
                confidence += 15
            elif rsi > 70:  # Overbought
                confidence += 15
            elif 40 < rsi < 60:  # Neutral zone
                confidence -= 10
        
        # MACD confirmation
        if features.get('macd_bullish') is not None:
            if features.get('macd_bullish'):
                confidence += 10
            else:
                confidence += 5
        
        # Momentum
        momentum_5d = features.get('momentum_5d', 0)
        if abs(momentum_5d) > 5:
            confidence += 10
        elif abs(momentum_5d) > 10:
            confidence += 20
        
        # Cap confidence between 0 and 100
        return max(0, min(100, confidence))
    
    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Optional[Signal]:
        """
        Generate signal for a single symbol
        
        Args:
            symbol: Stock symbol
            data: OHLCV DataFrame
        
        Returns:
            Signal object or None if no signal
        """
        if data is None or data.empty or len(data) < 50:
            return None
        
        # Calculate indicators
        indicators = TechnicalIndicators(data)
        features = indicators.get_signal_features()
        
        if not features:
            return None
        
        # Get signal action
        action = quick_signals(data)
        
        # Calculate confidence
        confidence = self.calculate_signal_confidence(features)
        
        # Only return signals with confidence > 60 for mobile app
        if confidence < 60:
            return None
        
        # Create signal
        signal_indicators = {
            'sma_20': indicators.latest.get('sma_20'),
            'sma_50': indicators.latest.get('sma_50'),
            'rsi': indicators.latest.get('rsi'),
            'macd': indicators.latest.get('macd'),
            'macd_signal': indicators.latest.get('macd_signal'),
            'momentum_5d': features.get('momentum_5d')
        }
        
        # Clean up None values for mobile app
        signal_indicators = {k: round(v, 2) if v is not None else None 
                           for k, v in signal_indicators.items()}
        
        return Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            price=float(data['Close'].iloc[-1]),
            indicators=signal_indicators
        )
    
    def scan(self, symbols: List[str] = None) -> Dict:
        """
        Scan symbols and generate signals
        
        Args:
            symbols: List of symbols to scan (default: all)
        
        Returns:
            Dictionary with scan results
        """
        start_time = time.time()
        
        if symbols is None:
            symbols = self.symbols
        
        # Fetch data in batch (much faster)
        print(f"Fetching data for {len(symbols)} symbols...")
        market_data = self.fetcher.fetch_batch(symbols, period="1mo")
        
        # Generate signals
        signals = []
        errors = []
        
        for symbol in symbols:
            try:
                if symbol in market_data:
                    signal = self.generate_signal(symbol, market_data[symbol])
                    if signal and signal.action != 'HOLD':
                        signals.append(signal)
            except Exception as e:
                errors.append({'symbol': symbol, 'error': str(e)})
        
        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Calculate metrics
        scan_time = time.time() - start_time
        data_quality = self.fetcher.get_data_quality_score(market_data)
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'scan_time_seconds': round(scan_time, 2),
            'symbols_scanned': len(symbols),
            'symbols_with_data': len(market_data),
            'data_quality_score': round(data_quality, 3),
            'signals_generated': len(signals),
            'buy_signals': len([s for s in signals if s.action == 'BUY']),
            'sell_signals': len([s for s in signals if s.action == 'SELL']),
            'top_signals': [s.to_dict() for s in signals[:10]],  # Top 10 for mobile
            'all_signals': [s.to_dict() for s in signals],
            'errors': errors[:5] if errors else []  # Limit error reporting
        }
        
        # Cache results
        self.last_scan_time = datetime.now()
        self.last_scan_results = results
        
        return results
    
    def quick_scan(self, top_n: int = 20) -> Dict:
        """
        Quick scan of top N symbols for mobile app
        
        Args:
            top_n: Number of top symbols to scan
        
        Returns:
            Simplified results for mobile
        """
        symbols = self.symbols[:top_n]
        results = self.scan(symbols)
        
        # Simplify for mobile
        return {
            'timestamp': results['timestamp'],
            'scan_time_ms': int(results['scan_time_seconds'] * 1000),
            'signals': results['top_signals'][:5],  # Top 5 signals only
            'summary': {
                'total_signals': results['signals_generated'],
                'buy_signals': results['buy_signals'],
                'sell_signals': results['sell_signals']
            }
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save scan results to JSON file
        
        Args:
            results: Scan results dictionary
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scan_results_{timestamp}.json"
        
        output_path = Path(__file__).parent.parent / "data" / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return str(output_path)


def run_daily_scan():
    """Run daily scan and save results"""
    scanner = MinimalScanner(max_symbols=100)
    results = scanner.scan()
    
    # Print summary
    print("\n" + "="*50)
    print("SCAN SUMMARY")
    print("="*50)
    print(f"Scan Time: {results['scan_time_seconds']:.2f} seconds")
    print(f"Symbols Scanned: {results['symbols_scanned']}")
    print(f"Data Quality: {results['data_quality_score']*100:.1f}%")
    print(f"Signals Generated: {results['signals_generated']}")
    print(f"  - Buy Signals: {results['buy_signals']}")
    print(f"  - Sell Signals: {results['sell_signals']}")
    
    if results['top_signals']:
        print("\nTOP 5 SIGNALS:")
        print("-"*50)
        for signal in results['top_signals'][:5]:
            print(f"{signal['symbol']:6} {signal['action']:4} "
                  f"Confidence: {signal['confidence']:5.1f}% "
                  f"Price: ${signal['price']:7.2f}")
    
    # Save results
    scanner.save_results(results)
    
    return results


if __name__ == "__main__":
    # Run daily scan when executed directly
    run_daily_scan()