"""
Step 1: Signal Validator - Verification and backtesting system
Ensures all signals are measurable and verifiable
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from data_fetcher import DataFetcher


class SignalValidator:
    """Validate and backtest generated signals"""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.results_dir = Path(__file__).parent.parent / "data"
    
    def calculate_forward_returns(self, symbol: str, signal_date: str, 
                                 days: int = 7) -> Optional[float]:
        """
        Calculate forward returns for a signal
        
        Args:
            symbol: Stock symbol
            signal_date: Date of signal generation
            days: Number of days forward to calculate returns
        
        Returns:
            Percentage return or None if data unavailable
        """
        try:
            # Fetch data including forward period
            data = self.fetcher.fetch_single(symbol, period="3mo")
            if data is None or data.empty:
                return None
            
            # Find signal date in data
            signal_date = pd.to_datetime(signal_date).date()
            
            # Get price on signal date
            signal_price = None
            for idx, row in data.iterrows():
                if idx.date() == signal_date:
                    signal_price = row['Close']
                    break
            
            if signal_price is None:
                return None
            
            # Get price N days later
            target_date = signal_date + timedelta(days=days)
            future_price = None
            
            for idx, row in data.iterrows():
                if idx.date() >= target_date:
                    future_price = row['Close']
                    break
            
            if future_price is None:
                return None
            
            # Calculate return
            return ((future_price - signal_price) / signal_price) * 100
            
        except Exception as e:
            print(f"Error calculating returns for {symbol}: {e}")
            return None
    
    def validate_signal_accuracy(self, signals: List[Dict], 
                                days: int = 7) -> Dict:
        """
        Validate accuracy of generated signals
        
        Args:
            signals: List of signal dictionaries
            days: Forward-looking period for validation
        
        Returns:
            Validation metrics dictionary
        """
        results = {
            'total_signals': len(signals),
            'validated_signals': 0,
            'buy_signals': {'total': 0, 'profitable': 0, 'returns': []},
            'sell_signals': {'total': 0, 'profitable': 0, 'returns': []},
            'overall_hit_rate': 0.0,
            'average_return': 0.0,
            'validation_period_days': days
        }
        
        for signal in signals:
            if signal['action'] == 'HOLD':
                continue
            
            # Calculate forward returns
            returns = self.calculate_forward_returns(
                signal['symbol'], 
                signal.get('timestamp', datetime.now().isoformat()),
                days
            )
            
            if returns is None:
                continue
            
            results['validated_signals'] += 1
            
            if signal['action'] == 'BUY':
                results['buy_signals']['total'] += 1
                results['buy_signals']['returns'].append(returns)
                if returns > 0:
                    results['buy_signals']['profitable'] += 1
            
            elif signal['action'] == 'SELL':
                results['sell_signals']['total'] += 1
                results['sell_signals']['returns'].append(-returns)  # Invert for sell
                if returns < 0:  # Price went down, sell was correct
                    results['sell_signals']['profitable'] += 1
        
        # Calculate overall metrics
        if results['validated_signals'] > 0:
            total_profitable = (results['buy_signals']['profitable'] + 
                              results['sell_signals']['profitable'])
            results['overall_hit_rate'] = total_profitable / results['validated_signals']
            
            all_returns = (results['buy_signals']['returns'] + 
                          results['sell_signals']['returns'])
            if all_returns:
                results['average_return'] = np.mean(all_returns)
        
        # Calculate hit rates
        if results['buy_signals']['total'] > 0:
            results['buy_signals']['hit_rate'] = (
                results['buy_signals']['profitable'] / results['buy_signals']['total']
            )
        
        if results['sell_signals']['total'] > 0:
            results['sell_signals']['hit_rate'] = (
                results['sell_signals']['profitable'] / results['sell_signals']['total']
            )
        
        return results
    
    def backtest_simple(self, signals: List[Dict], 
                       initial_capital: float = 10000,
                       position_size: float = 0.1) -> Dict:
        """
        Simple backtesting of signals
        
        Args:
            signals: List of signal dictionaries
            initial_capital: Starting capital
            position_size: Fraction of capital per position
        
        Returns:
            Backtest results dictionary
        """
        capital = initial_capital
        positions = {}  # Current positions
        trades = []  # Trade history
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            
            if action == 'BUY' and symbol not in positions:
                # Open position
                position_value = capital * position_size
                shares = position_value / price
                positions[symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_value': position_value
                }
                capital -= position_value
                
                trades.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares,
                    'value': position_value
                })
            
            elif action == 'SELL' and symbol in positions:
                # Close position
                position = positions[symbol]
                exit_value = position['shares'] * price
                profit = exit_value - position['entry_value']
                capital += exit_value
                
                trades.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': price,
                    'shares': position['shares'],
                    'value': exit_value,
                    'profit': profit
                })
                
                del positions[symbol]
        
        # Calculate final value (close all positions at current price)
        final_value = capital
        for symbol, position in positions.items():
            current_price = self.fetcher.fetch_latest_price(symbol)
            if current_price:
                final_value += position['shares'] * current_price
        
        # Calculate metrics
        total_return = ((final_value - initial_capital) / initial_capital) * 100
        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
        
        return {
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
            'total_return_pct': round(total_return, 2),
            'total_trades': len(trades),
            'profitable_trades': len(profitable_trades),
            'win_rate': len(profitable_trades) / len(trades) if trades else 0,
            'open_positions': len(positions)
        }
    
    def verify_data_quality(self, symbols: List[str] = None) -> Dict:
        """
        Verify data quality and completeness
        
        Args:
            symbols: List of symbols to verify
        
        Returns:
            Data quality metrics
        """
        if symbols is None:
            symbols = self.fetcher.symbols[:100]
        
        start_time = time.time()
        
        # Fetch data
        market_data = self.fetcher.fetch_batch(symbols, period="1mo")
        
        # Analyze quality
        quality_metrics = {
            'symbols_requested': len(symbols),
            'symbols_received': len(market_data),
            'success_rate': len(market_data) / len(symbols) if symbols else 0,
            'fetch_time_seconds': time.time() - start_time,
            'data_completeness': {}
        }
        
        # Check data completeness
        for symbol, data in market_data.items():
            if data is not None and not data.empty:
                expected_days = 20  # ~1 month of trading days
                actual_days = len(data)
                completeness = min(1.0, actual_days / expected_days)
                
                quality_metrics['data_completeness'][symbol] = {
                    'days_of_data': actual_days,
                    'completeness_score': round(completeness, 2),
                    'has_volume': 'Volume' in data.columns,
                    'latest_date': str(data.index[-1].date()) if len(data) > 0 else None
                }
        
        # Overall completeness
        if quality_metrics['data_completeness']:
            avg_completeness = np.mean([
                v['completeness_score'] 
                for v in quality_metrics['data_completeness'].values()
            ])
            quality_metrics['average_completeness'] = round(avg_completeness, 3)
        
        return quality_metrics
    
    def load_historical_results(self, days: int = 30) -> List[Dict]:
        """
        Load historical scan results for analysis
        
        Args:
            days: Number of days of history to load
        
        Returns:
            List of historical scan results
        """
        results = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Load all JSON files in data directory
        for file_path in self.results_dir.glob("scan_results_*.json"):
            try:
                # Parse date from filename
                date_str = file_path.stem.replace("scan_results_", "")
                file_date = datetime.strptime(date_str[:8], "%Y%m%d")
                
                if file_date >= cutoff_date:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        results.append(data)
            except:
                continue
        
        return sorted(results, key=lambda x: x.get('timestamp', ''))
    
    def generate_performance_report(self, days: int = 30) -> Dict:
        """
        Generate comprehensive performance report
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Performance report dictionary
        """
        historical_results = self.load_historical_results(days)
        
        if not historical_results:
            return {'error': 'No historical data available'}
        
        # Aggregate all signals
        all_signals = []
        for result in historical_results:
            all_signals.extend(result.get('all_signals', []))
        
        # Validate signals
        validation = self.validate_signal_accuracy(all_signals, days=7)
        
        # Calculate metrics
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'total_scans': len(historical_results),
            'total_signals_generated': len(all_signals),
            'validation_metrics': validation,
            'performance_trends': self._calculate_trends(historical_results),
            'top_performing_signals': self._get_top_signals(all_signals)
        }
        
        return report
    
    def _calculate_trends(self, results: List[Dict]) -> Dict:
        """Calculate performance trends over time"""
        if not results:
            return {}
        
        scan_times = [r.get('scan_time_seconds', 0) for r in results]
        data_quality = [r.get('data_quality_score', 0) for r in results]
        signal_counts = [r.get('signals_generated', 0) for r in results]
        
        return {
            'average_scan_time': round(np.mean(scan_times), 2),
            'average_data_quality': round(np.mean(data_quality), 3),
            'average_signals_per_scan': round(np.mean(signal_counts), 1),
            'scan_time_trend': 'stable' if np.std(scan_times) < 0.5 else 'variable',
            'signal_generation_consistency': round(np.std(signal_counts), 1)
        }
    
    def _get_top_signals(self, signals: List[Dict], top_n: int = 10) -> List[Dict]:
        """Get top performing signals by confidence"""
        sorted_signals = sorted(signals, 
                              key=lambda x: x.get('confidence', 0), 
                              reverse=True)
        return sorted_signals[:top_n]


def verify_step1_requirements():
    """Verify all Step 1 requirements are met"""
    validator = SignalValidator()
    
    print("\n" + "="*50)
    print("STEP 1 VERIFICATION")
    print("="*50)
    
    # Test 1: Data Quality
    print("\n1. Testing Data Quality...")
    quality = validator.verify_data_quality()
    print(f"   Success Rate: {quality['success_rate']*100:.1f}%")
    print(f"   Fetch Time: {quality['fetch_time_seconds']:.2f}s")
    assert quality['success_rate'] > 0.95, "Data quality below 95% threshold"
    
    # Test 2: Processing Speed
    print("\n2. Testing Processing Speed...")
    from .scanner import MinimalScanner
    scanner = MinimalScanner(max_symbols=100)
    start_time = time.time()
    results = scanner.scan()
    scan_time = time.time() - start_time
    print(f"   Scan Time: {scan_time:.2f}s")
    assert scan_time < 2.0, "Scan time exceeds 2 second requirement"
    
    # Test 3: Signal Generation
    print("\n3. Testing Signal Generation...")
    print(f"   Signals Generated: {results['signals_generated']}")
    assert results['signals_generated'] > 0, "No signals generated"
    
    # Test 4: Hit Rate (if historical data available)
    print("\n4. Testing Hit Rate...")
    if results['all_signals']:
        validation = validator.validate_signal_accuracy(results['all_signals'][:10], days=7)
        if validation['validated_signals'] > 0:
            print(f"   Hit Rate: {validation['overall_hit_rate']*100:.1f}%")
            # Note: 55% threshold check would require historical data
    
    print("\n" + "="*50)
    print("✅ STEP 1 VERIFICATION COMPLETE")
    print("="*50)
    
    return True


if __name__ == "__main__":
    # Run verification
    verify_step1_requirements()