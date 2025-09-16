"""
Step 2 Signal Generator
Converts Step 2 predictions into signal_visualization format
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# Add signal_visualization to path
signal_viz_path = str(Path(__file__).parent.parent.parent / "signal_visualization")
sys.path.append(signal_viz_path)

# Import signal renderer
try:
    from signal_renderer import (
        SignalRenderer, SignalData, SignalType, SignalPriority,
        ChartData, KeyStat, StrategyInfo
    )
except ImportError:
    print("Warning: Could not import signal_renderer. Signal generation may be limited.")

from step2_engine import Step2PredictionEngine
from prediction_strategies import PredictionResult

class Step2SignalGenerator:
    """
    Generates mobile-optimized signal visualizations from Step 2 predictions
    Integrates with existing signal_visualization system
    """
    
    def __init__(self, output_dir: str = "step2/signals"):
        self.prediction_engine = Step2PredictionEngine()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize signal renderer if available
        try:
            self.signal_renderer = SignalRenderer(output_dir)
        except:
            self.signal_renderer = None
        
        # Strategy to signal type mapping
        self.strategy_signal_types = {
            'Technical-LSTM Hybrid': SignalType.UNUSUAL_OPTIONS,
            'ARIMA-Momentum Ensemble': SignalType.PRE_MARKET,
            'Event-Driven Volatility': SignalType.FDA_EVENT,
            'Ensemble Meta-Learner': SignalType.YOLO_CALLS
        }
    
    def generate_signal_for_ticker(self, ticker: str, strategy: str = "ensemble") -> Dict:
        """
        Generate complete signal visualization for a ticker
        
        Args:
            ticker: Stock symbol
            strategy: Prediction strategy to use
            
        Returns:
            Complete signal data including HTML path
        """
        try:
            # Get prediction from Step 2 engine
            prediction_result = self.prediction_engine.generate_predictions(ticker, strategy)
            
            if not prediction_result['success']:
                return self._generate_error_signal(ticker, prediction_result['error'])
            
            # Convert to SignalData format
            signal_data = self._convert_to_signal_data(prediction_result)
            
            # Generate HTML signal if renderer available
            html_path = None
            if self.signal_renderer and signal_data:
                html_path = self.signal_renderer.render_signal(signal_data)
            
            return {
                'success': True,
                'ticker': ticker,
                'strategy': strategy,
                'signal_data': signal_data,
                'html_path': html_path,
                'prediction_result': prediction_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._generate_error_signal(ticker, f"Signal generation failed: {str(e)}")
    
    def generate_multi_strategy_signals(self, ticker: str) -> Dict:
        """
        Generate signals for all strategies for comparison
        
        Returns:
            Dictionary with signals from all strategies
        """
        try:
            # Get multi-strategy predictions
            multi_predictions = self.prediction_engine.get_multiple_strategy_predictions(ticker)
            
            if not multi_predictions['success']:
                return {'success': False, 'error': 'Multi-strategy prediction failed'}
            
            strategy_signals = {}
            
            for strategy_name, strategy_data in multi_predictions['strategies'].items():
                if 'error' not in strategy_data and strategy_data['prediction']:
                    # Create mini prediction result for each strategy
                    mini_result = {
                        'success': True,
                        'ticker': ticker,
                        'strategy': strategy_name,
                        'prediction': strategy_data['prediction'],
                        'visualization': strategy_data['visualization'],
                        'events': multi_predictions['events'],
                        'timestamp': multi_predictions['timestamp']
                    }
                    
                    # Convert to signal data
                    signal_data = self._convert_to_signal_data(mini_result)
                    
                    # Generate HTML
                    html_path = None
                    if self.signal_renderer and signal_data:
                        filename = f"{ticker.lower()}_{strategy_name.lower().replace(' ', '_')}.html"
                        html_path = self.signal_renderer.render_signal(signal_data, filename)
                    
                    strategy_signals[strategy_name] = {
                        'signal_data': signal_data,
                        'html_path': html_path,
                        'prediction': strategy_data['prediction']
                    }
                else:
                    strategy_signals[strategy_name] = {
                        'error': strategy_data.get('error', 'Unknown error'),
                        'signal_data': None,
                        'html_path': None
                    }
            
            return {
                'success': True,
                'ticker': ticker,
                'strategies': strategy_signals,
                'events': multi_predictions['events'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Multi-strategy signal generation failed: {str(e)}"}
    
    def _convert_to_signal_data(self, prediction_result: Dict) -> Optional['SignalData']:
        """Convert Step 2 prediction to SignalData format"""
        try:
            if not prediction_result['success']:
                return None
            
            prediction = prediction_result['prediction']
            visualization = prediction_result['visualization']
            
            # Determine signal type based on strategy
            signal_type = self.strategy_signal_types.get(
                prediction.strategy_name, 
                SignalType.UNUSUAL_OPTIONS
            )
            
            # Determine priority based on confidence and events
            priority = self._determine_priority(prediction, prediction_result.get('events', []))
            
            # Calculate price change
            current_price = prediction.current_price
            target_price = prediction.predicted_prices[-1] if prediction.predicted_prices else current_price
            price_change = target_price - current_price
            price_change_percent = (price_change / current_price) * 100 if current_price > 0 else 0
            
            # Create chart data
            chart_data = ChartData(
                historical_data=visualization['chart_data']['historical_data'],
                prediction_upper=visualization['chart_data']['prediction_upper'],
                prediction_base=visualization['chart_data']['prediction_base'],
                prediction_lower=visualization['chart_data']['prediction_lower'],
                event_label=visualization['chart_data']['event_label'],
                chart_color=visualization['chart_data']['chart_color']
            )
            
            # Convert key stats
            key_stats = [
                KeyStat(
                    value=stat['value'],
                    label=stat['label'],
                    is_positive=stat['is_positive']
                ) for stat in visualization['key_stats']
            ]
            
            # Create strategy info
            strategy_info = StrategyInfo(
                title=visualization['strategy_info']['title'],
                description=visualization['strategy_info']['description'],
                link_text="Algorithm Details →",
                link_url=visualization['strategy_info']['link_url']
            )
            
            # Determine company name (simplified for MVP)
            company_name = self._get_company_name(prediction.ticker)
            
            # Determine special flags
            is_yolo = signal_type == SignalType.YOLO_CALLS or prediction.confidence > 85
            has_animation = signal_type in [SignalType.YOLO_CALLS, SignalType.FDA_EVENT]
            border_style = "dashed" if signal_type == SignalType.PRE_MARKET else "solid"
            
            # Create SignalData
            signal_data = SignalData(
                ticker=prediction.ticker,
                company_name=company_name,
                signal_type=signal_type,
                current_price=current_price,
                price_change=price_change,
                price_change_percent=price_change_percent,
                priority=priority,
                key_stats=key_stats,
                strategy=strategy_info,
                chart_data=chart_data,
                timestamp=self._format_timestamp(prediction_result['timestamp']),
                notifications_enabled=True,
                is_yolo=is_yolo,
                has_animation=has_animation,
                border_style=border_style
            )
            
            return signal_data
            
        except Exception as e:
            print(f"Error converting to SignalData: {e}")
            return None
    
    def _determine_priority(self, prediction: 'PredictionResult', events: List[Dict]) -> 'SignalPriority':
        """Determine signal priority based on confidence and events"""
        try:
            from signal_renderer import SignalPriority
            
            # High confidence predictions
            if prediction.confidence >= 85:
                return SignalPriority.HOT
            elif prediction.confidence >= 75:
                return SignalPriority.URGENT
            
            # Event-driven priorities
            for event in events:
                if event.get('impact_score', 0) >= 8:
                    return SignalPriority.HOT
                elif event.get('impact_score', 0) >= 6:
                    return SignalPriority.URGENT
            
            # Default based on confidence
            if prediction.confidence >= 60:
                return SignalPriority.NORMAL
            else:
                return SignalPriority.WATCH
                
        except:
            # Fallback if SignalPriority not available
            class FallbackPriority:
                NORMAL = "NORMAL"
            return FallbackPriority.NORMAL
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for ticker (simplified mapping for MVP)"""
        company_names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla, Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms, Inc.',
            'NFLX': 'Netflix, Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'SAVA': 'Cassava Sciences',
            'MRNA': 'Moderna, Inc.',
            'BNTX': 'BioNTech SE',
            'AMD': 'Advanced Micro Devices'
        }
        return company_names.get(ticker.upper(), f"{ticker.upper()} Corporation")
    
    def _format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now()
            diff = now - dt.replace(tzinfo=None)
            
            if diff.total_seconds() < 60:
                return "Just now"
            elif diff.total_seconds() < 3600:
                minutes = int(diff.total_seconds() / 60)
                return f"{minutes} minutes ago"
            elif diff.total_seconds() < 86400:
                hours = int(diff.total_seconds() / 3600)
                return f"{hours} hours ago"
            else:
                return dt.strftime("%m/%d %H:%M")
        except:
            return "Recently"
    
    def _generate_error_signal(self, ticker: str, error_message: str) -> Dict:
        """Generate error response for signal generation"""
        return {
            'success': False,
            'ticker': ticker,
            'error': error_message,
            'signal_data': None,
            'html_path': None,
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_batch_signals(self, tickers: List[str], strategy: str = "ensemble") -> Dict:
        """Generate signals for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            print(f"Generating signal for {ticker}...")
            results[ticker] = self.generate_signal_for_ticker(ticker, strategy)
        
        successful_signals = [r for r in results.values() if r['success']]
        
        return {
            'success': True,
            'total_tickers': len(tickers),
            'successful_signals': len(successful_signals),
            'failed_signals': len(tickers) - len(successful_signals),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_signal_data(self, ticker: str, strategy: str = "ensemble", format: str = "json") -> str:
        """Export signal data in various formats"""
        signal_result = self.generate_signal_for_ticker(ticker, strategy)
        
        if format == "json":
            filename = f"{self.output_dir}/{ticker.lower()}_{strategy}_signal.json"
            with open(filename, 'w') as f:
                # Convert signal_data to dict for JSON serialization
                exportable_data = {
                    'ticker': signal_result.get('ticker'),
                    'strategy': signal_result.get('strategy'),
                    'prediction_summary': {
                        'current_price': signal_result['prediction_result']['prediction'].current_price if signal_result['success'] else None,
                        'target_price': signal_result['prediction_result']['prediction'].predicted_prices[-1] if signal_result['success'] and signal_result['prediction_result']['prediction'].predicted_prices else None,
                        'confidence': signal_result['prediction_result']['prediction'].confidence if signal_result['success'] else None,
                        'rationale': signal_result['prediction_result']['prediction'].rationale if signal_result['success'] else None
                    },
                    'timestamp': signal_result.get('timestamp'),
                    'success': signal_result['success']
                }
                json.dump(exportable_data, f, indent=2)
            return filename
        
        return None

# Demo and testing functionality
if __name__ == "__main__":
    print("🎨 Step 2 Signal Generator Demo")
    print("=" * 50)
    
    # Initialize signal generator
    generator = Step2SignalGenerator()
    
    # Test tickers
    test_tickers = ['AAPL', 'TSLA', 'SAVA']
    
    for ticker in test_tickers:
        print(f"\n📊 Generating signal for {ticker}...")
        
        # Generate single strategy signal
        result = generator.generate_signal_for_ticker(ticker, "ensemble")
        
        if result['success']:
            print(f"✅ Signal generated successfully")
            print(f"HTML Path: {result['html_path']}")
            
            prediction = result['prediction_result']['prediction']
            print(f"Current Price: ${prediction.current_price:.2f}")
            print(f"Target Price: ${prediction.predicted_prices[-1]:.2f}")
            print(f"Confidence: {prediction.confidence:.0f}%")
        else:
            print(f"❌ Error: {result['error']}")
    
    # Test multi-strategy generation
    print(f"\n🎭 Multi-Strategy Signals for AAPL...")
    multi_result = generator.generate_multi_strategy_signals('AAPL')
    
    if multi_result['success']:
        for strategy_name, strategy_data in multi_result['strategies'].items():
            if 'error' not in strategy_data:
                print(f"✅ {strategy_name}: {strategy_data['html_path']}")
            else:
                print(f"❌ {strategy_name}: {strategy_data['error']}")
    
    # Test batch generation
    print(f"\n📦 Batch Signal Generation...")
    batch_result = generator.generate_batch_signals(['AAPL', 'MSFT'], "technical-lstm")
    print(f"Generated {batch_result['successful_signals']}/{batch_result['total_tickers']} signals")
    
    print(f"\n✅ Step 2 Signal Generator Demo Complete!")
    print("Ready for mobile app integration!")