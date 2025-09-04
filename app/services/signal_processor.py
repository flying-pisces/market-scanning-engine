"""
Real-time signal processing service
Consumes market data from Kafka and generates trading signals
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

from app.core.kafka_client import KafkaConsumerClient, get_producer, KafkaTopics, kafka_config
from app.models.market import MarketDataPoint, AssetClass
from app.models.signal import SignalCreate, SignalType, TimeFrame
from app.services.risk_scoring import RiskScoringService
from app.services.matching import MatchingService
from app.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    volume_sma: Optional[float] = None


class PriceHistory:
    """Maintains price history for technical analysis"""
    
    def __init__(self, max_periods: int = 200):
        self.max_periods = max_periods
        self.prices = deque(maxlen=max_periods)
        self.volumes = deque(maxlen=max_periods)
        self.timestamps = deque(maxlen=max_periods)
    
    def add_data_point(self, price: float, volume: int, timestamp: datetime):
        """Add new price/volume data point"""
        self.prices.append(price)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)
    
    def get_prices(self, periods: int = None) -> List[float]:
        """Get recent prices"""
        if periods is None:
            return list(self.prices)
        return list(self.prices)[-periods:] if len(self.prices) >= periods else list(self.prices)
    
    def get_volumes(self, periods: int = None) -> List[int]:
        """Get recent volumes"""
        if periods is None:
            return list(self.volumes)
        return list(self.volumes)[-periods:] if len(self.volumes) >= periods else list(self.volumes)


class TechnicalAnalyzer:
    """Technical analysis calculations"""
    
    @staticmethod
    def sma(prices: List[float], periods: int) -> Optional[float]:
        """Simple Moving Average"""
        if len(prices) < periods:
            return None
        return np.mean(prices[-periods:])
    
    @staticmethod
    def ema(prices: List[float], periods: int, prev_ema: Optional[float] = None) -> Optional[float]:
        """Exponential Moving Average"""
        if len(prices) == 0:
            return None
        
        current_price = prices[-1]
        
        if prev_ema is None or len(prices) < periods:
            # Initialize with SMA
            return TechnicalAnalyzer.sma(prices, min(len(prices), periods))
        
        multiplier = 2 / (periods + 1)
        return (current_price * multiplier) + (prev_ema * (1 - multiplier))
    
    @staticmethod
    def rsi(prices: List[float], periods: int = 14) -> Optional[float]:
        """Relative Strength Index"""
        if len(prices) <= periods:
            return None
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD calculation"""
        if len(prices) < slow:
            return None, None, None
        
        ema_fast = TechnicalAnalyzer.ema(prices, fast)
        ema_slow = TechnicalAnalyzer.ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None, None, None
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, would need MACD history
        # Simplified for demo
        signal_line = macd_line * 0.9  # Mock signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: List[float], periods: int = 20, std_dev: float = 2.0) -> tuple:
        """Bollinger Bands"""
        if len(prices) < periods:
            return None, None, None
        
        sma = TechnicalAnalyzer.sma(prices, periods)
        if sma is None:
            return None, None, None
        
        std = np.std(prices[-periods:])
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band


class SignalGenerator:
    """Generate trading signals from technical analysis"""
    
    def __init__(self):
        self.risk_scorer = RiskScoringService()
    
    def generate_signals(
        self, 
        symbol: str,
        asset_class: AssetClass,
        current_price: float,
        indicators: TechnicalIndicators,
        price_history: PriceHistory
    ) -> List[SignalCreate]:
        """Generate signals based on technical indicators"""
        
        signals = []
        current_time = datetime.now(timezone.utc)
        
        # RSI-based signals
        if indicators.rsi is not None:
            if indicators.rsi < 30:  # Oversold
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.BUY,
                    description=f"RSI Oversold ({indicators.rsi:.1f}) - Potential Bounce",
                    entry_price=current_price,
                    confidence=min(95, 70 + (30 - indicators.rsi)),
                    timeframe=TimeFrame.INTRADAY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
            
            elif indicators.rsi > 70:  # Overbought
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.SELL,
                    description=f"RSI Overbought ({indicators.rsi:.1f}) - Potential Pullback",
                    entry_price=current_price,
                    confidence=min(95, 50 + (indicators.rsi - 70)),
                    timeframe=TimeFrame.INTRADAY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
        
        # Moving average crossover signals
        if indicators.sma_20 is not None and indicators.sma_50 is not None:
            if current_price > indicators.sma_20 > indicators.sma_50:
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.BUY,
                    description="Price above rising 20-day SMA",
                    entry_price=current_price,
                    confidence=75,
                    timeframe=TimeFrame.DAILY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
        
        # MACD signals
        if indicators.macd is not None and indicators.macd_signal is not None:
            if indicators.macd > indicators.macd_signal and indicators.macd_histogram > 0:
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.BUY,
                    description="MACD Bullish Crossover",
                    entry_price=current_price,
                    confidence=70,
                    timeframe=TimeFrame.DAILY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
        
        # Bollinger Bands signals
        if indicators.bollinger_lower is not None and indicators.bollinger_upper is not None:
            if current_price <= indicators.bollinger_lower:
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.BUY,
                    description="Price at Bollinger Band Support",
                    entry_price=current_price,
                    confidence=65,
                    timeframe=TimeFrame.INTRADAY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
            
            elif current_price >= indicators.bollinger_upper:
                signal = self._create_signal(
                    symbol=symbol,
                    asset_class=asset_class,
                    signal_type=SignalType.SELL,
                    description="Price at Bollinger Band Resistance",
                    entry_price=current_price,
                    confidence=65,
                    timeframe=TimeFrame.INTRADAY,
                    indicators=indicators,
                    current_time=current_time
                )
                signals.append(signal)
        
        return signals
    
    def _create_signal(
        self,
        symbol: str,
        asset_class: AssetClass,
        signal_type: SignalType,
        description: str,
        entry_price: float,
        confidence: float,
        timeframe: TimeFrame,
        indicators: TechnicalIndicators,
        current_time: datetime
    ) -> SignalCreate:
        """Create a signal with risk scoring"""
        
        # Calculate risk score
        risk_score = self.risk_scorer.calculate_risk_score(
            asset_class=asset_class,
            volatility=self._estimate_volatility(indicators),
            liquidity_score=80,  # Mock - would calculate from volume
            time_to_expiry=None,
            market_regime="normal"
        )
        
        # Set profit targets and stop losses based on asset class
        profit_target = self._calculate_profit_target(asset_class, entry_price, signal_type)
        stop_loss = self._calculate_stop_loss(asset_class, entry_price, signal_type)
        
        return SignalCreate(
            symbol=symbol,
            asset_class=asset_class,
            signal_type=signal_type,
            entry_price=entry_price,
            target_price=profit_target,
            stop_loss=stop_loss,
            confidence=confidence / 100.0,
            risk_score=risk_score,
            timeframe=timeframe,
            description=description,
            expires_at=current_time + timedelta(hours=self._get_signal_duration(timeframe)),
            metadata={
                "rsi": indicators.rsi,
                "sma_20": indicators.sma_20,
                "sma_50": indicators.sma_50,
                "macd": indicators.macd,
                "bollinger_upper": indicators.bollinger_upper,
                "bollinger_lower": indicators.bollinger_lower
            }
        )
    
    def _estimate_volatility(self, indicators: TechnicalIndicators) -> float:
        """Estimate volatility from indicators"""
        if indicators.bollinger_upper and indicators.bollinger_lower and indicators.sma_20:
            band_width = (indicators.bollinger_upper - indicators.bollinger_lower) / indicators.sma_20
            return min(100, band_width * 100)
        return 20  # Default moderate volatility
    
    def _calculate_profit_target(self, asset_class: AssetClass, entry_price: float, signal_type: SignalType) -> float:
        """Calculate profit target based on asset class"""
        target_pct = {
            AssetClass.DAILY_OPTIONS: 0.50,  # 50% for options
            AssetClass.STOCKS: 0.10,         # 10% for stocks
            AssetClass.ETFS: 0.05,           # 5% for ETFs
            AssetClass.BONDS: 0.02,          # 2% for bonds
            AssetClass.SAFE_ASSETS: 0.01     # 1% for safe assets
        }.get(asset_class, 0.05)
        
        if signal_type == SignalType.BUY:
            return entry_price * (1 + target_pct)
        else:
            return entry_price * (1 - target_pct)
    
    def _calculate_stop_loss(self, asset_class: AssetClass, entry_price: float, signal_type: SignalType) -> float:
        """Calculate stop loss based on asset class"""
        stop_pct = {
            AssetClass.DAILY_OPTIONS: 0.30,  # 30% stop for options
            AssetClass.STOCKS: 0.05,         # 5% stop for stocks
            AssetClass.ETFS: 0.03,           # 3% stop for ETFs
            AssetClass.BONDS: 0.02,          # 2% stop for bonds
            AssetClass.SAFE_ASSETS: 0.01     # 1% stop for safe assets
        }.get(asset_class, 0.03)
        
        if signal_type == SignalType.BUY:
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)
    
    def _get_signal_duration(self, timeframe: TimeFrame) -> int:
        """Get signal duration in hours"""
        durations = {
            TimeFrame.INTRADAY: 4,    # 4 hours
            TimeFrame.DAILY: 24,      # 1 day
            TimeFrame.WEEKLY: 168,    # 1 week
            TimeFrame.MONTHLY: 720    # 30 days
        }
        return durations.get(timeframe, 24)


class SignalProcessor:
    """Main signal processing service"""
    
    def __init__(self):
        self.price_histories: Dict[str, PriceHistory] = defaultdict(lambda: PriceHistory())
        self.technical_analyzer = TechnicalAnalyzer()
        self.signal_generator = SignalGenerator()
        self.matching_service = MatchingService()
        self.producer = None
        self.is_running = False
        
        # Track indicators for each symbol
        self.indicator_cache: Dict[str, TechnicalIndicators] = defaultdict(TechnicalIndicators)
    
    async def start(self):
        """Start signal processing service"""
        logger.info("Starting signal processing service")
        
        self.producer = get_producer()
        if not self.producer:
            raise RuntimeError("Kafka producer not available")
        
        self.is_running = True
        
        # Set up consumers
        market_data_consumer = KafkaConsumerClient(
            config=kafka_config,
            group_id="signal-processor-market-data",
            topics=[KafkaTopics.MARKET_DATA_RAW]
        )
        
        options_consumer = KafkaConsumerClient(
            config=kafka_config,
            group_id="signal-processor-options",
            topics=[KafkaTopics.OPTIONS_DATA]
        )
        
        # Start processing tasks
        tasks = [
            self._process_market_data(market_data_consumer),
            self._process_options_data(options_consumer),
            self._cleanup_expired_signals()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop signal processing service"""
        logger.info("Stopping signal processing service")
        self.is_running = False
    
    async def _process_market_data(self, consumer: KafkaConsumerClient):
        """Process incoming market data and generate signals"""
        
        async def handle_market_data(topic: str, message: Dict[str, Any]):
            try:
                # Parse market data
                symbol = message.get("symbol")
                price = float(message.get("price", 0))
                volume = int(message.get("volume", 0))
                asset_class = AssetClass(message.get("asset_class", AssetClass.STOCKS.value))
                timestamp = datetime.fromisoformat(message.get("timestamp").replace("Z", "+00:00"))
                
                # Update price history
                price_history = self.price_histories[symbol]
                price_history.add_data_point(price, volume, timestamp)
                
                # Calculate technical indicators
                indicators = await self._calculate_indicators(symbol, price_history)
                self.indicator_cache[symbol] = indicators
                
                # Generate signals
                signals = self.signal_generator.generate_signals(
                    symbol=symbol,
                    asset_class=asset_class,
                    current_price=price,
                    indicators=indicators,
                    price_history=price_history
                )
                
                # Publish signals
                for signal in signals:
                    await self._publish_signal(signal)
                
                if signals:
                    logger.info(f"Generated {len(signals)} signals for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing market data: {e}")
        
        async for message in consumer.consume_messages(handle_market_data):
            if not self.is_running:
                break
    
    async def _process_options_data(self, consumer: KafkaConsumerClient):
        """Process options data for high-risk signals"""
        
        async def handle_options_data(topic: str, message: Dict[str, Any]):
            try:
                # Options-specific signal generation
                symbol = message.get("underlying_symbol")
                option_type = message.get("option_type")
                strike = float(message.get("strike", 0))
                expiry = message.get("expiry_date")
                iv = float(message.get("implied_volatility", 0.25))
                
                # Generate options-specific signals based on Greeks and IV
                if iv > 0.40:  # High IV - potential mean reversion
                    signal = SignalCreate(
                        symbol=f"{symbol}_{strike}_{option_type}",
                        asset_class=AssetClass.DAILY_OPTIONS,
                        signal_type=SignalType.SELL,
                        entry_price=float(message.get("bid", 1.0)),
                        target_price=float(message.get("bid", 1.0)) * 0.5,
                        stop_loss=float(message.get("bid", 1.0)) * 1.5,
                        confidence=0.75,
                        risk_score=90,  # High risk for options
                        timeframe=TimeFrame.INTRADAY,
                        description=f"High IV ({iv:.1%}) - Sell Premium",
                        expires_at=datetime.fromisoformat(expiry.replace("Z", "+00:00")),
                        metadata=message
                    )
                    
                    await self._publish_signal(signal)
                
            except Exception as e:
                logger.error(f"Error processing options data: {e}")
        
        async for message in consumer.consume_messages(handle_options_data):
            if not self.is_running:
                break
    
    async def _calculate_indicators(self, symbol: str, price_history: PriceHistory) -> TechnicalIndicators:
        """Calculate technical indicators for symbol"""
        prices = price_history.get_prices()
        volumes = price_history.get_volumes()
        
        if len(prices) < 10:
            return TechnicalIndicators()  # Not enough data
        
        indicators = TechnicalIndicators()
        
        # Moving averages
        indicators.sma_20 = self.technical_analyzer.sma(prices, 20)
        indicators.sma_50 = self.technical_analyzer.sma(prices, 50)
        indicators.ema_12 = self.technical_analyzer.ema(prices, 12)
        indicators.ema_26 = self.technical_analyzer.ema(prices, 26)
        
        # RSI
        indicators.rsi = self.technical_analyzer.rsi(prices, 14)
        
        # MACD
        macd, signal, histogram = self.technical_analyzer.macd(prices)
        indicators.macd = macd
        indicators.macd_signal = signal
        indicators.macd_histogram = histogram
        
        # Bollinger Bands
        upper, middle, lower = self.technical_analyzer.bollinger_bands(prices)
        indicators.bollinger_upper = upper
        indicators.bollinger_lower = lower
        
        # Volume SMA
        if volumes:
            indicators.volume_sma = self.technical_analyzer.sma(volumes, 20)
        
        return indicators
    
    async def _publish_signal(self, signal: SignalCreate):
        """Publish generated signal to Kafka"""
        try:
            # Convert to dict for JSON serialization
            signal_data = {
                "symbol": signal.symbol,
                "asset_class": signal.asset_class.value,
                "signal_type": signal.signal_type.value,
                "entry_price": signal.entry_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "confidence": signal.confidence,
                "risk_score": signal.risk_score,
                "timeframe": signal.timeframe.value,
                "description": signal.description,
                "expires_at": signal.expires_at.isoformat(),
                "metadata": signal.metadata or {},
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Publish to signals topic
            await self.producer.send_message(
                KafkaTopics.SIGNALS_GENERATED,
                signal_data,
                key=signal.symbol
            )
            
            # Trigger matching for this signal
            await self._trigger_matching(signal_data)
            
        except Exception as e:
            logger.error(f"Error publishing signal: {e}")
    
    async def _trigger_matching(self, signal_data: Dict[str, Any]):
        """Trigger user-signal matching for new signal"""
        try:
            # This would typically query users from database
            # For now, publish to matching topic for processing
            await self.producer.send_message(
                KafkaTopics.SIGNALS_MATCHED,
                {
                    "signal": signal_data,
                    "trigger": "new_signal",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                key=f"match_{signal_data['symbol']}"
            )
            
        except Exception as e:
            logger.error(f"Error triggering matching: {e}")
    
    async def _cleanup_expired_signals(self):
        """Periodically cleanup expired signals"""
        while self.is_running:
            try:
                # This would clean up expired signals from database
                # For now, just log
                logger.info("Cleaning up expired signals")
                
                await asyncio.sleep(3600)  # Every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")


async def main():
    """Main entry point for signal processing service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    processor = SignalProcessor()
    
    try:
        await processor.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())