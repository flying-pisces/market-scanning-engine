"""
Complete Signal Generation Framework Example
Author: Claude Code (System Architect)
Version: 1.0

Comprehensive example demonstrating how to use the signal generation framework
for multi-asset class trading with risk-aware signal generation.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict
from uuid import UUID, uuid4

# Framework imports
from signal_generation.core.orchestrator import SignalOrchestrator, OrchestrationConfig, OrchestrationMode
from signal_generation.core.signal_factory import SignalGeneratorFactory
from signal_generation.config.strategy_config import ConfigurationManager
from signal_generation.strategies.technical import MovingAverageSignalGenerator, RSISignalGenerator
from signal_generation.strategies.options import OptionsFlowSignalGenerator, GammaExposureSignalGenerator

# Data model imports
from data_models.python.core_models import Asset, MarketData, TechnicalIndicators, OptionsData, OptionType
from data_models.python.signal_models import Signal, SignalCategory


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("signal_framework_example")


class MockDataProvider:
    """Mock data provider for demonstration purposes"""
    
    def __init__(self):
        self.assets = self._create_sample_assets()
        self.market_data = self._create_sample_market_data()
        self.technical_indicators = self._create_sample_technical_indicators()
        self.options_data = self._create_sample_options_data()
    
    def _create_sample_assets(self) -> List[Asset]:
        """Create sample assets for different asset classes"""
        assets = []
        
        # Daily Options Assets
        spy_asset = Asset(
            id=uuid4(),
            symbol="SPY",
            name="SPDR S&P 500 ETF Trust",
            exchange="ARCA",
            currency="USD",
            sector="ETF",
            market_cap=50000000000000,  # $500B
            avg_volume_30d=50000000,
            is_active=True,
            metadata={"asset_class": "daily_options", "options_available": True}
        )
        assets.append(spy_asset)
        
        # Large Cap Stock
        aapl_asset = Asset(
            id=uuid4(),
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            currency="USD",
            sector="Technology",
            market_cap=300000000000000,  # $3T
            avg_volume_30d=75000000,
            is_active=True,
            metadata={"asset_class": "large_cap_stocks"}
        )
        assets.append(aapl_asset)
        
        # Small Cap Stock
        small_cap_asset = Asset(
            id=uuid4(),
            symbol="SFIX",
            name="Stitch Fix, Inc.",
            exchange="NASDAQ",
            currency="USD",
            sector="Consumer Discretionary",
            market_cap=80000000000,  # $800M
            avg_volume_30d=1500000,
            is_active=True,
            metadata={"asset_class": "small_cap_stocks"}
        )
        assets.append(small_cap_asset)
        
        # Bond ETF
        tlt_asset = Asset(
            id=uuid4(),
            symbol="TLT",
            name="iShares 20+ Year Treasury Bond ETF",
            exchange="NASDAQ",
            currency="USD",
            sector="Fixed Income",
            market_cap=2000000000000,  # $20B
            avg_volume_30d=15000000,
            is_active=True,
            metadata={"asset_class": "bonds"}
        )
        assets.append(tlt_asset)
        
        return assets
    
    def _create_sample_market_data(self) -> Dict[UUID, MarketData]:
        """Create sample market data"""
        market_data = {}
        
        for asset in self.assets:
            # Generate realistic price data based on asset class
            if asset.symbol == "SPY":
                price_cents = 45000  # $450.00
            elif asset.symbol == "AAPL":
                price_cents = 17500   # $175.00
            elif asset.symbol == "SFIX":
                price_cents = 800     # $8.00
            elif asset.symbol == "TLT":
                price_cents = 9500    # $95.00
            else:
                price_cents = 10000   # Default $100.00
            
            # Create OHLC data with some variation
            open_price = int(price_cents * 0.995)
            high_price = int(price_cents * 1.012)
            low_price = int(price_cents * 0.988)
            close_price = price_cents
            
            # Volume based on asset's average
            volume = int(asset.avg_volume_30d * (0.8 + 0.4 * 0.7))  # 80-120% of average
            
            market_data[asset.id] = MarketData(
                id=uuid4(),
                asset_id=asset.id,
                timestamp=datetime.utcnow(),
                open_price_cents=open_price,
                high_price_cents=high_price,
                low_price_cents=low_price,
                close_price_cents=close_price,
                volume=volume,
                bid_price_cents=close_price - 5,
                ask_price_cents=close_price + 5,
                bid_size=100,
                ask_size=100,
                data_source="mock_provider",
                data_quality_score=95
            )
        
        return market_data
    
    def _create_sample_technical_indicators(self) -> Dict[UUID, TechnicalIndicators]:
        """Create sample technical indicators"""
        technical_indicators = {}
        
        for asset in self.assets:
            price = self.market_data[asset.id].close_price_cents
            
            # Generate realistic technical indicators
            technical_indicators[asset.id] = TechnicalIndicators(
                id=uuid4(),
                asset_id=asset.id,
                timestamp=datetime.utcnow(),
                timeframe="1d",
                
                # Moving averages
                sma_20=int(price * 1.02),   # Slightly above current price (bullish)
                sma_50=int(price * 1.05),   # Further above (longer term bullish)
                sma_200=int(price * 1.08),  # Even higher (strong trend)
                ema_12=int(price * 1.01),
                ema_26=int(price * 1.03),
                
                # Momentum indicators
                rsi_14=45.5,  # Slightly oversold (bullish signal potential)
                macd_line=int(price * 0.001),       # MACD in cents
                macd_signal=int(price * 0.0008),    # Signal line
                macd_histogram=int(price * 0.0002), # Positive histogram
                
                # Bollinger Bands
                bollinger_upper=int(price * 1.04),
                bollinger_middle=int(price * 1.02),
                bollinger_lower=int(price * 1.00),
                atr_14=int(price * 0.025),  # 2.5% ATR
                
                # Volume indicators
                volume_sma_20=asset.avg_volume_30d,
                on_balance_volume=asset.avg_volume_30d * 100,
                
                # Support/Resistance (simplified)
                support_level=int(price * 0.95),
                resistance_level=int(price * 1.08)
            )
        
        return technical_indicators
    
    def _create_sample_options_data(self) -> Dict[UUID, List[OptionsData]]:
        """Create sample options data for assets that support options"""
        options_data = {}
        
        for asset in self.assets:
            if asset.metadata.get("options_available"):
                current_price = self.market_data[asset.id].close_price_cents
                options_chain = []
                
                # Create options chain with various strikes around current price
                base_strikes = [0.95, 0.98, 1.00, 1.02, 1.05, 1.08, 1.10]
                expirations = [7, 14, 30, 45]  # Days to expiry
                
                for dte in expirations:
                    expiration_date = datetime.utcnow().date() + timedelta(days=dte)
                    
                    for strike_mult in base_strikes:
                        strike_price = int(current_price * strike_mult)
                        
                        # Create call option
                        call_option = OptionsData(
                            id=uuid4(),
                            underlying_asset_id=asset.id,
                            option_symbol=f"{asset.symbol}{expiration_date.strftime('%y%m%d')}C{strike_price/100:.0f}",
                            expiration_date=expiration_date,
                            strike_price_cents=strike_price,
                            option_type=OptionType.CALL,
                            contract_size=100,
                            timestamp=datetime.utcnow(),
                            
                            # Realistic options pricing (simplified)
                            bid_price_cents=max(5, int(abs(current_price - strike_price) * 0.1)),
                            ask_price_cents=max(10, int(abs(current_price - strike_price) * 0.12)),
                            last_price_cents=max(8, int(abs(current_price - strike_price) * 0.11)),
                            
                            # Volume and OI (higher for ATM options)
                            volume=max(10, int(1000 / (1 + abs(strike_mult - 1) * 10))),
                            open_interest=max(50, int(5000 / (1 + abs(strike_mult - 1) * 20))),
                            
                            # Greeks (scaled by 10000)
                            delta=int((strike_mult - 1) * 2000 + 5000),  # Simplified delta
                            gamma=int(1000 / (1 + abs(strike_mult - 1) * 5)),
                            theta=-int(50 + dte),
                            vega=int(200 - dte * 2),
                            rho=int(dte * 10),
                            
                            # IV (scaled by 10000)
                            implied_volatility=int(2500 + abs(strike_mult - 1) * 500),  # 25% base IV
                            
                            data_source="mock_options_provider"
                        )
                        options_chain.append(call_option)
                        
                        # Create put option
                        put_option = OptionsData(
                            id=uuid4(),
                            underlying_asset_id=asset.id,
                            option_symbol=f"{asset.symbol}{expiration_date.strftime('%y%m%d')}P{strike_price/100:.0f}",
                            expiration_date=expiration_date,
                            strike_price_cents=strike_price,
                            option_type=OptionType.PUT,
                            contract_size=100,
                            timestamp=datetime.utcnow(),
                            
                            # Put pricing
                            bid_price_cents=max(5, int(abs(strike_price - current_price) * 0.1)),
                            ask_price_cents=max(10, int(abs(strike_price - current_price) * 0.12)),
                            last_price_cents=max(8, int(abs(strike_price - current_price) * 0.11)),
                            
                            # Volume and OI
                            volume=max(10, int(800 / (1 + abs(strike_mult - 1) * 8))),
                            open_interest=max(50, int(4000 / (1 + abs(strike_mult - 1) * 15))),
                            
                            # Put Greeks
                            delta=int((1 - strike_mult) * 2000 - 5000),  # Negative delta for puts
                            gamma=int(1000 / (1 + abs(strike_mult - 1) * 5)),
                            theta=-int(50 + dte),
                            vega=int(200 - dte * 2),
                            rho=-int(dte * 10),
                            
                            implied_volatility=int(2500 + abs(strike_mult - 1) * 500),
                            
                            data_source="mock_options_provider"
                        )
                        options_chain.append(put_option)
                
                options_data[asset.id] = options_chain
        
        return options_data
    
    async def get_assets(self) -> List[Asset]:
        """Mock asset provider"""
        return self.assets
    
    async def get_market_data(self, assets: List[Asset]) -> Dict[UUID, MarketData]:
        """Mock market data provider"""
        return {asset.id: self.market_data[asset.id] for asset in assets if asset.id in self.market_data}
    
    async def get_technical_indicators(self, assets: List[Asset]) -> Dict[UUID, TechnicalIndicators]:
        """Mock technical indicators provider"""
        return {asset.id: self.technical_indicators[asset.id] for asset in assets if asset.id in self.technical_indicators}
    
    async def get_options_data(self, assets: List[Asset]) -> Dict[UUID, List[OptionsData]]:
        """Mock options data provider"""
        return {asset.id: self.options_data[asset.id] for asset in assets if asset.id in self.options_data}


async def output_handler(signals: List[Signal]) -> None:
    """Handle generated signals output"""
    logger.info(f"Received {len(signals)} signals for output")
    
    for signal in signals:
        logger.info(
            f"Signal: {signal.signal_name} | "
            f"Asset: {signal.asset_id} | "
            f"Direction: {signal.direction} | "
            f"Risk: {signal.risk_score} | "
            f"Confidence: {signal.confidence_score} | "
            f"Entry: ${signal.entry_price_cents/100 if signal.entry_price_cents else 'N/A'}"
        )


async def notification_handler(high_priority_signals: List[Signal]) -> None:
    """Handle high-priority signal notifications"""
    logger.info(f"High-priority notification: {len(high_priority_signals)} signals")
    
    for signal in high_priority_signals:
        composite_score = (signal.confidence_score * signal.profit_potential_score) / 100
        logger.warning(
            f"HIGH PRIORITY: {signal.signal_name} | "
            f"Composite Score: {composite_score:.1f} | "
            f"Risk: {signal.risk_score}"
        )


async def main():
    """Main example demonstrating the complete signal generation framework"""
    
    logger.info("Starting Signal Generation Framework Example")
    
    # 1. Initialize Configuration Manager
    logger.info("1. Initializing Configuration Manager")
    config_manager = ConfigurationManager()
    
    # Save sample configuration file
    config_file_path = "/tmp/signal_config_example.json"
    config_manager.save_config_file(config_file_path, format="json")
    logger.info(f"Sample configuration saved to: {config_file_path}")
    
    # 2. Setup Mock Data Provider
    logger.info("2. Setting up Mock Data Provider")
    data_provider = MockDataProvider()
    
    # 3. Register Signal Generators with Factory
    logger.info("3. Registering Signal Generators")
    
    # Register generator classes
    SignalGeneratorFactory.register(MovingAverageSignalGenerator, "MovingAverageSignalGenerator")
    SignalGeneratorFactory.register(RSISignalGenerator, "RSISignalGenerator")
    SignalGeneratorFactory.register(OptionsFlowSignalGenerator, "OptionsFlowSignalGenerator")
    SignalGeneratorFactory.register(GammaExposureSignalGenerator, "GammaExposureSignalGenerator")
    
    # Create generator instances for different asset classes
    generators = []
    
    # Large cap stock generators
    large_cap_ma = SignalGeneratorFactory.create_technical_generator(
        "MovingAverageSignalGenerator", "large_cap_trend", "large_cap_stocks"
    )
    generators.append(large_cap_ma)
    
    large_cap_rsi = SignalGeneratorFactory.create_technical_generator(
        "RSISignalGenerator", "large_cap_momentum", "large_cap_stocks"
    )
    generators.append(large_cap_rsi)
    
    # Daily options generators
    options_flow = SignalGeneratorFactory.create_options_generator(
        "OptionsFlowSignalGenerator", "spy_flow"
    )
    generators.append(options_flow)
    
    gamma_exposure = SignalGeneratorFactory.create_options_generator(
        "GammaExposureSignalGenerator", "spy_gamma"
    )
    generators.append(gamma_exposure)
    
    logger.info(f"Created {len(generators)} signal generators")
    
    # 4. Configure Orchestrator
    logger.info("4. Configuring Signal Orchestrator")
    
    orchestration_config = OrchestrationConfig(
        mode=OrchestrationMode.ON_DEMAND,  # Manual execution for example
        max_concurrent_generators=5,
        max_signals_per_run=100,
        enable_validation=True,
        enable_risk_assessment=True,
        enable_deduplication=True,
        min_confidence_score=50,
        max_risk_score=90,
        min_quality_score=60
    )
    
    orchestrator = SignalOrchestrator(orchestration_config)
    
    # Register generators
    for generator in generators:
        orchestrator.register_generator(generator)
    
    # Register data providers
    orchestrator.register_data_provider("assets", data_provider.get_assets)
    orchestrator.register_data_provider("market_data", data_provider.get_market_data)
    orchestrator.register_data_provider("technical_indicators", data_provider.get_technical_indicators)
    orchestrator.register_data_provider("options_data", data_provider.get_options_data)
    
    # Register handlers
    orchestrator.register_output_handler(output_handler)
    orchestrator.register_notification_handler(notification_handler)
    
    # 5. Execute Signal Generation
    logger.info("5. Executing Signal Generation")
    
    # Execute single run
    metrics = await orchestrator.execute_single_run()
    
    logger.info("Execution Complete!")
    logger.info(f"Run ID: {metrics.run_id}")
    logger.info(f"Assets Processed: {metrics.assets_processed}")
    logger.info(f"Generators Executed: {metrics.generators_executed}")
    logger.info(f"Signals Generated: {metrics.signals_generated}")
    logger.info(f"Signals Validated: {metrics.signals_validated}")
    logger.info(f"Signals Filtered: {metrics.signals_filtered}")
    logger.info(f"Signals Output: {metrics.signals_output}")
    logger.info(f"Execution Time: {metrics.execution_time_ms:.1f}ms")
    logger.info(f"Errors: {metrics.errors_count}")
    
    # 6. Display Performance Statistics
    logger.info("6. Performance Statistics")
    
    orchestrator_stats = orchestrator.get_status()
    performance_stats = orchestrator.get_performance_metrics()
    
    logger.info("Orchestrator Status:")
    logger.info(json.dumps(orchestrator_stats, indent=2, default=str))
    
    logger.info("Performance Metrics:")
    logger.info(json.dumps(performance_stats, indent=2, default=str))
    
    # 7. Demonstrate Real-time Processing (Brief Example)
    logger.info("7. Demonstrating Real-time Processing (10 seconds)")
    
    # Configure for real-time with shorter interval
    rt_config = OrchestrationConfig(
        mode=OrchestrationMode.REAL_TIME,
        execution_interval_seconds=3,  # Run every 3 seconds
        max_concurrent_generators=5,
        enable_validation=True,
        enable_risk_assessment=True
    )
    
    rt_orchestrator = SignalOrchestrator(rt_config)
    
    # Setup with fewer generators for faster execution
    rt_orchestrator.register_generator(large_cap_ma)
    rt_orchestrator.register_generator(options_flow)
    
    # Register data providers
    rt_orchestrator.register_data_provider("assets", data_provider.get_assets)
    rt_orchestrator.register_data_provider("market_data", data_provider.get_market_data)
    rt_orchestrator.register_data_provider("technical_indicators", data_provider.get_technical_indicators)
    rt_orchestrator.register_data_provider("options_data", data_provider.get_options_data)
    
    # Register simple output handler
    async def rt_output_handler(signals):
        logger.info(f"Real-time: Generated {len(signals)} signals")
    
    rt_orchestrator.register_output_handler(rt_output_handler)
    
    # Start real-time processing
    await rt_orchestrator.start_real_time_processing()
    
    # Let it run for 10 seconds
    await asyncio.sleep(10)
    
    # Stop real-time processing
    await rt_orchestrator.stop_processing()
    
    logger.info("Real-time processing stopped")
    
    # 8. Configuration Management Example
    logger.info("8. Configuration Management Example")
    
    config_summary = config_manager.get_configuration_summary()
    logger.info("Configuration Summary:")
    logger.info(json.dumps(config_summary, indent=2, default=str))
    
    # Validate configuration
    validation_errors = config_manager.validate_configuration()
    if validation_errors:
        logger.warning(f"Configuration validation errors: {validation_errors}")
    else:
        logger.info("Configuration validation passed")
    
    # 9. Factory Pattern Examples
    logger.info("9. Factory Pattern Examples")
    
    # Create asset class specific generator suites
    stock_generators = SignalGeneratorFactory.create_asset_class_suite("stocks")
    logger.info(f"Created {len(stock_generators)} generators for stocks")
    
    options_generators = SignalGeneratorFactory.create_asset_class_suite("daily_options")
    logger.info(f"Created {len(options_generators)} generators for daily options")
    
    # List all registered generator types
    available_generators = SignalGeneratorFactory.list_generators()
    logger.info(f"Available generator types: {available_generators}")
    
    logger.info("Signal Generation Framework Example Complete!")


if __name__ == "__main__":
    asyncio.run(main())