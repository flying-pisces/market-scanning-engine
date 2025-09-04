"""
Real-time market data ingestion service
Fetches data from multiple sources and publishes to Kafka
"""

import asyncio
import json
import logging
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass

import aiohttp
import yfinance as yf
from alpha_vantage.async_client import AsyncClient as AlphaVantageClient

from app.core.kafka_client import init_kafka, get_producer, KafkaTopics
from app.models.market import MarketDataPoint, AssetClass

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for market data sources"""
    alpha_vantage_api_key: str = "demo"
    finnhub_api_key: str = "demo" 
    polygon_api_key: str = "demo"
    update_interval: int = 5  # seconds
    batch_size: int = 100


class MarketDataIngestion:
    """Real-time market data ingestion pipeline"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.producer = None
        self.is_running = False
        
        # Asset symbols to track
        self.equity_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        self.options_symbols = ["SPY", "QQQ", "SPX", "XSP", "NDX"]  # Options on these
        self.bond_symbols = ["TLT", "SHY", "IEF", "LQD"]  # Bond ETFs
        self.safe_symbols = ["SHV", "BIL", "MINT"]  # Money market ETFs
        
    async def start(self):
        """Start the data ingestion service"""
        logger.info("Starting market data ingestion service")
        
        # Initialize Kafka
        if not await init_kafka():
            raise RuntimeError("Failed to initialize Kafka")
        
        self.producer = get_producer()
        if not self.producer:
            raise RuntimeError("Failed to get Kafka producer")
        
        self.is_running = True
        
        # Start data collection tasks
        tasks = [
            self._ingest_stock_data(),
            self._ingest_options_data(),
            self._ingest_bond_data(),
            self._monitor_market_hours()
        ]
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the data ingestion service"""
        logger.info("Stopping market data ingestion service")
        self.is_running = False
        
        if self.producer:
            self.producer.close()
    
    async def _ingest_stock_data(self):
        """Ingest real-time stock data"""
        while self.is_running:
            try:
                stock_data = await self._fetch_stock_quotes(
                    self.equity_symbols + self.bond_symbols + self.safe_symbols
                )
                
                for symbol, data in stock_data.items():
                    market_data = MarketDataPoint(
                        symbol=symbol,
                        asset_class=self._get_asset_class(symbol),
                        price=data.get("price", 0.0),
                        volume=data.get("volume", 0),
                        bid=data.get("bid"),
                        ask=data.get("ask"),
                        high=data.get("high"),
                        low=data.get("low"),
                        open_price=data.get("open"),
                        previous_close=data.get("previous_close"),
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    # Publish to Kafka
                    await self.producer.send_message(
                        KafkaTopics.MARKET_DATA_RAW,
                        market_data.model_dump(),
                        key=symbol
                    )
                
                logger.info(f"Published data for {len(stock_data)} symbols")
                
            except Exception as e:
                logger.error(f"Error ingesting stock data: {e}")
            
            await asyncio.sleep(self.config.update_interval)
    
    async def _ingest_options_data(self):
        """Ingest options data for high-risk signals"""
        while self.is_running:
            try:
                # For demo purposes, simulate options data
                # In production, would connect to options data provider
                options_data = await self._fetch_simulated_options_data()
                
                for option_data in options_data:
                    # Publish to options topic
                    await self.producer.send_message(
                        KafkaTopics.OPTIONS_DATA,
                        option_data,
                        key=option_data["symbol"]
                    )
                
                logger.info(f"Published options data for {len(options_data)} contracts")
                
            except Exception as e:
                logger.error(f"Error ingesting options data: {e}")
            
            await asyncio.sleep(self.config.update_interval * 2)  # Less frequent
    
    async def _ingest_bond_data(self):
        """Ingest bond and treasury data"""
        while self.is_running:
            try:
                # Fetch bond ETF data and treasury rates
                bond_data = await self._fetch_treasury_rates()
                
                for data_point in bond_data:
                    await self.producer.send_message(
                        KafkaTopics.MARKET_DATA_RAW,
                        data_point,
                        key=data_point["symbol"]
                    )
                
                logger.info(f"Published bond data for {len(bond_data)} instruments")
                
            except Exception as e:
                logger.error(f"Error ingesting bond data: {e}")
            
            await asyncio.sleep(self.config.update_interval * 4)  # Even less frequent
    
    async def _monitor_market_hours(self):
        """Monitor market hours and adjust data collection"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                is_market_hours = self._is_market_hours(now)
                
                # Publish market status
                market_status = {
                    "is_open": is_market_hours,
                    "timestamp": now.isoformat(),
                    "session": "regular" if is_market_hours else "closed"
                }
                
                await self.producer.send_message(
                    KafkaTopics.SYSTEM_METRICS,
                    market_status,
                    key="market_hours"
                )
                
            except Exception as e:
                logger.error(f"Error monitoring market hours: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _fetch_stock_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch real-time stock quotes"""
        try:
            # Using yfinance for demo - in production use professional API
            tickers = yf.Tickers(" ".join(symbols))
            quotes = {}
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    # Get current price from fast_info if available
                    try:
                        current_price = ticker.fast_info.last_price
                    except:
                        current_price = info.get("regularMarketPrice", info.get("currentPrice", 0))
                    
                    quotes[symbol] = {
                        "price": float(current_price) if current_price else 0.0,
                        "volume": info.get("regularMarketVolume", info.get("volume", 0)),
                        "bid": info.get("bid"),
                        "ask": info.get("ask"),
                        "high": info.get("regularMarketDayHigh", info.get("dayHigh")),
                        "low": info.get("regularMarketDayLow", info.get("dayLow")),
                        "open": info.get("regularMarketOpen", info.get("open")),
                        "previous_close": info.get("regularMarketPreviousClose", info.get("previousClose"))
                    }
                    
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")
                    # Provide default values
                    quotes[symbol] = {
                        "price": 100.0,  # Mock price
                        "volume": 1000000,
                        "bid": None,
                        "ask": None,
                        "high": None,
                        "low": None,
                        "open": None,
                        "previous_close": None
                    }
            
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching stock quotes: {e}")
            # Return mock data for demo
            return {symbol: {"price": 100.0, "volume": 1000000} for symbol in symbols}
    
    async def _fetch_simulated_options_data(self) -> List[Dict[str, Any]]:
        """Simulate options data for development"""
        options_data = []
        
        for symbol in self.options_symbols:
            # Generate some mock options contracts
            for days_to_exp in [0, 1, 2, 7]:  # Daily and weekly options
                for strike_offset in [-5, 0, 5]:  # ITM, ATM, OTM
                    base_price = 400  # Mock underlying price
                    strike = base_price + strike_offset
                    
                    option_data = {
                        "symbol": f"{symbol}",
                        "underlying_symbol": symbol,
                        "option_type": "call",
                        "strike": strike,
                        "expiry_date": (datetime.now() + 
                                      asyncio.create_task(asyncio.sleep(0)).get_loop().time() + 
                                      days_to_exp * 24 * 3600).isoformat(),
                        "bid": max(0.01, base_price - strike + (5 - days_to_exp)),
                        "ask": max(0.02, base_price - strike + (6 - days_to_exp)),
                        "volume": 1000,
                        "open_interest": 5000,
                        "implied_volatility": 0.25 + (days_to_exp * 0.05),
                        "delta": 0.5,
                        "gamma": 0.1,
                        "theta": -0.05,
                        "vega": 0.2,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    options_data.append(option_data)
        
        return options_data[:20]  # Limit for demo
    
    async def _fetch_treasury_rates(self) -> List[Dict[str, Any]]:
        """Fetch treasury rates and bond data"""
        # Mock treasury data for development
        treasury_data = [
            {
                "symbol": "^IRX",  # 3-month treasury
                "asset_class": "SAFE_ASSETS",
                "yield": 5.25,
                "price": 100.0,
                "maturity": "3M",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "symbol": "^FVX",  # 5-year treasury  
                "asset_class": "BONDS",
                "yield": 4.75,
                "price": 100.0,
                "maturity": "5Y",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "symbol": "^TNX",  # 10-year treasury
                "asset_class": "BONDS", 
                "yield": 4.50,
                "price": 100.0,
                "maturity": "10Y",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        return treasury_data
    
    def _get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol"""
        if symbol in self.equity_symbols:
            return AssetClass.STOCKS.value
        elif symbol in self.bond_symbols:
            return AssetClass.BONDS.value
        elif symbol in self.safe_symbols:
            return AssetClass.SAFE_ASSETS.value
        else:
            return AssetClass.ETFS.value
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """Check if markets are open (simplified)"""
        # US market hours: 9:30 AM - 4:00 PM ET
        # Convert to ET and check
        weekday = dt.weekday()
        
        # Weekend
        if weekday >= 5:
            return False
        
        # Simplified hour check (would need proper timezone handling in production)
        hour = dt.hour
        return 14 <= hour < 21  # Approximate ET hours in UTC


async def main():
    """Main entry point for data ingestion service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = DataSourceConfig(
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", "demo"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", "demo"),
        polygon_api_key=os.getenv("POLYGON_API_KEY", "demo"),
        update_interval=int(os.getenv("UPDATE_INTERVAL", "5")),
        batch_size=int(os.getenv("BATCH_SIZE", "100"))
    )
    
    ingestion_service = MarketDataIngestion(config)
    
    try:
        await ingestion_service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await ingestion_service.stop()


if __name__ == "__main__":
    asyncio.run(main())