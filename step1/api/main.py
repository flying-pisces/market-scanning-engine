"""
Step 1: FastAPI Backend for Mobile App
Enhanced with real-time ticker queries and multiple API support
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import sys
from pathlib import Path
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scanner import MinimalScanner
from data_fetcher import get_minimal_data
from validator import SignalValidator
from enhanced_data_fetcher import EnhancedDataFetcher

# Initialize FastAPI app
app = FastAPI(
    title="Market Scanner Step 1 API",
    description="Minimal API for mobile app integration",
    version="1.0.0"
)

# CORS for mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your mobile app in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (singleton for performance)
scanner = MinimalScanner(max_symbols=100)
validator = SignalValidator()

# Enhanced data fetcher with multiple APIs
enhanced_fetcher = EnhancedDataFetcher()

# Load environment variables from .env file
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                if value and value != f'your_{key.lower()}_here':
                    os.environ[key] = value


# Pydantic models for API
class SignalResponse(BaseModel):
    symbol: str
    action: str
    confidence: float
    price: float
    indicators: Dict
    timestamp: str


class QuickScanResponse(BaseModel):
    timestamp: str
    scan_time_ms: int
    signals: List[SignalResponse]
    summary: Dict


class SymbolDataResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    scanner_ready: bool
    apis_available: List[str]

class RealTimePriceResponse(BaseModel):
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: Optional[int] = None
    timestamp: str
    source: str
    market_status: str
    bid: Optional[float] = None
    ask: Optional[float] = None


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for mobile app"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        scanner_ready=True,
        apis_available=list(enhanced_fetcher.fetchers.keys())
    )


@app.get("/api/v1/scan/quick", response_model=QuickScanResponse)
async def quick_scan(top_n: int = 20):
    """
    Quick scan for mobile app - returns top signals only
    
    Args:
        top_n: Number of top symbols to scan (max 50 for quick response)
    """
    if top_n > 50:
        raise HTTPException(status_code=400, detail="top_n must be <= 50 for quick scan")
    
    try:
        results = scanner.quick_scan(top_n)
        
        # Convert to response model
        signals = [SignalResponse(**signal) for signal in results['signals']]
        
        return QuickScanResponse(
            timestamp=results['timestamp'],
            scan_time_ms=results['scan_time_ms'],
            signals=signals,
            summary=results['summary']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/scan/full")
async def full_scan(background_tasks: BackgroundTasks):
    """
    Trigger full scan (runs in background)
    Returns immediately with task ID
    """
    # In production, this would queue a background job
    background_tasks.add_task(scanner.scan)
    
    return {
        "status": "scanning",
        "message": "Full scan started in background",
        "timestamp": datetime.now().isoformat()
    }


# NEW: Real-time price endpoints with ticker as postfix
@app.get("/api/v1/price/{symbol}", response_model=RealTimePriceResponse)
async def get_real_time_price(symbol: str):
    """
    Get real-time price for any ticker symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT, GOOGL)
    
    Returns:
        Real-time price data with source information
    """
    symbol = symbol.upper()
    
    try:
        # Get real-time data from enhanced fetcher
        data = enhanced_fetcher.get_real_time_data(symbol)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"No real-time data available for {symbol}")
        
        # Determine the source that was used
        best_fetcher = enhanced_fetcher.get_best_fetcher(for_real_time=True)
        
        # Get market status
        market_status = enhanced_fetcher.get_market_status()
        
        return RealTimePriceResponse(
            symbol=symbol,
            price=data['price'],
            change=data['change'],
            change_pct=data['change_pct'],
            volume=data.get('volume'),
            timestamp=data['timestamp'],
            source=best_fetcher,
            market_status="open" if market_status.get('isOpen', False) else "closed",
            bid=data.get('bid'),
            ask=data.get('ask')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price for {symbol}: {str(e)}")


@app.get("/api/v1/quote/{symbol}")
async def get_quote(symbol: str):
    """
    Get detailed quote for any ticker symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL)
    
    Returns:
        Detailed quote with bid/ask, volume, etc.
    """
    symbol = symbol.upper()
    
    try:
        data = enhanced_fetcher.get_real_time_data(symbol)
        
        if data is None:
            raise HTTPException(status_code=404, detail=f"No quote data available for {symbol}")
        
        # Enhanced quote with additional details
        quote_data = {
            "symbol": symbol,
            "price": data['price'],
            "change": data['change'],
            "change_pct": data['change_pct'],
            "volume": data.get('volume', 0),
            "timestamp": data['timestamp'],
            "source": enhanced_fetcher.get_best_fetcher(for_real_time=True),
            "market_status": enhanced_fetcher.get_market_status(),
            "capabilities": enhanced_fetcher.get_capabilities()
        }
        
        # Add bid/ask if available
        if data.get('bid') is not None:
            quote_data['bid'] = data['bid']
            quote_data['ask'] = data['ask']
            quote_data['bid_size'] = data.get('bid_size')
            quote_data['ask_size'] = data.get('ask_size')
        
        # Add additional fields if available
        for field in ['high', 'low', 'open', 'prev_close']:
            if field in data:
                quote_data[field] = data[field]
        
        return quote_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching quote for {symbol}: {str(e)}")


@app.get("/api/v1/signals/{symbol}")
async def get_symbol_signal(symbol: str):
    """
    Get trading signal for a specific symbol
    
    Args:
        symbol: Stock symbol (e.g., AAPL)
    """
    symbol = symbol.upper()
    
    try:
        # Generate signal for single symbol
        results = scanner.scan([symbol])
        
        if results['all_signals']:
            signal = results['all_signals'][0]
            return SignalResponse(**signal)
        else:
            # Get real-time price even if no signal
            price_data = enhanced_fetcher.get_real_time_data(symbol)
            current_price = price_data['price'] if price_data else 0
            
            return {
                "symbol": symbol,
                "action": "HOLD",
                "confidence": 0,
                "price": current_price,
                "indicators": {},
                "timestamp": datetime.now().isoformat(),
                "message": "No strong signal at this time"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/data/{symbol}", response_model=SymbolDataResponse)
async def get_symbol_data(symbol: str):
    """
    Get minimal market data for a symbol
    
    Args:
        symbol: Stock symbol
    """
    symbol = symbol.upper()
    
    try:
        data = get_minimal_data(symbol)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        return SymbolDataResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/symbols")
async def get_symbols():
    """Get list of available SP500 symbols"""
    return {
        "symbols": scanner.symbols,
        "count": len(scanner.symbols)
    }


@app.get("/api/v1/performance")
async def get_performance_metrics():
    """
    Get performance metrics for the scanner
    """
    try:
        # Get last scan results
        if scanner.last_scan_results:
            return {
                "last_scan": scanner.last_scan_time.isoformat() if scanner.last_scan_time else None,
                "scan_time_seconds": scanner.last_scan_results.get('scan_time_seconds'),
                "data_quality_score": scanner.last_scan_results.get('data_quality_score'),
                "signals_generated": scanner.last_scan_results.get('signals_generated'),
                "symbols_scanned": scanner.last_scan_results.get('symbols_scanned')
            }
        else:
            return {
                "message": "No scans performed yet",
                "status": "idle"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/validate")
async def validate_signals(days: int = 7):
    """
    Validate recent signals (for testing/monitoring)
    
    Args:
        days: Number of days forward to validate
    """
    try:
        if scanner.last_scan_results and scanner.last_scan_results.get('all_signals'):
            validation = validator.validate_signal_accuracy(
                scanner.last_scan_results['all_signals'],
                days=days
            )
            return validation
        else:
            return {"error": "No signals to validate"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mobile-optimized endpoints
@app.get("/api/v1/mobile/dashboard")
async def mobile_dashboard():
    """
    Optimized dashboard data for mobile app
    Returns everything needed for main screen in one call
    """
    try:
        # Quick scan top 10
        quick_results = scanner.quick_scan(10)
        
        # Market summary
        market_summary = {
            "spy": get_minimal_data("SPY"),
            "qqq": get_minimal_data("QQQ"),
            "dia": get_minimal_data("DIA")
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "top_signals": quick_results['signals'][:3],  # Top 3 only
            "market_summary": market_summary,
            "stats": quick_results['summary']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NEW: Batch ticker endpoint
@app.get("/api/v1/prices")
async def get_multiple_prices(symbols: str):
    """
    Get real-time prices for multiple tickers
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")
    
    Returns:
        Dictionary of ticker prices
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")][:20]  # Max 20 for performance
    
    prices = {}
    source_used = enhanced_fetcher.get_best_fetcher(for_real_time=True)
    
    for symbol in symbol_list:
        try:
            data = enhanced_fetcher.get_real_time_data(symbol)
            if data:
                prices[symbol] = {
                    "price": data['price'],
                    "change": data['change'],
                    "change_pct": data['change_pct'],
                    "timestamp": data['timestamp']
                }
        except Exception as e:
            prices[symbol] = {"error": str(e)}
    
    return {
        "prices": prices,
        "source": source_used,
        "timestamp": datetime.now().isoformat(),
        "count": len(prices)
    }


@app.get("/api/v1/mobile/watchlist")
async def get_watchlist(symbols: str):
    """
    Enhanced watchlist with real-time data from multiple APIs
    
    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")][:10]  # Max 10
    
    watchlist_data = []
    for symbol in symbol_list:
        try:
            # Get real-time data from enhanced fetcher
            data = enhanced_fetcher.get_real_time_data(symbol)
            if data:
                # Add signal if available
                signal_results = scanner.scan([symbol])
                if signal_results['all_signals']:
                    data['signal'] = signal_results['all_signals'][0]['action']
                    data['confidence'] = signal_results['all_signals'][0]['confidence']
                else:
                    data['signal'] = 'HOLD'
                    data['confidence'] = 0
                
                watchlist_data.append(data)
        except Exception as e:
            # Fallback to basic data
            try:
                fallback_data = get_minimal_data(symbol)
                if fallback_data:
                    fallback_data['signal'] = 'HOLD'
                    fallback_data['confidence'] = 0
                    watchlist_data.append(fallback_data)
            except:
                continue
    
    return {
        "watchlist": watchlist_data,
        "source": enhanced_fetcher.get_best_fetcher(for_real_time=True),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python api/main.py
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )