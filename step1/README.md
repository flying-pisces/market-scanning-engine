# Step 1: Minimal SP500 Market Scanner

A lightweight, verifiable market scanning engine optimized for mobile app backends. Zero infrastructure cost, pure Python implementation with free data sources.

## 🎯 Key Features

- **Fast**: Scans 100 stocks in <2 seconds
- **Free**: Uses yfinance for zero-cost market data
- **Simple**: Only 3 indicators (SMA, RSI, MACD)
- **Verifiable**: Every signal is measurable and testable
- **Mobile-Ready**: FastAPI backend for iOS/Android apps

## 📊 Performance Metrics

- **Speed**: <2 seconds for 100 stocks
- **Accuracy**: Target >55% hit rate on 7-day returns
- **Data Quality**: >95% successful API calls
- **Memory**: <500MB RAM usage
- **API Latency**: <500ms response time

## 🚀 Quick Start

### Installation

```bash
cd step1
pip install -r requirements.txt
```

### Run Daily Scan

```bash
python scripts/run_scan.py
```

### Start API Server

```bash
python api/main.py
```

The API will be available at `http://localhost:8000`

### Run Performance Benchmark

```bash
python scripts/benchmark.py
```

### Verify Signal Performance

```bash
python scripts/verify_signals.py --period 30
```

## 📱 API Endpoints

### Mobile-Optimized Endpoints

- `GET /api/v1/mobile/dashboard` - Complete dashboard data
- `GET /api/v1/mobile/watchlist?symbols=AAPL,MSFT,GOOGL` - Watchlist data

### Core Endpoints

- `GET /api/v1/scan/quick` - Quick scan (top 20 symbols)
- `GET /api/v1/signals/{symbol}` - Get signal for specific symbol
- `GET /api/v1/data/{symbol}` - Get market data for symbol
- `GET /api/v1/symbols` - List available symbols

## 📂 Project Structure

```
step1/
├── src/
│   ├── data_fetcher.py    # yfinance data fetching
│   ├── indicators.py       # SMA, RSI, MACD calculations
│   ├── scanner.py          # Core scanning engine
│   └── validator.py        # Signal verification
├── api/
│   └── main.py            # FastAPI mobile backend
├── tests/
│   ├── test_indicators.py  # Indicator tests
│   └── test_scanner.py     # Scanner tests
├── scripts/
│   ├── run_scan.py        # Daily scan runner
│   ├── benchmark.py       # Performance benchmark
│   └── verify_signals.py  # Signal verification
└── data/
    └── sp500_symbols.json  # Top 100 SP500 stocks
```

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Test specific component:

```bash
pytest tests/test_indicators.py -v
pytest tests/test_scanner.py -v
```

## 📈 Signal Generation Logic

Signals are generated based on multiple technical indicators:

### Buy Signals
- Price above both SMA20 and SMA50
- RSI < 30 (oversold)
- MACD bullish crossover
- Positive 5-day momentum

### Sell Signals
- Price below both SMA20 and SMA50
- RSI > 70 (overbought)
- MACD bearish crossover
- Negative 5-day momentum

### Confidence Scoring
- Base confidence: 50%
- Trend alignment: +10%
- RSI extremes: +15%
- MACD confirmation: +10%
- Strong momentum: +10-20%

Only signals with >60% confidence are returned.

## 🔄 Daily Workflow

1. **Morning Scan** (9:00 AM EST)
   ```bash
   python scripts/run_scan.py
   ```

2. **Verify Performance** (After market close)
   ```bash
   python scripts/verify_signals.py --period 7
   ```

3. **Weekly Benchmark** (Fridays)
   ```bash
   python scripts/benchmark.py
   ```

## 📊 Sample Output

```
==================================================
SCAN SUMMARY
==================================================
Scan Time: 1.85 seconds
Symbols Scanned: 100
Data Quality: 98.0%
Signals Generated: 23
  - Buy Signals: 15
  - Sell Signals: 8

TOP 5 SIGNALS:
--------------------------------------------------
NVDA   BUY  Confidence: 85.2% Price: $875.50
AAPL   BUY  Confidence: 78.5% Price: $185.25
MSFT   SELL Confidence: 72.3% Price: $425.10
GOOGL  BUY  Confidence: 68.9% Price: $155.75
TSLA   HOLD Confidence: 45.2% Price: $245.30
```

## 🚦 Requirements Verification

The system automatically verifies all Step 1 requirements:

- ✅ Process 100 stocks in <2 seconds
- ✅ Generate 15-30 signals daily
- ✅ >95% data fetch success rate
- ✅ <500MB memory usage
- ✅ Zero infrastructure cost

## 🔮 Next Steps (Step 2-5)

- **Step 2**: Enhanced accuracy with more indicators
- **Step 3**: Real-time processing with WebSockets
- **Step 4**: Premium data integration (Polygon.io)
- **Step 5**: Production scale with ML enhancement

## 📝 License

MIT License - Free for commercial use

## 🤝 Contributing

This is Step 1 of a progressive development plan. Keep it simple, fast, and verifiable.