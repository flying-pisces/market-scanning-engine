# Market Scanning Engine

A comprehensive risk-based market scanning engine that generates and matches trading signals based on user risk tolerance (0-100 scale).

## Quick Start

### Installation

```bash
# Install the package in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Basic Usage

```python
from signal_generation import SignalOrchestrator, OrchestrationConfig
from data_models.python.core_models import Asset

# Create a simple signal generation example
config = OrchestrationConfig(enable_validation=True)
orchestrator = SignalOrchestrator(config)

# Run signal generation
# (Full example coming in Phase 1 implementation)
```

## Development Phases

### Phase 1: Foundation (Week 1-2) ✅ In Progress
- **Status**: Setting up package structure and basic functionality
- **Goal**: Working REST API with basic risk scoring and signal generation

### Phase 2: Data Pipeline (Week 3-4) 
- **Goal**: Real-time Kafka processing and multiple technical indicators

### Phase 3: Advanced Features (Week 5-8)
- **Goal**: Multi-asset class support, ML personalization, production deployment

## Architecture

The system follows a microservices architecture with:
- **Signal Generation**: Multi-strategy signal generators for different asset classes
- **Risk Assessment**: 0-100 risk scoring across 5 factors (volatility, liquidity, time decay, market regime, position size)
- **User Matching**: Sophisticated matching algorithm connecting signals to users based on risk tolerance
- **Real-time Processing**: Kafka-based streaming for high-throughput signal processing

## Asset Classes Supported

- **Daily Options** (Risk: 70-95): SPY, QQQ, SPX, XSP, NDX
- **Stocks** (Risk: 30-80): Large cap, mid cap, small cap
- **ETFs** (Risk: 20-70): Sector, regional, thematic
- **Bonds** (Risk: 5-40): Treasury, corporate, municipal
- **Safe Assets** (Risk: 0-20): T-bills, CDs, stable value funds

## Contributing

This project is in active development. The current focus is on implementing the MVP functionality outlined in Phase 1.

## License

MIT License - see LICENSE file for details.