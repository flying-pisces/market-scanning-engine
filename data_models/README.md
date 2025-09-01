# Market Scanning Engine - Data Models

## Overview

This directory contains comprehensive data models for the Market Scanning Engine, designed around a **0-100 risk scoring system** and **real-time signal matching** architecture. All monetary values are stored in cents to avoid floating-point precision issues, and all scoring uses consistent 0-100 integer scales.

## Architecture Principles

### 1. **Unified Risk Scoring (0-100)**
- Risk Score: 0 (lowest risk) to 100 (highest risk)
- Profit Potential: 0 (lowest potential) to 100 (highest potential)
- Confidence Score: 0 (lowest confidence) to 100 (highest confidence)
- User Risk Tolerance: 0 (conservative) to 100 (aggressive)

### 2. **Monetary Precision**
- All prices stored in cents (BIGINT) to eliminate floating-point errors
- Position sizes and P&L calculations use cents
- Percentages scaled appropriately (e.g., Greeks scaled by 10,000)

### 3. **Real-time Architecture**
- Kafka streaming for real-time data distribution
- Optimized indexes for high-frequency queries
- Partitioning strategy for time-series data
- Efficient signal matching algorithms

## Directory Structure

```
data_models/
├── README.md                           # This file
├── schemas/
│   ├── postgresql/                     # Database schemas
│   │   ├── 01_core_tables.sql         # Core entities (assets, users, market data)
│   │   ├── 02_signal_tables.sql       # Signal generation and risk assessment
│   │   └── 03_matching_tables.sql     # User matching and portfolio tracking
│   ├── json/                          # API schemas
│   │   ├── signal_schemas.json        # Signal and risk assessment schemas
│   │   └── portfolio_schemas.json     # Portfolio and execution schemas
│   └── kafka/                         # Streaming schemas
│       └── message_schemas.json       # Real-time message formats
├── python/                            # Pydantic models
│   ├── core_models.py                 # Base models and asset definitions
│   ├── signal_models.py               # Signal and risk models
│   └── portfolio_models.py            # Portfolio and execution models
└── performance/                       # Optimization guides
    ├── indexing_strategy.sql          # Database indexing strategy
    └── query_patterns.md              # Common query patterns and optimization
```

## Data Model Categories

### 1. **Core Models** (`core_models.py`, `01_core_tables.sql`)

#### Asset Management
- **AssetClass**: Asset class definitions (equity, fixed_income, derivatives, etc.)
- **Asset**: Individual securities with metadata and classification
- **MarketData**: Real-time and historical OHLCV data
- **TechnicalIndicators**: Computed technical analysis metrics
- **OptionsData**: Options chain data with Greeks calculations

#### User Management  
- **UserProfile**: Comprehensive user preferences and risk settings
- **UserAssetPreference**: Asset class allocation preferences with weights

### 2. **Signal Models** (`signal_models.py`, `02_signal_tables.sql`)

#### Signal Generation
- **SignalType**: Signal category definitions
- **Signal**: Core trading signals with comprehensive scoring
- **SignalFactor**: Detailed breakdown of signal contributing factors
- **SignalPerformance**: Historical performance tracking

#### Risk Assessment
- **RiskAssessment**: Multi-factor risk analysis for assets
- **RiskFactor**: Individual risk components and weights
- **MarketRegimeData**: Market condition classification

### 3. **Portfolio Models** (`portfolio_models.py`, `03_matching_tables.sql`)

#### Matching and Execution
- **SignalMatch**: Signal-to-user matching results with scoring
- **UserSignalInteraction**: User engagement tracking
- **TradeExecution**: Detailed trade execution records
- **Position**: Current and historical portfolio positions

#### Analytics and Reporting
- **PortfolioSnapshot**: Daily portfolio performance snapshots  
- **PortfolioAllocation**: Asset allocation breakdowns
- **PositionHistory**: Position modification audit trail

## Key Features

### **Signal Matching Algorithm**
The system matches signals to users based on:
- Risk tolerance alignment (0-100 scale compatibility)
- Asset class preferences with weights
- Position sizing constraints
- Time horizon compatibility
- Portfolio concentration limits

### **Real-time Data Flow**
```
Market Data → Technical Analysis → Signal Generation → 
User Matching → Notifications → Trade Execution → 
Portfolio Updates → Risk Monitoring
```

### **Performance Optimizations**
- **Composite Indexes**: Multi-column indexes for common query patterns
- **Partial Indexes**: Filtered indexes for active signals and open positions
- **Time-series Partitioning**: Monthly partitions for high-volume tables
- **Connection Pooling**: Separate pools for different workload types
- **Strategic Caching**: Redis caching for frequently accessed data

## Getting Started

### 1. **Database Setup**
```bash
# Create database and extensions
psql -c "CREATE DATABASE market_scanning_engine;"
psql -d market_scanning_engine -f schemas/postgresql/01_core_tables.sql
psql -d market_scanning_engine -f schemas/postgresql/02_signal_tables.sql
psql -d market_scanning_engine -f schemas/postgresql/03_matching_tables.sql
```

### 2. **Python Models Usage**
```python
from data_models.python.signal_models import Signal, SignalMatch
from data_models.python.portfolio_models import Position, TradeExecution

# Create a new signal
signal = Signal(
    asset_id="123e4567-e89b-12d3-a456-426614174000",
    signal_type_id=1,
    signal_name="Technical Breakout - AAPL",
    direction="BUY",
    risk_score=35,           # 0-100 scale
    profit_potential_score=75,
    confidence_score=80,
    entry_price_cents=15000,  # $150.00
    target_price_cents=16500, # $165.00
    stop_loss_price_cents=14000, # $140.00
    signal_source="technical_analysis_v2"
)
```

### 3. **API Integration**
```python
from fastapi import FastAPI
from data_models.python.signal_models import Signal

app = FastAPI()

@app.post("/signals/", response_model=Signal)
async def create_signal(signal: Signal):
    # Validate signal data automatically with Pydantic
    return await save_signal_to_database(signal)
```

### 4. **Kafka Streaming**
```python
from kafka import KafkaProducer
import json

# Send real-time signal updates
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

signal_data = {
    "signal_id": "123e4567-e89b-12d3-a456-426614174000",
    "asset_id": "456e7890-e89b-12d3-a456-426614174000", 
    "symbol": "AAPL",
    "direction": "BUY",
    "risk_score": 35,
    "profit_potential_score": 75,
    "confidence_score": 80
}

producer.send('signals-generated', value=signal_data)
```

## API Schema Integration

### OpenAPI/Swagger Integration
The JSON schemas in `schemas/json/` are designed for direct use with OpenAPI 3.0 specifications:

```yaml
# openapi.yaml excerpt
components:
  schemas:
    Signal:
      $ref: 'data_models/schemas/json/signal_schemas.json#/Signal'
    PortfolioSnapshot:
      $ref: 'data_models/schemas/json/portfolio_schemas.json#/PortfolioSnapshot'
```

### Validation and Documentation
- **Automatic validation** of API requests/responses
- **Interactive API documentation** with Swagger UI
- **Type safety** with Pydantic model integration
- **Consistent data formats** across all services

## Performance Considerations

### **High-Frequency Operations**
- Signal generation: 1000+ signals/minute
- Market data updates: 10,000+ updates/second  
- Signal matching: 100+ matches/second
- Portfolio valuations: 200+ calculations/second

### **Optimization Strategies**
- **Indexed queries**: All common access patterns have optimized indexes
- **Partitioned tables**: Time-series data partitioned by month
- **Materialized views**: Pre-computed aggregations for reporting
- **Read replicas**: Separate read workloads from write operations

### **Caching Strategy**
- **User profiles**: 1-hour TTL
- **Market data**: 30-second TTL  
- **Signal matches**: 15-minute TTL
- **Portfolio snapshots**: 1-hour TTL

## Monitoring and Maintenance

### **Performance Monitoring**
```sql
-- Check index usage
SELECT * FROM index_usage_stats WHERE usage_category = 'UNUSED';

-- Monitor slow queries  
SELECT * FROM slow_query_analysis WHERE mean_time > 1000;

-- Check table bloat
SELECT * FROM table_bloat_stats WHERE bloat_percent > 20;
```

### **Automated Maintenance**
- **Daily statistics updates**: `ANALYZE` on all major tables
- **Weekly index maintenance**: `REINDEX` on frequently used indexes
- **Monthly partition cleanup**: Remove old time-series partitions
- **Continuous monitoring**: Alert on performance degradation

## Data Quality and Validation

### **Input Validation**
- All scores validated within 0-100 range
- Monetary values must be positive cents
- Required fields enforced at database and API level
- Foreign key relationships maintained

### **Data Integrity**
- **Referential integrity**: All foreign keys enforced
- **Check constraints**: Business rule validation at database level
- **Audit trails**: All modifications logged with timestamps
- **Backup strategy**: Point-in-time recovery capability

## Migration and Versioning

### **Schema Migrations**
- **Version-controlled DDL**: All schema changes tracked in Git
- **Backward compatibility**: API versioning strategy implemented
- **Zero-downtime deployments**: Online schema modification support
- **Rollback procedures**: Quick rollback capability for failed deployments

### **Data Migration Tools**
```bash
# Example migration command
./migrate.py --from-version 1.0 --to-version 1.1 --dry-run
./migrate.py --from-version 1.0 --to-version 1.1 --execute
```

This comprehensive data model provides the foundation for a scalable, high-performance market scanning engine that can handle real-time signal generation, user matching, and portfolio management at enterprise scale.