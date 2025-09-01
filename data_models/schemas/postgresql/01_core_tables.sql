-- Market Scanning Engine - Core Data Tables
-- PostgreSQL Schema Definition
-- Author: Claude Code (System Architect)
-- Version: 1.0

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- ASSET MANAGEMENT TABLES
-- ============================================================================

-- Asset classes lookup table
CREATE TABLE asset_classes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(30) NOT NULL, -- 'equity', 'fixed_income', 'derivatives', 'commodity', 'crypto'
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Assets master table with comprehensive asset information
CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    asset_class_id INTEGER REFERENCES asset_classes(id),
    exchange VARCHAR(20) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    sector VARCHAR(50),
    industry VARCHAR(100),
    market_cap BIGINT, -- in cents to avoid floating point issues
    avg_volume_30d BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    listing_date DATE,
    delisting_date DATE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(symbol, exchange)
);

-- ============================================================================
-- USER MANAGEMENT TABLES
-- ============================================================================

-- User profiles with comprehensive preference management
CREATE TABLE user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL, -- External auth system ID
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    
    -- Risk preferences (0-100 scale)
    risk_tolerance INTEGER CHECK (risk_tolerance >= 0 AND risk_tolerance <= 100),
    max_position_size_pct DECIMAL(5,2) CHECK (max_position_size_pct >= 0 AND max_position_size_pct <= 100),
    max_daily_loss_pct DECIMAL(5,2) CHECK (max_daily_loss_pct >= 0 AND max_daily_loss_pct <= 100),
    
    -- Time horizon preferences
    min_holding_period_hours INTEGER DEFAULT 1,
    max_holding_period_hours INTEGER DEFAULT 8760, -- 1 year
    
    -- Portfolio constraints
    max_open_positions INTEGER DEFAULT 10,
    min_trade_amount_cents BIGINT DEFAULT 100000, -- $1000 minimum
    max_trade_amount_cents BIGINT DEFAULT 10000000, -- $100k maximum
    
    -- Notification settings
    notification_preferences JSONB DEFAULT '{}',
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- User asset class preferences with weights
CREATE TABLE user_asset_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    asset_class_id INTEGER REFERENCES asset_classes(id),
    preference_weight DECIMAL(3,2) CHECK (preference_weight >= 0 AND preference_weight <= 1),
    is_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, asset_class_id)
);

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Real-time and historical price data
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- OHLCV data (in cents to avoid floating point precision issues)
    open_price_cents BIGINT NOT NULL,
    high_price_cents BIGINT NOT NULL,
    low_price_cents BIGINT NOT NULL,
    close_price_cents BIGINT NOT NULL,
    volume BIGINT NOT NULL,
    
    -- Additional market data
    bid_price_cents BIGINT,
    ask_price_cents BIGINT,
    bid_size INTEGER,
    ask_size INTEGER,
    
    -- Metadata
    data_source VARCHAR(50) NOT NULL,
    data_quality_score INTEGER CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Technical indicators computed data
CREATE TABLE technical_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '15m', '1h', '4h', '1d'
    
    -- Moving averages (in cents)
    sma_20 BIGINT,
    sma_50 BIGINT,
    sma_200 BIGINT,
    ema_12 BIGINT,
    ema_26 BIGINT,
    
    -- Momentum indicators
    rsi_14 DECIMAL(5,2),
    macd_line BIGINT,
    macd_signal BIGINT,
    macd_histogram BIGINT,
    
    -- Volatility indicators
    bollinger_upper BIGINT,
    bollinger_middle BIGINT,
    bollinger_lower BIGINT,
    atr_14 BIGINT,
    
    -- Volume indicators
    volume_sma_20 BIGINT,
    on_balance_volume BIGINT,
    
    -- Additional technical data
    support_level BIGINT,
    resistance_level BIGINT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(asset_id, timestamp, timeframe)
);

-- ============================================================================
-- OPTIONS DATA TABLES (for derivatives)
-- ============================================================================

-- Options chain data
CREATE TABLE options_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    underlying_asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    option_symbol VARCHAR(30) UNIQUE NOT NULL,
    
    -- Contract specifications
    expiration_date DATE NOT NULL,
    strike_price_cents BIGINT NOT NULL,
    option_type VARCHAR(4) CHECK (option_type IN ('CALL', 'PUT')),
    contract_size INTEGER DEFAULT 100,
    
    -- Market data timestamp
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Pricing data (in cents)
    bid_price_cents BIGINT,
    ask_price_cents BIGINT,
    last_price_cents BIGINT,
    
    -- Volume and open interest
    volume INTEGER DEFAULT 0,
    open_interest INTEGER DEFAULT 0,
    
    -- Greeks (scaled by 10000 for precision, e.g., delta of 0.5 stored as 5000)
    delta INTEGER, -- -10000 to 10000 (representing -1.0 to 1.0)
    gamma INTEGER, -- 0 to 100000 (representing 0.0 to 10.0)
    theta INTEGER, -- -10000 to 0 (representing -1.0 to 0.0)
    vega INTEGER, -- 0 to 100000 (representing 0.0 to 10.0)
    rho INTEGER, -- -10000 to 10000 (representing -1.0 to 1.0)
    
    -- Implied volatility (scaled by 10000, e.g., 20% stored as 2000)
    implied_volatility INTEGER,
    
    -- Metadata
    data_source VARCHAR(50) NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- NEWS AND SENTIMENT DATA
-- ============================================================================

-- News articles and events
CREATE TABLE news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    headline VARCHAR(500) NOT NULL,
    content TEXT,
    source VARCHAR(100) NOT NULL,
    author VARCHAR(100),
    published_at TIMESTAMPTZ NOT NULL,
    
    -- Sentiment analysis (0-100 scale)
    sentiment_score INTEGER CHECK (sentiment_score >= 0 AND sentiment_score <= 100),
    sentiment_label VARCHAR(20), -- 'bullish', 'bearish', 'neutral'
    confidence_score INTEGER CHECK (confidence_score >= 0 AND confidence_score <= 100),
    
    -- Article categorization
    category VARCHAR(50),
    importance_score INTEGER CHECK (importance_score >= 0 AND importance_score <= 100),
    
    -- Full-text search vector
    search_vector tsvector,
    
    -- Metadata
    url VARCHAR(1000),
    tags VARCHAR(255)[],
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- News to asset relationships (many-to-many)
CREATE TABLE news_asset_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    news_article_id UUID REFERENCES news_articles(id) ON DELETE CASCADE,
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    relevance_score INTEGER CHECK (relevance_score >= 0 AND relevance_score <= 100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(news_article_id, asset_id)
);

-- Economic indicators
CREATE TABLE economic_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    indicator_name VARCHAR(100) NOT NULL,
    indicator_code VARCHAR(20) UNIQUE NOT NULL,
    release_date TIMESTAMPTZ NOT NULL,
    period_date DATE NOT NULL, -- The period this data refers to
    
    -- Value storage (scaled appropriately)
    actual_value DECIMAL(20,6),
    forecast_value DECIMAL(20,6),
    previous_value DECIMAL(20,6),
    
    -- Impact assessment
    market_impact_score INTEGER CHECK (market_impact_score >= 0 AND market_impact_score <= 100),
    surprise_index DECIMAL(5,2), -- (actual - forecast) / forecast * 100
    
    -- Metadata
    data_source VARCHAR(50) NOT NULL,
    country VARCHAR(3) DEFAULT 'USA',
    frequency VARCHAR(20), -- 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(indicator_code, period_date)
);