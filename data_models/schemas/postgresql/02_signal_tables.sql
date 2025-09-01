-- Market Scanning Engine - Signal and Risk Tables
-- PostgreSQL Schema Definition
-- Author: Claude Code (System Architect)
-- Version: 1.0

-- ============================================================================
-- SIGNAL GENERATION TABLES
-- ============================================================================

-- Signal types lookup
CREATE TABLE signal_types (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(30) NOT NULL, -- 'technical', 'fundamental', 'options', 'news', 'macro'
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Core signals table with comprehensive scoring
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    signal_type_id INTEGER REFERENCES signal_types(id),
    
    -- Signal identification
    signal_name VARCHAR(100) NOT NULL,
    direction VARCHAR(5) CHECK (direction IN ('BUY', 'SELL', 'HOLD')) NOT NULL,
    
    -- Timing information
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ, -- Signal expiration
    target_entry_time TIMESTAMPTZ,
    recommended_holding_period_hours INTEGER,
    
    -- Core scoring (0-100 scale)
    risk_score INTEGER CHECK (risk_score >= 0 AND risk_score <= 100) NOT NULL,
    profit_potential_score INTEGER CHECK (profit_potential_score >= 0 AND profit_potential_score <= 100) NOT NULL,
    confidence_score INTEGER CHECK (confidence_score >= 0 AND confidence_score <= 100) NOT NULL,
    
    -- Risk-adjusted scoring
    sharpe_ratio DECIMAL(6,3),
    max_drawdown_pct DECIMAL(5,2),
    win_rate_pct DECIMAL(5,2),
    
    -- Price targets (in cents)
    entry_price_cents BIGINT,
    target_price_cents BIGINT,
    stop_loss_price_cents BIGINT,
    
    -- Position sizing recommendations
    recommended_position_size_pct DECIMAL(5,2) CHECK (recommended_position_size_pct >= 0 AND recommended_position_size_pct <= 100),
    max_position_size_pct DECIMAL(5,2) CHECK (max_position_size_pct >= 0 AND max_position_size_pct <= 100),
    
    -- Signal source and methodology
    signal_source VARCHAR(100) NOT NULL, -- Algorithm/model that generated signal
    methodology_version VARCHAR(20),
    signal_strength VARCHAR(20) CHECK (signal_strength IN ('WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG')),
    
    -- Backtesting results
    backtest_return_pct DECIMAL(8,4),
    backtest_volatility_pct DECIMAL(6,3),
    backtest_sample_size INTEGER,
    backtest_period_start DATE,
    backtest_period_end DATE,
    
    -- Asset class specific data (stored as JSONB for flexibility)
    asset_specific_data JSONB DEFAULT '{}',
    
    -- Signal metadata
    tags VARCHAR(50)[],
    notes TEXT,
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'EXPIRED', 'TRIGGERED', 'CANCELLED')),
    is_paper_trading BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signal factors - detailed breakdown of what contributed to the signal
CREATE TABLE signal_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    factor_name VARCHAR(100) NOT NULL,
    factor_category VARCHAR(50), -- 'technical', 'fundamental', 'sentiment', 'momentum', 'mean_reversion'
    
    -- Factor contribution (0-100 scale)
    contribution_score INTEGER CHECK (contribution_score >= 0 AND contribution_score <= 100),
    weight DECIMAL(5,4) CHECK (weight >= 0 AND weight <= 1),
    
    -- Factor-specific data
    factor_value DECIMAL(15,6),
    factor_percentile INTEGER CHECK (factor_percentile >= 0 AND factor_percentile <= 100),
    factor_z_score DECIMAL(6,3),
    
    -- Factor metadata
    calculation_method VARCHAR(100),
    data_source VARCHAR(50),
    lookback_periods INTEGER,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Signal performance tracking
CREATE TABLE signal_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    
    -- Performance timestamp
    evaluation_date DATE NOT NULL,
    days_since_signal INTEGER NOT NULL,
    
    -- Price performance (in cents)
    entry_price_cents BIGINT,
    current_price_cents BIGINT,
    high_since_signal_cents BIGINT,
    low_since_signal_cents BIGINT,
    
    -- Performance metrics
    unrealized_pnl_pct DECIMAL(8,4),
    realized_pnl_pct DECIMAL(8,4),
    max_favorable_excursion_pct DECIMAL(8,4),
    max_adverse_excursion_pct DECIMAL(8,4),
    
    -- Risk metrics
    current_risk_score INTEGER CHECK (current_risk_score >= 0 AND current_risk_score <= 100),
    volatility_realized_pct DECIMAL(6,3),
    
    -- Status
    position_status VARCHAR(20) CHECK (position_status IN ('OPEN', 'CLOSED', 'STOPPED_OUT', 'TARGET_HIT')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(signal_id, evaluation_date)
);

-- ============================================================================
-- RISK ASSESSMENT TABLES
-- ============================================================================

-- Risk factors configuration
CREATE TABLE risk_factors (
    id SERIAL PRIMARY KEY,
    factor_name VARCHAR(100) UNIQUE NOT NULL,
    factor_category VARCHAR(50) NOT NULL, -- 'market', 'liquidity', 'credit', 'operational', 'model'
    description TEXT,
    weight DECIMAL(5,4) CHECK (weight >= 0 AND weight <= 1),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Multi-factor risk assessments
CREATE TABLE risk_assessments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    assessment_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Overall risk score (0-100)
    overall_risk_score INTEGER CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100) NOT NULL,
    
    -- Component risk scores (0-100 each)
    market_risk_score INTEGER CHECK (market_risk_score >= 0 AND market_risk_score <= 100),
    liquidity_risk_score INTEGER CHECK (liquidity_risk_score >= 0 AND liquidity_risk_score <= 100),
    credit_risk_score INTEGER CHECK (credit_risk_score >= 0 AND credit_risk_score <= 100),
    volatility_risk_score INTEGER CHECK (volatility_risk_score >= 0 AND volatility_risk_score <= 100),
    concentration_risk_score INTEGER CHECK (concentration_risk_score >= 0 AND concentration_risk_score <= 100),
    
    -- Quantitative risk metrics
    beta DECIMAL(6,4),
    var_1d_pct DECIMAL(6,3), -- 1-day Value at Risk (95% confidence)
    var_5d_pct DECIMAL(6,3), -- 5-day Value at Risk (95% confidence)
    expected_shortfall_pct DECIMAL(6,3), -- Conditional VaR
    
    -- Volatility measures
    realized_volatility_30d_pct DECIMAL(6,3),
    implied_volatility_pct DECIMAL(6,3),
    volatility_skew DECIMAL(6,3),
    
    -- Liquidity measures
    bid_ask_spread_bps INTEGER, -- Basis points
    average_daily_volume_20d BIGINT,
    turnover_ratio DECIMAL(6,4),
    market_impact_score INTEGER CHECK (market_impact_score >= 0 AND market_impact_score <= 100),
    
    -- Market regime indicators
    market_regime VARCHAR(30), -- 'bull_market', 'bear_market', 'sideways', 'high_vol', 'low_vol'
    regime_confidence_pct INTEGER CHECK (regime_confidence_pct >= 0 AND regime_confidence_pct <= 100),
    
    -- Model metadata
    model_version VARCHAR(20),
    calculation_method VARCHAR(100),
    data_quality_score INTEGER CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk factor contributions detail
CREATE TABLE risk_factor_contributions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    risk_assessment_id UUID REFERENCES risk_assessments(id) ON DELETE CASCADE,
    risk_factor_id INTEGER REFERENCES risk_factors(id),
    
    -- Factor contribution
    factor_score INTEGER CHECK (factor_score >= 0 AND factor_score <= 100),
    contribution_weight DECIMAL(5,4) CHECK (contribution_weight >= 0 AND contribution_weight <= 1),
    marginal_contribution DECIMAL(6,3), -- Marginal contribution to overall risk
    
    -- Factor-specific metrics
    factor_value DECIMAL(15,6),
    factor_percentile INTEGER CHECK (factor_percentile >= 0 AND factor_percentile <= 100),
    factor_z_score DECIMAL(6,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(risk_assessment_id, risk_factor_id)
);

-- ============================================================================
-- MARKET REGIME AND MACRO ENVIRONMENT
-- ============================================================================

-- Market regime tracking
CREATE TABLE market_regimes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    regime_date DATE NOT NULL UNIQUE,
    
    -- Regime classification
    primary_regime VARCHAR(30) NOT NULL, -- 'bull', 'bear', 'sideways', 'crisis', 'recovery'
    volatility_regime VARCHAR(20) NOT NULL, -- 'low', 'normal', 'elevated', 'high', 'extreme'
    liquidity_regime VARCHAR(20) NOT NULL, -- 'abundant', 'normal', 'tight', 'stressed'
    
    -- Regime confidence scores (0-100)
    regime_confidence INTEGER CHECK (regime_confidence >= 0 AND regime_confidence <= 100),
    regime_stability INTEGER CHECK (regime_stability >= 0 AND regime_stability <= 100),
    
    -- Market-wide risk metrics
    market_stress_index INTEGER CHECK (market_stress_index >= 0 AND market_stress_index <= 100),
    correlation_regime VARCHAR(20), -- 'low', 'normal', 'high', 'extreme'
    
    -- Supporting indicators
    vix_level DECIMAL(5,2),
    credit_spreads_bps INTEGER,
    yield_curve_slope_bps INTEGER,
    dollar_strength_index DECIMAL(6,3),
    
    -- Regime metadata
    model_version VARCHAR(20),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- ============================================================================

-- Core table indexes
CREATE INDEX CONCURRENTLY idx_signals_asset_generated ON signals (asset_id, generated_at DESC);
CREATE INDEX CONCURRENTLY idx_signals_risk_score ON signals (risk_score DESC);
CREATE INDEX CONCURRENTLY idx_signals_profit_potential ON signals (profit_potential_score DESC);
CREATE INDEX CONCURRENTLY idx_signals_confidence ON signals (confidence_score DESC);
CREATE INDEX CONCURRENTLY idx_signals_status_active ON signals (status) WHERE status = 'ACTIVE';
CREATE INDEX CONCURRENTLY idx_signals_composite_score ON signals ((risk_score + profit_potential_score + confidence_score) DESC);

-- Signal factors indexes
CREATE INDEX CONCURRENTLY idx_signal_factors_signal ON signal_factors (signal_id);
CREATE INDEX CONCURRENTLY idx_signal_factors_category ON signal_factors (factor_category);
CREATE INDEX CONCURRENTLY idx_signal_factors_contribution ON signal_factors (contribution_score DESC);

-- Risk assessment indexes
CREATE INDEX CONCURRENTLY idx_risk_assessments_asset_timestamp ON risk_assessments (asset_id, assessment_timestamp DESC);
CREATE INDEX CONCURRENTLY idx_risk_assessments_overall_risk ON risk_assessments (overall_risk_score);
CREATE INDEX CONCURRENTLY idx_risk_assessments_market_regime ON risk_assessments (market_regime);

-- Performance tracking indexes
CREATE INDEX CONCURRENTLY idx_signal_performance_signal ON signal_performance (signal_id, evaluation_date DESC);
CREATE INDEX CONCURRENTLY idx_signal_performance_pnl ON signal_performance (unrealized_pnl_pct DESC);

-- Market regimes indexes
CREATE INDEX CONCURRENTLY idx_market_regimes_date ON market_regimes (regime_date DESC);
CREATE INDEX CONCURRENTLY idx_market_regimes_primary ON market_regimes (primary_regime);

-- GIN indexes for JSON data and arrays
CREATE INDEX CONCURRENTLY idx_signals_asset_specific_data ON signals USING gin (asset_specific_data);
CREATE INDEX CONCURRENTLY idx_signals_tags ON signals USING gin (tags);
CREATE INDEX CONCURRENTLY idx_news_articles_tags ON news_articles USING gin (tags);