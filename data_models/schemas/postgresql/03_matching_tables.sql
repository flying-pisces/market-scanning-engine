-- Market Scanning Engine - User Matching and Execution Tables
-- PostgreSQL Schema Definition
-- Author: Claude Code (System Architect)
-- Version: 1.0

-- ============================================================================
-- USER SIGNAL MATCHING TABLES
-- ============================================================================

-- Signal matching results - tracks which signals match which users
CREATE TABLE signal_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    
    -- Matching timestamp
    matched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Match quality scores (0-100)
    overall_match_score INTEGER CHECK (overall_match_score >= 0 AND overall_match_score <= 100) NOT NULL,
    risk_tolerance_match INTEGER CHECK (risk_tolerance_match >= 0 AND risk_tolerance_match <= 100),
    asset_preference_match INTEGER CHECK (asset_preference_match >= 0 AND asset_preference_match <= 100),
    time_horizon_match INTEGER CHECK (time_horizon_match >= 0 AND time_horizon_match <= 100),
    position_size_match INTEGER CHECK (position_size_match >= 0 AND position_size_match <= 100),
    
    -- Recommended position parameters
    recommended_position_size_pct DECIMAL(5,2),
    recommended_position_size_dollars_cents BIGINT,
    adjusted_stop_loss_cents BIGINT,
    adjusted_target_price_cents BIGINT,
    
    -- Match reasoning
    match_factors JSONB DEFAULT '{}', -- Detailed breakdown of matching factors
    exclusion_reasons VARCHAR(255)[], -- Array of reasons if not matched
    
    -- User notification status
    notification_status VARCHAR(20) DEFAULT 'PENDING' CHECK (notification_status IN ('PENDING', 'SENT', 'READ', 'DISMISSED', 'ACTED_ON')),
    notification_sent_at TIMESTAMPTZ,
    user_response VARCHAR(20) CHECK (user_response IN ('INTERESTED', 'NOT_INTERESTED', 'MAYBE', 'ALREADY_HAVE')),
    user_response_at TIMESTAMPTZ,
    
    -- Match expiration
    expires_at TIMESTAMPTZ,
    is_expired BOOLEAN GENERATED ALWAYS AS (expires_at < NOW()) STORED,
    
    -- Metadata
    matching_algorithm_version VARCHAR(20),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(signal_id, user_id)
);

-- User signal interaction tracking
CREATE TABLE user_signal_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_match_id UUID REFERENCES signal_matches(id) ON DELETE CASCADE,
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(id) ON DELETE CASCADE,
    
    -- Interaction details
    interaction_type VARCHAR(50) NOT NULL, -- 'view', 'favorite', 'share', 'dismiss', 'execute', 'modify'
    interaction_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Interaction context
    interaction_source VARCHAR(50), -- 'web_app', 'mobile_app', 'email', 'api'
    session_id VARCHAR(100),
    
    -- User modifications (if any)
    modified_position_size_pct DECIMAL(5,2),
    modified_stop_loss_cents BIGINT,
    modified_target_price_cents BIGINT,
    modification_reason TEXT,
    
    -- Interaction metadata
    device_type VARCHAR(20),
    user_agent TEXT,
    ip_address INET,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- EXECUTION AND PORTFOLIO TRACKING
-- ============================================================================

-- Trade execution records
CREATE TABLE trade_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_match_id UUID REFERENCES signal_matches(id),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(id),
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    
    -- Trade identification
    external_trade_id VARCHAR(100), -- Broker's trade ID
    order_id VARCHAR(100), -- Our internal order ID
    
    -- Trade details
    trade_type VARCHAR(10) CHECK (trade_type IN ('BUY', 'SELL', 'SELL_SHORT', 'BUY_TO_COVER')) NOT NULL,
    quantity INTEGER NOT NULL,
    execution_price_cents BIGINT NOT NULL,
    
    -- Timing
    order_submitted_at TIMESTAMPTZ,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Trade sizing and risk management
    position_size_dollars_cents BIGINT NOT NULL,
    position_size_pct_of_portfolio DECIMAL(5,2),
    stop_loss_price_cents BIGINT,
    take_profit_price_cents BIGINT,
    
    -- Execution quality metrics
    slippage_bps INTEGER, -- Basis points of slippage vs expected price
    execution_time_ms INTEGER, -- Milliseconds from order to execution
    
    -- Costs and fees (in cents)
    commission_cents BIGINT DEFAULT 0,
    sec_fees_cents BIGINT DEFAULT 0,
    other_fees_cents BIGINT DEFAULT 0,
    total_cost_cents BIGINT GENERATED ALWAYS AS (commission_cents + sec_fees_cents + other_fees_cents) STORED,
    
    -- Trade status
    status VARCHAR(20) DEFAULT 'EXECUTED' CHECK (status IN ('PENDING', 'PARTIAL', 'EXECUTED', 'CANCELLED', 'REJECTED')),
    
    -- Execution metadata
    broker_name VARCHAR(50),
    execution_venue VARCHAR(50),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Position tracking (current holdings)
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    
    -- Position details
    quantity INTEGER NOT NULL, -- Positive for long, negative for short
    average_cost_cents BIGINT NOT NULL,
    current_price_cents BIGINT,
    
    -- Position dates
    opened_at TIMESTAMPTZ NOT NULL,
    last_updated_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ, -- NULL for open positions
    
    -- Risk management
    stop_loss_price_cents BIGINT,
    take_profit_price_cents BIGINT,
    trailing_stop_pct DECIMAL(5,2),
    
    -- Position metrics
    unrealized_pnl_cents BIGINT GENERATED ALWAYS AS (
        CASE 
            WHEN closed_at IS NULL AND current_price_cents IS NOT NULL 
            THEN (current_price_cents - average_cost_cents) * quantity
            ELSE NULL
        END
    ) STORED,
    
    realized_pnl_cents BIGINT, -- Set when position is closed
    
    -- Position sizing
    position_value_cents BIGINT GENERATED ALWAYS AS (
        CASE 
            WHEN closed_at IS NULL AND current_price_cents IS NOT NULL 
            THEN current_price_cents * ABS(quantity)
            ELSE NULL
        END
    ) STORED,
    
    -- Associated signals (can be multiple if position was increased)
    originating_signal_ids UUID[],
    
    -- Position status
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CLOSING')),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, asset_id) WHERE status = 'OPEN' -- One open position per user per asset
);

-- Position history and modifications
CREATE TABLE position_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    
    -- Historical snapshot
    snapshot_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    quantity INTEGER NOT NULL,
    price_cents BIGINT NOT NULL,
    
    -- Change details
    change_type VARCHAR(30) NOT NULL, -- 'OPEN', 'INCREASE', 'DECREASE', 'CLOSE', 'PRICE_UPDATE', 'STOP_LOSS_UPDATE'
    quantity_change INTEGER DEFAULT 0,
    
    -- Associated trade (if applicable)
    trade_execution_id UUID REFERENCES trade_executions(id),
    
    -- Reason for change
    change_reason VARCHAR(255),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PORTFOLIO ANALYTICS TABLES
-- ============================================================================

-- Daily portfolio snapshots
CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    
    -- Portfolio values (in cents)
    total_value_cents BIGINT NOT NULL,
    cash_balance_cents BIGINT NOT NULL,
    equity_value_cents BIGINT NOT NULL,
    unrealized_pnl_cents BIGINT DEFAULT 0,
    realized_pnl_cents BIGINT DEFAULT 0,
    
    -- Portfolio metrics
    number_of_positions INTEGER DEFAULT 0,
    largest_position_pct DECIMAL(5,2),
    portfolio_beta DECIMAL(6,4),
    portfolio_var_1d_pct DECIMAL(6,3),
    
    -- Risk metrics
    overall_risk_score INTEGER CHECK (overall_risk_score >= 0 AND overall_risk_score <= 100),
    concentration_risk_score INTEGER CHECK (concentration_risk_score >= 0 AND concentration_risk_score <= 100),
    liquidity_risk_score INTEGER CHECK (liquidity_risk_score >= 0 AND liquidity_risk_score <= 100),
    
    -- Performance metrics (relative to benchmark)
    daily_return_pct DECIMAL(8,4),
    mtd_return_pct DECIMAL(8,4),
    ytd_return_pct DECIMAL(8,4),
    total_return_pct DECIMAL(8,4),
    
    -- Drawdown tracking
    peak_value_cents BIGINT,
    current_drawdown_pct DECIMAL(6,3),
    max_drawdown_pct DECIMAL(6,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, snapshot_date)
);

-- Portfolio asset allocation snapshots
CREATE TABLE portfolio_allocations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_snapshot_id UUID REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
    asset_id UUID REFERENCES assets(id) ON DELETE CASCADE,
    asset_class_id INTEGER REFERENCES asset_classes(id),
    
    -- Allocation details
    position_value_cents BIGINT NOT NULL,
    weight_pct DECIMAL(5,2) NOT NULL,
    quantity INTEGER NOT NULL,
    average_cost_cents BIGINT NOT NULL,
    current_price_cents BIGINT NOT NULL,
    
    -- Position performance
    unrealized_pnl_cents BIGINT,
    unrealized_pnl_pct DECIMAL(8,4),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- Signal matching indexes
CREATE INDEX CONCURRENTLY idx_signal_matches_user_matched ON signal_matches (user_id, matched_at DESC);
CREATE INDEX CONCURRENTLY idx_signal_matches_signal_score ON signal_matches (signal_id, overall_match_score DESC);
CREATE INDEX CONCURRENTLY idx_signal_matches_notification_status ON signal_matches (notification_status) WHERE notification_status IN ('PENDING', 'SENT');
CREATE INDEX CONCURRENTLY idx_signal_matches_expires ON signal_matches (expires_at) WHERE expires_at > NOW();

-- User interactions indexes
CREATE INDEX CONCURRENTLY idx_user_interactions_user_timestamp ON user_signal_interactions (user_id, interaction_timestamp DESC);
CREATE INDEX CONCURRENTLY idx_user_interactions_signal ON user_signal_interactions (signal_id, interaction_type);
CREATE INDEX CONCURRENTLY idx_user_interactions_session ON user_signal_interactions (session_id);

-- Trade execution indexes
CREATE INDEX CONCURRENTLY idx_trade_executions_user_executed ON trade_executions (user_id, executed_at DESC);
CREATE INDEX CONCURRENTLY idx_trade_executions_signal ON trade_executions (signal_id);
CREATE INDEX CONCURRENTLY idx_trade_executions_asset ON trade_executions (asset_id, executed_at DESC);
CREATE INDEX CONCURRENTLY idx_trade_executions_status ON trade_executions (status);

-- Position indexes
CREATE INDEX CONCURRENTLY idx_positions_user_open ON positions (user_id) WHERE status = 'OPEN';
CREATE INDEX CONCURRENTLY idx_positions_asset_open ON positions (asset_id) WHERE status = 'OPEN';
CREATE INDEX CONCURRENTLY idx_positions_unrealized_pnl ON positions (unrealized_pnl_cents DESC) WHERE status = 'OPEN';

-- Portfolio snapshot indexes
CREATE INDEX CONCURRENTLY idx_portfolio_snapshots_user_date ON portfolio_snapshots (user_id, snapshot_date DESC);
CREATE INDEX CONCURRENTLY idx_portfolio_snapshots_date ON portfolio_snapshots (snapshot_date DESC);

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_signals_matches_composite ON signal_matches (user_id, overall_match_score DESC, matched_at DESC) WHERE notification_status = 'PENDING';
CREATE INDEX CONCURRENTLY idx_positions_value_composite ON positions (user_id, position_value_cents DESC, last_updated_at DESC) WHERE status = 'OPEN';