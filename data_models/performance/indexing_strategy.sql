-- Market Scanning Engine - Database Indexing and Performance Strategy
-- PostgreSQL Optimization Guide
-- Author: Claude Code (System Architect)
-- Version: 1.0

-- ============================================================================
-- PERFORMANCE OPTIMIZATION STRATEGY
-- ============================================================================

/*
INDEXING PHILOSOPHY:
1. Query Pattern Analysis: Index based on actual query patterns, not theoretical needs
2. Selective Indexing: Focus on high-frequency queries and join conditions
3. Composite Indexes: Use multi-column indexes for common query combinations
4. Partial Indexes: Filter indexes for selective conditions to reduce size
5. Expression Indexes: Index computed values and functions used in queries
6. Avoid Over-indexing: Each index costs write performance and storage

QUERY OPTIMIZATION PRINCIPLES:
- All monetary values in cents (BIGINT) to avoid floating-point precision issues
- All scores use 0-100 INTEGER scale for consistency and efficient comparison
- Timestamp queries optimized with proper timezone handling
- Foreign key relationships with proper referential integrity
- Partitioning for large time-series tables
*/

-- ============================================================================
-- PRIMARY PERFORMANCE INDEXES (Already defined in schema)
-- ============================================================================

/*
These indexes are already included in the schema files but documented here for reference:

SIGNAL PERFORMANCE INDEXES:
- idx_signals_asset_generated: (asset_id, generated_at DESC) - Asset signal history
- idx_signals_risk_score: (risk_score DESC) - Risk-based signal filtering
- idx_signals_profit_potential: (profit_potential_score DESC) - Profit-based filtering
- idx_signals_confidence: (confidence_score DESC) - Confidence-based filtering
- idx_signals_status_active: (status) WHERE status = 'ACTIVE' - Active signals only
- idx_signals_composite_score: ((risk_score + profit_potential_score + confidence_score) DESC)

MATCHING PERFORMANCE INDEXES:
- idx_signal_matches_user_matched: (user_id, matched_at DESC) - User match history
- idx_signal_matches_signal_score: (signal_id, overall_match_score DESC) - Signal quality
- idx_signal_matches_notification_status: (notification_status) - Pending notifications
- idx_signal_matches_expires: (expires_at) WHERE expires_at > NOW() - Active matches

PORTFOLIO INDEXES:
- idx_positions_user_open: (user_id) WHERE status = 'OPEN' - User open positions
- idx_portfolio_snapshots_user_date: (user_id, snapshot_date DESC) - Portfolio history
*/

-- ============================================================================
-- ADVANCED PERFORMANCE INDEXES
-- ============================================================================

-- Time-series data partitioning and indexing for high-volume tables
-- Market data partitioning by month for efficient time-based queries
CREATE INDEX CONCURRENTLY idx_market_data_time_series 
ON market_data (timestamp DESC, asset_id) 
WHERE timestamp >= NOW() - INTERVAL '1 year';

-- Technical indicators with time-series optimization
CREATE INDEX CONCURRENTLY idx_technical_indicators_latest
ON technical_indicators (asset_id, timeframe, timestamp DESC)
WHERE timestamp >= NOW() - INTERVAL '6 months';

-- Options data with expiration-based partitioning
CREATE INDEX CONCURRENTLY idx_options_data_expiration
ON options_data (underlying_asset_id, expiration_date, option_type, strike_price_cents)
WHERE expiration_date >= CURRENT_DATE;

-- News sentiment with recency bias
CREATE INDEX CONCURRENTLY idx_news_articles_recent_sentiment
ON news_articles (published_at DESC, sentiment_score DESC, importance_score DESC)
WHERE published_at >= NOW() - INTERVAL '30 days';

-- Risk assessments with asset-time clustering
CREATE INDEX CONCURRENTLY idx_risk_assessments_asset_recent
ON risk_assessments (asset_id, assessment_timestamp DESC)
WHERE assessment_timestamp >= NOW() - INTERVAL '7 days';

-- ============================================================================
-- QUERY-SPECIFIC PERFORMANCE INDEXES
-- ============================================================================

-- Signal matching optimization - find signals matching user risk tolerance
CREATE INDEX CONCURRENTLY idx_signals_user_risk_matching
ON signals (risk_score, asset_id, status, generated_at DESC)
WHERE status = 'ACTIVE' AND valid_until > NOW();

-- Portfolio performance analysis - efficient P&L calculations
CREATE INDEX CONCURRENTLY idx_portfolio_pnl_analysis
ON portfolio_snapshots (user_id, snapshot_date DESC, total_return_pct DESC)
WHERE snapshot_date >= CURRENT_DATE - INTERVAL '1 year';

-- Real-time position monitoring - current positions with P&L
CREATE INDEX CONCURRENTLY idx_positions_realtime_monitoring
ON positions (user_id, last_updated_at DESC, unrealized_pnl_cents DESC)
WHERE status = 'OPEN';

-- Signal performance tracking - success rate analysis
CREATE INDEX CONCURRENTLY idx_signal_performance_success_rate
ON signal_performance (signal_id, evaluation_date DESC, unrealized_pnl_pct DESC);

-- Trade execution analysis - slippage and timing analysis
CREATE INDEX CONCURRENTLY idx_trade_executions_performance_analysis
ON trade_executions (user_id, executed_at DESC, slippage_bps, execution_time_ms)
WHERE status = 'EXECUTED';

-- ============================================================================
-- FUNCTIONAL/EXPRESSION INDEXES
-- ============================================================================

-- Signal composite scoring for ranking
CREATE INDEX CONCURRENTLY idx_signals_weighted_score
ON signals (
    (
        (risk_score * 0.3) + 
        (profit_potential_score * 0.4) + 
        (confidence_score * 0.3)
    ) DESC
)
WHERE status = 'ACTIVE';

-- Position value calculations
CREATE INDEX CONCURRENTLY idx_positions_calculated_values
ON positions (
    user_id,
    (current_price_cents * quantity) DESC, -- position value
    ((current_price_cents - average_cost_cents) * quantity) DESC -- unrealized PnL
)
WHERE status = 'OPEN';

-- Risk-adjusted returns calculation
CREATE INDEX CONCURRENTLY idx_portfolio_risk_adjusted_returns
ON portfolio_snapshots (
    user_id,
    snapshot_date DESC,
    CASE 
        WHEN portfolio_var_1d_pct > 0 THEN daily_return_pct / portfolio_var_1d_pct 
        ELSE NULL 
    END DESC
);

-- Asset performance percentiles
CREATE INDEX CONCURRENTLY idx_signals_performance_percentile
ON signals (
    asset_id,
    PERCENT_RANK() OVER (
        PARTITION BY asset_id 
        ORDER BY profit_potential_score
    ) DESC
)
WHERE status IN ('TRIGGERED', 'EXPIRED');

-- ============================================================================
-- PARTIAL INDEXES FOR SPECIFIC CONDITIONS
-- ============================================================================

-- High-confidence, low-risk signals only
CREATE INDEX CONCURRENTLY idx_signals_premium_quality
ON signals (asset_id, generated_at DESC, profit_potential_score DESC)
WHERE status = 'ACTIVE' 
  AND confidence_score >= 80 
  AND risk_score <= 30
  AND profit_potential_score >= 70;

-- Large positions requiring special monitoring
CREATE INDEX CONCURRENTLY idx_positions_large_monitoring
ON positions (user_id, last_updated_at DESC, position_value_cents DESC)
WHERE status = 'OPEN' 
  AND position_value_cents >= 1000000; -- $10,000+

-- Recent high-impact news
CREATE INDEX CONCURRENTLY idx_news_high_impact_recent
ON news_articles (published_at DESC, sentiment_score, importance_score DESC)
WHERE published_at >= NOW() - INTERVAL '24 hours'
  AND importance_score >= 70;

-- Failed trade executions for analysis
CREATE INDEX CONCURRENTLY idx_trade_executions_failures
ON trade_executions (user_id, executed_at DESC, slippage_bps DESC)
WHERE status IN ('CANCELLED', 'REJECTED') 
  OR slippage_bps > 100;

-- ============================================================================
-- GIN INDEXES FOR JSON AND ARRAY DATA
-- ============================================================================

-- Signal factors and metadata search
CREATE INDEX CONCURRENTLY idx_signals_factors_gin
ON signals USING gin (
    (asset_specific_data || jsonb_build_object('tags', tags))
);

-- User notification preferences
CREATE INDEX CONCURRENTLY idx_user_profiles_notification_prefs
ON user_profiles USING gin (notification_preferences)
WHERE notification_preferences IS NOT NULL;

-- Risk factor contributions search
CREATE INDEX CONCURRENTLY idx_risk_assessments_metadata
ON risk_assessments USING gin (
    jsonb_build_object(
        'market_regime', market_regime,
        'model_version', model_version
    )
);

-- ============================================================================
-- FULL-TEXT SEARCH INDEXES
-- ============================================================================

-- News article full-text search with ranking
CREATE INDEX CONCURRENTLY idx_news_articles_fulltext
ON news_articles USING gin (search_vector);

-- Update trigger to maintain search vector
CREATE OR REPLACE FUNCTION update_news_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := 
        setweight(to_tsvector('english', COALESCE(NEW.headline, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.content, '')), 'B') ||
        setweight(to_tsvector('english', array_to_string(COALESCE(NEW.tags, ARRAY[]::VARCHAR[]), ' ')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER news_articles_search_vector_update
    BEFORE INSERT OR UPDATE ON news_articles
    FOR EACH ROW EXECUTE FUNCTION update_news_search_vector();

-- ============================================================================
-- TIME-SERIES PARTITIONING STRATEGY
-- ============================================================================

/*
PARTITIONING RECOMMENDATIONS:

1. MARKET_DATA table - Partition by month
   - High insert volume (millions of records daily)
   - Queries typically focus on recent data
   - Older data can be compressed or archived

2. TECHNICAL_INDICATORS - Partition by month
   - Similar pattern to market data
   - Time-based queries are common

3. SIGNAL_PERFORMANCE - Partition by quarter
   - Medium volume table
   - Performance analysis often quarterly

4. NEWS_ARTICLES - Partition by week
   - High volume during market hours
   - Relevance decreases quickly over time

Example partitioning for market_data:
*/

-- Create partitioned table (example - would replace existing table)
-- CREATE TABLE market_data_partitioned (LIKE market_data INCLUDING ALL) 
-- PARTITION BY RANGE (timestamp);

-- Create monthly partitions
-- CREATE TABLE market_data_2024_01 PARTITION OF market_data_partitioned
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ============================================================================
-- QUERY OPTIMIZATION HINTS AND PATTERNS
-- ============================================================================

/*
OPTIMIZED QUERY PATTERNS:

1. Signal Matching Query:
*/
-- Example: Find top signals for user based on preferences
-- EXPLAIN (ANALYZE, BUFFERS) 
SELECT 
    s.id,
    s.signal_name,
    s.risk_score,
    s.profit_potential_score,
    s.confidence_score,
    (s.risk_score * up.risk_tolerance / 100.0) as risk_match_score
FROM signals s
JOIN assets a ON s.asset_id = a.id
JOIN asset_classes ac ON a.asset_class_id = ac.id
JOIN user_asset_preferences uap ON ac.id = uap.asset_class_id
JOIN user_profiles up ON uap.user_id = up.id
WHERE s.status = 'ACTIVE'
  AND s.valid_until > NOW()
  AND up.user_id = 'target_user_id'
  AND s.risk_score <= (up.risk_tolerance + 10) -- Allow 10 point tolerance
  AND uap.is_enabled = TRUE
ORDER BY 
    (s.profit_potential_score * s.confidence_score * uap.preference_weight) DESC,
    s.generated_at DESC
LIMIT 20;

/*
2. Portfolio Performance Query:
*/
-- Example: Calculate portfolio performance metrics
-- WITH portfolio_metrics AS (
SELECT 
    ps.user_id,
    ps.snapshot_date,
    ps.total_value_cents,
    ps.daily_return_pct,
    LAG(ps.total_value_cents) OVER (
        PARTITION BY ps.user_id 
        ORDER BY ps.snapshot_date
    ) as previous_value,
    STDDEV(ps.daily_return_pct) OVER (
        PARTITION BY ps.user_id 
        ORDER BY ps.snapshot_date 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as volatility_30d
FROM portfolio_snapshots ps
WHERE ps.user_id = 'target_user_id'
  AND ps.snapshot_date >= CURRENT_DATE - INTERVAL '1 year'
ORDER BY ps.snapshot_date DESC;

/*
3. Risk Assessment Query:
*/
-- Example: Real-time risk monitoring across portfolio
-- SELECT 
    p.asset_id,
    a.symbol,
    p.position_value_cents,
    ra.overall_risk_score,
    ra.var_1d_pct,
    (p.position_value_cents * ra.var_1d_pct / 100.0) as position_var_cents
FROM positions p
JOIN assets a ON p.asset_id = a.id
JOIN LATERAL (
    SELECT * FROM risk_assessments ra2
    WHERE ra2.asset_id = p.asset_id
    ORDER BY ra2.assessment_timestamp DESC
    LIMIT 1
) ra ON true
WHERE p.user_id = 'target_user_id'
  AND p.status = 'OPEN'
ORDER BY position_var_cents DESC;

-- ============================================================================
-- MAINTENANCE AND MONITORING
-- ============================================================================

-- Index usage monitoring query
CREATE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE 
        WHEN idx_scan = 0 THEN 'UNUSED'
        WHEN idx_scan < 100 THEN 'LOW_USAGE'
        WHEN idx_scan < 1000 THEN 'MODERATE_USAGE'
        ELSE 'HIGH_USAGE'
    END as usage_category
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Query to find slow queries
CREATE VIEW slow_query_analysis AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE mean_time > 1000 -- Queries slower than 1 second
ORDER BY mean_time DESC;

-- Table bloat monitoring
CREATE VIEW table_bloat_stats AS
SELECT
    schemaname,
    tablename,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    n_dead_tup as dead_tuples,
    CASE 
        WHEN n_live_tup > 0 THEN 
            round(100.0 * n_dead_tup / (n_live_tup + n_dead_tup), 2)
        ELSE 0 
    END as bloat_percent
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY bloat_percent DESC;

-- ============================================================================
-- AUTOMATED MAINTENANCE TASKS
-- ============================================================================

-- Auto-vacuum settings for high-frequency tables
ALTER TABLE market_data SET (
    autovacuum_vacuum_scale_factor = 0.01,
    autovacuum_analyze_scale_factor = 0.005,
    autovacuum_vacuum_cost_delay = 10
);

ALTER TABLE signals SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE signal_matches SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_analyze_scale_factor = 0.01
);

-- Weekly index maintenance (example cron job commands)
/*
CRON SCHEDULE for database maintenance:

# Reindex frequently used indexes weekly (Sunday 2 AM)
0 2 * * 0 psql -d market_scanning_engine -c "REINDEX INDEX CONCURRENTLY idx_signals_asset_generated;"

# Update table statistics daily (2:30 AM)
30 2 * * * psql -d market_scanning_engine -c "ANALYZE signals, market_data, positions, portfolio_snapshots;"

# Clean up old partitions monthly (1st day, 3 AM)
0 3 1 * * psql -d market_scanning_engine -c "DROP TABLE IF EXISTS market_data_$(date -d '13 months ago' +'%Y_%m');"

# Vacuum large tables weekly (Sunday 3 AM)
0 3 * * 0 psql -d market_scanning_engine -c "VACUUM ANALYZE market_data, technical_indicators;"
*/

-- ============================================================================
-- CONNECTION POOLING AND CACHING RECOMMENDATIONS
-- ============================================================================

/*
CONNECTION POOLING CONFIGURATION (PgBouncer):

[databases]
market_scanning_engine = host=localhost port=5432 dbname=market_scanning_engine

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 20
max_db_connections = 100
ignore_startup_parameters = extra_float_digits

# Application-specific pools
signals_pool = max_db_connections=50   # High concurrency for signal generation
portfolio_pool = max_db_connections=30 # Medium concurrency for portfolio operations  
reporting_pool = max_db_connections=10 # Lower concurrency for analytical queries

REDIS CACHING STRATEGY:

1. Cache frequently accessed user profiles (TTL: 1 hour)
   Key pattern: user:profile:{user_id}

2. Cache asset metadata and current prices (TTL: 5 minutes)
   Key pattern: asset:metadata:{asset_id}
   Key pattern: price:{asset_id}

3. Cache signal match results (TTL: 15 minutes)
   Key pattern: signal:matches:{user_id}

4. Cache portfolio snapshots (TTL: 1 hour)
   Key pattern: portfolio:snapshot:{user_id}:{date}

5. Cache market regime data (TTL: 4 hours)
   Key pattern: market:regime:current
*/