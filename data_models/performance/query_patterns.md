# Market Scanning Engine - Query Patterns and Performance Guide

## Query Pattern Analysis and Optimization

This document outlines the most common query patterns in the Market Scanning Engine and their optimization strategies. All queries are designed around the 0-100 risk scoring system and cent-based monetary values for precision.

### 1. Signal Discovery and Matching Patterns

#### 1.1 Find Active Signals for User Risk Profile
**Frequency**: Very High (100+ queries/second)  
**Optimization**: Composite index on (risk_score, status, valid_until)

```sql
-- Optimized signal matching query
SELECT 
    s.id,
    s.signal_name,
    s.asset_id,
    a.symbol,
    s.risk_score,
    s.profit_potential_score,
    s.confidence_score,
    s.entry_price_cents,
    s.target_price_cents,
    s.stop_loss_price_cents
FROM signals s
JOIN assets a ON s.asset_id = a.id
WHERE s.status = 'ACTIVE'
  AND s.valid_until > NOW()
  AND s.risk_score BETWEEN $1 AND $2  -- User risk tolerance range
  AND s.profit_potential_score >= $3   -- Minimum profit threshold
  AND s.confidence_score >= $4         -- Minimum confidence threshold
ORDER BY 
    (s.profit_potential_score * s.confidence_score / s.risk_score) DESC,
    s.generated_at DESC
LIMIT 50;
```

**Key Indexes Used**:
- `idx_signals_status_active`
- `idx_signals_composite_score`
- `idx_signals_premium_quality` (for high-quality signals)

#### 1.2 Real-time Signal Matching Engine
**Frequency**: High (50+ queries/second)  
**Pattern**: Match new signals to all eligible users

```sql
-- Find users matching signal criteria
WITH signal_criteria AS (
    SELECT 
        risk_score,
        profit_potential_score,
        confidence_score,
        asset_id
    FROM signals 
    WHERE id = $1
),
eligible_users AS (
    SELECT DISTINCT
        up.id as user_id,
        up.risk_tolerance,
        uap.preference_weight,
        up.max_position_size_pct
    FROM user_profiles up
    JOIN user_asset_preferences uap ON up.id = uap.user_id
    JOIN assets a ON a.asset_class_id = uap.asset_class_id
    CROSS JOIN signal_criteria sc
    WHERE a.id = sc.asset_id
      AND up.is_active = TRUE
      AND uap.is_enabled = TRUE
      AND sc.risk_score <= (up.risk_tolerance + 15)  -- 15 point tolerance
      AND sc.profit_potential_score >= 50
      AND sc.confidence_score >= 60
)
SELECT * FROM eligible_users
ORDER BY 
    (risk_tolerance + preference_weight * 100) DESC
LIMIT 1000;
```

### 2. Portfolio Management Patterns

#### 2.1 Real-time Portfolio Valuation
**Frequency**: High (200+ queries/second)  
**Pattern**: Calculate current portfolio values and P&L

```sql
-- Current portfolio positions with real-time P&L
SELECT 
    p.id,
    p.asset_id,
    a.symbol,
    p.quantity,
    p.average_cost_cents,
    md.close_price_cents as current_price_cents,
    (md.close_price_cents - p.average_cost_cents) * p.quantity as unrealized_pnl_cents,
    (md.close_price_cents * ABS(p.quantity)) as position_value_cents,
    CASE 
        WHEN p.quantity > 0 THEN 'LONG'
        ELSE 'SHORT'
    END as position_type
FROM positions p
JOIN assets a ON p.asset_id = a.id
JOIN LATERAL (
    SELECT close_price_cents
    FROM market_data md2
    WHERE md2.asset_id = p.asset_id
    ORDER BY md2.timestamp DESC
    LIMIT 1
) md ON true
WHERE p.user_id = $1
  AND p.status = 'OPEN'
ORDER BY position_value_cents DESC;
```

**Key Indexes Used**:
- `idx_positions_user_open`
- `idx_market_data_time_series`
- `idx_positions_realtime_monitoring`

#### 2.2 Portfolio Risk Assessment
**Frequency**: Medium (20+ queries/second)  
**Pattern**: Calculate portfolio-wide risk metrics

```sql
-- Portfolio risk analysis with VaR calculation
WITH position_risks AS (
    SELECT 
        p.user_id,
        p.asset_id,
        (md.close_price_cents * ABS(p.quantity)) as position_value_cents,
        ra.overall_risk_score,
        ra.var_1d_pct,
        ra.beta,
        ((md.close_price_cents * ABS(p.quantity)) * ra.var_1d_pct / 100.0) as position_var_cents
    FROM positions p
    JOIN LATERAL (
        SELECT close_price_cents
        FROM market_data md2
        WHERE md2.asset_id = p.asset_id
        ORDER BY md2.timestamp DESC
        LIMIT 1
    ) md ON true
    JOIN LATERAL (
        SELECT overall_risk_score, var_1d_pct, beta
        FROM risk_assessments ra2
        WHERE ra2.asset_id = p.asset_id
        ORDER BY ra2.assessment_timestamp DESC
        LIMIT 1
    ) ra ON true
    WHERE p.user_id = $1 AND p.status = 'OPEN'
)
SELECT 
    user_id,
    SUM(position_value_cents) as total_portfolio_value_cents,
    SQRT(SUM(position_var_cents * position_var_cents)) as portfolio_var_cents,
    AVG(overall_risk_score) as avg_risk_score,
    MAX(position_value_cents) as largest_position_cents,
    COUNT(*) as num_positions
FROM position_risks
GROUP BY user_id;
```

### 3. Signal Performance Tracking Patterns

#### 3.1 Signal Success Rate Analysis
**Frequency**: Medium (10+ queries/minute)  
**Pattern**: Analyze signal performance over time

```sql
-- Signal performance analysis with success metrics
WITH signal_outcomes AS (
    SELECT 
        s.id,
        s.signal_name,
        s.signal_source,
        s.risk_score,
        s.profit_potential_score,
        s.confidence_score,
        sp.unrealized_pnl_pct,
        sp.max_favorable_excursion_pct,
        sp.max_adverse_excursion_pct,
        CASE 
            WHEN sp.unrealized_pnl_pct > 0 THEN 1
            WHEN sp.unrealized_pnl_pct < 0 THEN -1
            ELSE 0
        END as outcome_flag
    FROM signals s
    LEFT JOIN signal_performance sp ON s.id = sp.signal_id
    WHERE s.generated_at >= $1  -- Date range
      AND s.generated_at <= $2
      AND s.status IN ('TRIGGERED', 'EXPIRED')
)
SELECT 
    signal_source,
    COUNT(*) as total_signals,
    AVG(profit_potential_score) as avg_profit_score,
    AVG(risk_score) as avg_risk_score,
    AVG(confidence_score) as avg_confidence_score,
    COUNT(*) FILTER (WHERE outcome_flag = 1) as winning_signals,
    COUNT(*) FILTER (WHERE outcome_flag = -1) as losing_signals,
    AVG(unrealized_pnl_pct) as avg_return_pct,
    STDDEV(unrealized_pnl_pct) as return_volatility_pct,
    AVG(unrealized_pnl_pct) / NULLIF(STDDEV(unrealized_pnl_pct), 0) as sharpe_ratio
FROM signal_outcomes
GROUP BY signal_source
HAVING COUNT(*) >= 10  -- Minimum sample size
ORDER BY sharpe_ratio DESC NULLS LAST;
```

### 4. Market Data and Technical Analysis Patterns

#### 4.1 Real-time Technical Indicator Updates
**Frequency**: Very High (1000+ queries/second)  
**Pattern**: Update technical indicators with new market data

```sql
-- Calculate RSI and moving averages for real-time updates
WITH price_series AS (
    SELECT 
        asset_id,
        timestamp,
        close_price_cents,
        LAG(close_price_cents) OVER (
            PARTITION BY asset_id 
            ORDER BY timestamp
        ) as prev_close_cents,
        AVG(close_price_cents) OVER (
            PARTITION BY asset_id 
            ORDER BY timestamp 
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) as sma_20_cents,
        AVG(close_price_cents) OVER (
            PARTITION BY asset_id 
            ORDER BY timestamp 
            ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
        ) as sma_50_cents,
        ROW_NUMBER() OVER (
            PARTITION BY asset_id 
            ORDER BY timestamp DESC
        ) as rn
    FROM market_data
    WHERE asset_id = $1
      AND timestamp >= NOW() - INTERVAL '60 days'
),
rsi_calculation AS (
    SELECT 
        asset_id,
        timestamp,
        close_price_cents,
        sma_20_cents,
        sma_50_cents,
        CASE 
            WHEN prev_close_cents IS NOT NULL THEN
                CASE 
                    WHEN close_price_cents > prev_close_cents 
                    THEN close_price_cents - prev_close_cents 
                    ELSE 0 
                END
            ELSE NULL
        END as gain_cents,
        CASE 
            WHEN prev_close_cents IS NOT NULL THEN
                CASE 
                    WHEN close_price_cents < prev_close_cents 
                    THEN prev_close_cents - close_price_cents 
                    ELSE 0 
                END
            ELSE NULL
        END as loss_cents
    FROM price_series
    WHERE rn <= 50  -- Last 50 periods
)
SELECT 
    asset_id,
    timestamp,
    close_price_cents,
    sma_20_cents::BIGINT,
    sma_50_cents::BIGINT,
    CASE 
        WHEN AVG(loss_cents) OVER (ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) > 0 THEN
            100 - (100 / (1 + 
                AVG(gain_cents) OVER (ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) / 
                AVG(loss_cents) OVER (ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
            ))
        ELSE NULL
    END as rsi_14
FROM rsi_calculation
ORDER BY timestamp DESC
LIMIT 1;
```

### 5. Risk Management and Alert Patterns

#### 5.1 Real-time Risk Monitoring
**Frequency**: High (100+ queries/second)  
**Pattern**: Monitor portfolio risk limits and generate alerts

```sql
-- Portfolio risk limit monitoring
WITH risk_metrics AS (
    SELECT 
        up.id as user_id,
        up.max_daily_loss_pct,
        up.max_position_size_pct,
        ps_today.total_value_cents as current_value_cents,
        ps_yesterday.total_value_cents as yesterday_value_cents,
        ((ps_today.total_value_cents::DECIMAL - ps_yesterday.total_value_cents) / 
         ps_yesterday.total_value_cents * 100) as daily_return_pct,
        p.position_value_cents,
        (p.position_value_cents::DECIMAL / ps_today.total_value_cents * 100) as position_size_pct
    FROM user_profiles up
    LEFT JOIN portfolio_snapshots ps_today ON up.id = ps_today.user_id 
        AND ps_today.snapshot_date = CURRENT_DATE
    LEFT JOIN portfolio_snapshots ps_yesterday ON up.id = ps_yesterday.user_id 
        AND ps_yesterday.snapshot_date = CURRENT_DATE - INTERVAL '1 day'
    LEFT JOIN positions p ON up.id = p.user_id AND p.status = 'OPEN'
    WHERE up.is_active = TRUE
)
SELECT 
    user_id,
    'DAILY_LOSS_LIMIT' as alert_type,
    'CRITICAL' as severity,
    daily_return_pct as current_value,
    max_daily_loss_pct as threshold_value,
    'Daily loss limit exceeded' as message
FROM risk_metrics
WHERE daily_return_pct < -ABS(max_daily_loss_pct)

UNION ALL

SELECT 
    user_id,
    'POSITION_SIZE_LIMIT' as alert_type,
    'WARNING' as severity,
    position_size_pct as current_value,
    max_position_size_pct as threshold_value,
    'Position size limit exceeded for individual position' as message
FROM risk_metrics
WHERE position_size_pct > max_position_size_pct;
```

### 6. Historical Analysis and Reporting Patterns

#### 6.1 Portfolio Performance Attribution
**Frequency**: Low (1-10 queries/hour)  
**Pattern**: Analyze performance attribution by asset class, time period

```sql
-- Monthly portfolio performance attribution
WITH monthly_performance AS (
    SELECT 
        ps.user_id,
        DATE_TRUNC('month', ps.snapshot_date) as month,
        FIRST_VALUE(ps.total_value_cents) OVER (
            PARTITION BY ps.user_id, DATE_TRUNC('month', ps.snapshot_date)
            ORDER BY ps.snapshot_date
        ) as month_start_value_cents,
        LAST_VALUE(ps.total_value_cents) OVER (
            PARTITION BY ps.user_id, DATE_TRUNC('month', ps.snapshot_date)
            ORDER BY ps.snapshot_date
            RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as month_end_value_cents,
        pa.asset_class_id,
        ac.name as asset_class_name,
        AVG(pa.weight_pct) as avg_weight_pct,
        SUM(pa.unrealized_pnl_cents) as total_pnl_cents
    FROM portfolio_snapshots ps
    JOIN portfolio_allocations pa ON ps.id = pa.portfolio_snapshot_id
    JOIN asset_classes ac ON pa.asset_class_id = ac.id
    WHERE ps.user_id = $1
      AND ps.snapshot_date >= $2  -- Start date
      AND ps.snapshot_date <= $3  -- End date
    GROUP BY 
        ps.user_id,
        DATE_TRUNC('month', ps.snapshot_date),
        pa.asset_class_id,
        ac.name
),
attribution_calc AS (
    SELECT 
        user_id,
        month,
        asset_class_id,
        asset_class_name,
        avg_weight_pct,
        total_pnl_cents,
        ((month_end_value_cents::DECIMAL - month_start_value_cents) / 
         month_start_value_cents * 100) as total_return_pct,
        (total_pnl_cents::DECIMAL / month_start_value_cents * 100) as contribution_pct
    FROM monthly_performance
)
SELECT 
    month,
    asset_class_name,
    avg_weight_pct,
    contribution_pct,
    (contribution_pct / avg_weight_pct) as relative_performance
FROM attribution_calc
WHERE user_id = $1
ORDER BY month DESC, contribution_pct DESC;
```

## Performance Optimization Guidelines

### 1. Query Design Principles
- **Always use prepared statements** with parameterized queries
- **Limit result sets** appropriately (use LIMIT clause)
- **Use appropriate data types**: BIGINT for cents, INTEGER for scores (0-100)
- **Avoid SELECT \*** - specify only needed columns
- **Use EXISTS instead of IN** for subqueries when possible

### 2. Index Strategy
- **Composite indexes** for multi-column WHERE clauses
- **Partial indexes** for filtered conditions (status = 'ACTIVE')
- **Expression indexes** for computed values used in WHERE/ORDER BY
- **Regular index maintenance** - monitor usage and rebuild as needed

### 3. Caching Strategy
- **User profiles and preferences**: Cache for 1 hour
- **Asset metadata**: Cache for 5 minutes
- **Market data**: Cache for 30 seconds
- **Signal matches**: Cache for 15 minutes
- **Portfolio snapshots**: Cache for 1 hour

### 4. Connection Management
- **Use connection pooling** (PgBouncer recommended)
- **Separate pools** for different workloads:
  - Real-time queries: Larger pool, transaction mode
  - Analytics queries: Smaller pool, session mode
  - Reporting: Dedicated pool with longer timeouts

### 5. Monitoring and Alerts
- **Track slow queries** (> 1 second)
- **Monitor index usage** regularly
- **Set up alerts** for connection pool exhaustion
- **Monitor table bloat** and schedule maintenance