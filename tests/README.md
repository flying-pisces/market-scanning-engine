# Risk Scoring System - Comprehensive Test Suite

This test suite provides comprehensive validation for the risk scoring system used in the market scanning engine. It ensures reliability, accuracy, and performance for live trading environments where risk assessment quality directly impacts financial outcomes.

## Test Architecture Overview

### Test Categories

1. **Unit Tests** (`test_risk_components.py`)
   - Individual risk component validation (volatility, liquidity, time decay, market regime, position size)
   - Mathematical calculation accuracy
   - Input boundary testing
   - Error handling validation

2. **Integration Tests** (`test_risk_integration.py`)
   - End-to-end risk assessment workflows
   - Cross-asset class consistency
   - Real-time data integration
   - Portfolio-level risk aggregation

3. **Performance Tests** (`test_risk_performance.py`)
   - Single request latency (<100ms requirement)
   - Batch processing throughput (>1000 calc/sec)
   - Concurrent processing capacity
   - Memory usage optimization

4. **Stress Tests** (`test_risk_stress.py`)
   - Market crash scenarios
   - System overload conditions
   - Corrupted data resilience
   - Security injection resistance

5. **Accuracy Tests** (`test_risk_accuracy.py`)
   - Historical correlation validation (R² > 0.75 target)
   - Cross-asset class calibration
   - Market regime sensitivity
   - Predictive accuracy measurement

## Quick Start

### Prerequisites

```bash
# Install Python 3.9+
python --version

# Install test dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
python test_framework.py

# Run specific categories
python test_framework.py --categories unit integration

# Run with custom configuration
python test_framework.py --config custom_config.json --output-dir results/
```

### CI/CD Integration

The framework integrates with major CI/CD systems:

```bash
# GitHub Actions
cp ci_config.yml ../.github/workflows/risk-tests.yml

# GitLab CI
cp ci_config.yml ../.gitlab-ci.yml

# Jenkins
# Use pipeline configuration from ci_config.yml
```

## Test Configuration

### Default Configuration

```json
{
  "test_categories": {
    "unit": {
      "enabled": true,
      "timeout_minutes": 10,
      "required_pass_rate": 0.95,
      "critical": true
    },
    "performance": {
      "enabled": true,
      "timeout_minutes": 20,
      "required_pass_rate": 0.85,
      "critical": true
    }
  },
  "performance_thresholds": {
    "max_latency_ms": 100,
    "min_throughput_ops_per_sec": 1000,
    "max_memory_usage_mb": 2000
  },
  "accuracy_thresholds": {
    "min_correlation": 0.4,
    "min_r_squared": 0.15,
    "min_calibration_score": 0.3
  }
}
```

### Asset Class Expectations

The test suite validates risk scores against expected ranges:

- **Options (SPY/QQQ)**: 70-95 (high volatility, time decay)
- **Stocks (S&P 500)**: 30-80 (moderate volatility)
- **Bonds (Treasury/Corporate)**: 5-40 (lower volatility)  
- **T-bills/CDs**: 0-20 (minimal risk)

## Performance Requirements

### Latency Benchmarks
- **Individual Risk Score**: <50ms (target: <25ms)
- **Batch Processing (100 assets)**: <2 seconds
- **Real-time Updates**: <100ms end-to-end
- **Portfolio Risk Assessment**: <200ms

### Throughput Benchmarks
- **Concurrent Calculations**: 1000+ simultaneous
- **Daily Score Updates**: 10,000+ assets
- **Peak Load Handling**: 5x normal capacity

### Memory Limits
- **Per Calculation**: <100KB
- **Batch Processing**: <2GB for 10K assets
- **Memory Growth**: <10MB per 10K calculations

## Accuracy Validation

### Historical Correlation
- **Volatility Prediction**: R² > 0.75 with realized volatility
- **Drawdown Prediction**: Correlation > 0.25 with actual drawdowns
- **Regime Classification**: >80% accuracy in market regime detection

### Cross-Asset Consistency
- Risk ordering: Options > Stocks > Bonds > T-bills
- Score calibration within expected ranges
- Consistent behavior across market conditions

## Test Data Management

### Synthetic Data Generation
- Realistic OHLCV data with proper statistical properties
- Various market regimes (bull, bear, sideways, high/low vol)
- Edge cases (crashes, flash crashes, circuit breakers)

### Historical Data (Optional)
- Download real market data for accuracy validation
- Configurable data sources and timeframes
- Automated data quality validation

## Continuous Integration

### Deployment Gates
Tests enforce strict quality gates for production deployment:

1. **Critical Test Pass Rate**: >95%
2. **Overall Pass Rate**: >90%
3. **Performance Requirements**: All latency/throughput thresholds met
4. **Accuracy Requirements**: Historical correlation > 0.4

### Failure Handling
- Immediate blocking for critical test failures
- Performance regression detection
- Automatic rollback triggers
- Detailed failure reporting and alerting

## Advanced Features

### Property-Based Testing
Uses Hypothesis for automatic edge case discovery:

```python
@given(volatility=floats(min_value=0.01, max_value=2.0))
def test_volatility_risk_properties(volatility):
    score = calculate_volatility_risk(volatility)
    assert 0 <= score <= 100
    assert score > 50 if volatility > 0.3 else score <= 50
```

### Mutation Testing
Validates test quality by introducing code mutations:

```bash
mutmut run --paths-to-mutate=risk_scoring/
```

### Performance Regression Detection
Tracks performance over time and alerts on regressions:

```bash
asv run --python=python3.9
asv compare main HEAD
```

## Troubleshooting

### Common Issues

**Memory Errors During Stress Tests**
```bash
# Increase system limits
ulimit -n 65536
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf
```

**Performance Test Inconsistency**
```bash
# Disable CPU frequency scaling
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Network Timeouts in CI**
```bash
# Increase timeout values
export TEST_TIMEOUT=3600
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run with verbose output
python test_framework.py --categories unit -v --tb=long
```

### Test Isolation

Run tests in isolated environments:

```bash
# Using tox
tox -e py39

# Using nox
nox -s tests

# Using Docker
docker run --rm -v $(pwd):/app python:3.9 bash -c "cd /app/tests && python test_framework.py"
```

## Reporting and Analytics

### HTML Reports
Generated automatically with test results, performance metrics, and deployment recommendations:

```
test_reports/
├── risk_test_report_20240301_143022.html
├── risk_test_report_20240301_143022.json
└── performance_history.json
```

### Metrics Collection
Integration with monitoring systems:

- Test execution metrics
- Performance trend analysis  
- Failure rate tracking
- Deployment success correlation

### Custom Dashboards
Template configurations for:
- Grafana dashboard
- DataDog monitors
- New Relic alerts
- PagerDuty escalations

## Contributing

### Adding New Tests

1. **Component Tests**: Add to `test_risk_components.py`
2. **Integration Tests**: Add to `test_risk_integration.py`
3. **Performance Tests**: Add to `test_risk_performance.py`
4. **Stress Tests**: Add to `test_risk_stress.py`
5. **Accuracy Tests**: Add to `test_risk_accuracy.py`

### Test Naming Convention

```python
def test_[component]_[functionality]_[condition]():
    """Test [component] [functionality] under [condition]"""
    pass

# Examples
def test_volatility_calculation_accuracy():
def test_liquidity_risk_extreme_spreads():
def test_performance_concurrent_load():
```

### Code Coverage

Maintain >90% test coverage:

```bash
pytest --cov=risk_scoring --cov-report=html
open htmlcov/index.html
```

## License and Support

This test suite is designed for the market scanning engine risk scoring system. For support:

1. Check test logs in `logs/` directory
2. Review HTML reports in `test_reports/`
3. Examine CI/CD pipeline outputs
4. Contact the QA engineering team

**Critical Production Note**: These tests validate systems handling real money in live trading environments. All test failures should be treated as potential financial risk and investigated immediately.