"""
Comprehensive Test Suite Runner
Runs all tests for the Market Scanning Engine with proper reporting
"""

import pytest
import sys
import os
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_test_suite(test_type=None, verbose=True, coverage=True):
    """
    Run the complete test suite or specific test categories
    
    Args:
        test_type: Specific test type to run (unit, integration, performance)
        verbose: Enable verbose output
        coverage: Enable coverage reporting
    """
    
    # Base pytest arguments
    pytest_args = []
    
    # Add verbosity
    if verbose:
        pytest_args.extend(['-v', '--tb=short'])
    
    # Add coverage if requested and available
    if coverage:
        try:
            import pytest_cov
            pytest_args.extend([
                '--cov=app',
                '--cov-report=html:tests/coverage_html',
                '--cov-report=term-missing',
                '--cov-fail-under=80'
            ])
        except ImportError:
            print("Warning: pytest-cov not installed, skipping coverage reporting")
            coverage = False
    
    # Determine which tests to run
    test_files = []
    
    if test_type is None or test_type == 'all':
        # Run all tests
        test_files.extend([
            'tests/test_framework.py',
            'tests/test_ml_comprehensive.py',
            'tests/test_portfolio_comprehensive.py',
            'tests/test_backtesting_comprehensive.py',
            'tests/test_risk_management_comprehensive.py'
        ])
    elif test_type == 'unit':
        # Run unit tests only (exclude integration and performance)
        pytest_args.extend(['-m', 'not integration and not performance'])
        test_files.extend([
            'tests/test_framework.py',
            'tests/test_ml_comprehensive.py',
            'tests/test_portfolio_comprehensive.py',
            'tests/test_backtesting_comprehensive.py',
            'tests/test_risk_management_comprehensive.py'
        ])
    elif test_type == 'integration':
        # Run integration tests only
        pytest_args.extend(['-m', 'integration'])
        test_files.extend([
            'tests/test_ml_comprehensive.py',
            'tests/test_portfolio_comprehensive.py',
            'tests/test_backtesting_comprehensive.py',
            'tests/test_risk_management_comprehensive.py'
        ])
    elif test_type == 'performance':
        # Run performance tests only
        pytest_args.extend(['-m', 'performance'])
        test_files.extend([
            'tests/test_ml_comprehensive.py',
            'tests/test_portfolio_comprehensive.py',
            'tests/test_backtesting_comprehensive.py',
            'tests/test_risk_management_comprehensive.py'
        ])
    elif test_type == 'ml':
        test_files.append('tests/test_ml_comprehensive.py')
    elif test_type == 'portfolio':
        test_files.append('tests/test_portfolio_comprehensive.py')
    elif test_type == 'backtesting':
        test_files.append('tests/test_backtesting_comprehensive.py')
    elif test_type == 'risk':
        test_files.append('tests/test_risk_management_comprehensive.py')
    elif test_type == 'framework':
        test_files.append('tests/test_framework.py')
    else:
        print(f"Unknown test type: {test_type}")
        return 1
    
    # Add test files to pytest args
    pytest_args.extend(test_files)
    
    # Add HTML report if available
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        import pytest_html
        pytest_args.extend([
            f'--html=tests/reports/test_report_{timestamp}.html',
            '--self-contained-html'
        ])
        html_report_enabled = True
    except ImportError:
        print("Warning: pytest-html not installed, skipping HTML report generation")
        html_report_enabled = False
    
    # Ensure reports directory exists
    os.makedirs('tests/reports', exist_ok=True)
    
    # Print test configuration
    print("="*80)
    print("MARKET SCANNING ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Test Type: {test_type or 'all'}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Files: {len(test_files)} file(s)")
    print(f"Coverage: {'Enabled' if coverage else 'Disabled'}")
    print(f"Verbose: {'Enabled' if verbose else 'Disabled'}")
    print("="*80)
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST EXECUTION COMPLETE")
    print("="*80)
    
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    
    print(f"Exit Code: {exit_code}")
    
    if coverage:
        print(f"Coverage Report: tests/coverage_html/index.html")
    
    if html_report_enabled:
        print(f"Test Report: tests/reports/test_report_{timestamp}.html")
    print("="*80)
    
    return exit_code

def run_quick_smoke_tests():
    """Run a quick subset of tests for rapid feedback"""
    print("Running Quick Smoke Tests...")
    
    pytest_args = [
        '-v',
        '--tb=short',
        '-x',  # Stop on first failure
        '-m', 'not performance and not integration',
        'tests/test_framework.py::TestDatabaseConnection::test_database_connection',
        'tests/test_framework.py::TestKafkaIntegration::test_kafka_connection',
        'tests/test_framework.py::TestRedisCache::test_redis_connection',
        'tests/test_ml_comprehensive.py::TestMLModels::test_model_initialization',
        'tests/test_portfolio_comprehensive.py::TestPortfolioOptimization::test_mean_variance_optimization',
        'tests/test_backtesting_comprehensive.py::TestBacktestEngine::test_perfect_execution_backtest',
        'tests/test_risk_management_comprehensive.py::TestRiskManager::test_historical_var_calculation'
    ]
    
    return pytest.main(pytest_args)

def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description='Market Scanning Engine Test Runner')
    parser.add_argument(
        '--type', 
        choices=['all', 'unit', 'integration', 'performance', 'ml', 'portfolio', 
                'backtesting', 'risk', 'framework', 'smoke'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument('--no-coverage', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    
    if args.type == 'smoke':
        exit_code = run_quick_smoke_tests()
    else:
        exit_code = run_test_suite(
            test_type=args.type,
            verbose=not args.quiet,
            coverage=not args.no_coverage
        )
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()