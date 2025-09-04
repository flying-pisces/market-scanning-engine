"""
Working Test Runner - Step by Step Approach
Runs tests incrementally, handling missing dependencies gracefully
"""

import pytest
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_phase_tests(phase: str, verbose: bool = True) -> int:
    """
    Run tests for a specific phase
    
    Args:
        phase: Test phase to run
        verbose: Enable verbose output
        
    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    
    # Base pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.extend(['-v', '--tb=short'])
    
    # Add markers to suppress warnings
    pytest_args.extend(['--disable-warnings'])
    
    # Determine which tests to run
    if phase == 'core':
        test_path = 'tests/test_step_by_step.py::TestPhase1Core'
        description = "Core System (Configuration, Models, Risk Management)"
    elif phase == 'ml':
        test_path = 'tests/test_step_by_step.py::TestPhase2MLFallbacks'
        description = "ML System (with fallbacks)"
    elif phase == 'portfolio':
        test_path = 'tests/test_step_by_step.py::TestPhase3PortfolioFallbacks'
        description = "Portfolio System (with fallbacks)"
    elif phase == 'backtesting':
        test_path = 'tests/test_step_by_step.py::TestPhase4BacktestingFallbacks'
        description = "Backtesting System (with fallbacks)"
    elif phase == 'integration':
        test_path = 'tests/test_step_by_step.py::TestPhase5Integration'
        description = "Integration & Summary"
    elif phase == 'working':
        test_path = 'tests/test_working_basics.py'
        description = "Working Basic Tests"
    elif phase == 'step-by-step':
        test_path = 'tests/test_step_by_step.py'
        description = "All Step-by-Step Tests"
    else:
        print(f"Unknown phase: {phase}")
        return 1
    
    # Add test path
    pytest_args.append(test_path)
    
    # Print test configuration
    print("=" * 80)
    print("MARKET SCANNING ENGINE - INCREMENTAL TEST RUNNER")
    print("=" * 80)
    print(f"Phase: {phase}")
    print(f"Description: {description}")
    print(f"Test Path: {test_path}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST EXECUTION COMPLETE")
    print("=" * 80)
    
    if exit_code == 0:
        print(f"✅ PHASE '{phase.upper()}' - ALL TESTS PASSED")
    else:
        print(f"❌ PHASE '{phase.upper()}' - SOME TESTS FAILED")
    
    print(f"Exit Code: {exit_code}")
    print("=" * 80)
    
    return exit_code


def run_all_phases(verbose: bool = True) -> dict:
    """
    Run all test phases incrementally
    
    Returns:
        Dictionary with results for each phase
    """
    phases = ['core', 'ml', 'portfolio', 'backtesting', 'integration']
    results = {}
    
    print("\n🚀 RUNNING ALL PHASES INCREMENTALLY")
    print("=" * 80)
    
    for phase in phases:
        print(f"\n▶️  Starting Phase: {phase.upper()}")
        exit_code = run_phase_tests(phase, verbose)
        results[phase] = exit_code
        
        if exit_code == 0:
            print(f"✅ Phase {phase} completed successfully")
        else:
            print(f"⚠️  Phase {phase} had issues (continuing...)")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL PHASE SUMMARY")
    print("=" * 80)
    
    passed_phases = []
    failed_phases = []
    
    for phase, exit_code in results.items():
        if exit_code == 0:
            passed_phases.append(phase)
            print(f"✅ {phase.upper()} - PASSED")
        else:
            failed_phases.append(phase)
            print(f"❌ {phase.upper()} - FAILED")
    
    print(f"\n📊 RESULTS: {len(passed_phases)}/{len(phases)} phases passed")
    print("=" * 80)
    
    return results


def run_dependency_check():
    """Check and report on system dependencies"""
    print("🔍 CHECKING SYSTEM DEPENDENCIES")
    print("=" * 60)
    
    # Check ML dependencies
    try:
        from app.ml.dependencies import get_dependency_status
        status = get_dependency_status()
        
        print("ML DEPENDENCIES:")
        for dep, available in status['status'].items():
            status_icon = "✅" if available else "❌"
            print(f"  {status_icon} {dep}")
        
        print(f"\nML Readiness: {status['percentage']:.1f}% ({status['available']}/{status['total']})")
        print(f"Production Ready: {'✅ Yes' if status['ready_for_production'] else '❌ No'}")
        
    except Exception as e:
        print(f"❌ Could not check ML dependencies: {e}")
    
    # Check core components
    core_components = [
        ("Configuration System", "app.core.config", "get_settings"),
        ("Database Models", "app.models.database", "User"),
        ("Risk Management", "app.risk.management", "RiskManager"),
    ]
    
    print(f"\nCORE COMPONENTS:")
    for name, module, attr in core_components:
        try:
            mod = __import__(module, fromlist=[attr])
            getattr(mod, attr)
            print(f"  ✅ {name}")
        except Exception:
            print(f"  ❌ {name}")
    
    print("=" * 60)


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description='Market Scanning Engine - Working Test Runner')
    parser.add_argument(
        '--phase', 
        choices=['core', 'ml', 'portfolio', 'backtesting', 'integration', 'working', 'step-by-step', 'all'],
        default='core',
        help='Test phase to run'
    )
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')
    
    args = parser.parse_args()
    
    if args.check_deps:
        run_dependency_check()
        return
    
    if args.phase == 'all':
        results = run_all_phases(verbose=not args.quiet)
        
        # Exit with non-zero if any phase failed
        failed_count = sum(1 for exit_code in results.values() if exit_code != 0)
        sys.exit(failed_count)
    else:
        exit_code = run_phase_tests(args.phase, verbose=not args.quiet)
        sys.exit(exit_code)


if __name__ == "__main__":
    main()