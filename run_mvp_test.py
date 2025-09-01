#!/usr/bin/env python3
"""
Market Scanning Engine MVP Test Runner
Simple test runner to validate core functionality
"""

import sys
import traceback
from tests.test_mvp_basic import TestRiskScoring, TestSignalMatching, TestEndToEndFlow


def run_test_suite():
    """Run the complete MVP test suite"""
    
    print("🚀 Market Scanning Engine MVP Test Suite")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    # Test categories
    test_categories = [
        ("Risk Scoring", TestRiskScoring, [
            "test_basic_risk_calculation",
            "test_asset_class_risk_ranges", 
            "test_symbol_specific_adjustments",
            "test_risk_factor_breakdown"
        ]),
        ("Signal Matching", TestSignalMatching, [
            "test_perfect_risk_match",
            "test_risk_tolerance_boundaries",
            "test_asset_class_preferences", 
            "test_position_size_compatibility",
            "test_confidence_boost"
        ]),
        ("End-to-End Flow", TestEndToEndFlow, [
            "test_signal_creation_to_scoring",
            "test_user_signal_compatibility"
        ])
    ]
    
    for category_name, test_class, test_methods in test_categories:
        print(f"\n📊 Testing {category_name}...")
        print("-" * 40)
        
        test_instance = test_class()
        category_passed = 0
        category_total = len(test_methods)
        
        for test_method_name in test_methods:
            total_tests += 1
            
            try:
                test_method = getattr(test_instance, test_method_name)
                test_method()
                
                print(f"✅ {test_method_name}")
                passed_tests += 1
                category_passed += 1
                
            except AssertionError as e:
                print(f"❌ {test_method_name}: {e}")
                failed_tests += 1
                
            except Exception as e:
                print(f"💥 {test_method_name}: Unexpected error - {e}")
                failed_tests += 1
        
        # Category summary
        if category_passed == category_total:
            print(f"🎉 {category_name}: ALL TESTS PASSED ({category_passed}/{category_total})")
        else:
            print(f"⚠️  {category_name}: {category_passed}/{category_total} tests passed")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📈 FINAL RESULTS")
    print("=" * 50)
    
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests} ✅")
    print(f"Failed: {failed_tests} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! MVP is ready for deployment.")
        print("\n✨ Core Features Validated:")
        print("   • Risk scoring algorithm (0-100 scale)")
        print("   • Multi-asset class support")
        print("   • User-signal matching logic")
        print("   • End-to-end signal processing")
        return True
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix issues.")
        return False


def validate_system_components():
    """Validate that system components can be imported and initialized"""
    
    print("\n🔧 System Component Validation")
    print("-" * 40)
    
    components_ok = True
    
    try:
        from app.services.risk_scoring import risk_scorer, AssetClass
        print("✅ Risk scoring service")
    except Exception as e:
        print(f"❌ Risk scoring service: {e}")
        components_ok = False
    
    try:
        from app.services.matching import SignalMatcher
        print("✅ Signal matching service")
    except Exception as e:
        print(f"❌ Signal matching service: {e}")
        components_ok = False
    
    try:
        from app.models.database import User, Signal, UserSignalMatch
        print("✅ Database models")
    except Exception as e:
        print(f"❌ Database models: {e}")
        components_ok = False
    
    try:
        from app.core.config import get_settings
        settings = get_settings()
        print("✅ Configuration management")
    except Exception as e:
        print(f"❌ Configuration management: {e}")
        components_ok = False
    
    return components_ok


if __name__ == "__main__":
    print("Market Scanning Engine - MVP Validation")
    print("Testing core functionality before deployment\n")
    
    # Validate system components first
    if not validate_system_components():
        print("\n💥 System component validation failed!")
        sys.exit(1)
    
    # Run test suite
    if run_test_suite():
        print("\n✅ MVP validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ MVP validation failed!")
        sys.exit(1)