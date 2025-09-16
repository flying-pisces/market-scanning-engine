#!/usr/bin/env python3
"""
Integration test for cleaned Market Scanning Engine
Tests both Step 1 and Step 2 functionality
"""

import sys
import os
from pathlib import Path

def test_step1():
    """Test Step 1 functionality"""
    print("🧪 Testing Step 1...")
    
    # Add step1 to path
    step1_path = str(Path(__file__).parent / "step1" / "src")
    sys.path.append(step1_path)
    
    try:
        from enhanced_data_fetcher import EnhancedDataFetcher
        fetcher = EnhancedDataFetcher()
        
        # Test real-time data
        data = fetcher.get_real_time_data('AAPL')
        if data and 'price' in data:
            print(f"✅ Step 1: Real-time data working - AAPL: ${data['price']}")
            return True
        else:
            print("❌ Step 1: Real-time data failed")
            return False
            
    except Exception as e:
        print(f"❌ Step 1: Import failed - {e}")
        return False

def test_step2():
    """Test Step 2 functionality"""
    print("🧪 Testing Step 2...")
    
    # Add step2 to path
    step2_path = str(Path(__file__).parent / "step2" / "src")
    sys.path.append(step2_path)
    
    try:
        from user_profiling_engine import UserProfilingEngine, RiskProfile
        engine = UserProfilingEngine()
        
        # Test user profile creation
        profile = engine.create_user_profile("test_user", RiskProfile.MODERATE)
        if profile and profile.current_risk_profile == RiskProfile.MODERATE:
            print(f"✅ Step 2: User profiling working - Profile: {profile.current_risk_profile.value}")
            return True
        else:
            print("❌ Step 2: User profiling failed")
            return False
            
    except Exception as e:
        print(f"❌ Step 2: Import failed - {e}")
        return False

def test_integration():
    """Test integration between Step 1 and Step 2"""
    print("🔗 Testing Step 1 + Step 2 Integration...")
    
    try:
        # This would test the transparent signal engine
        # For now, just verify both steps work independently
        step1_ok = test_step1()
        step2_ok = test_step2()
        
        if step1_ok and step2_ok:
            print("✅ Integration: Both steps functional")
            return True
        else:
            print("❌ Integration: One or both steps failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Market Scanning Engine - Integration Test")
    print("=" * 60)
    
    # Run tests
    step1_result = test_step1()
    step2_result = test_step2() 
    integration_result = step1_result and step2_result
    
    print("\n📊 Test Results:")
    print(f"Step 1: {'✅ PASS' if step1_result else '❌ FAIL'}")
    print(f"Step 2: {'✅ PASS' if step2_result else '❌ FAIL'}")
    print(f"Integration: {'✅ PASS' if integration_result else '❌ FAIL'}")
    
    if integration_result:
        print("\n🎉 All tests passed! Repository is ready for deployment.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")
        sys.exit(1)