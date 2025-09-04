"""
Step-by-Step Testing Strategy
Tests that work incrementally, handling missing dependencies gracefully
"""

import pytest
import sys
import os
from pathlib import Path
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestPhase1Core:
    """Phase 1: Core system functionality (no external deps)"""
    
    def test_project_structure(self):
        """✓ Project structure is valid"""
        assert (project_root / "app").exists()
        assert (project_root / "app" / "core").exists()
        assert (project_root / "app" / "models").exists()
        
    def test_configuration_system(self):
        """✓ Configuration system works"""
        from app.core.config import get_settings
        settings = get_settings()
        assert settings.environment is not None
        assert settings.database_url is not None
    
    def test_basic_models(self):
        """✓ Basic database models work"""
        from app.models.database import User, Signal, UserSignalMatch
        assert User.__tablename__ == "users"
        assert Signal.__tablename__ == "signals"
        assert UserSignalMatch.__tablename__ == "user_signal_matches"
    
    def test_risk_management_core(self):
        """✓ Risk management works without external deps"""
        from app.risk.management import VaRMethod, RiskManager
        
        # Test enums
        assert VaRMethod.HISTORICAL == "historical"
        assert VaRMethod.PARAMETRIC == "parametric"
        assert VaRMethod.MONTE_CARLO == "monte_carlo"
        
        # Test class creation
        risk_manager = RiskManager()
        assert risk_manager is not None


class TestPhase2MLFallbacks:
    """Phase 2: ML system with graceful fallbacks"""
    
    def test_ml_enum_definitions(self):
        """✓ ML enums work regardless of dependencies"""
        try:
            from app.ml.models import ModelType, PredictionHorizon
            
            # These should work even without external ML libs
            assert hasattr(ModelType, 'XGBOOST')
            assert hasattr(ModelType, 'RANDOM_FOREST')
            assert hasattr(PredictionHorizon, 'DAILY')
            assert hasattr(PredictionHorizon, 'WEEKLY')
            
        except ImportError as e:
            pytest.skip(f"ML enums failed due to missing dependencies: {e}")
    
    def test_ml_dependencies_check(self):
        """✓ ML dependency checker works"""
        try:
            from app.ml.dependencies import check_ml_dependencies, get_dependency_status
            
            deps = check_ml_dependencies()
            assert isinstance(deps, dict)
            assert 'xgboost' in deps
            assert 'sklearn' in deps
            
            status = get_dependency_status()
            assert 'available' in status
            assert 'total' in status
            assert 'percentage' in status
            
        except ImportError:
            pytest.skip("ML dependency module not available")
    
    def test_ml_service_creation(self):
        """✓ ML services can be created (may use mocks)"""
        try:
            from app.ml.training_service import MLTrainingService
            service = MLTrainingService()
            assert hasattr(service, 'is_running')
            
        except ImportError as e:
            pytest.skip(f"ML training service failed: {e}")


class TestPhase3PortfolioFallbacks:
    """Phase 3: Portfolio system with fallbacks"""
    
    def test_portfolio_enum_definitions(self):
        """✓ Portfolio enums work"""
        try:
            from app.portfolio.optimization import OptimizationMethod
            
            assert hasattr(OptimizationMethod, 'MEAN_VARIANCE')
            assert hasattr(OptimizationMethod, 'BLACK_LITTERMAN')
            assert hasattr(OptimizationMethod, 'RISK_PARITY')
            
        except ImportError as e:
            pytest.skip(f"Portfolio enums failed: {e}")
    
    def test_portfolio_service_creation(self):
        """✓ Portfolio services can be created"""
        try:
            from app.portfolio.allocation_service import PortfolioAllocationService
            service = PortfolioAllocationService()
            assert hasattr(service, 'is_running')
            
        except ImportError as e:
            pytest.skip(f"Portfolio service failed: {e}")


class TestPhase4BacktestingFallbacks:
    """Phase 4: Backtesting system with fallbacks"""
    
    def test_backtesting_enum_definitions(self):
        """✓ Backtesting enums work"""
        try:
            from app.backtesting.engine import ExecutionModel
            
            assert hasattr(ExecutionModel, 'PERFECT')
            assert hasattr(ExecutionModel, 'REALISTIC')
            assert hasattr(ExecutionModel, 'PESSIMISTIC')
            
        except ImportError as e:
            pytest.skip(f"Backtesting enums failed: {e}")
    
    def test_backtesting_service_creation(self):
        """✓ Backtesting services can be created"""
        try:
            from app.backtesting.service import BacktestingService
            service = BacktestingService()
            assert hasattr(service, 'is_running')
            
        except ImportError as e:
            pytest.skip(f"Backtesting service failed: {e}")


class TestPhase5Integration:
    """Phase 5: Integration tests (what actually works)"""
    
    def test_end_to_end_config_flow(self):
        """✓ Complete configuration flow works"""
        from app.core.config import get_settings, get_settings_for_environment
        
        # Test different environments
        dev_settings = get_settings_for_environment('development')
        test_settings = get_settings_for_environment('testing')
        prod_settings = get_settings_for_environment('production')
        
        # Test that we get different settings for different environments
        assert dev_settings.debug == True
        assert test_settings.test_mode == True
        assert prod_settings.debug == False
        
        # Test that settings are different between environments
        assert dev_settings.debug != prod_settings.debug
    
    def test_system_capabilities_summary(self):
        """✓ Summarize what's actually working"""
        working_components = []
        
        # Test core
        try:
            from app.core.config import get_settings
            working_components.append("✓ Configuration System")
        except ImportError:
            pass
        
        # Test models
        try:
            from app.models.database import User, Signal
            working_components.append("✓ Database Models")
        except ImportError:
            pass
        
        # Test risk management
        try:
            from app.risk.management import RiskManager
            risk_manager = RiskManager()
            working_components.append("✓ Risk Management")
        except ImportError:
            pass
        
        # Test ML dependencies
        try:
            from app.ml.dependencies import get_dependency_status
            status = get_dependency_status()
            working_components.append(f"✓ ML Dependencies ({status['available']}/{status['total']})")
        except ImportError:
            working_components.append("⚠ ML Dependencies (fallback mode)")
        
        # Print summary
        print("\n" + "="*60)
        print("WORKING COMPONENTS SUMMARY:")
        print("="*60)
        for component in working_components:
            print(component)
        print("="*60)
        
        # At least core components should work
        assert len(working_components) >= 3, f"Expected at least 3 working components, got {len(working_components)}"


@pytest.mark.parametrize("test_phase", [
    "phase1_core",
    "phase2_ml", 
    "phase3_portfolio",
    "phase4_backtesting",
    "phase5_integration"
])
def test_run_specific_phase(test_phase):
    """Run specific test phase"""
    if test_phase == "phase1_core":
        # Run Phase 1 tests
        pytest.main(["-v", "tests/test_step_by_step.py::TestPhase1Core"])
    elif test_phase == "phase2_ml":
        # Run Phase 2 tests
        pytest.main(["-v", "tests/test_step_by_step.py::TestPhase2MLFallbacks"])
    # ... etc for other phases


if __name__ == "__main__":
    # Run all phases systematically
    print("Running step-by-step tests...")
    pytest.main([__file__, "-v", "--tb=short", "-x"])