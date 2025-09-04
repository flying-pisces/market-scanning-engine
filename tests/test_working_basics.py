"""
Working Basic Tests - Step 1
Tests that should work immediately without external dependencies
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestProjectStructure:
    """Test basic project structure is valid"""
    
    def test_python_version(self):
        """Test Python version is compatible"""
        assert sys.version_info >= (3, 8), f"Python 3.8+ required, got {sys.version_info}"
    
    def test_project_root_exists(self):
        """Test project root directory structure"""
        assert project_root.exists(), "Project root should exist"
        assert (project_root / "app").exists(), "app directory should exist"
        assert (project_root / "tests").exists(), "tests directory should exist"
    
    def test_core_directories(self):
        """Test core application directories exist"""
        core_dirs = ["core", "models", "api", "services"]
        
        for dir_name in core_dirs:
            dir_path = project_root / "app" / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
    
    def test_advanced_directories(self):
        """Test advanced feature directories exist"""
        advanced_dirs = ["ml", "portfolio", "backtesting", "risk"]
        
        for dir_name in advanced_dirs:
            dir_path = project_root / "app" / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"


class TestConfigurationSystem:
    """Test configuration system works"""
    
    def test_config_module_imports(self):
        """Test config module can be imported"""
        try:
            from app.core.config import Settings, get_settings
            assert Settings is not None
            assert get_settings is not None
        except ImportError as e:
            pytest.fail(f"Failed to import config: {e}")
    
    def test_default_settings(self):
        """Test default settings can be loaded"""
        from app.core.config import get_settings
        
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, 'environment')
        assert hasattr(settings, 'database_url')
        assert hasattr(settings, 'redis_url')
        assert hasattr(settings, 'kafka_bootstrap_servers')
    
    def test_environment_detection(self):
        """Test environment detection works"""
        from app.core.config import get_settings_for_environment
        
        # Test different environments
        dev_settings = get_settings_for_environment('development')
        test_settings = get_settings_for_environment('testing')
        prod_settings = get_settings_for_environment('production')
        
        assert dev_settings.debug == True
        assert test_settings.test_mode == True
        assert prod_settings.debug == False


class TestBasicModels:
    """Test basic model imports work"""
    
    def test_database_models_import(self):
        """Test original database models can be imported"""
        try:
            from app.models.database import User, Signal, UserSignalMatch
            assert User is not None
            assert Signal is not None
            assert UserSignalMatch is not None
        except ImportError as e:
            pytest.fail(f"Failed to import database models: {e}")
    
    def test_database_base_import(self):
        """Test database base can be imported"""
        try:
            from app.core.database import Base
            assert Base is not None
        except ImportError as e:
            pytest.fail(f"Failed to import database base: {e}")


class TestBasicEnums:
    """Test basic enum definitions work"""
    
    def test_basic_enums_exist(self):
        """Test we can access basic enums without complex dependencies"""
        # Test that the files exist and basic structure is there
        enum_files = [
            project_root / "app" / "ml" / "models.py",
            project_root / "app" / "portfolio" / "optimization.py",
            project_root / "app" / "backtesting" / "engine.py",
            project_root / "app" / "risk" / "management.py"
        ]
        
        for enum_file in enum_files:
            assert enum_file.exists(), f"Enum file {enum_file} should exist"
    
    def test_risk_enums_work(self):
        """Test risk enums work (no external dependencies)"""
        try:
            from app.risk.management import VaRMethod
            
            assert hasattr(VaRMethod, 'HISTORICAL')
            assert hasattr(VaRMethod, 'PARAMETRIC') 
            assert hasattr(VaRMethod, 'MONTE_CARLO')
            
            assert VaRMethod.HISTORICAL == "historical"
            assert VaRMethod.PARAMETRIC == "parametric"
            assert VaRMethod.MONTE_CARLO == "monte_carlo"
            
        except ImportError as e:
            pytest.fail(f"Risk enums failed: {e}")


class TestMainApplication:
    """Test main application can be loaded"""
    
    def test_main_app_exists(self):
        """Test main.py exists and has basic structure"""
        main_file = project_root / "app" / "main.py"
        assert main_file.exists(), "main.py should exist"
        
        # Read and check it has basic FastAPI structure
        content = main_file.read_text()
        assert "FastAPI" in content, "main.py should contain FastAPI"
        assert "create_app" in content, "main.py should have create_app function"
    
    def test_basic_app_imports(self):
        """Test we can import basic app components"""
        try:
            from app.main import create_app
            assert create_app is not None
        except (ImportError, TypeError) as e:
            # This might fail due to dependencies, which is expected
            pytest.skip(f"App import failed due to dependencies: {e}")


if __name__ == "__main__":
    # Run just these basic tests
    pytest.main([__file__, "-v", "-x"])