"""
Strategy Configuration Management System
Author: Claude Code (System Architect)
Version: 1.0

Comprehensive configuration management for signal generation strategies including:
- Asset class specific configurations
- Strategy parameter management
- Environment-based configuration
- Performance optimization settings
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from signal_generation.core.base_generator import GeneratorConfig
from signal_generation.core.orchestrator import OrchestrationConfig, OrchestrationMode
from data_models.python.signal_models import SignalCategory


class ConfigurationSource(str, Enum):
    """Configuration sources"""
    FILE = "file"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    API = "api"
    DEFAULT = "default"


@dataclass
class AssetClassConfig:
    """Configuration for a specific asset class"""
    name: str
    base_risk_score: int
    min_confidence_threshold: float = 0.6
    max_signals_per_hour: int = 100
    signal_ttl_hours: int = 24
    supported_strategies: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    risk_adjustments: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class StrategyConfig:
    """Configuration for a specific strategy"""
    name: str
    category: SignalCategory
    enabled: bool = True
    asset_classes: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, int] = field(default_factory=dict)
    
    def to_generator_config(self, asset_class_config: AssetClassConfig) -> GeneratorConfig:
        """Convert to GeneratorConfig for signal generator"""
        return GeneratorConfig(
            generator_name=self.name,
            category=self.category,
            enabled=self.enabled,
            base_risk_score=asset_class_config.base_risk_score,
            min_confidence_threshold=asset_class_config.min_confidence_threshold,
            max_signals_per_hour=asset_class_config.max_signals_per_hour,
            signal_ttl_hours=asset_class_config.signal_ttl_hours,
            supported_asset_classes=asset_class_config.supported_strategies,
            strategy_params=self.parameters
        )


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration"""
    name: str  # development, staging, production
    max_concurrent_generators: int = 10
    execution_interval_seconds: int = 60
    enable_validation: bool = True
    enable_risk_assessment: bool = True
    enable_notifications: bool = False
    log_level: str = "INFO"
    performance_monitoring: bool = True
    
    def to_orchestration_config(self) -> OrchestrationConfig:
        """Convert to OrchestrationConfig"""
        return OrchestrationConfig(
            mode=OrchestrationMode.REAL_TIME,
            execution_interval_seconds=self.execution_interval_seconds,
            max_concurrent_generators=self.max_concurrent_generators,
            enable_validation=self.enable_validation,
            enable_risk_assessment=self.enable_risk_assessment,
            enable_notifications=self.enable_notifications
        )


class ConfigurationManager:
    """
    Centralized configuration management system
    
    Manages configurations for:
    - Asset classes
    - Trading strategies
    - Environment settings
    - Performance parameters
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent
        self.logger = logging.getLogger("config_manager")
        
        # Configuration storage
        self.asset_class_configs: Dict[str, AssetClassConfig] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        self.environment_config: Optional[EnvironmentConfig] = None
        
        # Configuration sources tracking
        self.config_sources: Dict[str, ConfigurationSource] = {}
        self.last_loaded: Dict[str, datetime] = {}
        
        # Load default configurations
        self._load_default_configs()
    
    def _load_default_configs(self) -> None:
        """Load default configurations for all asset classes and strategies"""
        
        # Default asset class configurations
        self._create_default_asset_class_configs()
        
        # Default strategy configurations  
        self._create_default_strategy_configs()
        
        # Default environment configuration
        self.environment_config = EnvironmentConfig(name="default")
    
    def _create_default_asset_class_configs(self) -> None:
        """Create default asset class configurations"""
        
        # Daily Options (High Risk/High Reward)
        self.asset_class_configs["daily_options"] = AssetClassConfig(
            name="daily_options",
            base_risk_score=85,
            min_confidence_threshold=0.7,
            max_signals_per_hour=50,
            signal_ttl_hours=8,
            supported_strategies=[
                "options_flow", "gamma_exposure", "put_call_ratio", 
                "unusual_volume", "technical_momentum"
            ],
            strategy_weights={
                "options_flow": 0.3,
                "gamma_exposure": 0.25,
                "put_call_ratio": 0.2,
                "unusual_volume": 0.15,
                "technical_momentum": 0.1
            },
            risk_adjustments={
                "high_volume": 10,
                "market_hours": -5,
                "low_liquidity": 15
            }
        )
        
        # Large Cap Stocks (Moderate Risk)
        self.asset_class_configs["large_cap_stocks"] = AssetClassConfig(
            name="large_cap_stocks",
            base_risk_score=45,
            min_confidence_threshold=0.6,
            max_signals_per_hour=100,
            signal_ttl_hours=24,
            supported_strategies=[
                "moving_average", "rsi_momentum", "macd", "bollinger_bands",
                "earnings", "valuation", "sector_rotation"
            ],
            strategy_weights={
                "moving_average": 0.2,
                "rsi_momentum": 0.15,
                "macd": 0.15,
                "bollinger_bands": 0.15,
                "earnings": 0.15,
                "valuation": 0.1,
                "sector_rotation": 0.1
            }
        )
        
        # Small Cap Stocks (Higher Risk)
        self.asset_class_configs["small_cap_stocks"] = AssetClassConfig(
            name="small_cap_stocks",
            base_risk_score=65,
            min_confidence_threshold=0.65,
            max_signals_per_hour=75,
            signal_ttl_hours=24,
            supported_strategies=[
                "moving_average", "rsi_momentum", "bollinger_bands",
                "earnings", "unusual_volume"
            ],
            strategy_weights={
                "moving_average": 0.25,
                "rsi_momentum": 0.2,
                "bollinger_bands": 0.2,
                "earnings": 0.2,
                "unusual_volume": 0.15
            },
            risk_adjustments={
                "low_liquidity": 20,
                "high_volatility": 15,
                "earnings_season": 10
            }
        )
        
        # ETFs (Lower Risk)
        self.asset_class_configs["etfs"] = AssetClassConfig(
            name="etfs",
            base_risk_score=35,
            min_confidence_threshold=0.55,
            max_signals_per_hour=80,
            signal_ttl_hours=48,
            supported_strategies=[
                "moving_average", "rsi_momentum", "sector_rotation",
                "macro_fed_policy", "macro_inflation"
            ],
            strategy_weights={
                "moving_average": 0.3,
                "rsi_momentum": 0.25,
                "sector_rotation": 0.2,
                "macro_fed_policy": 0.15,
                "macro_inflation": 0.1
            }
        )
        
        # Bonds (Low Risk)
        self.asset_class_configs["bonds"] = AssetClassConfig(
            name="bonds",
            base_risk_score=20,
            min_confidence_threshold=0.5,
            max_signals_per_hour=30,
            signal_ttl_hours=72,
            supported_strategies=[
                "macro_fed_policy", "macro_inflation", "yield_curve", "moving_average"
            ],
            strategy_weights={
                "macro_fed_policy": 0.4,
                "macro_inflation": 0.3,
                "yield_curve": 0.2,
                "moving_average": 0.1
            }
        )
    
    def _create_default_strategy_configs(self) -> None:
        """Create default strategy configurations"""
        
        # Technical Analysis Strategies
        self.strategy_configs["moving_average"] = StrategyConfig(
            name="moving_average",
            category=SignalCategory.TECHNICAL,
            asset_classes=["large_cap_stocks", "small_cap_stocks", "etfs", "bonds"],
            parameters={
                "short_ma_period": 20,
                "long_ma_period": 50,
                "ma_type": "sma",
                "trend_confirmation_period": 3,
                "volume_confirmation": True
            },
            performance_thresholds={
                "min_win_rate": 0.55,
                "min_sharpe_ratio": 1.0,
                "max_drawdown": 0.15
            },
            risk_limits={
                "max_position_size": 10,
                "max_correlation": 70
            }
        )
        
        self.strategy_configs["rsi_momentum"] = StrategyConfig(
            name="rsi_momentum",
            category=SignalCategory.TECHNICAL,
            asset_classes=["large_cap_stocks", "small_cap_stocks", "etfs", "daily_options"],
            parameters={
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
                "extreme_oversold": 20,
                "extreme_overbought": 80,
                "divergence_detection": False
            },
            performance_thresholds={
                "min_win_rate": 0.60,
                "min_sharpe_ratio": 1.2
            }
        )
        
        self.strategy_configs["bollinger_bands"] = StrategyConfig(
            name="bollinger_bands",
            category=SignalCategory.TECHNICAL,
            asset_classes=["large_cap_stocks", "small_cap_stocks"],
            parameters={
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "squeeze_threshold": 0.1,
                "band_touch_sensitivity": 0.02
            },
            performance_thresholds={
                "min_win_rate": 0.58,
                "max_drawdown": 0.12
            }
        )
        
        # Options Strategies
        self.strategy_configs["options_flow"] = StrategyConfig(
            name="options_flow",
            category=SignalCategory.OPTIONS,
            asset_classes=["daily_options"],
            parameters={
                "volume_threshold_multiplier": 3.0,
                "min_trade_size": 50,
                "premium_threshold": 100000,
                "time_decay_days": 30,
                "moneyness_range": 0.2,
                "sweep_detection": True,
                "block_trade_threshold": 100
            },
            performance_thresholds={
                "min_win_rate": 0.65,
                "min_sharpe_ratio": 1.5
            },
            risk_limits={
                "max_position_size": 5,
                "max_time_decay": 30
            }
        )
        
        self.strategy_configs["gamma_exposure"] = StrategyConfig(
            name="gamma_exposure",
            category=SignalCategory.OPTIONS,
            asset_classes=["daily_options"],
            parameters={
                "gamma_threshold": 1000000,
                "strike_spacing": 500,
                "max_dte": 45,
                "min_open_interest": 100
            },
            performance_thresholds={
                "min_win_rate": 0.70,
                "min_sharpe_ratio": 1.8
            }
        )
        
        # Fundamental Strategies
        self.strategy_configs["earnings"] = StrategyConfig(
            name="earnings",
            category=SignalCategory.FUNDAMENTAL,
            asset_classes=["large_cap_stocks", "small_cap_stocks"],
            parameters={
                "earnings_surprise_threshold": 0.05,
                "revenue_surprise_threshold": 0.03,
                "guidance_weight": 0.3,
                "analyst_revision_weight": 0.4,
                "historical_reaction_weight": 0.3
            },
            performance_thresholds={
                "min_win_rate": 0.62,
                "min_sharpe_ratio": 1.3
            }
        )
        
        # Macro Strategies
        self.strategy_configs["macro_fed_policy"] = StrategyConfig(
            name="macro_fed_policy",
            category=SignalCategory.MACRO,
            asset_classes=["etfs", "bonds"],
            parameters={
                "fomc_meeting_weight": 0.5,
                "fed_speak_weight": 0.3,
                "economic_data_weight": 0.2,
                "rate_change_sensitivity": 0.25,
                "forward_guidance_weight": 0.4
            },
            performance_thresholds={
                "min_win_rate": 0.58,
                "min_sharpe_ratio": 1.1
            }
        )
    
    def load_config_file(self, file_path: str) -> bool:
        """Load configuration from file (JSON or YAML)"""
        try:
            path = Path(file_path)
            
            if not path.exists():
                self.logger.error(f"Configuration file not found: {file_path}")
                return False
            
            # Determine file format
            if path.suffix.lower() in ['.json']:
                with open(path, 'r') as f:
                    config_data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                self.logger.error(f"Unsupported configuration file format: {path.suffix}")
                return False
            
            # Parse configuration sections
            if 'asset_classes' in config_data:
                for name, data in config_data['asset_classes'].items():
                    self.asset_class_configs[name] = AssetClassConfig(**data)
                    self.config_sources[f"asset_class.{name}"] = ConfigurationSource.FILE
            
            if 'strategies' in config_data:
                for name, data in config_data['strategies'].items():
                    # Convert category string to enum
                    if 'category' in data and isinstance(data['category'], str):
                        data['category'] = SignalCategory(data['category'])
                    
                    self.strategy_configs[name] = StrategyConfig(**data)
                    self.config_sources[f"strategy.{name}"] = ConfigurationSource.FILE
            
            if 'environment' in config_data:
                self.environment_config = EnvironmentConfig(**config_data['environment'])
                self.config_sources["environment"] = ConfigurationSource.FILE
            
            self.last_loaded[file_path] = datetime.utcnow()
            self.logger.info(f"Configuration loaded successfully from {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {file_path}: {str(e)}")
            return False
    
    def save_config_file(self, file_path: str, format: str = "json") -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                "asset_classes": {
                    name: config.to_dict() 
                    for name, config in self.asset_class_configs.items()
                },
                "strategies": {
                    name: {
                        **asdict(config),
                        "category": config.category.value
                    }
                    for name, config in self.strategy_configs.items()
                },
                "environment": asdict(self.environment_config) if self.environment_config else {}
            }
            
            path = Path(file_path)
            
            if format.lower() == "json":
                with open(path, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            elif format.lower() in ["yaml", "yml"]:
                with open(path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False
            
            self.logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {file_path}: {str(e)}")
            return False
    
    def get_asset_class_config(self, asset_class: str) -> Optional[AssetClassConfig]:
        """Get configuration for specific asset class"""
        return self.asset_class_configs.get(asset_class)
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for specific strategy"""
        return self.strategy_configs.get(strategy_name)
    
    def get_strategies_for_asset_class(self, asset_class: str) -> List[StrategyConfig]:
        """Get all strategies configured for a specific asset class"""
        return [
            config for config in self.strategy_configs.values()
            if asset_class in config.asset_classes
        ]
    
    def create_generator_configs(self, asset_class: str) -> List[GeneratorConfig]:
        """Create generator configurations for an asset class"""
        asset_config = self.get_asset_class_config(asset_class)
        if not asset_config:
            self.logger.error(f"Asset class configuration not found: {asset_class}")
            return []
        
        strategies = self.get_strategies_for_asset_class(asset_class)
        generator_configs = []
        
        for strategy in strategies:
            if strategy.enabled:
                generator_config = strategy.to_generator_config(asset_config)
                generator_configs.append(generator_config)
        
        return generator_configs
    
    def update_strategy_performance(self, 
                                  strategy_name: str,
                                  performance_metrics: Dict[str, float]) -> bool:
        """Update strategy performance metrics"""
        strategy = self.get_strategy_config(strategy_name)
        if not strategy:
            return False
        
        # Update performance thresholds based on actual performance
        for metric, value in performance_metrics.items():
            if metric in strategy.performance_thresholds:
                # Simple adaptive adjustment (could be more sophisticated)
                current_threshold = strategy.performance_thresholds[metric]
                new_threshold = current_threshold * 0.9 + value * 0.1
                strategy.performance_thresholds[metric] = new_threshold
        
        self.logger.info(f"Updated performance metrics for strategy: {strategy_name}")
        return True
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate all configurations for consistency and completeness"""
        validation_errors = {}
        
        # Validate asset class configurations
        for name, config in self.asset_class_configs.items():
            errors = []
            
            if not 0 <= config.base_risk_score <= 100:
                errors.append(f"Invalid base_risk_score: {config.base_risk_score}")
            
            if not 0 <= config.min_confidence_threshold <= 1:
                errors.append(f"Invalid min_confidence_threshold: {config.min_confidence_threshold}")
            
            if config.max_signals_per_hour <= 0:
                errors.append(f"Invalid max_signals_per_hour: {config.max_signals_per_hour}")
            
            if errors:
                validation_errors[f"asset_class.{name}"] = errors
        
        # Validate strategy configurations
        for name, config in self.strategy_configs.items():
            errors = []
            
            if not config.asset_classes:
                errors.append("No asset classes specified")
            
            # Check if referenced asset classes exist
            for asset_class in config.asset_classes:
                if asset_class not in self.asset_class_configs:
                    errors.append(f"Unknown asset class: {asset_class}")
            
            if errors:
                validation_errors[f"strategy.{name}"] = errors
        
        # Validate environment configuration
        if self.environment_config:
            errors = []
            
            if self.environment_config.max_concurrent_generators <= 0:
                errors.append("Invalid max_concurrent_generators")
            
            if self.environment_config.execution_interval_seconds <= 0:
                errors.append("Invalid execution_interval_seconds")
            
            if errors:
                validation_errors["environment"] = errors
        
        return validation_errors
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        return {
            "asset_classes": {
                "count": len(self.asset_class_configs),
                "names": list(self.asset_class_configs.keys())
            },
            "strategies": {
                "count": len(self.strategy_configs),
                "enabled": len([s for s in self.strategy_configs.values() if s.enabled]),
                "by_category": {
                    category.value: len([
                        s for s in self.strategy_configs.values() 
                        if s.category == category
                    ])
                    for category in SignalCategory
                }
            },
            "environment": {
                "name": self.environment_config.name if self.environment_config else "none",
                "validation_enabled": self.environment_config.enable_validation if self.environment_config else False,
                "risk_assessment_enabled": self.environment_config.enable_risk_assessment if self.environment_config else False
            },
            "configuration_sources": self.config_sources,
            "last_loaded": {
                path: timestamp.isoformat() 
                for path, timestamp in self.last_loaded.items()
            }
        }