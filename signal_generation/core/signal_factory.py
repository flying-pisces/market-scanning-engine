"""
Signal Generator Factory - Registry and Factory for All Signal Generators
Author: Claude Code (System Architect)
Version: 1.0

Manages registration, instantiation, and discovery of signal generators.
Provides a centralized registry for all available signal generator types.
"""

import logging
from typing import Dict, Type, List, Optional, Any
from enum import Enum

from .base_generator import BaseSignalGenerator, GeneratorConfig
from data_models.python.signal_models import SignalCategory


class AssetClassRiskProfiles(Enum):
    """Predefined risk profiles for different asset classes"""
    DAILY_OPTIONS = {"base_risk": 80, "min_risk": 70, "max_risk": 95}
    STOCKS = {"base_risk": 55, "min_risk": 30, "max_risk": 80}
    ETFS = {"base_risk": 45, "min_risk": 20, "max_risk": 70}
    BONDS = {"base_risk": 25, "min_risk": 5, "max_risk": 40}
    SAFE_ASSETS = {"base_risk": 10, "min_risk": 0, "max_risk": 20}


class SignalGeneratorFactory:
    """
    Factory for creating and managing signal generators.
    
    Provides:
    - Registration of generator classes
    - Creation of generator instances with proper configuration
    - Asset class specific risk profile assignment
    - Generator discovery and listing
    """
    
    _generators: Dict[str, Type[BaseSignalGenerator]] = {}
    _instances: Dict[str, BaseSignalGenerator] = {}
    _logger = logging.getLogger("signal_factory")
    
    # Predefined configurations for different asset classes
    ASSET_CLASS_CONFIGS = {
        "daily_options": {
            "spy": {"symbols": ["SPY"], "base_risk": 85, "max_signals_per_hour": 50},
            "qqq": {"symbols": ["QQQ"], "base_risk": 85, "max_signals_per_hour": 50}, 
            "spx": {"symbols": ["SPX"], "base_risk": 90, "max_signals_per_hour": 30},
            "xsp": {"symbols": ["XSP"], "base_risk": 88, "max_signals_per_hour": 30},
            "ndx": {"symbols": ["NDX"], "base_risk": 90, "max_signals_per_hour": 30}
        },
        "stocks": {
            "large_cap": {"base_risk": 45, "min_market_cap_cents": 1000000000000},  # $10B+
            "mid_cap": {"base_risk": 60, "min_market_cap_cents": 200000000000},     # $2B-$10B
            "small_cap": {"base_risk": 75, "min_market_cap_cents": 0},             # <$2B
            "growth": {"base_risk": 65},
            "value": {"base_risk": 40},
            "dividend": {"base_risk": 35}
        },
        "etfs": {
            "sector": {"base_risk": 50},
            "regional": {"base_risk": 55}, 
            "thematic": {"base_risk": 65},
            "commodity": {"base_risk": 70},
            "broad_market": {"base_risk": 35}
        },
        "bonds": {
            "treasury": {"base_risk": 15},
            "corporate": {"base_risk": 30},
            "municipal": {"base_risk": 25},
            "high_yield": {"base_risk": 40}
        },
        "safe_assets": {
            "tbills": {"base_risk": 5},
            "cds": {"base_risk": 8},
            "stable_value": {"base_risk": 12}
        }
    }
    
    @classmethod
    def register(cls, generator_class: Type[BaseSignalGenerator], name: Optional[str] = None) -> None:
        """Register a signal generator class"""
        class_name = name or generator_class.__name__
        
        if class_name in cls._generators:
            cls._logger.warning(f"Overriding existing generator: {class_name}")
        
        cls._generators[class_name] = generator_class
        cls._logger.info(f"Registered signal generator: {class_name}")
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a signal generator"""
        if name in cls._generators:
            del cls._generators[name]
            cls._logger.info(f"Unregistered signal generator: {name}")
        
        if name in cls._instances:
            del cls._instances[name]
    
    @classmethod
    def create_generator(cls, 
                        generator_type: str,
                        config: GeneratorConfig,
                        instance_name: Optional[str] = None) -> BaseSignalGenerator:
        """Create a signal generator instance"""
        if generator_type not in cls._generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        generator_class = cls._generators[generator_type]
        instance = generator_class(config)
        
        # Store instance for reuse if named
        if instance_name:
            cls._instances[instance_name] = instance
        
        cls._logger.info(f"Created generator instance: {generator_type} -> {config.generator_name}")
        return instance
    
    @classmethod
    def create_default_config(cls,
                             generator_name: str,
                             category: SignalCategory,
                             asset_class: str = "stocks",
                             asset_subclass: str = "large_cap",
                             **overrides) -> GeneratorConfig:
        """Create default configuration for a generator based on asset class"""
        
        # Get base configuration for asset class
        base_config = cls.ASSET_CLASS_CONFIGS.get(asset_class, {}).get(asset_subclass, {})
        
        # Default configuration
        config_dict = {
            "generator_name": generator_name,
            "category": category,
            "enabled": True,
            "base_risk_score": base_config.get("base_risk", 50),
            "min_confidence_threshold": 0.6,
            "max_signals_per_hour": base_config.get("max_signals_per_hour", 100),
            "signal_ttl_hours": 24,
            "lookback_periods": 20,
            "supported_asset_classes": base_config.get("symbols", []),
            "min_market_cap_cents": base_config.get("min_market_cap_cents"),
            "min_avg_volume": base_config.get("min_avg_volume"),
            "strategy_params": {}
        }
        
        # Apply overrides
        config_dict.update(overrides)
        
        return GeneratorConfig(**config_dict)
    
    @classmethod
    def create_technical_generator(cls, 
                                  generator_type: str,
                                  strategy_name: str,
                                  asset_class: str = "stocks",
                                  **strategy_params) -> BaseSignalGenerator:
        """Create a technical analysis signal generator with defaults"""
        config = cls.create_default_config(
            generator_name=f"technical_{strategy_name}",
            category=SignalCategory.TECHNICAL,
            asset_class=asset_class,
            strategy_params=strategy_params
        )
        return cls.create_generator(generator_type, config)
    
    @classmethod
    def create_options_generator(cls,
                                generator_type: str,
                                strategy_name: str,
                                **strategy_params) -> BaseSignalGenerator:
        """Create an options analysis signal generator with defaults"""
        config = cls.create_default_config(
            generator_name=f"options_{strategy_name}",
            category=SignalCategory.OPTIONS,
            asset_class="daily_options",
            asset_subclass="spy",
            strategy_params=strategy_params
        )
        return cls.create_generator(generator_type, config)
    
    @classmethod
    def create_fundamental_generator(cls,
                                   generator_type: str, 
                                   strategy_name: str,
                                   asset_class: str = "stocks",
                                   **strategy_params) -> BaseSignalGenerator:
        """Create a fundamental analysis signal generator with defaults"""
        config = cls.create_default_config(
            generator_name=f"fundamental_{strategy_name}",
            category=SignalCategory.FUNDAMENTAL,
            asset_class=asset_class,
            strategy_params=strategy_params
        )
        return cls.create_generator(generator_type, config)
    
    @classmethod
    def create_macro_generator(cls,
                              generator_type: str,
                              strategy_name: str,
                              **strategy_params) -> BaseSignalGenerator:
        """Create a macro analysis signal generator with defaults"""
        config = cls.create_default_config(
            generator_name=f"macro_{strategy_name}",
            category=SignalCategory.MACRO,
            asset_class="stocks",  # Macro affects all assets
            strategy_params=strategy_params
        )
        return cls.create_generator(generator_type, config)
    
    @classmethod
    def get_instance(cls, instance_name: str) -> Optional[BaseSignalGenerator]:
        """Get a stored generator instance by name"""
        return cls._instances.get(instance_name)
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """List all registered generator types"""
        return list(cls._generators.keys())
    
    @classmethod
    def list_instances(cls) -> List[str]:
        """List all created instances"""
        return list(cls._instances.keys())
    
    @classmethod
    def get_generator_info(cls, generator_type: str) -> Dict[str, Any]:
        """Get information about a registered generator type"""
        if generator_type not in cls._generators:
            return {}
        
        generator_class = cls._generators[generator_type]
        return {
            "name": generator_type,
            "class": generator_class.__name__,
            "module": generator_class.__module__,
            "docstring": generator_class.__doc__
        }
    
    @classmethod
    def create_asset_class_suite(cls, asset_class: str) -> List[BaseSignalGenerator]:
        """
        Create a complete suite of generators for a specific asset class.
        
        Args:
            asset_class: One of 'daily_options', 'stocks', 'etfs', 'bonds', 'safe_assets'
            
        Returns:
            List of configured generators suitable for the asset class
        """
        generators = []
        
        if asset_class == "daily_options":
            # High-frequency options-specific generators
            generators.extend([
                cls.create_options_generator("OptionsFlowSignalGenerator", "flow"),
                cls.create_options_generator("GammaExposureSignalGenerator", "gamma"),
                cls.create_options_generator("PutCallRatioSignalGenerator", "put_call"),
                cls.create_options_generator("UnusualVolumeSignalGenerator", "unusual_volume"),
                cls.create_technical_generator("MovingAverageSignalGenerator", "ma_crossover", 
                                             asset_class="daily_options"),
                cls.create_technical_generator("RSISignalGenerator", "rsi_momentum", 
                                             asset_class="daily_options")
            ])
            
        elif asset_class == "stocks":
            # Comprehensive stock analysis generators  
            generators.extend([
                cls.create_technical_generator("MovingAverageSignalGenerator", "trend_following"),
                cls.create_technical_generator("RSISignalGenerator", "momentum"),
                cls.create_technical_generator("MACDSignalGenerator", "macd"),
                cls.create_technical_generator("BollingerBandsSignalGenerator", "mean_reversion"),
                cls.create_fundamental_generator("EarningsSignalGenerator", "earnings"),
                cls.create_fundamental_generator("ValuationSignalGenerator", "valuation"),
                cls.create_fundamental_generator("SectorRotationSignalGenerator", "sector_rotation"),
            ])
            
        elif asset_class == "etfs":
            # ETF-specific generators focusing on sector/theme analysis
            generators.extend([
                cls.create_technical_generator("MovingAverageSignalGenerator", "trend", 
                                             asset_class="etfs"),
                cls.create_technical_generator("RSISignalGenerator", "momentum", 
                                             asset_class="etfs"),
                cls.create_fundamental_generator("SectorRotationSignalGenerator", "rotation", 
                                                asset_class="etfs"),
                cls.create_macro_generator("FedPolicySignalGenerator", "fed_policy")
            ])
            
        elif asset_class == "bonds":
            # Fixed income focused generators
            generators.extend([
                cls.create_macro_generator("FedPolicySignalGenerator", "fed_policy"),
                cls.create_macro_generator("InflationSignalGenerator", "inflation"), 
                cls.create_macro_generator("YieldCurveSignalGenerator", "yield_curve"),
                cls.create_technical_generator("MovingAverageSignalGenerator", "bond_trend",
                                             asset_class="bonds")
            ])
            
        elif asset_class == "safe_assets":
            # Conservative, macro-driven generators
            generators.extend([
                cls.create_macro_generator("FedPolicySignalGenerator", "safe_haven"),
                cls.create_macro_generator("InflationSignalGenerator", "real_rates")
            ])
        
        cls._logger.info(f"Created {len(generators)} generators for asset class: {asset_class}")
        return generators
    
    @classmethod
    def validate_all_generators(cls) -> Dict[str, List[str]]:
        """Validate all registered generators"""
        validation_results = {}
        
        for name, generator_class in cls._generators.items():
            try:
                # Create a minimal config for validation
                config = GeneratorConfig(
                    generator_name=f"test_{name}",
                    category=SignalCategory.TECHNICAL,
                    base_risk_score=50
                )
                
                instance = generator_class(config)
                is_valid, errors = instance.validate_configuration()
                validation_results[name] = errors if not is_valid else []
                
            except Exception as e:
                validation_results[name] = [f"Failed to instantiate: {str(e)}"]
        
        return validation_results