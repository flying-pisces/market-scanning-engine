"""
Signal Generation Framework
Author: Claude Code (System Architect)
Version: 1.0

High-performance signal generation framework for multi-asset class trading
with comprehensive risk scoring and quality control.
"""

__version__ = "1.0.0"
__author__ = "Claude Code (System Architect)"

# Core framework components
from .core.base_generator import BaseSignalGenerator
from .core.signal_factory import SignalGeneratorFactory
from .core.orchestrator import SignalOrchestrator, OrchestrationConfig, OrchestrationMode
from .core.validator import SignalValidator
from .core.risk_scorer import RiskScorer

# Strategy generators - Phase 1 implementations
from .strategies.technical import (
    MovingAverageSignalGenerator,
    RSISignalGenerator,
    MACDSignalGenerator,
    BollingerBandsSignalGenerator
)

from .strategies.options import (
    OptionsFlowSignalGenerator,
    GammaExposureSignalGenerator,
    PutCallRatioSignalGenerator,
    UnusualVolumeSignalGenerator
)

# Configuration management
from .config.strategy_config import ConfigurationManager

# Export main classes for easy access
__all__ = [
    # Core framework
    "BaseSignalGenerator",
    "SignalGeneratorFactory", 
    "SignalOrchestrator",
    "OrchestrationConfig",
    "OrchestrationMode",
    "SignalValidator",
    "RiskScorer",
    "ConfigurationManager",
    
    # Technical strategy generators
    "MovingAverageSignalGenerator",
    "RSISignalGenerator", 
    "MACDSignalGenerator",
    "BollingerBandsSignalGenerator",
    
    # Options strategy generators
    "OptionsFlowSignalGenerator",
    "GammaExposureSignalGenerator",
    "PutCallRatioSignalGenerator",
    "UnusualVolumeSignalGenerator",
]