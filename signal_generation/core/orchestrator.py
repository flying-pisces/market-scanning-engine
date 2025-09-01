"""
Signal Orchestration System  
Author: Claude Code (System Architect)
Version: 1.0

High-performance signal orchestration system that coordinates:
- Multiple signal generators execution
- Real-time data feed management
- Signal validation and filtering
- Risk assessment and calibration
- Output distribution and notification
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Callable
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json

from .base_generator import BaseSignalGenerator, SignalGenerationResult
from .signal_factory import SignalGeneratorFactory
from .validator import SignalValidator, ValidationResult
from .risk_scorer import RiskScorer, RiskAssessment
from data_models.python.core_models import Asset, MarketData, TechnicalIndicators, OptionsData
from data_models.python.signal_models import Signal, SignalStatus


class OrchestrationMode(str, Enum):
    """Orchestration execution modes"""
    REAL_TIME = "real_time"        # Continuous real-time processing
    SCHEDULED = "scheduled"        # Run at specific intervals
    ON_DEMAND = "on_demand"       # Manual triggering
    EVENT_DRIVEN = "event_driven" # Triggered by market events


class ExecutionStatus(str, Enum):
    """Execution status for orchestration runs"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class OrchestrationConfig:
    """Configuration for signal orchestration"""
    mode: OrchestrationMode = OrchestrationMode.REAL_TIME
    execution_interval_seconds: int = 60
    max_concurrent_generators: int = 10
    max_signals_per_run: int = 1000
    enable_validation: bool = True
    enable_risk_assessment: bool = True
    enable_deduplication: bool = True
    
    # Quality filters
    min_confidence_score: int = 50
    max_risk_score: int = 95
    min_quality_score: int = 60
    
    # Performance settings
    generator_timeout_seconds: int = 30
    batch_size: int = 100
    max_retry_attempts: int = 3
    
    # Notification settings
    enable_notifications: bool = False
    notification_channels: List[str] = field(default_factory=list)
    high_priority_threshold: int = 80


@dataclass
class ExecutionMetrics:
    """Metrics for a single orchestration run"""
    run_id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    generators_executed: int = 0
    assets_processed: int = 0
    signals_generated: int = 0
    signals_validated: int = 0
    signals_filtered: int = 0
    signals_output: int = 0
    errors_count: int = 0
    execution_time_ms: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        return self.end_time is not None
    
    @property
    def signals_per_second(self) -> float:
        if self.execution_time_ms > 0:
            return (self.signals_generated / self.execution_time_ms) * 1000
        return 0.0


class SignalOrchestrator:
    """
    High-performance signal orchestration system
    
    Coordinates the entire signal generation pipeline:
    1. Data gathering and preparation
    2. Signal generator execution (parallel)
    3. Signal validation and quality control
    4. Risk assessment and calibration
    5. Deduplication and filtering
    6. Output distribution
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = logging.getLogger("signal_orchestrator")
        
        # Core components
        self.validator = SignalValidator() if config.enable_validation else None
        self.risk_scorer = RiskScorer() if config.enable_risk_assessment else None
        
        # State management
        self.status = ExecutionStatus.IDLE
        self.generators: List[BaseSignalGenerator] = []
        self.data_providers: Dict[str, Callable] = {}
        self.output_handlers: List[Callable] = []
        self.notification_handlers: List[Callable] = []
        
        # Performance tracking
        self.execution_history: deque = deque(maxlen=100)
        self.current_run_metrics: Optional[ExecutionMetrics] = None
        
        # Signal deduplication
        self.signal_cache: Dict[str, datetime] = {}  # signal_hash -> timestamp
        self.cache_ttl_seconds = 3600  # 1 hour
        
        # Error handling
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.generator_failures: Dict[str, int] = defaultdict(int)
        
        # Concurrency control
        self.execution_lock = asyncio.Lock()
        self.generator_semaphore = asyncio.Semaphore(config.max_concurrent_generators)
        
        # Scheduling
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
    
    def register_generator(self, generator: BaseSignalGenerator) -> None:
        """Register a signal generator with the orchestrator"""
        if generator not in self.generators:
            self.generators.append(generator)
            self.logger.info(f"Registered generator: {generator.name}")
    
    def unregister_generator(self, generator_name: str) -> bool:
        """Unregister a signal generator"""
        for i, gen in enumerate(self.generators):
            if gen.name == generator_name:
                self.generators.pop(i)
                self.logger.info(f"Unregistered generator: {generator_name}")
                return True
        return False
    
    def register_data_provider(self, name: str, provider: Callable) -> None:
        """Register a data provider function"""
        self.data_providers[name] = provider
        self.logger.info(f"Registered data provider: {name}")
    
    def register_output_handler(self, handler: Callable) -> None:
        """Register an output handler for processed signals"""
        self.output_handlers.append(handler)
        self.logger.info("Registered output handler")
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification handler"""
        self.notification_handlers.append(handler)
        self.logger.info("Registered notification handler")
    
    async def start_real_time_processing(self) -> None:
        """Start real-time signal processing"""
        if self.status != ExecutionStatus.IDLE:
            raise RuntimeError(f"Cannot start processing while status is {self.status}")
        
        self.status = ExecutionStatus.RUNNING
        self._stop_event.clear()
        
        if self.config.mode == OrchestrationMode.REAL_TIME:
            self._scheduler_task = asyncio.create_task(self._real_time_loop())
        elif self.config.mode == OrchestrationMode.SCHEDULED:
            self._scheduler_task = asyncio.create_task(self._scheduled_loop())
        
        self.logger.info(f"Started signal orchestration in {self.config.mode} mode")
    
    async def stop_processing(self) -> None:
        """Stop signal processing gracefully"""
        self.status = ExecutionStatus.STOPPING
        self._stop_event.set()
        
        if self._scheduler_task:
            try:
                await asyncio.wait_for(self._scheduler_task, timeout=10.0)
            except asyncio.TimeoutError:
                self._scheduler_task.cancel()
                self.logger.warning("Scheduler task cancelled due to timeout")
        
        self.status = ExecutionStatus.IDLE
        self.logger.info("Signal orchestration stopped")
    
    async def pause_processing(self) -> None:
        """Pause signal processing"""
        self.status = ExecutionStatus.PAUSED
        self.logger.info("Signal orchestration paused")
    
    async def resume_processing(self) -> None:
        """Resume signal processing"""
        if self.status == ExecutionStatus.PAUSED:
            self.status = ExecutionStatus.RUNNING
            self.logger.info("Signal orchestration resumed")
    
    async def execute_single_run(self, 
                                assets: Optional[List[Asset]] = None,
                                force_execution: bool = False) -> ExecutionMetrics:
        """Execute a single orchestration run"""
        
        async with self.execution_lock:
            # Initialize run metrics
            run_id = uuid4()
            metrics = ExecutionMetrics(
                run_id=run_id,
                start_time=datetime.utcnow()
            )
            self.current_run_metrics = metrics
            
            try:
                start_time = time.time()
                
                # Step 1: Data gathering
                self.logger.info(f"Starting orchestration run {run_id}")
                
                if not assets:
                    assets = await self._gather_assets()
                
                market_data = await self._gather_market_data(assets)
                technical_indicators = await self._gather_technical_indicators(assets)
                options_data = await self._gather_options_data(assets) if self._has_options_generators() else {}
                
                metrics.assets_processed = len(assets)
                
                # Step 2: Generate signals
                all_signals = await self._execute_generators(
                    assets, market_data, technical_indicators, options_data, metrics
                )
                
                metrics.signals_generated = len(all_signals)
                
                # Step 3: Validation and quality control
                if self.config.enable_validation and all_signals:
                    validated_signals = await self._validate_signals(
                        all_signals, assets, market_data, technical_indicators, metrics
                    )
                else:
                    validated_signals = all_signals
                
                metrics.signals_validated = len(validated_signals)
                
                # Step 4: Risk assessment
                if self.config.enable_risk_assessment and validated_signals:
                    risk_assessed_signals = await self._assess_risks(
                        validated_signals, assets, market_data, technical_indicators
                    )
                else:
                    risk_assessed_signals = validated_signals
                
                # Step 5: Deduplication and filtering
                filtered_signals = await self._filter_and_deduplicate(risk_assessed_signals)
                metrics.signals_filtered = len(risk_assessed_signals) - len(filtered_signals)
                
                # Step 6: Output distribution
                await self._distribute_signals(filtered_signals)
                metrics.signals_output = len(filtered_signals)
                
                # Finalize metrics
                metrics.end_time = datetime.utcnow()
                metrics.execution_time_ms = (time.time() - start_time) * 1000
                
                self.execution_history.append(metrics)
                
                self.logger.info(
                    f"Orchestration run {run_id} completed: "
                    f"{metrics.signals_generated} generated, "
                    f"{metrics.signals_output} output, "
                    f"{metrics.execution_time_ms:.1f}ms"
                )
                
                # Send notifications for high-priority signals
                if self.config.enable_notifications:
                    await self._send_notifications(filtered_signals)
                
                return metrics
                
            except Exception as e:
                self.error_count += 1
                self.last_error = e
                metrics.errors_count += 1
                metrics.end_time = datetime.utcnow()
                
                self.logger.error(f"Orchestration run {run_id} failed: {str(e)}", exc_info=True)
                
                if not force_execution:
                    self.status = ExecutionStatus.ERROR
                
                return metrics
            
            finally:
                self.current_run_metrics = None
    
    async def _real_time_loop(self) -> None:
        """Main real-time processing loop"""
        while not self._stop_event.is_set():
            if self.status == ExecutionStatus.RUNNING:
                try:
                    await self.execute_single_run()
                except Exception as e:
                    self.logger.error(f"Error in real-time loop: {str(e)}", exc_info=True)
                    await asyncio.sleep(5)  # Brief pause on error
            
            # Wait for next execution cycle
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), 
                    timeout=self.config.execution_interval_seconds
                )
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue loop
    
    async def _scheduled_loop(self) -> None:
        """Scheduled processing loop"""
        while not self._stop_event.is_set():
            if self.status == ExecutionStatus.RUNNING:
                # Check if it's time to run (simplified scheduling)
                now = datetime.utcnow()
                
                # Run every interval (could be enhanced with cron-like scheduling)
                try:
                    await self.execute_single_run()
                except Exception as e:
                    self.logger.error(f"Error in scheduled loop: {str(e)}", exc_info=True)
            
            # Wait for next scheduled execution
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.execution_interval_seconds
                )
            except asyncio.TimeoutError:
                pass
    
    async def _gather_assets(self) -> List[Asset]:
        """Gather assets for processing"""
        if "assets" in self.data_providers:
            return await self.data_providers["assets"]()
        else:
            self.logger.warning("No asset data provider registered")
            return []
    
    async def _gather_market_data(self, assets: List[Asset]) -> Dict[UUID, MarketData]:
        """Gather market data for assets"""
        if "market_data" in self.data_providers:
            return await self.data_providers["market_data"](assets)
        else:
            self.logger.warning("No market data provider registered")
            return {}
    
    async def _gather_technical_indicators(self, assets: List[Asset]) -> Dict[UUID, TechnicalIndicators]:
        """Gather technical indicators for assets"""
        if "technical_indicators" in self.data_providers:
            return await self.data_providers["technical_indicators"](assets)
        else:
            self.logger.warning("No technical indicators provider registered")
            return {}
    
    async def _gather_options_data(self, assets: List[Asset]) -> Dict[UUID, List[OptionsData]]:
        """Gather options data for assets"""
        if "options_data" in self.data_providers:
            return await self.data_providers["options_data"](assets)
        else:
            return {}
    
    def _has_options_generators(self) -> bool:
        """Check if any generators require options data"""
        return any("options" in gen.name.lower() for gen in self.generators)
    
    async def _execute_generators(self,
                                 assets: List[Asset],
                                 market_data: Dict[UUID, MarketData],
                                 technical_indicators: Dict[UUID, TechnicalIndicators],
                                 options_data: Dict[UUID, List[OptionsData]],
                                 metrics: ExecutionMetrics) -> List[Signal]:
        """Execute all signal generators in parallel"""
        
        if not self.generators:
            self.logger.warning("No generators registered")
            return []
        
        # Create execution tasks
        tasks = []
        for generator in self.generators:
            if generator.is_enabled:
                task = self._execute_single_generator(
                    generator, assets, market_data, technical_indicators, options_data
                )
                tasks.append(task)
        
        if not tasks:
            self.logger.warning("No enabled generators")
            return []
        
        # Execute generators concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect signals and handle errors
        all_signals = []
        for i, result in enumerate(results):
            generator = self.generators[i]
            
            if isinstance(result, Exception):
                self.generator_failures[generator.name] += 1
                metrics.errors_count += 1
                self.logger.error(f"Generator {generator.name} failed: {result}")
            elif isinstance(result, SignalGenerationResult):
                all_signals.extend(result.signals)
                metrics.generators_executed += 1
                
                if result.errors:
                    metrics.errors_count += len(result.errors)
                    self.logger.warning(f"Generator {generator.name} had {len(result.errors)} errors")
        
        return all_signals
    
    async def _execute_single_generator(self,
                                       generator: BaseSignalGenerator,
                                       assets: List[Asset],
                                       market_data: Dict[UUID, MarketData],
                                       technical_indicators: Dict[UUID, TechnicalIndicators],
                                       options_data: Dict[UUID, List[OptionsData]]) -> SignalGenerationResult:
        """Execute a single generator with timeout and error handling"""
        
        async with self.generator_semaphore:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    generator.run_generation(
                        assets, market_data, technical_indicators, options_data
                    ),
                    timeout=self.config.generator_timeout_seconds
                )
                
                return result
                
            except asyncio.TimeoutError:
                raise RuntimeError(f"Generator {generator.name} timed out after {self.config.generator_timeout_seconds}s")
            
            except Exception as e:
                raise RuntimeError(f"Generator {generator.name} execution failed: {str(e)}")
    
    async def _validate_signals(self,
                               signals: List[Signal],
                               assets: List[Asset],
                               market_data: Dict[UUID, MarketData],
                               technical_indicators: Dict[UUID, TechnicalIndicators],
                               metrics: ExecutionMetrics) -> List[Signal]:
        """Validate signals and filter by quality"""
        
        if not self.validator or not signals:
            return signals
        
        # Create asset lookup
        asset_lookup = {asset.id: asset for asset in assets}
        
        # Validate signals in batches
        validated_signals = []
        
        for i in range(0, len(signals), self.config.batch_size):
            batch = signals[i:i + self.config.batch_size]
            
            validation_results = await self.validator.validate_signal_batch(
                batch, asset_lookup, market_data, technical_indicators
            )
            
            # Filter based on validation results
            for j, result in enumerate(validation_results):
                signal = batch[j]
                
                if (result.is_valid and 
                    result.quality_score >= self.config.min_quality_score and
                    signal.confidence_score >= self.config.min_confidence_score and
                    signal.risk_score <= self.config.max_risk_score):
                    
                    validated_signals.append(signal)
                else:
                    metrics.signals_filtered += 1
        
        return validated_signals
    
    async def _assess_risks(self,
                           signals: List[Signal],
                           assets: List[Asset],
                           market_data: Dict[UUID, MarketData],
                           technical_indicators: Dict[UUID, TechnicalIndicators]) -> List[Signal]:
        """Assess and calibrate risk scores for signals"""
        
        if not self.risk_scorer or not signals:
            return signals
        
        # Create asset lookup
        asset_lookup = {asset.id: asset for asset in assets}
        
        # Assess risks for each signal
        risk_assessed_signals = []
        
        for signal in signals:
            asset = asset_lookup.get(signal.asset_id)
            mkt_data = market_data.get(signal.asset_id)
            tech_data = technical_indicators.get(signal.asset_id)
            
            if asset and mkt_data:
                try:
                    risk_assessment = await self.risk_scorer.assess_risk(
                        signal, asset, mkt_data, tech_data
                    )
                    
                    # Update signal with calibrated risk score
                    signal.risk_score = risk_assessment.calibrated_risk_score
                    
                    # Add risk assessment metadata
                    signal.asset_specific_data = signal.asset_specific_data or {}
                    signal.asset_specific_data.update({
                        "risk_assessment": {
                            "calibrated_score": risk_assessment.calibrated_risk_score,
                            "original_score": risk_assessment.original_risk_score,
                            "confidence_level": risk_assessment.confidence_level,
                            "assessment_timestamp": risk_assessment.assessment_timestamp.isoformat()
                        }
                    })
                    
                    risk_assessed_signals.append(signal)
                    
                except Exception as e:
                    self.logger.warning(f"Risk assessment failed for signal {signal.id}: {e}")
                    risk_assessed_signals.append(signal)  # Keep original signal
            else:
                risk_assessed_signals.append(signal)
        
        return risk_assessed_signals
    
    async def _filter_and_deduplicate(self, signals: List[Signal]) -> List[Signal]:
        """Filter and deduplicate signals"""
        
        if not signals:
            return signals
        
        # Clean expired entries from cache
        now = datetime.utcnow()
        expired_keys = [
            key for key, timestamp in self.signal_cache.items()
            if (now - timestamp).total_seconds() > self.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self.signal_cache[key]
        
        # Deduplicate and filter signals
        filtered_signals = []
        seen_hashes = set()
        
        for signal in signals:
            # Create hash for deduplication
            signal_hash = self._create_signal_hash(signal)
            
            # Check for duplicates
            if self.config.enable_deduplication:
                if signal_hash in seen_hashes or signal_hash in self.signal_cache:
                    continue
                
                seen_hashes.add(signal_hash)
                self.signal_cache[signal_hash] = now
            
            # Apply additional filters
            if self._passes_final_filters(signal):
                filtered_signals.append(signal)
        
        # Limit total signals per run
        if len(filtered_signals) > self.config.max_signals_per_run:
            # Sort by confidence * profit potential and take top signals
            filtered_signals.sort(
                key=lambda s: s.confidence_score * s.profit_potential_score,
                reverse=True
            )
            filtered_signals = filtered_signals[:self.config.max_signals_per_run]
        
        return filtered_signals
    
    def _create_signal_hash(self, signal: Signal) -> str:
        """Create a hash for signal deduplication"""
        # Use asset, direction, and price for basic deduplication
        hash_components = [
            str(signal.asset_id),
            signal.direction.value,
            str(signal.entry_price_cents) if signal.entry_price_cents else "no_price",
            signal.signal_source
        ]
        
        return "_".join(hash_components)
    
    def _passes_final_filters(self, signal: Signal) -> bool:
        """Apply final quality filters to signal"""
        
        # Check expiration
        if signal.valid_until and signal.valid_until <= datetime.utcnow():
            return False
        
        # Check minimum confidence
        if signal.confidence_score < self.config.min_confidence_score:
            return False
        
        # Check maximum risk
        if signal.risk_score > self.config.max_risk_score:
            return False
        
        # Check status
        if signal.status != SignalStatus.ACTIVE:
            return False
        
        return True
    
    async def _distribute_signals(self, signals: List[Signal]) -> None:
        """Distribute signals to output handlers"""
        
        if not signals or not self.output_handlers:
            return
        
        # Send signals to all registered output handlers
        for handler in self.output_handlers:
            try:
                await handler(signals)
            except Exception as e:
                self.logger.error(f"Output handler failed: {str(e)}", exc_info=True)
    
    async def _send_notifications(self, signals: List[Signal]) -> None:
        """Send notifications for high-priority signals"""
        
        if not self.notification_handlers:
            return
        
        # Filter for high-priority signals
        high_priority_signals = [
            signal for signal in signals
            if (signal.confidence_score * signal.profit_potential_score / 100) >= self.config.high_priority_threshold
        ]
        
        if not high_priority_signals:
            return
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                await handler(high_priority_signals)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {str(e)}", exc_info=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        
        avg_execution_time = 0.0
        total_signals = 0
        
        if self.execution_history:
            recent_runs = list(self.execution_history)[-10:]  # Last 10 runs
            avg_execution_time = sum(run.execution_time_ms for run in recent_runs) / len(recent_runs)
            total_signals = sum(run.signals_output for run in recent_runs)
        
        return {
            "status": self.status.value,
            "generators_registered": len(self.generators),
            "generators_enabled": len([g for g in self.generators if g.is_enabled]),
            "data_providers": list(self.data_providers.keys()),
            "output_handlers": len(self.output_handlers),
            "notification_handlers": len(self.notification_handlers),
            "execution_history_size": len(self.execution_history),
            "average_execution_time_ms": avg_execution_time,
            "total_signals_recent": total_signals,
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "generator_failures": dict(self.generator_failures),
            "current_run": {
                "run_id": str(self.current_run_metrics.run_id),
                "start_time": self.current_run_metrics.start_time.isoformat(),
                "generators_executed": self.current_run_metrics.generators_executed,
                "signals_generated": self.current_run_metrics.signals_generated
            } if self.current_run_metrics else None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        runs = list(self.execution_history)
        
        # Calculate aggregate metrics
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.errors_count == 0])
        
        execution_times = [r.execution_time_ms for r in runs]
        signals_generated = [r.signals_generated for r in runs]
        signals_output = [r.signals_output for r in runs]
        
        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate_pct": (successful_runs / total_runs) * 100,
            "execution_time_stats": {
                "mean_ms": sum(execution_times) / len(execution_times),
                "min_ms": min(execution_times),
                "max_ms": max(execution_times),
                "p95_ms": sorted(execution_times)[int(len(execution_times) * 0.95)] if execution_times else 0
            },
            "signals_stats": {
                "total_generated": sum(signals_generated),
                "total_output": sum(signals_output),
                "avg_per_run": sum(signals_output) / len(signals_output) if signals_output else 0,
                "output_rate_pct": (sum(signals_output) / max(sum(signals_generated), 1)) * 100
            },
            "generator_performance": {
                name: {
                    "failures": failures,
                    "failure_rate_pct": (failures / total_runs) * 100
                }
                for name, failures in self.generator_failures.items()
            }
        }
    
    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics"""
        self.execution_history.clear()
        self.error_count = 0
        self.last_error = None
        self.generator_failures.clear()
        self.signal_cache.clear()
        
        # Reset generator metrics
        for generator in self.generators:
            generator.reset_performance_metrics()
        
        # Reset validator metrics
        if self.validator:
            self.validator.reset_statistics()
        
        # Reset risk scorer metrics  
        if self.risk_scorer:
            self.risk_scorer.reset_statistics()
        
        self.logger.info("Performance metrics reset")