"""
Risk Monitoring Service
Real-time risk monitoring and alerting system
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import logging

from ..models.user import User, RiskTolerance
from ..models.portfolio import Portfolio
from .management import RiskManager, RiskMetrics


logger = logging.getLogger(__name__)


class RiskMonitoringService:
    """Real-time risk monitoring and alerting service"""
    
    def __init__(self):
        self.risk_manager = RiskManager()
        self.is_running = False
        self.monitored_portfolios: Dict[str, Dict[str, Any]] = {}
        self.risk_limits = self._get_default_risk_limits()
        self.monitoring_task: Optional[asyncio.Task] = None
        
    def _get_default_risk_limits(self) -> Dict[str, Dict[str, float]]:
        """Get default risk limits by risk tolerance"""
        return {
            RiskTolerance.CONSERVATIVE: {
                'max_var_95': 0.03,  # 3% of portfolio
                'max_drawdown': -0.10,  # 10% max drawdown
                'max_position_concentration': 0.15,  # 15% max single position
                'min_liquidity_score': 0.8,
                'max_correlation': 0.5,
                'max_beta': 0.8
            },
            RiskTolerance.MODERATE: {
                'max_var_95': 0.05,  # 5% of portfolio
                'max_drawdown': -0.15,  # 15% max drawdown
                'max_position_concentration': 0.25,  # 25% max single position
                'min_liquidity_score': 0.6,
                'max_correlation': 0.7,
                'max_beta': 1.2
            },
            RiskTolerance.AGGRESSIVE: {
                'max_var_95': 0.08,  # 8% of portfolio
                'max_drawdown': -0.25,  # 25% max drawdown
                'max_position_concentration': 0.35,  # 35% max single position
                'min_liquidity_score': 0.4,
                'max_correlation': 0.8,
                'max_beta': 1.5
            },
            RiskTolerance.VERY_AGGRESSIVE: {
                'max_var_95': 0.12,  # 12% of portfolio
                'max_drawdown': -0.35,  # 35% max drawdown
                'max_position_concentration': 0.50,  # 50% max single position
                'min_liquidity_score': 0.2,
                'max_correlation': 0.9,
                'max_beta': 2.0
            }
        }
    
    async def start(self):
        """Start the risk monitoring service"""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Starting Risk Monitoring Service")
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop the risk monitoring service"""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Stopping Risk Monitoring Service")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def add_portfolio_monitoring(
        self,
        user: User,
        portfolio: Portfolio,
        risk_limits: Optional[Dict[str, float]] = None
    ):
        """Add portfolio to risk monitoring"""
        try:
            # Get risk limits based on user's risk tolerance
            if risk_limits is None:
                risk_limits = self.risk_limits.get(
                    user.risk_tolerance,
                    self.risk_limits[RiskTolerance.MODERATE]
                ).copy()
            
            # Store monitoring configuration
            self.monitored_portfolios[user.id] = {
                'user': user,
                'portfolio': portfolio,
                'risk_limits': risk_limits,
                'last_check': datetime.now(timezone.utc),
                'alert_history': [],
                'auto_rebalancing': False,
                'rebalance_threshold': 0.05
            }
            
            logger.info(f"Added portfolio monitoring for user {user.id}")
            
        except Exception as e:
            logger.error(f"Error adding portfolio monitoring: {e}")
    
    async def remove_portfolio_monitoring(self, user_id: str):
        """Remove portfolio from monitoring"""
        if user_id in self.monitored_portfolios:
            del self.monitored_portfolios[user_id]
            logger.info(f"Removed portfolio monitoring for user {user_id}")
    
    async def update_risk_limits(self, user_id: str, risk_limits: Dict[str, float]):
        """Update risk limits for a user"""
        if user_id in self.monitored_portfolios:
            self.monitored_portfolios[user_id]['risk_limits'].update(risk_limits)
            logger.info(f"Updated risk limits for user {user_id}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check all monitored portfolios
                for user_id, config in self.monitored_portfolios.items():
                    await self._check_portfolio_risk(user_id, config)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_portfolio_risk(self, user_id: str, config: Dict[str, Any]):
        """Check risk metrics for a single portfolio"""
        try:
            portfolio = config['portfolio']
            risk_limits = config['risk_limits']
            
            # Mock current risk metrics calculation
            # In real implementation, this would use live market data
            current_metrics = await self._calculate_current_risk_metrics(portfolio)
            
            # Check for violations
            violations = self._check_risk_violations(user_id, current_metrics)
            
            # Generate alerts for violations
            for violation in violations:
                await self._generate_alert(user_id, violation)
            
            # Check if rebalancing is needed
            if config.get('auto_rebalancing', False):
                if await self._check_rebalancing_needed(user_id, {}):
                    await self._trigger_rebalancing(user_id)
            
            # Update last check time
            config['last_check'] = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk for user {user_id}: {e}")
    
    async def _calculate_current_risk_metrics(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate current risk metrics for portfolio"""
        # Mock implementation - in reality would use live market data
        # and the risk manager to calculate actual metrics
        return RiskMetrics(
            var_95=2000.0,
            var_99=3000.0,
            expected_shortfall=2500.0,
            current_drawdown=-0.05,
            max_drawdown=-0.08,
            volatility=0.15,
            beta=1.1,
            max_position_concentration=0.25,
            liquidity_score=0.75,
            portfolio_value=100000.0
        )
    
    def _check_risk_violations(
        self,
        user_id: str,
        current_metrics: RiskMetrics
    ) -> List[Dict[str, Any]]:
        """Check for risk limit violations"""
        violations = []
        
        if user_id not in self.monitored_portfolios:
            return violations
        
        risk_limits = self.monitored_portfolios[user_id]['risk_limits']
        portfolio_value = current_metrics.portfolio_value
        
        # Check VaR limits
        if current_metrics.var_95:
            var_percentage = current_metrics.var_95 / portfolio_value
            if var_percentage > risk_limits.get('max_var_95', 1.0):
                violations.append({
                    'limit_type': 'max_var_95',
                    'current_value': var_percentage,
                    'limit_value': risk_limits['max_var_95'],
                    'severity': 'HIGH' if var_percentage > risk_limits['max_var_95'] * 1.5 else 'MEDIUM',
                    'message': f'Portfolio VaR ({var_percentage:.2%}) exceeds risk limit ({risk_limits["max_var_95"]:.2%})'
                })
        
        # Check drawdown limits
        if current_metrics.current_drawdown:
            if current_metrics.current_drawdown < risk_limits.get('max_drawdown', -1.0):
                violations.append({
                    'limit_type': 'max_drawdown',
                    'current_value': current_metrics.current_drawdown,
                    'limit_value': risk_limits['max_drawdown'],
                    'severity': 'HIGH',
                    'message': f'Portfolio drawdown ({current_metrics.current_drawdown:.2%}) exceeds limit ({risk_limits["max_drawdown"]:.2%})'
                })
        
        # Check position concentration
        if current_metrics.max_position_concentration:
            if current_metrics.max_position_concentration > risk_limits.get('max_position_concentration', 1.0):
                violations.append({
                    'limit_type': 'max_position_concentration',
                    'current_value': current_metrics.max_position_concentration,
                    'limit_value': risk_limits['max_position_concentration'],
                    'severity': 'MEDIUM',
                    'message': f'Position concentration ({current_metrics.max_position_concentration:.2%}) exceeds limit'
                })
        
        # Check liquidity score
        if current_metrics.liquidity_score:
            if current_metrics.liquidity_score < risk_limits.get('min_liquidity_score', 0.0):
                violations.append({
                    'limit_type': 'min_liquidity_score',
                    'current_value': current_metrics.liquidity_score,
                    'limit_value': risk_limits['min_liquidity_score'],
                    'severity': 'MEDIUM',
                    'message': f'Portfolio liquidity score ({current_metrics.liquidity_score:.2f}) below minimum'
                })
        
        return violations
    
    async def _generate_alert(self, user_id: str, violation: Dict[str, Any]):
        """Generate risk alert"""
        try:
            alert = {
                'user_id': user_id,
                'alert_type': 'RISK_LIMIT_VIOLATION',
                'severity': violation['severity'],
                'message': violation['message'],
                'violation_details': violation,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Store alert in history
            if user_id in self.monitored_portfolios:
                self.monitored_portfolios[user_id]['alert_history'].append(alert)
                
                # Keep only last 100 alerts
                alert_history = self.monitored_portfolios[user_id]['alert_history']
                if len(alert_history) > 100:
                    self.monitored_portfolios[user_id]['alert_history'] = alert_history[-100:]
            
            # Send alert (mock implementation)
            await self._send_alert(alert)
            
            # Publish to Kafka
            await self._publish_risk_alert(alert)
            
            logger.warning(f"Risk alert generated for user {user_id}: {violation['message']}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to user (email, SMS, push notification)"""
        # Mock implementation - in reality would integrate with notification services
        logger.info(f"Sending alert: {alert['message']}")
    
    async def _publish_risk_alert(self, alert: Dict[str, Any]):
        """Publish risk alert to Kafka"""
        try:
            # Mock Kafka publishing
            logger.info(f"Publishing risk alert to Kafka: {alert['alert_type']}")
        except Exception as e:
            logger.error(f"Error publishing risk alert to Kafka: {e}")
    
    async def _check_rebalancing_needed(
        self,
        user_id: str,
        current_allocation: Dict[str, float]
    ) -> bool:
        """Check if portfolio rebalancing is needed"""
        if user_id not in self.monitored_portfolios:
            return False
        
        config = self.monitored_portfolios[user_id]
        target_allocation = config.get('target_allocation', {})
        threshold = config.get('rebalance_threshold', 0.05)
        
        # Check if any position has drifted beyond threshold
        for symbol, target_weight in target_allocation.items():
            current_weight = current_allocation.get(symbol, 0.0)
            drift = abs(current_weight - target_weight)
            
            if drift > threshold:
                return True
        
        return False
    
    async def _trigger_rebalancing(self, user_id: str):
        """Trigger portfolio rebalancing"""
        try:
            logger.info(f"Triggering portfolio rebalancing for user {user_id}")
            
            # Mock rebalancing trigger
            rebalancing_event = {
                'user_id': user_id,
                'event_type': 'AUTO_REBALANCING_TRIGGERED',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'reason': 'Portfolio drift exceeded threshold'
            }
            
            # Publish rebalancing event
            await self._publish_rebalancing_event(rebalancing_event)
            
        except Exception as e:
            logger.error(f"Error triggering rebalancing: {e}")
    
    async def _publish_rebalancing_event(self, event: Dict[str, Any]):
        """Publish rebalancing event to Kafka"""
        try:
            # Mock Kafka publishing
            logger.info(f"Publishing rebalancing event to Kafka")
        except Exception as e:
            logger.error(f"Error publishing rebalancing event: {e}")
    
    def calculate_risk_score(
        self,
        portfolio: Portfolio,
        risk_metrics: RiskMetrics,
        risk_tolerance: RiskTolerance
    ) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            risk_limits = self.risk_limits.get(risk_tolerance, self.risk_limits[RiskTolerance.MODERATE])
            
            scores = []
            
            # VaR score
            if risk_metrics.var_95 and risk_metrics.portfolio_value > 0:
                var_pct = risk_metrics.var_95 / risk_metrics.portfolio_value
                var_limit = risk_limits['max_var_95']
                var_score = min(100, (var_pct / var_limit) * 50)  # 50% weight
                scores.append(var_score)
            
            # Drawdown score
            if risk_metrics.current_drawdown:
                dd_limit = risk_limits['max_drawdown']
                dd_score = min(100, abs(risk_metrics.current_drawdown / dd_limit) * 30)  # 30% weight
                scores.append(dd_score)
            
            # Concentration score
            if risk_metrics.max_position_concentration:
                conc_limit = risk_limits['max_position_concentration']
                conc_score = min(100, (risk_metrics.max_position_concentration / conc_limit) * 20)  # 20% weight
                scores.append(conc_score)
            
            # Overall score
            if scores:
                return min(100, sum(scores))
            else:
                return 50.0  # Default moderate risk score
                
        except Exception:
            return 50.0  # Default score on error
    
    def get_portfolio_performance(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio performance statistics"""
        if user_id not in self.monitored_portfolios:
            return None
        
        config = self.monitored_portfolios[user_id]
        portfolio = config['portfolio']
        
        # Mock performance statistics
        return {
            'total_return': 0.12,
            'ytd_return': 0.08,
            'current_value': float(portfolio.total_value) if hasattr(portfolio, 'total_value') else 0.0,
            'positions': len(portfolio.positions) if hasattr(portfolio, 'positions') else 0,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
    
    def aggregate_risk_metrics(
        self,
        portfolio_metrics: Dict[str, RiskMetrics]
    ) -> Dict[str, Any]:
        """Aggregate risk metrics across multiple portfolios"""
        if not portfolio_metrics:
            return {}
        
        total_value = sum(metrics.portfolio_value for metrics in portfolio_metrics.values())
        
        # Weighted average calculations
        weighted_var = sum(
            (metrics.var_95 or 0) * (metrics.portfolio_value / total_value)
            for metrics in portfolio_metrics.values()
        ) if total_value > 0 else 0
        
        avg_volatility = sum(
            (metrics.volatility or 0) for metrics in portfolio_metrics.values()
        ) / len(portfolio_metrics)
        
        return {
            'total_var': weighted_var,
            'average_volatility': avg_volatility,
            'total_portfolio_value': total_value,
            'portfolio_count': len(portfolio_metrics)
        }