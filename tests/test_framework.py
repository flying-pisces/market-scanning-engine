"""
Risk Scoring System - Automated Test Framework
Author: Claude Code (QA Engineer)  
Version: 1.0

Comprehensive test automation framework for continuous integration.
Orchestrates all risk scoring tests with reporting, metrics collection,
and automated pass/fail determination for production deployments.
"""

import pytest
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import logging
from contextlib import contextmanager

# Import test modules
from test_risk_components import *
from test_risk_integration import *
from test_risk_performance import *
from test_risk_stress import *
from test_risk_accuracy import *


@dataclass
class TestResult:
    """Container for individual test results"""
    test_name: str
    test_category: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_seconds: float
    error_message: Optional[str] = None
    warnings: List[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}


@dataclass
class TestSuiteReport:
    """Comprehensive test suite report"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    overall_status: str
    test_results: List[TestResult]
    performance_metrics: Dict[str, Any]
    coverage_metrics: Dict[str, float]
    deployment_recommendation: str
    
    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.passed + self.skipped) / self.total_tests if self.total_tests > 0 else 0.0


class RiskTestFramework:
    """Main test framework orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.results: List[TestResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            "test_categories": {
                "unit": {
                    "enabled": True,
                    "timeout_minutes": 10,
                    "required_pass_rate": 0.95,
                    "critical": True
                },
                "integration": {
                    "enabled": True, 
                    "timeout_minutes": 15,
                    "required_pass_rate": 0.90,
                    "critical": True
                },
                "performance": {
                    "enabled": True,
                    "timeout_minutes": 20,
                    "required_pass_rate": 0.85,
                    "critical": True
                },
                "stress": {
                    "enabled": True,
                    "timeout_minutes": 30,
                    "required_pass_rate": 0.80,
                    "critical": False
                },
                "accuracy": {
                    "enabled": True,
                    "timeout_minutes": 25,
                    "required_pass_rate": 0.85,
                    "critical": True
                }
            },
            "performance_thresholds": {
                "max_latency_ms": 100,
                "min_throughput_ops_per_sec": 1000,
                "max_memory_usage_mb": 2000,
                "min_cpu_efficiency": 0.5
            },
            "accuracy_thresholds": {
                "min_correlation": 0.4,
                "min_r_squared": 0.15,
                "min_calibration_score": 0.3
            },
            "deployment_gates": {
                "critical_test_pass_rate": 0.95,
                "overall_pass_rate": 0.90,
                "performance_requirement_met": True,
                "accuracy_requirement_met": True
            },
            "reporting": {
                "output_directory": "test_reports",
                "generate_html_report": True,
                "generate_json_report": True,
                "upload_metrics": False
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test framework"""
        logger = logging.getLogger("risk_test_framework")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler(f"logs/risk_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def run_test_suite(self, categories: Optional[List[str]] = None) -> TestSuiteReport:
        """Run complete test suite"""
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting risk scoring test suite at {self.start_time}")
        
        if categories is None:
            categories = [cat for cat, config in self.config["test_categories"].items() 
                         if config["enabled"]]
        
        self.logger.info(f"Running test categories: {categories}")
        
        # Run each test category
        for category in categories:
            if category in self.config["test_categories"]:
                self._run_category(category)
            else:
                self.logger.warning(f"Unknown test category: {category}")
        
        self.end_time = datetime.utcnow()
        
        # Generate comprehensive report
        report = self._generate_report()
        
        # Save reports
        self._save_reports(report)
        
        # Log summary
        self._log_summary(report)
        
        return report
    
    def _run_category(self, category: str) -> None:
        """Run tests for a specific category"""
        category_config = self.config["test_categories"][category]
        timeout_seconds = category_config["timeout_minutes"] * 60
        
        self.logger.info(f"Running {category} tests (timeout: {timeout_seconds}s)")
        
        # Map categories to test modules/patterns
        test_patterns = {
            "unit": "test_risk_components.py",
            "integration": "test_risk_integration.py", 
            "performance": "test_risk_performance.py",
            "stress": "test_risk_stress.py",
            "accuracy": "test_risk_accuracy.py"
        }
        
        if category not in test_patterns:
            self.logger.error(f"No test pattern defined for category: {category}")
            return
        
        test_file = test_patterns[category]
        
        try:
            # Run pytest for the category
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file=test_reports/{category}_results.json",
                f"--timeout={timeout_seconds}"
            ]
            
            # Add category-specific options
            if category == "performance":
                cmd.extend(["-m", "not slow"])  # Skip very slow tests in CI
            elif category == "stress":
                cmd.extend(["-s"])  # Show output for stress tests
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=os.path.dirname(__file__)
            )
            end_time = time.time()
            
            # Parse results
            self._parse_pytest_results(category, result, end_time - start_time)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"{category} tests timed out after {timeout_seconds} seconds")
            self.results.append(TestResult(
                test_name=f"{category}_timeout",
                test_category=category,
                status="ERROR",
                duration_seconds=timeout_seconds,
                error_message="Test suite timed out"
            ))
            
        except Exception as e:
            self.logger.error(f"Error running {category} tests: {e}")
            self.results.append(TestResult(
                test_name=f"{category}_error",
                test_category=category,
                status="ERROR", 
                duration_seconds=0.0,
                error_message=str(e)
            ))
    
    def _parse_pytest_results(self, category: str, result: subprocess.CompletedProcess, duration: float) -> None:
        """Parse pytest results and extract test information"""
        
        # Try to load JSON report if available
        json_file = f"test_reports/{category}_results.json"
        
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    pytest_data = json.load(f)
                
                # Extract test results from JSON
                for test in pytest_data.get('tests', []):
                    test_result = TestResult(
                        test_name=test.get('nodeid', 'unknown'),
                        test_category=category,
                        status=test.get('outcome', 'UNKNOWN').upper(),
                        duration_seconds=test.get('duration', 0.0),
                        error_message=test.get('longrepr') if test.get('outcome') != 'passed' else None
                    )
                    self.results.append(test_result)
                    
                self.logger.info(f"{category} tests completed - {pytest_data.get('summary', {})}")
            else:
                # Fallback to parsing stdout/stderr
                self._parse_pytest_output(category, result, duration)
                
        except Exception as e:
            self.logger.warning(f"Could not parse {category} test results: {e}")
            # Create summary result
            status = "PASS" if result.returncode == 0 else "FAIL"
            self.results.append(TestResult(
                test_name=f"{category}_suite",
                test_category=category,
                status=status,
                duration_seconds=duration,
                error_message=result.stderr if result.returncode != 0 else None
            ))
    
    def _parse_pytest_output(self, category: str, result: subprocess.CompletedProcess, duration: float) -> None:
        """Fallback method to parse pytest stdout output"""
        
        lines = result.stdout.split('\n') if result.stdout else []
        
        # Simple parsing - look for test results
        test_count = 0
        passed = 0
        failed = 0
        
        for line in lines:
            if " PASSED " in line:
                test_count += 1
                passed += 1
            elif " FAILED " in line:
                test_count += 1
                failed += 1
        
        # Create summary result
        if test_count > 0:
            for i in range(passed):
                self.results.append(TestResult(
                    test_name=f"{category}_test_{i+1}",
                    test_category=category,
                    status="PASS",
                    duration_seconds=duration / test_count
                ))
            
            for i in range(failed):
                self.results.append(TestResult(
                    test_name=f"{category}_failed_{i+1}",
                    test_category=category,
                    status="FAIL",
                    duration_seconds=duration / test_count,
                    error_message="Test failed (details in logs)"
                ))
        else:
            # No individual tests detected
            status = "PASS" if result.returncode == 0 else "FAIL"
            self.results.append(TestResult(
                test_name=f"{category}_suite",
                test_category=category,
                status=status,
                duration_seconds=duration,
                error_message=result.stderr if result.returncode != 0 else None
            ))
    
    def _generate_report(self) -> TestSuiteReport:
        """Generate comprehensive test suite report"""
        
        if not self.start_time or not self.end_time:
            raise ValueError("Test suite timing not recorded")
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        skipped = sum(1 for r in self.results if r.status == "SKIP")
        errors = sum(1 for r in self.results if r.status == "ERROR")
        
        # Determine overall status
        overall_status = self._determine_overall_status(passed, failed, errors, total_tests)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Calculate coverage metrics (simplified)
        coverage_metrics = self._calculate_coverage_metrics()
        
        # Generate deployment recommendation
        deployment_recommendation = self._generate_deployment_recommendation(
            overall_status, performance_metrics, coverage_metrics
        )
        
        report = TestSuiteReport(
            suite_name="Risk Scoring System Test Suite",
            start_time=self.start_time,
            end_time=self.end_time,
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            overall_status=overall_status,
            test_results=self.results,
            performance_metrics=performance_metrics,
            coverage_metrics=coverage_metrics,
            deployment_recommendation=deployment_recommendation
        )
        
        return report
    
    def _determine_overall_status(self, passed: int, failed: int, errors: int, total: int) -> str:
        """Determine overall test suite status"""
        if total == 0:
            return "NO_TESTS"
        
        if errors > 0:
            return "ERROR"
        
        if failed == 0:
            return "PASS"
        
        pass_rate = passed / total
        if pass_rate >= 0.95:
            return "PASS_WITH_WARNINGS"
        elif pass_rate >= 0.80:
            return "MARGINAL"
        else:
            return "FAIL"
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from test results"""
        performance_tests = [r for r in self.results if r.test_category == "performance"]
        
        if not performance_tests:
            return {"no_performance_data": True}
        
        # Extract performance data from test metrics
        latencies = []
        throughputs = []
        memory_usage = []
        
        for test in performance_tests:
            if test.metrics:
                if "latency_ms" in test.metrics:
                    latencies.append(test.metrics["latency_ms"])
                if "throughput_ops_per_sec" in test.metrics:
                    throughputs.append(test.metrics["throughput_ops_per_sec"])
                if "memory_usage_mb" in test.metrics:
                    memory_usage.append(test.metrics["memory_usage_mb"])
        
        metrics = {}
        
        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            metrics["max_latency_ms"] = max(latencies)
            metrics["p95_latency_ms"] = np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0]
        
        if throughputs:
            metrics["avg_throughput"] = sum(throughputs) / len(throughputs)
            metrics["min_throughput"] = min(throughputs)
        
        if memory_usage:
            metrics["avg_memory_mb"] = sum(memory_usage) / len(memory_usage)
            metrics["peak_memory_mb"] = max(memory_usage)
        
        # Check against thresholds
        thresholds = self.config["performance_thresholds"]
        metrics["meets_latency_requirement"] = metrics.get("max_latency_ms", 0) <= thresholds["max_latency_ms"]
        metrics["meets_throughput_requirement"] = metrics.get("min_throughput", float('inf')) >= thresholds["min_throughput_ops_per_sec"]
        metrics["meets_memory_requirement"] = metrics.get("peak_memory_mb", 0) <= thresholds["max_memory_usage_mb"]
        
        return metrics
    
    def _calculate_coverage_metrics(self) -> Dict[str, float]:
        """Calculate test coverage metrics"""
        
        # Count tests by category
        category_counts = {}
        category_passed = {}
        
        for result in self.results:
            category = result.test_category
            category_counts[category] = category_counts.get(category, 0) + 1
            if result.status == "PASS":
                category_passed[category] = category_passed.get(category, 0) + 1
        
        coverage = {}
        for category in category_counts:
            passed = category_passed.get(category, 0)
            total = category_counts[category]
            coverage[f"{category}_pass_rate"] = passed / total if total > 0 else 0.0
        
        # Overall coverage
        total_tests = sum(category_counts.values())
        total_passed = sum(category_passed.values())
        coverage["overall_pass_rate"] = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Critical component coverage
        critical_categories = [cat for cat, config in self.config["test_categories"].items() 
                              if config.get("critical", False)]
        
        critical_passed = sum(category_passed.get(cat, 0) for cat in critical_categories)
        critical_total = sum(category_counts.get(cat, 0) for cat in critical_categories)
        coverage["critical_pass_rate"] = critical_passed / critical_total if critical_total > 0 else 0.0
        
        return coverage
    
    def _generate_deployment_recommendation(
        self, 
        overall_status: str, 
        performance_metrics: Dict[str, Any],
        coverage_metrics: Dict[str, float]
    ) -> str:
        """Generate deployment recommendation based on test results"""
        
        gates = self.config["deployment_gates"]
        
        # Check critical tests
        critical_pass_rate = coverage_metrics.get("critical_pass_rate", 0.0)
        if critical_pass_rate < gates["critical_test_pass_rate"]:
            return f"BLOCK_DEPLOYMENT - Critical test pass rate {critical_pass_rate:.1%} below threshold {gates['critical_test_pass_rate']:.1%}"
        
        # Check overall pass rate
        overall_pass_rate = coverage_metrics.get("overall_pass_rate", 0.0)
        if overall_pass_rate < gates["overall_pass_rate"]:
            return f"BLOCK_DEPLOYMENT - Overall pass rate {overall_pass_rate:.1%} below threshold {gates['overall_pass_rate']:.1%}"
        
        # Check performance requirements
        if gates["performance_requirement_met"]:
            perf_issues = []
            if not performance_metrics.get("meets_latency_requirement", True):
                perf_issues.append("latency")
            if not performance_metrics.get("meets_throughput_requirement", True):
                perf_issues.append("throughput")
            if not performance_metrics.get("meets_memory_requirement", True):
                perf_issues.append("memory")
            
            if perf_issues:
                return f"BLOCK_DEPLOYMENT - Performance requirements not met: {', '.join(perf_issues)}"
        
        # Check accuracy requirements (if accuracy tests were run)
        accuracy_tests = [r for r in self.results if r.test_category == "accuracy"]
        if accuracy_tests and gates.get("accuracy_requirement_met", True):
            accuracy_pass_rate = sum(1 for r in accuracy_tests if r.status == "PASS") / len(accuracy_tests)
            if accuracy_pass_rate < 0.8:  # 80% accuracy test pass rate
                return f"BLOCK_DEPLOYMENT - Accuracy test pass rate {accuracy_pass_rate:.1%} too low"
        
        # All gates passed
        if overall_status == "PASS":
            return "APPROVE_DEPLOYMENT - All test criteria met"
        elif overall_status == "PASS_WITH_WARNINGS":
            return "APPROVE_DEPLOYMENT_WITH_MONITORING - Minor issues detected, monitor closely"
        else:
            return "CONDITIONAL_DEPLOYMENT - Review failed tests before deployment"
    
    def _save_reports(self, report: TestSuiteReport) -> None:
        """Save test reports in various formats"""
        
        output_dir = Path(self.config["reporting"]["output_directory"])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        if self.config["reporting"]["generate_json_report"]:
            json_file = output_dir / f"risk_test_report_{timestamp}.json"
            with open(json_file, 'w') as f:
                # Convert dataclasses to dict for JSON serialization
                report_dict = asdict(report)
                # Handle datetime serialization
                report_dict['start_time'] = report.start_time.isoformat()
                report_dict['end_time'] = report.end_time.isoformat()
                
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"JSON report saved to {json_file}")
        
        # HTML report
        if self.config["reporting"]["generate_html_report"]:
            html_file = output_dir / f"risk_test_report_{timestamp}.html"
            self._generate_html_report(report, html_file)
            self.logger.info(f"HTML report saved to {html_file}")
        
        # Upload metrics (if configured)
        if self.config["reporting"]["upload_metrics"]:
            self._upload_metrics(report)
    
    def _generate_html_report(self, report: TestSuiteReport, output_file: Path) -> None:
        """Generate HTML test report"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Risk Scoring System Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric {{ background-color: #e8f4f8; margin: 10px; padding: 15px; border-radius: 5px; min-width: 200px; }}
        .test-results {{ margin: 20px 0; }}
        .test {{ margin: 5px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .pass {{ border-left-color: #28a745; }}
        .fail {{ border-left-color: #dc3545; }}
        .error {{ border-left-color: #fd7e14; }}
        .skip {{ border-left-color: #6c757d; }}
        .recommendation {{ padding: 20px; margin: 20px 0; border-radius: 5px; font-weight: bold; }}
        .approve {{ background-color: #d4edda; color: #155724; }}
        .block {{ background-color: #f8d7da; color: #721c24; }}
        .conditional {{ background-color: #fff3cd; color: #856404; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Risk Scoring System Test Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {report.duration_seconds:.1f} seconds</p>
        <p><strong>Overall Status:</strong> {report.overall_status}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Tests</strong><br>
                {report.total_tests}
            </div>
            <div class="metric">
                <strong>Passed</strong><br>
                {report.passed} ({report.pass_rate:.1%})
            </div>
            <div class="metric">
                <strong>Failed</strong><br>
                {report.failed}
            </div>
            <div class="metric">
                <strong>Errors</strong><br>
                {report.errors}
            </div>
            <div class="metric">
                <strong>Skipped</strong><br>
                {report.skipped}
            </div>
        </div>
    </div>
    
    <div class="performance">
        <h2>Performance Metrics</h2>
        <div class="metrics">
        """
        
        for key, value in report.performance_metrics.items():
            if isinstance(value, (int, float)):
                html_content += f"""
            <div class="metric">
                <strong>{key.replace('_', ' ').title()}</strong><br>
                {value:.2f}
            </div>
                """
        
        html_content += f"""
        </div>
    </div>
    
    <div class="recommendation {'approve' if 'APPROVE' in report.deployment_recommendation else 'block' if 'BLOCK' in report.deployment_recommendation else 'conditional'}">
        <h2>Deployment Recommendation</h2>
        <p>{report.deployment_recommendation}</p>
    </div>
    
    <div class="test-results">
        <h2>Test Results</h2>
        """
        
        # Group tests by category
        by_category = {}
        for test in report.test_results:
            if test.test_category not in by_category:
                by_category[test.test_category] = []
            by_category[test.test_category].append(test)
        
        for category, tests in by_category.items():
            html_content += f"<h3>{category.title()} Tests</h3>"
            
            for test in tests:
                status_class = test.status.lower()
                html_content += f"""
        <div class="test {status_class}">
            <strong>{test.test_name}</strong> - {test.status} ({test.duration_seconds:.2f}s)
            {f'<br><em>Error: {test.error_message}</em>' if test.error_message else ''}
        </div>
                """
        
        html_content += """
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _upload_metrics(self, report: TestSuiteReport) -> None:
        """Upload test metrics to external system (placeholder)"""
        self.logger.info("Metrics upload not configured - skipping")
        # This would integrate with monitoring systems like Datadog, New Relic, etc.
        pass
    
    def _log_summary(self, report: TestSuiteReport) -> None:
        """Log test summary to console and file"""
        
        summary = f"""
========================================
RISK SCORING SYSTEM TEST SUMMARY
========================================
Duration: {report.duration_seconds:.1f} seconds
Overall Status: {report.overall_status}

Test Results:
  Total: {report.total_tests}
  Passed: {report.passed} ({report.pass_rate:.1%})
  Failed: {report.failed}
  Errors: {report.errors}
  Skipped: {report.skipped}

Performance Summary:"""
        
        for key, value in report.performance_metrics.items():
            if isinstance(value, (int, float)) and not key.startswith('meets_'):
                summary += f"\n  {key.replace('_', ' ').title()}: {value:.2f}"
        
        summary += f"""

Deployment Recommendation:
{report.deployment_recommendation}

========================================
        """
        
        self.logger.info(summary)
        print(summary)


# CLI interface for running tests
def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Risk Scoring System Test Framework")
    parser.add_argument("--categories", nargs="+", 
                       choices=["unit", "integration", "performance", "stress", "accuracy"],
                       help="Test categories to run (default: all enabled)")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first failure")
    
    args = parser.parse_args()
    
    # Override config with CLI args
    config_overrides = {}
    if args.output_dir:
        config_overrides["reporting"] = {"output_directory": args.output_dir}
    
    # Create and run framework
    framework = RiskTestFramework(config_path=args.config)
    
    if config_overrides:
        framework.config.update(config_overrides)
    
    try:
        report = framework.run_test_suite(categories=args.categories)
        
        # Set exit code based on results
        if "BLOCK_DEPLOYMENT" in report.deployment_recommendation:
            sys.exit(1)  # Critical failure
        elif report.overall_status == "FAIL":
            sys.exit(2)  # Test failures
        elif report.overall_status == "ERROR":
            sys.exit(3)  # Test errors
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test framework error: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()