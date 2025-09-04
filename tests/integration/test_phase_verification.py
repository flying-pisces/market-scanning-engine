"""
Phase Verification Tests
Automated verification for each development phase completion
"""

import unittest
import requests
import json
import time
from typing import Dict, List, Tuple
import subprocess
import os


class PhaseVerification:
    """Base class for phase verification"""
    
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def verify(self, test_name: str, condition: bool, message: str = ""):
        """Record verification result"""
        status = "PASS" if condition else "FAIL"
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message
        })
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        
        print(f"[{status}] {test_name}: {message}")
        return condition
    
    def get_summary(self) -> Dict:
        """Get verification summary"""
        return {
            'phase': self.phase_name,
            'total': len(self.results),
            'passed': self.passed,
            'failed': self.failed,
            'success_rate': (self.passed / len(self.results) * 100) if self.results else 0,
            'results': self.results
        }
    
    def print_summary(self):
        """Print verification summary"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print(f"PHASE {self.phase_name} VERIFICATION SUMMARY")
        print("="*50)
        print(f"Total Tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print("="*50)
        
        if summary['success_rate'] >= 90:
            print("✅ PHASE VERIFICATION PASSED")
        else:
            print("❌ PHASE VERIFICATION FAILED")
            print("\nFailed Tests:")
            for result in summary['results']:
                if result['status'] == 'FAIL':
                    print(f"  - {result['test']}: {result['message']}")


class Phase1Verification(PhaseVerification):
    """Verification tests for Phase 1 (MVP)"""
    
    def __init__(self):
        super().__init__("Phase 1 - MVP")
        self.base_url = os.getenv('APP_URL', 'http://localhost:3000')
        self.api_url = f"{self.base_url}/api"
    
    def verify_infrastructure(self):
        """Verify zero-cost infrastructure"""
        print("\n🔍 Verifying Infrastructure...")
        
        # Check Vercel deployment
        try:
            response = requests.get(self.base_url, timeout=10)
            self.verify(
                "Vercel Deployment",
                response.status_code == 200,
                f"Status code: {response.status_code}"
            )
        except:
            self.verify("Vercel Deployment", False, "Failed to connect")
        
        # Check no database connections
        self.verify(
            "No Database",
            not os.path.exists('.env') or 'DATABASE_URL' not in open('.env').read(),
            "No database configuration found"
        )
        
        # Check for IndexedDB usage in frontend
        self.verify(
            "Client-side Storage",
            os.path.exists('src/services/storage.ts'),
            "IndexedDB implementation exists"
        )
    
    def verify_free_apis(self):
        """Verify free API integrations"""
        print("\n🔍 Verifying Free APIs...")
        
        # Test yfinance integration
        try:
            response = requests.get(f"{self.api_url}/market-data?symbols=AAPL")
            data = response.json()
            self.verify(
                "yfinance API",
                response.status_code == 200 and 'AAPL' in data,
                "Successfully fetched AAPL data"
            )
        except:
            self.verify("yfinance API", False, "API call failed")
        
        # Check for rate limiting
        start_time = time.time()
        for _ in range(5):
            requests.get(f"{self.api_url}/market-data?symbols=MSFT")
        elapsed = time.time() - start_time
        
        self.verify(
            "Rate Limiting",
            elapsed > 1,  # Should take more than 1 second for 5 requests
            f"Rate limiting active (took {elapsed:.2f}s)"
        )
    
    def verify_features(self):
        """Verify MVP features"""
        print("\n🔍 Verifying MVP Features...")
        
        # Test market scanning
        try:
            response = requests.get(f"{self.api_url}/scan")
            data = response.json()
            self.verify(
                "Market Scanner",
                len(data.get('stocks', [])) >= 50,
                f"Scanning {len(data.get('stocks', []))} stocks"
            )
        except:
            self.verify("Market Scanner", False, "Scanner API failed")
        
        # Test signal generation
        try:
            response = requests.get(f"{self.api_url}/signals?symbol=AAPL")
            data = response.json()
            self.verify(
                "Signal Generation",
                'signal' in data and data['signal'] in ['BUY', 'SELL', 'HOLD'],
                f"Generated signal: {data.get('signal')}"
            )
        except:
            self.verify("Signal Generation", False, "Signal API failed")
        
        # Test technical indicators
        indicators = ['SMA', 'RSI', 'MACD']
        for indicator in indicators:
            try:
                response = requests.get(f"{self.api_url}/indicators/{indicator}?symbol=AAPL")
                self.verify(
                    f"{indicator} Indicator",
                    response.status_code == 200,
                    "Indicator calculated successfully"
                )
            except:
                self.verify(f"{indicator} Indicator", False, "Calculation failed")
    
    def verify_performance(self):
        """Verify performance requirements"""
        print("\n🔍 Verifying Performance...")
        
        # Test page load time
        start = time.time()
        response = requests.get(self.base_url)
        load_time = time.time() - start
        
        self.verify(
            "Page Load Time",
            load_time < 3,
            f"Load time: {load_time:.2f}s"
        )
        
        # Test signal calculation time
        start = time.time()
        requests.get(f"{self.api_url}/signals?symbol=AAPL")
        calc_time = time.time() - start
        
        self.verify(
            "Signal Calculation",
            calc_time < 1,
            f"Calculation time: {calc_time:.2f}s"
        )
    
    def verify_pwa(self):
        """Verify PWA functionality"""
        print("\n🔍 Verifying PWA...")
        
        # Check manifest file
        self.verify(
            "PWA Manifest",
            os.path.exists('public/manifest.json'),
            "Manifest file exists"
        )
        
        # Check service worker
        self.verify(
            "Service Worker",
            os.path.exists('src/serviceWorker.ts'),
            "Service worker implemented"
        )
    
    def run_all(self):
        """Run all Phase 1 verification tests"""
        print(f"\n{'='*50}")
        print(f"STARTING PHASE 1 VERIFICATION")
        print(f"{'='*50}")
        
        self.verify_infrastructure()
        self.verify_free_apis()
        self.verify_features()
        self.verify_performance()
        self.verify_pwa()
        
        self.print_summary()
        return self.get_summary()


class Phase2Verification(PhaseVerification):
    """Verification tests for Phase 2 (Premium)"""
    
    def __init__(self):
        super().__init__("Phase 2 - Premium")
        self.base_url = os.getenv('APP_URL', 'http://localhost:3000')
        self.api_url = f"{self.base_url}/api"
    
    def verify_payment_system(self):
        """Verify payment integration"""
        print("\n🔍 Verifying Payment System...")
        
        # Check RevenueCat integration
        self.verify(
            "RevenueCat SDK",
            os.path.exists('src/services/payments.ts'),
            "Payment service implemented"
        )
        
        # Test credit purchase flow (sandbox)
        try:
            response = requests.post(f"{self.api_url}/purchase", json={
                'package': 'test_100_credits',
                'sandbox': True
            })
            self.verify(
                "Credit Purchase",
                response.status_code == 200,
                "Sandbox purchase successful"
            )
        except:
            self.verify("Credit Purchase", False, "Purchase API failed")
        
        # Test credit balance check
        try:
            response = requests.get(f"{self.api_url}/credits/balance")
            self.verify(
                "Credit Balance",
                'balance' in response.json(),
                f"Balance API working"
            )
        except:
            self.verify("Credit Balance", False, "Balance API failed")
    
    def verify_lambda_functions(self):
        """Verify Lambda infrastructure"""
        print("\n🔍 Verifying Lambda Functions...")
        
        # Check Lambda deployment
        lambda_functions = [
            'premium-signal',
            'options-flow',
            'sentiment-analysis'
        ]
        
        for func in lambda_functions:
            try:
                response = requests.post(f"{self.api_url}/lambda/{func}", json={
                    'symbol': 'AAPL',
                    'test': True
                })
                self.verify(
                    f"Lambda: {func}",
                    response.status_code in [200, 402],  # 402 for insufficient credits
                    f"Lambda function responding"
                )
            except:
                self.verify(f"Lambda: {func}", False, "Lambda not deployed")
    
    def verify_premium_features(self):
        """Verify premium features"""
        print("\n🔍 Verifying Premium Features...")
        
        # Test advanced indicators
        premium_indicators = ['bollinger', 'fibonacci', 'ichimoku']
        for indicator in premium_indicators:
            self.verify(
                f"Premium: {indicator}",
                os.path.exists(f'src/services/premium/{indicator}.ts'),
                "Premium indicator implemented"
            )
        
        # Test options flow data
        try:
            response = requests.get(f"{self.api_url}/premium/options-flow?symbol=AAPL")
            self.verify(
                "Options Flow",
                response.status_code in [200, 402],
                "Options flow API available"
            )
        except:
            self.verify("Options Flow", False, "Options API failed")
    
    def verify_scalability(self):
        """Verify scalability improvements"""
        print("\n🔍 Verifying Scalability...")
        
        # Test concurrent requests
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{self.api_url}/signals?symbol=AAPL")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        success_count = sum(1 for r in results if r.status_code == 200)
        self.verify(
            "Concurrent Requests",
            success_count >= 8,
            f"{success_count}/10 requests successful"
        )
    
    def run_all(self):
        """Run all Phase 2 verification tests"""
        print(f"\n{'='*50}")
        print(f"STARTING PHASE 2 VERIFICATION")
        print(f"{'='*50}")
        
        self.verify_payment_system()
        self.verify_lambda_functions()
        self.verify_premium_features()
        self.verify_scalability()
        
        self.print_summary()
        return self.get_summary()


class Phase3Verification(PhaseVerification):
    """Verification tests for Phase 3 (Pro)"""
    
    def __init__(self):
        super().__init__("Phase 3 - Pro")
        self.base_url = os.getenv('APP_URL', 'http://localhost:3000')
        self.ws_url = os.getenv('WS_URL', 'ws://localhost:8080')
    
    def verify_websocket(self):
        """Verify WebSocket streaming"""
        print("\n🔍 Verifying WebSocket...")
        
        try:
            import websocket
            
            def on_message(ws, message):
                data = json.loads(message)
                self.verify(
                    "WebSocket Message",
                    'symbol' in data and 'price' in data,
                    "Receiving market data"
                )
                ws.close()
            
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=on_message
            )
            
            # Test connection (timeout after 5 seconds)
            import threading
            timer = threading.Timer(5.0, ws.close)
            timer.start()
            ws.run_forever()
            
        except:
            self.verify("WebSocket Connection", False, "Failed to connect")
    
    def verify_real_time_data(self):
        """Verify real-time data streaming"""
        print("\n🔍 Verifying Real-time Data...")
        
        # Test latency
        latencies = []
        for _ in range(5):
            start = time.time()
            response = requests.get(f"{self.base_url}/api/realtime/quote?symbol=AAPL")
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        self.verify(
            "Streaming Latency",
            avg_latency < 100,
            f"Average latency: {avg_latency:.2f}ms"
        )
    
    def verify_pro_features(self):
        """Verify pro tier features"""
        print("\n🔍 Verifying Pro Features...")
        
        # Test custom alerts
        try:
            response = requests.post(f"{self.base_url}/api/alerts", json={
                'symbol': 'AAPL',
                'condition': 'price > 150',
                'action': 'email'
            })
            self.verify(
                "Custom Alerts",
                response.status_code == 201,
                "Alert created successfully"
            )
        except:
            self.verify("Custom Alerts", False, "Alert API failed")
        
        # Test backtesting
        try:
            response = requests.post(f"{self.base_url}/api/backtest", json={
                'strategy': 'sma_crossover',
                'period': '1Y',
                'symbols': ['AAPL', 'MSFT']
            })
            self.verify(
                "Backtesting Engine",
                response.status_code == 200 and 'results' in response.json(),
                "Backtest completed"
            )
        except:
            self.verify("Backtesting Engine", False, "Backtest API failed")
        
        # Test API access
        headers = {'X-API-Key': 'test_key'}
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/market/scan",
                headers=headers
            )
            self.verify(
                "API Access",
                response.status_code in [200, 401],
                "API endpoint available"
            )
        except:
            self.verify("API Access", False, "API not accessible")
    
    def verify_reliability(self):
        """Verify reliability metrics"""
        print("\n🔍 Verifying Reliability...")
        
        # Test uptime (simplified)
        failures = 0
        for _ in range(10):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code != 200:
                    failures += 1
            except:
                failures += 1
            time.sleep(0.5)
        
        uptime = ((10 - failures) / 10) * 100
        self.verify(
            "Uptime Check",
            uptime >= 99,
            f"Uptime: {uptime}%"
        )
    
    def run_all(self):
        """Run all Phase 3 verification tests"""
        print(f"\n{'='*50}")
        print(f"STARTING PHASE 3 VERIFICATION")
        print(f"{'='*50}")
        
        self.verify_websocket()
        self.verify_real_time_data()
        self.verify_pro_features()
        self.verify_reliability()
        
        self.print_summary()
        return self.get_summary()


def main():
    """Main verification runner"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_phase_verification.py [phase1|phase2|phase3|all]")
        sys.exit(1)
    
    phase = sys.argv[1].lower()
    
    results = []
    
    if phase in ['phase1', 'all']:
        verifier = Phase1Verification()
        results.append(verifier.run_all())
    
    if phase in ['phase2', 'all']:
        verifier = Phase2Verification()
        results.append(verifier.run_all())
    
    if phase in ['phase3', 'all']:
        verifier = Phase3Verification()
        results.append(verifier.run_all())
    
    # Final summary
    if results:
        print("\n" + "="*60)
        print("OVERALL VERIFICATION SUMMARY")
        print("="*60)
        
        total_passed = sum(r['passed'] for r in results)
        total_failed = sum(r['failed'] for r in results)
        total_tests = sum(r['total'] for r in results)
        
        for result in results:
            print(f"\n{result['phase']}:")
            print(f"  Success Rate: {result['success_rate']:.1f}%")
            print(f"  Status: {'✅ PASSED' if result['success_rate'] >= 90 else '❌ FAILED'}")
        
        print(f"\nTotal: {total_passed}/{total_tests} tests passed")
        
        if all(r['success_rate'] >= 90 for r in results):
            print("\n🎉 ALL PHASES VERIFIED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n⚠️  SOME PHASES FAILED VERIFICATION")
            sys.exit(1)


if __name__ == "__main__":
    main()