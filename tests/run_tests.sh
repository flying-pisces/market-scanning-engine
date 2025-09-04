#!/bin/bash

# Market Scanner Test Runner
# Automated test execution script with logging

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="$(dirname "$0")"
LOG_DIR="$TEST_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_run_$TIMESTAMP.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "${2}[${1}]${NC} ${3}" | tee -a "$LOG_FILE"
}

# Function to run test suite
run_test_suite() {
    local suite_name=$1
    local test_command=$2
    
    print_status "INFO" "$YELLOW" "Running $suite_name..."
    
    if eval "$test_command" >> "$LOG_FILE" 2>&1; then
        print_status "PASS" "$GREEN" "$suite_name completed successfully"
        return 0
    else
        print_status "FAIL" "$RED" "$suite_name failed"
        return 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "INFO" "$YELLOW" "Checking prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_status "ERROR" "$RED" "Node.js is not installed"
        return 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_status "ERROR" "$RED" "Python 3 is not installed"
        return 1
    fi
    
    # Check npm packages
    if [ ! -d "node_modules" ]; then
        print_status "WARN" "$YELLOW" "Node modules not installed. Running npm install..."
        npm install >> "$LOG_FILE" 2>&1
    fi
    
    # Check Python packages
    if ! python3 -c "import yfinance" 2>/dev/null; then
        print_status "WARN" "$YELLOW" "Python packages not installed. Installing..."
        pip3 install yfinance pytest pytest-cov >> "$LOG_FILE" 2>&1
    fi
    
    print_status "PASS" "$GREEN" "Prerequisites check completed"
    return 0
}

# Main test execution
main() {
    echo "=====================================" | tee "$LOG_FILE"
    echo "Market Scanner Test Suite" | tee -a "$LOG_FILE"
    echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
    echo "=====================================" | tee -a "$LOG_FILE"
    
    # Check prerequisites
    if ! check_prerequisites; then
        print_status "ERROR" "$RED" "Prerequisites check failed. Exiting."
        exit 1
    fi
    
    # Initialize test results
    TOTAL_TESTS=0
    PASSED_TESTS=0
    FAILED_TESTS=0
    
    # Phase 1 Tests (MVP)
    if [ "$1" == "phase1" ] || [ -z "$1" ]; then
        print_status "INFO" "$YELLOW" "=== PHASE 1 TESTS ==="
        
        # Unit tests
        if run_test_suite "Unit Tests" "npm test -- --coverage --watchAll=false"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
        
        # API tests
        if run_test_suite "API Tests" "python3 -m pytest tests/unit/test_api.py -v"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
        
        # Storage tests
        if run_test_suite "Storage Tests" "npm test -- tests/unit/storage.test.ts"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
    
    # Phase 2 Tests (Premium)
    if [ "$1" == "phase2" ]; then
        print_status "INFO" "$YELLOW" "=== PHASE 2 TESTS ==="
        
        # Payment integration tests
        if run_test_suite "Payment Tests" "npm test -- tests/integration/payment.test.ts"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
        
        # Lambda function tests
        if run_test_suite "Lambda Tests" "python3 -m pytest tests/integration/test_lambda.py -v"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
    
    # Phase 3 Tests (Pro)
    if [ "$1" == "phase3" ]; then
        print_status "INFO" "$YELLOW" "=== PHASE 3 TESTS ==="
        
        # WebSocket tests
        if run_test_suite "WebSocket Tests" "npm test -- tests/integration/websocket.test.ts"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
        
        # Performance tests
        if run_test_suite "Performance Tests" "npm run test:performance"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
    
    # End-to-end tests
    if [ "$1" == "e2e" ] || [ "$1" == "all" ]; then
        print_status "INFO" "$YELLOW" "=== E2E TESTS ==="
        
        if run_test_suite "E2E Tests" "npm run test:e2e"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
    
    # Print test summary
    echo "=====================================" | tee -a "$LOG_FILE"
    echo "TEST SUMMARY" | tee -a "$LOG_FILE"
    echo "=====================================" | tee -a "$LOG_FILE"
    echo "Total Tests: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo "Passed: $PASSED_TESTS" | tee -a "$LOG_FILE"
    echo "Failed: $FAILED_TESTS" | tee -a "$LOG_FILE"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        print_status "SUCCESS" "$GREEN" "All tests passed!"
        echo "Log file: $LOG_FILE"
        exit 0
    else
        print_status "FAILURE" "$RED" "Some tests failed. Check log for details."
        echo "Log file: $LOG_FILE"
        exit 1
    fi
}

# Parse command line arguments
case "$1" in
    phase1|phase2|phase3|e2e|all)
        main "$1"
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|phase3|e2e|all]"
        echo "  phase1 - Run Phase 1 (MVP) tests"
        echo "  phase2 - Run Phase 2 (Premium) tests"
        echo "  phase3 - Run Phase 3 (Pro) tests"
        echo "  e2e    - Run end-to-end tests"
        echo "  all    - Run all tests"
        echo "  (none) - Run Phase 1 tests by default"
        main ""
        ;;
esac