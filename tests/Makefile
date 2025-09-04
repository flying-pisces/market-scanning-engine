# Market Scanner Test Automation Makefile
# Simplifies test execution and verification

.PHONY: all test verify clean help

# Default target
all: help

# Help command
help:
	@echo "Market Scanner Test Commands:"
	@echo "================================"
	@echo "make test-unit        - Run unit tests"
	@echo "make test-integration - Run integration tests"
	@echo "make test-e2e        - Run end-to-end tests"
	@echo "make test-all        - Run all tests"
	@echo ""
	@echo "Phase Verification:"
	@echo "make verify-phase1   - Verify Phase 1 (MVP)"
	@echo "make verify-phase2   - Verify Phase 2 (Premium)"
	@echo "make verify-phase3   - Verify Phase 3 (Pro)"
	@echo "make verify-all      - Verify all phases"
	@echo ""
	@echo "Utilities:"
	@echo "make clean          - Clean test artifacts"
	@echo "make report         - Generate test report"
	@echo "make coverage       - Run coverage analysis"

# Test commands
test-unit:
	@echo "Running unit tests..."
	@npm test -- --coverage --watchAll=false
	@python3 -m pytest unit/ -v

test-integration:
	@echo "Running integration tests..."
	@python3 -m pytest integration/ -v

test-e2e:
	@echo "Running E2E tests..."
	@npm run test:e2e

test-all: test-unit test-integration test-e2e
	@echo "All tests completed!"

# Phase verification
verify-phase1:
	@echo "Verifying Phase 1 (MVP)..."
	@./run_tests.sh phase1
	@python3 integration/test_phase_verification.py phase1

verify-phase2:
	@echo "Verifying Phase 2 (Premium)..."
	@./run_tests.sh phase2
	@python3 integration/test_phase_verification.py phase2

verify-phase3:
	@echo "Verifying Phase 3 (Pro)..."
	@./run_tests.sh phase3
	@python3 integration/test_phase_verification.py phase3

verify-all:
	@echo "Running complete verification..."
	@python3 integration/test_phase_verification.py all

# Utilities
clean:
	@echo "Cleaning test artifacts..."
	@rm -rf logs/*.log
	@rm -rf coverage/
	@rm -rf .pytest_cache/
	@rm -rf __pycache__/
	@echo "Clean complete!"

report:
	@echo "Generating test report..."
	@python3 integration/test_phase_verification.py all > logs/verification_report_$(shell date +%Y%m%d_%H%M%S).txt
	@echo "Report saved to logs/"

coverage:
	@echo "Running coverage analysis..."
	@npm test -- --coverage
	@python3 -m pytest --cov=. --cov-report=html
	@echo "Coverage reports generated!"

# Installation check
check-deps:
	@echo "Checking dependencies..."
	@command -v node >/dev/null 2>&1 || { echo "Node.js not installed"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "Python 3 not installed"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "npm not installed"; exit 1; }
	@echo "All dependencies found!"

# Quick start for development
setup:
	@echo "Setting up test environment..."
	@npm install
	@pip3 install -r requirements-test.txt
	@chmod +x run_tests.sh
	@mkdir -p logs
	@echo "Setup complete!"