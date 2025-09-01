#!/usr/bin/env python3
"""
Market Scanning Engine MVP Startup Script
Quick start script for development and testing
"""

import asyncio
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import fastapi
        import uvicorn
        import sqlalchemy
        import asyncpg
        import pydantic
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def run_tests():
    """Run MVP tests before starting"""
    print("\n🧪 Running MVP tests...")
    
    result = subprocess.run([
        sys.executable, "run_mvp_test.py"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Tests failed:")
        print(result.stdout)
        print(result.stderr)
        return False

def start_development_server():
    """Start the FastAPI development server"""
    print("\n🚀 Starting Market Scanning Engine...")
    print("=" * 50)
    
    # Start the server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def show_startup_info():
    """Display startup information and available endpoints"""
    print("""
🎯 Market Scanning Engine MVP - Ready!

📋 Available Endpoints:
   • GET    /health                    - Health check
   • GET    /docs                      - Interactive API docs
   • GET    /api/v1/                   - API information
   
   👥 Users:
   • POST   /api/v1/users              - Create user
   • GET    /api/v1/users              - List users  
   • GET    /api/v1/users/{id}         - Get user
   • PUT    /api/v1/users/{id}         - Update user
   • GET    /api/v1/users/{id}/stats   - User statistics
   
   📊 Signals:
   • POST   /api/v1/signals            - Create signal
   • GET    /api/v1/signals            - List signals
   • GET    /api/v1/signals/{id}       - Get signal
   • POST   /api/v1/signals/generate   - Generate signals
   
   🔗 Matching:
   • POST   /api/v1/matching/signals/{id}/users     - Find users for signal
   • POST   /api/v1/matching/users/{id}/signals     - Find signals for user
   • POST   /api/v1/matching/create-match           - Create match record

🎨 Features:
   • 0-100 risk scoring system
   • Multi-asset class support (options, stocks, bonds, safe assets)
   • Intelligent user-signal matching
   • Real-time signal generation
   • RESTful API with async operations

📖 Usage Examples:
   
   1. Create a conservative user:
   curl -X POST http://localhost:8000/api/v1/users \\
     -H "Content-Type: application/json" \\
     -d '{"email":"conservative@example.com","risk_tolerance":20,"asset_preferences":{"bonds":0.7,"safe_assets":0.3}}'
   
   2. Generate signals for safe assets:
   curl -X POST http://localhost:8000/api/v1/signals/generate \\
     -H "Content-Type: application/json" \\
     -d '{"asset_classes":["safe_assets"],"max_signals":10}'
   
   3. Find matching signals for user:
   curl -X POST http://localhost:8000/api/v1/matching/users/{user_id}/signals

🌐 Web Interface:
   Open http://localhost:8000/docs for interactive API documentation

Press Ctrl+C to stop the server
""")

def main():
    """Main startup sequence"""
    print("🚀 Market Scanning Engine MVP Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests
    print("\n🧪 Running validation tests...")
    if not run_tests():
        response = input("\n⚠️  Tests failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(1)
    
    # Show startup info
    show_startup_info()
    
    # Start server
    start_development_server()

if __name__ == "__main__":
    main()