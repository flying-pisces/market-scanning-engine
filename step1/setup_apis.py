#!/usr/bin/env python3
"""
Setup script for API configuration
Helps configure Alpaca and Finnhub API keys
"""

import os
import sys
from pathlib import Path


def setup_environment_file():
    """Create .env file with API key templates"""
    env_file = Path('.env')
    
    if env_file.exists():
        print("📄 .env file already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if 'ALPACA_API_KEY' in content:
                print("✅ Alpaca keys already configured")
            if 'FINNHUB_API_KEY' in content:
                print("✅ Finnhub key already configured")
        return
    
    env_template = """# Step 1 Market Scanner API Keys
# Get free API keys from the providers below

# Alpaca Markets API (Free tier with IEX data + after hours)
# Sign up at: https://alpaca.markets/
# Free tier: 200 requests/minute, IEX data, extended hours support
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# Finnhub API (Free tier with real-time data)
# Sign up at: https://finnhub.io/
# Free tier: 60 requests/minute, real-time quotes, after hours data
FINNHUB_API_KEY=your_finnhub_api_key_here

# Optional: Other API keys for future use
# POLYGON_API_KEY=your_polygon_api_key_here
# ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
"""
    
    with open(env_file, 'w') as f:
        f.write(env_template)
    
    print("📄 Created .env file template")
    print("📝 Edit .env file with your actual API keys")


def test_alpaca_connection():
    """Test Alpaca API connection"""
    try:
        sys.path.append('src')
        from alpaca_fetcher import AlpacaDataFetcher
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key or api_key == 'your_alpaca_api_key_here':
            print("❌ Alpaca API keys not configured")
            print("   Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
            return False
        
        fetcher = AlpacaDataFetcher(api_key, secret_key)
        if fetcher.test_connection():
            print("✅ Alpaca API connection successful")
            
            # Test real-time data
            data = fetcher.get_real_time_data('AAPL')
            if data:
                print(f"   Real-time AAPL: ${data['price']:.2f} ({data['change']:+.2f})")
            return True
        else:
            print("❌ Alpaca API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Alpaca test error: {e}")
        return False


def test_finnhub_connection():
    """Test Finnhub API connection"""
    try:
        sys.path.append('src')
        from finnhub_fetcher import FinnhubDataFetcher
        
        api_key = os.getenv('FINNHUB_API_KEY')
        
        if not api_key or api_key == 'your_finnhub_api_key_here':
            print("❌ Finnhub API key not configured")
            print("   Set FINNHUB_API_KEY in .env file")
            return False
        
        fetcher = FinnhubDataFetcher(api_key)
        if fetcher.test_connection():
            print("✅ Finnhub API connection successful")
            
            # Test real-time data
            data = fetcher.get_real_time_data('AAPL')
            if data:
                print(f"   Real-time AAPL: ${data['price']:.2f} ({data['change']:+.2f})")
            return True
        else:
            print("❌ Finnhub API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Finnhub test error: {e}")
        return False


def test_enhanced_fetcher():
    """Test enhanced fetcher with all available APIs"""
    try:
        sys.path.append('src')
        from enhanced_data_fetcher import EnhancedDataFetcher
        
        print("\n🔧 Testing Enhanced Data Fetcher...")
        fetcher = EnhancedDataFetcher()
        
        capabilities = fetcher.get_capabilities()
        print(f"📊 Available APIs: {list(fetcher.fetchers.keys())}")
        print(f"🕐 After-hours capable: {capabilities['after_hours']}")
        print(f"⚡ Real-time capable: {capabilities['real_time']}")
        
        # Test real-time data
        data = fetcher.get_real_time_data('AAPL')
        if data:
            print(f"✅ Real-time AAPL: ${data['price']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced fetcher test error: {e}")
        return False


def print_api_instructions():
    """Print instructions for getting API keys"""
    print("\n" + "="*60)
    print("🔑 FREE API KEY SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n📈 ALPACA MARKETS (Recommended for after-hours)")
    print("   🌐 Website: https://alpaca.markets/")
    print("   ✅ Free tier: 200 requests/minute")
    print("   ✅ IEX real-time data (2% of market volume)")
    print("   ✅ Extended hours support (9:00-9:30 AM, 4:00-6:00 PM EST)")
    print("   ✅ After-hours trading data")
    print("   📝 Steps:")
    print("      1. Sign up for free paper trading account")
    print("      2. Go to 'API Keys' in dashboard")
    print("      3. Generate API key and secret")
    print("      4. Add to .env file")
    
    print("\n📊 FINNHUB (Best for real-time quotes)")
    print("   🌐 Website: https://finnhub.io/")
    print("   ✅ Free tier: 60 requests/minute")
    print("   ✅ Real-time stock quotes")
    print("   ✅ After-hours data included")
    print("   ✅ Market news and earnings data")
    print("   📝 Steps:")
    print("      1. Sign up for free account")
    print("      2. Go to dashboard and copy API key")
    print("      3. Add to .env file")
    
    print("\n⚡ QUICK START")
    print("   1. Run: python setup_apis.py")
    print("   2. Edit .env file with your API keys")
    print("   3. Run: python setup_apis.py test")
    print("   4. Run: python demo.py")
    
    print("\n💡 FOR YOUR ALPACA KEYS:")
    print("   You mentioned you have Alpaca API key and secret")
    print("   Add them to the .env file to enable after-hours data!")


def main():
    """Main setup function"""
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test existing configuration
        print("🧪 Testing API connections...")
        
        # Load environment variables from .env file
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        if value and value != f'your_{key.lower()}_here':
                            os.environ[key] = value
        
        alpaca_ok = test_alpaca_connection()
        finnhub_ok = test_finnhub_connection()
        
        if alpaca_ok or finnhub_ok:
            test_enhanced_fetcher()
            print("\n🎉 At least one API is working!")
            print("   Your scanner now has access to after-hours data!")
        else:
            print("\n❌ No APIs configured. See instructions below.")
            print_api_instructions()
        
    else:
        # Setup mode
        print("🚀 Step 1 API Setup")
        print("="*40)
        
        setup_environment_file()
        print_api_instructions()
        
        print("\n🔄 After adding your API keys, run:")
        print("   python setup_apis.py test")


if __name__ == "__main__":
    main()