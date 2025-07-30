# test_api_adapters.py
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import os

# Load environment variables FIRST
load_dotenv()

# NOW import the modules that need env vars
from api_adapters.yfinance_adapter import YFinanceAdapter
from api_adapters.polygon_adapter import PolygonAdapter
from api_adapters.newsapi_adapter import NewsAPIAdapter
from api_config import api_config

# Setup logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

def test_adapter(adapter, adapter_name, ticker="AAPL", use_recent=True):
    """Test a single adapter"""
    print(f"\n{'='*50}")
    print(f"Testing {adapter_name}...")
    print('='*50)
    
    # Test connection
    if adapter.test_connection():
        print("✓ Connection successful")
    else:
        print("✗ Connection failed")
        return
    
    # Test news fetch
    if use_recent:
        date = datetime.now()
    else:
        date = datetime.now() - timedelta(days=7)
    
    print(f"Fetching news for {ticker} around {date.strftime('%Y-%m-%d')}")
    articles = adapter.get_news(ticker, date)
    print(f"\nFound {len(articles)} articles")
    
    # Show sample articles
    for i, article in enumerate(articles[:3], 1):
        print(f"\nArticle {i}:")
        print(f"  Title: {article['title'][:80]}...")
        print(f"  Date: {article['published_date']}")
        print(f"  Source: {article['source']}")
        if article.get('summary'):
            print(f"  Summary: {article['summary'][:100]}...")

def main():
    print("Testing News API Adapters")
    print("="*50)
    
    # Test YFinance
    print("\n1. Testing YFinance (Recent news only)")
    adapter = YFinanceAdapter()
    test_adapter(adapter, "YFinance", use_recent=True)
    
    # Test Polygon
    polygon_key = api_config.keys.get('polygon')
    if polygon_key:
        print("\n2. Testing Polygon (Historical + Recent)")
        adapter = PolygonAdapter(polygon_key)
        test_adapter(adapter, "Polygon.io", use_recent=False)
    
    # Test NewsAPI
    newsapi_key = api_config.keys.get('newsapi')
    if newsapi_key:
        print("\n3. Testing NewsAPI (Last 30 days)")
        adapter = NewsAPIAdapter(newsapi_key)
        test_adapter(adapter, "NewsAPI", use_recent=True)
    
    print("\n" + "="*50)
    print("Testing complete!")
    print("\nSummary:")
    print("- YFinance: Best for real-time news (last 1-3 days)")
    print("- Polygon: Best for historical analysis")
    print("- NewsAPI: Good for recent news (last 30 days)")

if __name__ == "__main__":
    main()