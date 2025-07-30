# debug_yfinance_dates.py
import yfinance as yf
from datetime import datetime, timedelta

def debug_yfinance_dates():
    print("Debugging YFinance Date Issues")
    print("="*50)
    
    ticker = yf.Ticker("AAPL")
    news = ticker.news
    
    print(f"Total articles found: {len(news)}")
    print(f"Current date: {datetime.now()}")
    print(f"Current date (date only): {datetime.now().date()}")
    
    if news:
        print("\nArticle dates:")
        for i, article in enumerate(news[:5], 1):
            timestamp = article.get('providerPublishTime', 0)
            if timestamp:
                pub_date = datetime.fromtimestamp(timestamp)
                age_days = (datetime.now() - pub_date).days
                print(f"\nArticle {i}:")
                print(f"  Title: {article.get('title', 'N/A')[:60]}...")
                print(f"  Published: {pub_date}")
                print(f"  Published date only: {pub_date.date()}")
                print(f"  Age: {age_days} days")
                
                # Test date comparison
                current_date = datetime.now()
                days_diff = abs((current_date.date() - pub_date.date()).days)
                print(f"  Days difference: {days_diff}")
                print(f"  Would include (<=7 days)? {days_diff <= 7}")

if __name__ == "__main__":
    debug_yfinance_dates()