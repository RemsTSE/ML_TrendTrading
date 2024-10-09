# test_news.py

from data_collection.news_collector import NewsCollector
from datetime import datetime, timedelta
import time

def test_news_collection():
    collector = NewsCollector()
    
    # Test different time ranges
    time_ranges = [7, 30, 90]
    
    # Test different keyword sets
    keyword_sets = [
        ['AAPL', 'Apple', 'iPhone'],
        ['tech', 'technology', 'smartphone'],
        ['market', 'stock', 'trading']
    ]
    
    for days in time_ranges:
        for keywords in keyword_sets:
            print(f"\n{'='*50}")
            print(f"Testing with {days} days and keywords: {keywords}")
            print('='*50)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            news_data = collector.fetch_news(keywords, start_date, end_date)
            
            if not news_data.empty:
                print(f"\nFound {len(news_data)} articles")
                print("\nSample titles:")
                for title in news_data['title'].head():
                    print(f"- {title}")
            
            print(f"\nWaiting a few seconds before next test...")
            time.sleep(3)

if __name__ == "__main__":
    test_news_collection()