import pandas as pd
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import time

class NewsCollector:
    def __init__(self):
        self.rss_feeds = {
            'reuters': 'https://www.rss.com/blog/reuters-rss-feeds/',
            'yahoo_finance': 'https://finance.yahoo.com/news/rssindex',
            'marketwatch': 'http://feeds.marketwatch.com/marketwatch/topstories/'
        }
    
    def fetch_news(self, keywords, start_date, end_date):
        articles = []
        
        for source, feed_url in self.rss_feeds.items():
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6])
                
                if start_date <= pub_date <= end_date:
                    if any(keyword.lower() in entry.title.lower() or 
                           keyword.lower() in entry.description.lower() 
                           for keyword in keywords):
                        articles.append({
                            'title': entry.title,
                            'description': entry.description,
                            'content': entry.content[0].value if 'content' in entry else entry.description,
                            'published_at': pub_date,
                            'source': source
                        })
            
            time.sleep(1)  # Be nice to servers
        
        return pd.DataFrame(articles)