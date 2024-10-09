import pandas as pd
import requests
from datetime import datetime

class NewsCollector:
    def __init__(self, api_key):
        # Set your API key and base URL for NewsAPI
        self.api_key = api_key
        self.base_url = 'https://newsapi.org/v2/everything'

    def fetch_news(self, keywords, start_date, end_date):
        # Convert dates to the required format
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Prepare query parameters
        query = ' OR '.join(keywords)  # Use OR to search for multiple keywords
        params = {
            'q': query,
            'from': start_date_str,
            'to': end_date_str,
            'sortBy': 'relevancy',
            'apiKey': self.api_key,
            'language': 'en',
            'pageSize': 100  # Maximum results per request
        }

        # Fetch news articles
        print(f"Fetching news for keywords: {keywords} from {start_date_str} to {end_date_str}")
        response = requests.get(self.base_url, params=params)
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            print(f"Found {len(articles)} articles")
            
            # Parse and return the news data as a DataFrame
            news_data = pd.DataFrame([
                {
                    'title': article['title'],
                    'content': article['content'],
                    'description': article['description'],
                    'published_at': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    'source': article['source']['name'],
                    'url': article['url'],
                    'author': article['author']
                }
                for article in articles
            ])
            
            return news_data
        else:
            print(f"Failed to fetch news: {response.status_code} - {response.text}")
            return pd.DataFrame()  # Return an empty DataFrame if the request fails

