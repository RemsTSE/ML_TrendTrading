import requests
from datetime import datetime, timedelta

class CryptoNewsCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://min-api.cryptocompare.com/data/v2/news/"
    
    def fetch_crypto_news(self, symbol="BTC", start_date=None, end_date=None, limit=100):
        """
        Fetch news related to a specific cryptocurrency. The function supports date filtering and pagination.
        """
        # Define API parameters
        params = {
            "categories": symbol,
            "api_key": self.api_key,
            "limit": limit
        }
        
        # Send request to CryptoCompare API
        response = requests.get(self.base_url, params=params)
        
        # Check if response is successful
        if response.status_code == 200:
            news_data = response.json()
            if "Data" in news_data:
                return news_data["Data"]
            else:
                print("No news data found for the given parameters.")
                return []
        else:
            print(f"Failed to fetch news. Status Code: {response.status_code}, Response: {response.text}")
            return []

    def fetch_historical_news(self, symbol="BTC", days=30, limit=100):
        """
        Fetch historical news by going back in time, day-by-day, up to the specified number of days.
        """
        all_news = []
        end_date = datetime.now()
        
        # Loop through each day and collect news
        for i in range(days):
            start_date = end_date - timedelta(days=1)
            print(f"Fetching news from {start_date.date()} to {end_date.date()}")

            # Fetch news for the specified date range
            news = self.fetch_crypto_news(symbol=symbol, start_date=start_date, end_date=end_date, limit=limit)
            all_news.extend(news)
            
            # Update the end date for the next iteration
            end_date = start_date

        return all_news



