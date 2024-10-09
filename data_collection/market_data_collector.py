import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
from io import StringIO

class MarketDataCollector:
    def __init__(self):
        self.alpha_vantage_api_key = 'demo'  # Using Alpha Vantage's demo API key
    
    def fetch_market_data(self, symbol, start_date, end_date):
        # Try multiple sources in case one fails
        try:
            return self._fetch_yahoo_finance(symbol, start_date, end_date)
        except:
            try:
                return self._fetch_alpha_vantage(symbol)
            except:
                return self._fetch_stooq(symbol, start_date, end_date)
    
    def _fetch_yahoo_finance(self, symbol, start_date, end_date):
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        return df
    
    def _fetch_alpha_vantage(self, symbol):
        ts = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        return data
    
    def _fetch_stooq(self, symbol, start_date, end_date):
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}&i=d"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text), parse_dates=['Date'], index_col='Date')
        return df