from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from config import Config

class FeatureEngineer:
    def __init__(self, max_features=Config.MODEL_PARAMS['max_features']):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.pca = PCA(n_components=50)
    
    def engineer_features(self, news_df, market_df):
        # Ensure that 'published_at' in news_df is converted to datetime
        news_df['published_at'] = pd.to_datetime(news_df['published_at'], errors='coerce')

        # Text features
        text_features = self.vectorizer.fit_transform(news_df['processed_text'])
        
        # Reduce dimensionality
        text_features_reduced = self.pca.fit_transform(text_features.toarray())
        
        # Create feature names
        feature_names = [f'text_pca_{i}' for i in range(text_features_reduced.shape[1])]
        
        # Convert to DataFrame
        daily_features = pd.DataFrame(
            text_features_reduced,
            columns=feature_names,
            index=news_df['published_at']
        )
        
        # Add sentiment features
        sentiment_features = pd.DataFrame(
            news_df['sentiment'].tolist(),
            index=news_df['published_at']
        )
        
        # Combine features
        daily_features = pd.concat([daily_features, sentiment_features], axis=1)
        
        # Aggregate features by day
        daily_features = daily_features.resample('D').agg({
            **{col: 'mean' for col in feature_names},
            'polarity': ['mean', 'std', 'min', 'max'],
            'subjectivity': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten multi-level columns after aggregation
        daily_features.columns = ['_'.join(col).strip() for col in daily_features.columns.values]
        
        # Ensure daily_features index is a datetime and tz-naive
        daily_features.index = pd.to_datetime(daily_features.index, errors='coerce')
        daily_features.index = daily_features.index.tz_localize(None)  # Make tz-naive
        
        # Add technical indicators to market data
        market_features = self._add_technical_indicators(market_df)
        
        # Ensure market_features index is a datetime and tz-naive
        market_features.index = pd.to_datetime(market_features.index, errors='coerce')
        market_features.index = market_features.index.tz_localize(None)  # Make tz-naive

        # Validate the indices before merging
        if not isinstance(daily_features.index, pd.DatetimeIndex):
            raise TypeError("daily_features index is not a valid DatetimeIndex")
        if not isinstance(market_features.index, pd.DatetimeIndex):
            raise TypeError("market_features index is not a valid DatetimeIndex")
        
        # Merge with market data
        merged_features = pd.merge(
            daily_features, 
            market_features,
            left_index=True, 
            right_index=True,
            how='inner'
        )
        
        return merged_features
    
    def _add_technical_indicators(self, df):
        df = df.copy()
        
        # Moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df


