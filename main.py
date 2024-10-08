from data_collection.news_collector import NewsCollector
from data_collection.market_data_collector import MarketDataCollector
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.feature_engineering import FeatureEngineer
from model.market_predictor import MarketPredictor
from backtesting.backtest import Backtester
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Initialize collectors
    news_collector = NewsCollector()
    market_collector = MarketDataCollector()
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Collect data
    print("Collecting data...")
    keywords = ['AAPL', 'Apple Inc', 'iPhone']
    news_data = news_collector.fetch_news(keywords, start_date, end_date)
    market_data = market_collector.fetch_market_data('AAPL', start_date, end_date)
    
    # Preprocess text
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    news_data['processed_text'] = news_data['content'].apply(preprocessor.preprocess_text)
    news_data['sentiment'] = news_data['content'].apply(preprocessor.get_sentiment)
    
    # Engineer features
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    features = feature_engineer.engineer_features(news_data, market_data)
    
    # Prepare data for model
    X = features.drop(['Close'], axis=1).values
    y = features['Close'].values
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    print("Training model...")
    predictor = MarketPredictor(input_shape=(X_train.shape[1],))
    histories = predictor.train(X_train, y_train)
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Backtest
    print("Running backtest...")
    backtester = Backtester()
    backtest_results = backtester.run_backtest(predictions.flatten(), y_test)
    
    print("Generating visualizations...")
    visualizer = Visualizer()
    
    test_dates = features.index[train_size:]
    
    # Plot predictions
    visualizer.plot_predictions(y_test, predictions.flatten(), test_dates)
    plt.savefig('predictions.png')
    
    # Plot feature importance
    feature_names = features.drop(['Close'], axis=1).columns
    visualizer.plot_feature_importance(feature_names, predictor.model)
    plt.savefig('feature_importance.png')
    
    # Plot training history
    visualizer.plot_training_history(histories)
    plt.savefig('training_history.png')
    
    # Plot backtest results
    visualizer.plot_backtest_results(backtest_results, test_dates)
    plt.savefig('backtest_results.png')

if __name__ == "__main__":
    main()
