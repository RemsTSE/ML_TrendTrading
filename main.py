from data_collection.news_collector import NewsCollector
from data_collection.crypto_news_collector import CryptoNewsCollector
from data_collection.market_data_collector import MarketDataCollector
from preprocessing.text_preprocessor import TextPreprocessor
from preprocessing.feature_engineering import FeatureEngineer
from model.market_predictor import MarketPredictor
from backtesting.backtest import Backtester
from visualization.plot_utils import Visualizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():

    
    # Initialize collectors
    market_collector = MarketDataCollector()
    
    # Your NewsAPI key (replace 'your_api_key' with an actual key)
    api_key = '870869f569104d5d84591d4572e7200e'
    
    # Initialize collectors with API key
    news_collector = NewsCollector(api_key=api_key)
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Collect data
    print("Collecting news data...")
    keywords = ['AAPL', 'Apple Inc', 'iPhone', 'stock']
    news_data = news_collector.fetch_news(keywords, start_date, end_date)
    
    if news_data.empty:
        print("No news articles found. Please check your keywords or date range.")
        return

    print("Collecting market data...")
    market_data = market_collector.fetch_market_data('AAPL', start_date, end_date)
    
    # Check if market data is empty and handle gracefully
    if market_data.empty:
        print("No market data found. Please check your stock symbol or date range.")
        return

    # Preprocess text
    print("Preprocessing text...")
    preprocessor = TextPreprocessor()
    # Check if 'content' column exists before applying text processing
    if 'content' not in news_data.columns:
        print("The 'content' column is missing in news data. Please verify the data collection step.")
        return
    
    # Apply text preprocessing and sentiment analysis
    news_data['processed_text'] = news_data['content'].apply(preprocessor.preprocess_text)
    news_data['sentiment'] = news_data['content'].apply(preprocessor.get_sentiment)
    
    # Engineer features
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    features = feature_engineer.engineer_features(news_data, market_data)
    
    # Ensure features and target columns are present
    if 'Close' not in features.columns:
        print("The 'Close' column is missing in the feature set. Check the feature engineering step.")
        return

    # Prepare data for model
    X = features.drop(['Close'], axis=1).values
    y = features['Close'].values

    # **Handle NaN values and normalize**
    X = np.nan_to_num(X)  # Replace NaNs with zeros
    y = np.nan_to_num(y)  # Replace NaNs with zeros

    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Scale all features in X
    print(f"Normalized values in X:\n{X[:5]}")

    # Check and remove constant features
    variances = np.var(X, axis=0)  # Calculate variance of each feature
    print("Feature variances:\n", variances)

    # Remove features with zero variance
    constant_features = np.where(variances == 0)[0]
    if len(constant_features) > 0:
        print(f"Removing constant features at indices: {constant_features}")
        X = np.delete(X, constant_features, axis=1)
        print(f"New shape of X after removing constant features: {X.shape}")

    # Update input_features to reflect the removal of constant features
    input_features = X.shape[1]

    # Calculate total elements
    total_elements = X.size  # This is the total number of elements in X
    print(f"Total elements in X after removing constant features: {total_elements}")

    # Set desired number of features per timestep (e.g., based on remaining features)
    desired_features_per_timestep = min(10, input_features)  # Choose a reasonable value for time steps

    # Calculate compatible time_steps and samples based on total_elements
    samples = len(X)  # The total number of rows in X is the number of samples
    time_steps = total_elements // (samples * input_features)  # Ensure compatibility with total elements

    print(f"Updated values: time_steps={time_steps}, input_features={input_features}, samples={samples}")

    # Ensure valid number of samples
    if samples <= 0 or time_steps <= 0:
        print("Error: The calculated number of samples or time steps is less than or equal to 0. Exiting the script.")
        return

    # Reshape X and adjust y accordingly
    try:
        X = X[:samples * time_steps * input_features].reshape(samples, time_steps, input_features)
        y = y[:samples]  # Ensure y has the same number of samples
    except ValueError as e:
        print(f"Reshape error: {e}. Exiting script.")
        return

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    print("Training model with modified parameters...")
    predictor = MarketPredictor(time_steps=time_steps, input_features=input_features)
    histories = predictor.train(X_train, y_train, epochs=50, batch_size=8)

    # Make predictions
    predictions = predictor.predict(X_test)
    print("Model Predictions on Test Data:")
    print(predictions.flatten())

    # Backtest
    print("Running backtest...")
    backtester = Backtester()
    backtest_results = backtester.run_backtest(predictions.flatten(), y_test)
    print("Backtest Results (Raw):")
    print(backtest_results)

    print("Generating visualizations...")
    visualizer = Visualizer()
    test_dates = features.index[train_size:]
    feature_names = features.drop(['Close'], axis=1).columns.to_list()  # Update after feature engineering

    # Visualizations
    visualizer.plot_predictions(y_test, predictions.flatten(), test_dates)
    plt.savefig('predictions.png')

    visualizer.plot_feature_importance(feature_names, predictor.model, X_train, y_train)
    plt.savefig('feature_importance.png')

    visualizer.plot_training_history(histories)
    plt.savefig('training_history.png')

    visualizer.plot_backtest_results(backtest_results, y_test, test_dates)
    plt.savefig('backtest_results.png')

    print("All visualizations have been saved successfully.")


if __name__ == "__main__":
    main()


