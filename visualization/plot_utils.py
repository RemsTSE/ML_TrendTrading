import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Visualizer:
    @staticmethod
    def plot_predictions(y_true, y_pred, dates):
        plt.figure(figsize=(15, 7))
        plt.plot(dates, y_true, label='Actual')
        plt.plot(dates, y_pred, label='Predicted')
        plt.title('Market Price Predictions vs Actual')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

    @staticmethod
    def plot_feature_importance(feature_names, model, X, y):
        # Flatten X to 2D if it's 3D (samples, time_steps, input_features)
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)  # Flatten to shape (samples, time_steps * input_features)

        # Check and handle NaN values before calculating correlation
        if np.isnan(X).any() or np.isnan(y).any():
            print("Warning: Detected NaN values in X or y. Replacing NaNs with zeros for correlation calculation.")
            X = np.nan_to_num(X)  # Replace NaNs with zeros
            y = np.nan_to_num(y)  # Replace NaNs with zeros

        # Calculate correlation-based feature importance
        print(f"Original shape of X: {X.shape}")
        importance = np.abs(np.corrcoef(X, y[:, np.newaxis], rowvar=False)[0, 1:])
        print(f"Feature importance values:\n{importance}")

        # Ensure feature_names matches the length of importance array
        if len(importance) != len(feature_names):
            print(f"Warning: Length of feature_names ({len(feature_names)}) does not match length of importance ({len(importance)}).")
            # Trim feature_names or importance to match
            min_length = min(len(feature_names), len(importance))
            feature_names = feature_names[:min_length]
            importance = importance[:min_length]

        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Based on Correlation')
        plt.tight_layout()

    @staticmethod
    def plot_training_history(histories):
        plt.figure(figsize=(15, 7))
        
        for i, history in enumerate(histories):
            plt.plot(history['val_mae'], label=f'Fold {i+1}')
        
        plt.title('Model Performance Across Folds')
        plt.xlabel('Epoch')
        plt.ylabel('Validation MAE')
        plt.legend()
        plt.tight_layout()

    @staticmethod
    def plot_backtest_results(backtest_results, y_test, dates):
        trades = pd.DataFrame(backtest_results['trades'])
        
        # Debug: Print trades structure to verify its content
        print("Trades DataFrame structure:")
        print(trades)

        # Check if 'type' column is present in trades DataFrame
        if 'type' not in trades.columns:
            print("Error: The 'type' column is missing in the trades DataFrame.")
            print(f"Available columns: {trades.columns}")
            return

        plt.figure(figsize=(15, 7))
        plt.plot(dates, y_test, label='Price')

        # Filter out 'buy' and 'sell' dates based on the 'type' column
        buy_dates = [dates[i] for i in range(len(dates)) if i in trades[trades['type'] == 'buy'].index]
        sell_dates = [dates[i] for i in range(len(dates)) if i in trades[trades['type'] == 'sell'].index]

        # Extract buy and sell prices
        buy_prices = trades[trades['type'] == 'buy']['price'] if 'price' in trades.columns else []
        sell_prices = trades[trades['type'] == 'sell']['price'] if 'price' in trades.columns else []

        # Plot buy and sell signals
        if len(buy_dates) > 0 and len(buy_prices) > 0:
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy')
        if len(sell_dates) > 0 and len(sell_prices) > 0:
            plt.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell')
        
        plt.title('Backtest Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
