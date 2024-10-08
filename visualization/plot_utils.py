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
    def plot_feature_importance(feature_names, model):
        # For LSTM, we'll use a simple correlation-based approach
        importance = np.abs(np.corrcoef(X, y[:, np.newaxis]))[0, 1:]
        
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
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
    def plot_backtest_results(backtest_results, dates):
        trades = pd.DataFrame(backtest_results['trades'])
        
        plt.figure(figsize=(15, 7))
        plt.plot(dates, y_test, label='Price')
        
        buy_dates = [dates[i] for i in range(len(dates)) if i in trades[trades['type'] == 'buy'].index]
        sell_dates = [dates[i] for i in range(len(dates)) if i in trades[trades['type'] == 'sell'].index]
        
        buy_prices = trades[trades['type'] == 'buy']['price']
        sell_prices = trades[trades['type'] == 'sell']['price']
        
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy')
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell')
        
        plt.title('Backtest Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()