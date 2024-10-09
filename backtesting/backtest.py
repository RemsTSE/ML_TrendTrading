from config import Config

class Backtester:
    def __init__(self, initial_capital=Config.BACKTEST_PARAMS['initial_capital'],
                 position_size=Config.BACKTEST_PARAMS['position_size']):
        self.initial_capital = initial_capital
        self.position_size = position_size
    
    def run_backtest(self, predictions, actual_prices):
        capital = self.initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(predictions)):
            # Simple strategy: Buy if predicted price is higher, sell if lower
            pred_return = predictions[i] / actual_prices[i-1] - 1
            
            if pred_return > 0 and position <= 0:
                # Buy
                position = (capital * self.position_size) / actual_prices[i]
                trades.append({
                    'type': 'buy',
                    'price': actual_prices[i],
                    'position': position,
                    'capital': capital
                })
            elif pred_return < 0 and position > 0:
                # Sell
                capital += position * actual_prices[i]
                position = 0
                trades.append({
                    'type': 'sell',
                    'price': actual_prices[i],
                    'position': position,
                    'capital': capital
                })
        
        # Final liquidation
        if position > 0:
            capital += position * actual_prices[-1]
        
        return {
            'final_capital': capital,
            'return': (capital - self.initial_capital) / self.initial_capital,
            'trades': trades
        }