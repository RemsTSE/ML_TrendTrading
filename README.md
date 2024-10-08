# Market Prediction Model

This project implements a machine learning model for predicting market prices using news data and technical indicators. It uses free data sources and implements various improvements including sentiment analysis, cross-validation, and backtesting.

## Features
- Free data collection from multiple sources
- Text preprocessing with sentiment analysis
- Advanced feature engineering
- LSTM-based prediction model with cross-validation
- Backtesting functionality
- Comprehensive visualizations

## Setup
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the model:
```
python main.py
```

## Output
The script will generate four visualization files:
- predictions.png: Shows predicted vs actual prices
- feature_importance.png: Displays the importance of different features
- training_history.png: Shows the model's performance across cross-validation folds
- backtest_results.png: Visualizes the backtesting results with buy/sell signals

## Notes
- This implementation uses free data sources which may have limitations
- The model's performance depends on the quality and quantity of available data
- Consider adjusting hyperparameters for your specific use case

## Future Improvements
- Implement more sophisticated trading strategies
- Add more data sources
- Experiment with different model architectures
- Implement real-time prediction capabilities