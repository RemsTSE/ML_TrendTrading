import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class MarketPredictor:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
    
    def _build_model(self, input_shape):
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model
    
    def train(self, X, y, epochs=Config.MODEL_PARAMS['epochs'], 
              batch_size=Config.MODEL_PARAMS['batch_size']):
        tscv = TimeSeriesSplit(n_splits=Config.MODEL_PARAMS['cv_folds'])
        histories = []
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            print(f"Training fold {fold+1}")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val)
            )
            histories.append(history.history)
        
        return histories
    
    def predict(self, X):
        return self.model.predict(X)