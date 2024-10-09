import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from config import Config

class MarketPredictor:
    def __init__(self, time_steps, input_features):
        self.time_steps = time_steps
        self.input_features = input_features
        self.scaler = StandardScaler()  # Use a scaler to scale the target variable
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.time_steps, self.input_features)),  # Define input layer
            LSTM(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # LSTM layer with L2 regularization
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Second LSTM layer
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dense(1)  # Output layer for regression
        ])
        
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def _reshape_input(self, X):
        """
        Reshape input data to 3D array with shape (samples, time_steps, features_per_timestep).
        """
        if X.size == 0:
            print("Warning: The input array X is empty.")
            return X

        print(f"Original X shape before reshaping: {X.shape}")

        # Calculate the number of samples based on total elements, time steps, and input features
        total_elements = X.size
        print(f"Total elements in X: {total_elements}")

        # Calculate the number of samples that perfectly fits the shape (samples, time_steps, input_features)
        samples = total_elements // (self.time_steps * self.input_features)

        if samples == 0 or samples * self.time_steps * self.input_features != total_elements:
            print(f"Warning: Calculated samples ({samples}) result in a shape mismatch.")
            # Try to adjust `time_steps` or `input_features` to fit the total elements
            remainder = total_elements % (self.time_steps * self.input_features)
            if remainder != 0:
                print(f"Total elements ({total_elements}) cannot be perfectly divided by time_steps * input_features.")
                # Calculate a compatible shape
                self.time_steps = total_elements // self.input_features
                print(f"Adjusting time_steps to {self.time_steps} to fit the total elements.")
                samples = total_elements // (self.time_steps * self.input_features)
                print(f"New calculated samples: {samples}")

        # Final check to ensure samples is positive
        if samples <= 0:
            print(f"Calculated samples is 0 or negative. Cannot reshape X with shape {X.shape}. Returning empty array.")
            return np.array([])

        # Reshape the array to (samples, time_steps, input_features)
        reshaped_X = X[:samples * self.time_steps * self.input_features].reshape(samples, self.time_steps, self.input_features)

        print(f"Reshaped X shape: {reshaped_X.shape}")
        return reshaped_X
    
    def train(self, X, y, epochs=Config.MODEL_PARAMS['epochs'], batch_size=Config.MODEL_PARAMS['batch_size']):
        # Reshape X to be compatible with LSTM input
        X = self._reshape_input(X)
        
        # Scale y for training
        y = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Ensure that X and y have the same number of samples
        if X.shape[0] != len(y):
            print(f"Mismatch between number of samples in X and y: X has {X.shape[0]}, y has {len(y)}.")
            print("Trimming y to match X.")
            y = y[:X.shape[0]]

        # Check that X has the expected number of features
        if X.shape[2] != self.input_features:
            print(f"Error: Expected input features to be {self.input_features}, but got {X.shape[2]}.")
            print(f"Reshaped X: {X}")
            return

        # Adjust the number of splits based on the number of samples in X
        n_samples = X.shape[0]
        n_splits = min(Config.MODEL_PARAMS['cv_folds'], n_samples - 1)  # Ensure n_splits is at least 2
        print(f"Adjusted n_splits for TimeSeriesSplit: {n_splits}")

        if n_splits < 2:
            print("Insufficient samples for TimeSeriesSplit. Using simple train-test split instead.")
            # Use 80% for training and 20% for validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")

            # Train and validate the model
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]  # Added early stopping
            )
            return [history.history]
        
        # Use TimeSeriesSplit if n_splits >= 2
        tscv = TimeSeriesSplit(n_splits=n_splits)
        histories = []
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            print(f"Training fold {fold+1} with {len(X_train)} samples")
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]  # Added early stopping
            )
            histories.append(history.history)
        
        return histories

    def predict(self, X):
        # Reshape X for prediction
        X = self._reshape_input(X)
        
        # Check if X is empty before predicting
        if X.size == 0:
            print("The input data X is empty. Skipping prediction.")
            return np.array([])

        predictions = self.model.predict(X)
        # Inverse transform to return to original scale
        return self.scaler.inverse_transform(predictions)


