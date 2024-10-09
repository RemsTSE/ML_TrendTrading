# test_setup.py

import pandas as pd
import numpy as np
import tensorflow as tf
import yfinance as yf
import feedparser
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def test_setup():
    print("Testing key components...")
    
    # Test data collection
    print("\n1. Testing data collection:")
    try:
        # Test stock data
        stock = yf.Ticker("AAPL")
        hist = stock.history(period="1d")
        print("✓ Stock data collection working")
        
        # Test news data
        feed = feedparser.parse("http://feeds.marketwatch.com/marketwatch/topstories/")
        print(f"✓ News feed collection working (got {len(feed.entries)} entries)")
    except Exception as e:
        print(f"✗ Data collection error: {str(e)}")

    # Test text processing
    print("\n2. Testing text processing:")
    try:
        text = "This is a test sentence for sentiment analysis."
        sentiment = TextBlob(text).sentiment
        print(f"✓ Sentiment analysis working (polarity: {sentiment.polarity})")
    except Exception as e:
        print(f"✗ Text processing error: {str(e)}")

    # Test TensorFlow
    print("\n3. Testing TensorFlow:")
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        print(f"✓ TensorFlow working (version {tf.__version__})")
    except Exception as e:
        print(f"✗ TensorFlow error: {str(e)}")

    # Test plotting
    print("\n4. Testing plotting:")
    try:
        plt.figure(figsize=(3, 2))
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.close()
        print("✓ Plotting working")
    except Exception as e:
        print(f"✗ Plotting error: {str(e)}")

if __name__ == "__main__":
    test_setup()

#C:\Users\remil\AppData\Local\Programs\Python\Python312>