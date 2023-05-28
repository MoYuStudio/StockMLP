
import pandas as pd
import matplotlib.pyplot as plt

# Load stock market data into a Pandas DataFrame
df = pd.read_csv('data/stock_data.csv')  # Replace 'stock_data.csv' with the path to your data file

# Data Summary
print(df.head())  # Display first few rows
print(df.info())  # Display column names and data types
print(df.describe())  # Display statistical summaries

# Plotting Time Series
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Stock Price Over Time')
plt.xticks(rotation=45)
plt.show()

# Histograms and Distributions
plt.figure(figsize=(8, 6))
plt.hist(df['Close'], bins=20, edgecolor='black')
plt.xlabel('Closing Price')
plt.ylabel('Frequency')
plt.title('Closing Price Distribution')
plt.show()

# Correlation Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Matrix')
plt.show()

# Moving Averages
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Closing Price')
plt.plot(df['Date'], df['Close'].rolling(window=20).mean(), label='20-day Moving Average')
plt.plot(df['Date'], df['Close'].rolling(window=50).mean(), label='50-day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Moving Averages')
plt.legend()
plt.xticks(rotation=45)
plt.show()
