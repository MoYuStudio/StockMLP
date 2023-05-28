
import pandas as pd
import numpy as np

# 加载股票市场数据到 Pandas DataFrame
df = pd.read_csv('data/stock_data.csv')

# 创建滞后特征
df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag7'] = df['Close'].shift(7)
df['Volume_Lag1'] = df['Volume'].shift(1)

# 计算移动平均线
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# 计算相对强弱指标(RSI)
price_change = df['Close'].diff()
gain = price_change.mask(price_change < 0, 0)
loss = -price_change.mask(price_change > 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
rsi = 100 - (100 / (1 + rs))
df['RSI'] = rsi

# 计算移动平均收敛/发散指标(MACD)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
df['MACD'] = macd

# 计算布林带(Bollinger Bands)
std = df['Close'].rolling(window=20).std()
df['BB_upper'] = df['SMA_20'] + 2 * std
df['BB_lower'] = df['SMA_20'] - 2 * std

# 计算真实波幅(ATR, Average True Range)
df['TR'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()))
df['ATR'] = df['TR'].rolling(window=14).mean()

# 计算历史波动率
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift())
df['Volatility'] = df['Log_Return'].rolling(window=252).std() * np.sqrt(252)

# 删除存在缺失值的行
df = df.dropna()

# 打印更新后的 DataFrame
print(df.head())
