
import pandas as pd
import numpy as np

# 加载股票市场数据到 Pandas DataFrame
df = pd.read_csv('data/stock_data.csv')  # 替换 'data/stock_data.csv' 为你的数据文件路径

# 创建滞后特征
df['收盘价滞后1天'] = df['收盘价'].shift(1)
df['收盘价滞后7天'] = df['收盘价'].shift(7)
df['成交量滞后1天'] = df['成交量'].shift(1)

# 计算移动平均线
df['SMA_20'] = df['收盘价'].rolling(window=20).mean()
df['SMA_50'] = df['收盘价'].rolling(window=50).mean()

# 计算相对强弱指标(RSI)
price_change = df['收盘价'].diff()
gain = price_change.mask(price_change < 0, 0)
loss = -price_change.mask(price_change > 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
rsi = 100 - (100 / (1 + rs))
df['RSI'] = rsi

# 计算移动平均收敛/发散指标(MACD)
ema_12 = df['收盘价'].ewm(span=12, adjust=False).mean()
ema_26 = df['收盘价'].ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
df['MACD'] = macd

# 计算布林带(Bollinger Bands)
std = df['收盘价'].rolling(window=20).std()
df['布林带上轨'] = df['SMA_20'] + 2 * std
df['布林带下轨'] = df['SMA_20'] - 2 * std

# 计算真实波幅(ATR, Average True Range)
df['真实波幅'] = np.maximum(df['最高价'] - df['最低价'], np.abs(df['最高价'] - df['收盘价'].shift()))
df['ATR'] = df['真实波幅'].rolling(window=14).mean()

# 计算历史波动率
log_returns = np.log(df['收盘价'] / df['收盘价'].shift())
df['波动率'] = log_returns.rolling(window=252).std() * np.sqrt(252)

# 删除存在缺失值的行
df = df.dropna()

# 打印更新后的 DataFrame
print(df.head())
