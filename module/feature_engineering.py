
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
涨跌 = df['收盘价'].diff()
涨幅 = 涨跌.mask(涨跌 < 0, 0)
跌幅 = -涨跌.mask(涨跌 > 0, 0)
涨幅均值 = 涨幅.rolling(window=14).mean()
跌幅均值 = 跌幅.rolling(window=14).mean()
相对强弱指标 = 100 - (100 / (1 + (涨幅均值 / 跌幅均值)))
df['RSI'] = 相对强弱指标

# 计算移动平均收敛/发散指标(MACD)
ema_12 = df['收盘价'].ewm(span=12, adjust=False).mean()
ema_26 = df['收盘价'].ewm(span=26, adjust=False).mean()
macd = ema_12 - ema_26
df['MACD'] = macd

# 计算布林带(Bollinger Bands)
标准差 = df['收盘价'].rolling(window=20).std()
df['布林带上轨'] = df['SMA_20'] + 2 * 标准差
df['布林带下轨'] = df['SMA_20'] - 2 * 标准差

# 计算真实波幅(ATR, Average True Range)
df['真实波幅'] = np.maximum(df['最高价'] - df['最低价'], np.abs(df['最高价'] - df['收盘价'].shift()))
df['ATR'] = df['真实波幅'].rolling(window=14).mean()

# 计算历史波动率
对数收益率 = np.log(df['收盘价'] / df['收盘价'].shift())
df['波动率'] = 对数收益率.rolling(window=252).std() * np.sqrt(252)

# 删除存在缺失值的行
df = df.dropna()

# 打印更新后的 DataFrame
print(df.head())
