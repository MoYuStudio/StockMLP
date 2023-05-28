
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

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
# print(df.head())

# 将数据分为特征和目标变量
X = df[['Close_Lag1', 'Close_Lag7', 'Volume_Lag1', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'Volatility']]
y = df['Close']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(X_train_scaled, y_train)

# 模型评估
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差（Mean Squared Error）: {mse}")

# 模型预测
new_data = pd.DataFrame({  # 替换为要进行预测的新数据
    'Close_Lag1': [123.45],
    'Close_Lag7': [120.0],
    'Volume_Lag1': [10000],
    'SMA_20': [125.0],
    'SMA_50': [130.0],
    'RSI': [60.0],
    'MACD': [2.5],
    'BB_upper': [135.0],
    'BB_lower': [120.0],
    'ATR': [5.0],
    'Volatility': [0.15]
})
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"预测结果：{prediction}")
