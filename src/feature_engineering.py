
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
x = df[['Close_Lag1', 'Close_Lag7', 'Volume_Lag1', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'ATR', 'Volatility']]
y = df['Close'] # 收盘价

# 将数据分为训练集和测试集 ===============================================================
# GPT解释
# 这里使用了 train_test_split函数来将特征数据 X和目标变量 y划分为训练集和测试集。
# 划分比例为80%的数据作为训练集，20%的数据作为测试集。 
# random_state参数用于设置随机种子，以保证每次运行代码时得到的划分结果一致。 
# 划分后，训练集的特征数据被赋值给 X_train，训练集的目标变量被赋值给 y_train，
# 测试集的特征数据被赋值给 X_test，测试集的目标变量被赋值给 y_test。
# 在模型训练过程中，使用 X_train和 y_train来训练模型，即通过对训练集数据进行学习和优化来拟合模型。
# 然后，使用训练好的模型在测试集 X_test上进行预测，并将预测结果与真实的测试集目标变量 y_test进行比较，
# 计算均方误差（MSE）等评估指标来评估模型的性能。

# WV人话
# 由上面可以看出 这里的 x 是基础数据(喂的(特征)数据)=>在这里 即 RSI,MACD,日均 等, y 是 求的数据 在这里 即 收盘价
# test_size = 0.2 里的[0.2]为 20% 从给的数据中每次 随机取出(random_state=42) 20%的数据遮住为答案去训练(用剩下的80%为素材(特征))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 特征缩放 =============================================================================
# GPT解释
# 这里使用了 MinMaxScaler类来进行特征缩放操作。 
# MinMaxScaler将特征缩放到指定的范围（默认为0到1之间）。 
# 首先，创建一个 MinMaxScaler对象 scaler。
# 然后，通过调用 fit_transform方法将训练集特征数据 X_train进行拟合和转换，得到缩放后的训练集特征数据 X_train_scaled。
# 拟合过程中， MinMaxScaler会计算训练集特征数据的最小值和最大值，并应用于转换操作。
# 接着，使用transform方法将测试集特征数据X_test进行缩放转换，得到缩放后的测试集特征数据X_test_scaled。
# 这里注意，测试集的缩放是基于训练集的最小值和最大值进行的，而不是重新计算测试集的最小值和最大值，以确保训练集和测试集的缩放方式保持一致。
# 特征缩放的目的是将不同范围和尺度的特征数据缩放到相同的范围，以避免某些特征对模型训练的影响过大。
# 常见的特征缩放方法包括最小-最大缩放、标准化等。在这个代码中，使用的是最小-最大缩放方法。

# WV人话
# MinMaxScaler 特征缩放(即 把素材(特征)归一化和标准化), fit_transform 拟合和转换，得到缩放后的素材(特征)
# PS 这里的 缩放 的 功能 可以理解为 2次清洗 素材(特征)，降低 异常数据 对结果的影响

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 模型训练 ===================================================================================
# GPT解释
# 训练过程会基于给定的隐藏层结构（hidden_layer_sizes）和激活函数（activation）来优化模型的参数。
# 在这个例子中，我们使用了包含100个神经元的第一个隐藏层和包含50个神经元的第二个隐藏层，激活函数选择了ReLU。

# hidden_layer_sizes=(100, 50)：这个参数指定了MLP模型的隐藏层结构。在这个例子中，
# 模型包含两个隐藏层，第一个隐藏层有100个神经元，第二个隐藏层有50个神经元。 

# activation='relu'：这个参数指定了MLP模型中的激活函数。
# 在这个例子中，我们使用的是ReLU激活函数，它将负输入值置为零，保持正输入值不变。
#ReLU（Rectified Linear Unit）是一种常用的激活函数，用于神经网络中的神经元的非线性变换。ReLU函数定义为：

# f(x) = max(0, x)

# 其中x是输入值。该函数将负输入值设为零，保持正输入值不变。
# 简而言之，如果输入值大于零，则ReLU函数输出输入值本身；如果输入值小于或等于零，则ReLU函数输出零。
# ReLU激活函数的主要优点是计算简单且具有良好的数值稳定性。
# 它能够在保持线性特性的同时引入非线性变换，从而帮助神经网络学习更复杂的函数关系。
# ReLU的非线性特性使其能够处理非线性数据和提取非线性特征，对于解决分类和回归等机器学习任务非常有效。
# 在MLP模型中，ReLU通常用作隐藏层的激活函数，因为它能够帮助网络学习更复杂的特征表示，并且在梯度计算时没有梯度消失问题。
# 然而，在输出层的激活函数选择上，根据任务的不同，可能需要使用其他适当的激活函数，例如Sigmoid或Softmax函数。

# random_state=42：这个参数指定了随机数生成器的种子，用于初始化模型的权重和偏差，以及每次迭代中的随机性。 

# WV人话
# hidden_layer_sizes 神经元(if)层数 第1层100 第2层50， activation激活函数， random_state随机数生成器的种子

model = MLPRegressor(hidden_layer_sizes=(10000, 7000, 5000, 3000, 1000, 500, 100), activation='relu', random_state=42)
model.fit(x_train_scaled, y_train)

# 模型评估
y_pred = model.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差（Mean Squared Error）: {mse}")

# 模型预测
new_data = pd.DataFrame({
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
