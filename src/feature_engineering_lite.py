
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = 'ZiYuYongSongTi-2.ttf'
font_prop = FontProperties(fname=font_path)

# 加载股票市场数据到 Pandas DataFrame
df = pd.read_csv('data/stock_data.csv')

# 创建滞后特征
df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag7'] = df['Close'].shift(7)

# 删除存在缺失值的行
df = df.dropna()

# 将数据分为特征和目标变量
x = df[['Close_Lag1', 'Close_Lag7']]
y = df['Close'] # 收盘价

# 将数据分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 模型训练
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
model.fit(x_train_scaled, y_train)

# 模型评估
y_pred = model.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差（Mean Squared Error）: {mse}")

# 模型预测
new_data = pd.DataFrame({
    'Close_Lag1': [123.45],
    'Close_Lag7': [120.0],

})
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"预测结果：{prediction}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='实际值 (Actual)', color='blue')
plt.plot(y_pred, label='预测值 (Predicted)', color='orange')
plt.xlabel('样本索引 (Sample Index)', fontproperties=font_prop)
plt.ylabel('收盘价 (Close Price)', fontproperties=font_prop)
plt.title('实际值 vs 预测值 (Actual vs Predicted)', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.show()
plt.show()