
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font_path = 'ZiYuYongSongTi-2.ttf'
font_prop = FontProperties(fname=font_path)

# 加载股票市场数据到 Pandas DataFrame
df = pd.read_csv('data/stock_data.csv')

# 数据摘要
print(df.head())  # 显示前几行
print(df.info())  # 显示列名和数据类型
print(df.describe())  # 显示统计摘要

# 绘制时间序列图
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'])
plt.xlabel('日期', fontproperties=font_prop)
plt.ylabel('收盘价', fontproperties=font_prop)
plt.title('股票价格随时间变化', fontproperties=font_prop)
plt.xticks(rotation=45)
plt.show()

# 绘制直方图和分布图
plt.figure(figsize=(8, 6))
plt.hist(df['Close'], bins=20, edgecolor='black')
plt.xlabel('收盘价', fontproperties=font_prop)
plt.ylabel('频率', fontproperties=font_prop)
plt.title('收盘价分布', fontproperties=font_prop)
plt.show()

# 相关性分析
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('相关性矩阵', fontproperties=font_prop)
plt.show()

# 移动平均线
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='收盘价')
plt.plot(df['Date'], df['Close'].rolling(window=20).mean(), label='20日移动平均线')
plt.plot(df['Date'], df['Close'].rolling(window=50).mean(), label='50日移动平均线')
plt.xlabel('日期', fontproperties=font_prop)
plt.ylabel('价格', fontproperties=font_prop)
plt.title('移动平均线', fontproperties=font_prop)
plt.legend()
plt.xticks(rotation=45)
plt.show()
