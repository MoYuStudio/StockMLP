
import yfinance as yf
import pandas as pd

# 定义要操作的多只股票代码
codes = ["0700.HK"]# , "3690.HK", "2269.HK", "1810.HK", "1024.HK", "1211.HK"]

# 定义空的DataFrame用于保存所有股票的结果
all_data = pd.DataFrame()

for code in codes:
    # 获取数据
    data = yf.download(code, start='2010-01-01', end='2023-05-1')

    print(data.tail(7))

    # 将数据保存到Excel文件xlsx
    # data.to_excel('data/stock_data.csv')
    data.to_csv('data/stock_data.csv')
