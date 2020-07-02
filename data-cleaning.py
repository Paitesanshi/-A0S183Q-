import pandas as pd
from datetime import datetime
from dateutil import parser

data_raw = pd.read_csv('2199262.csv', encoding='utf-8')
data_raw['date'] = data_raw['DATE'].apply(parser.parse)
data_raw['tmax'] = data_raw['TMAX'].astype(float)
data_raw['tmin'] = data_raw['TMIN'].astype(float)
data = data_raw.loc[:, ['date', 'tmax', 'tmin']]
data = data[(pd.Series.notnull(data['tmax'])) & (pd.Series.notnull(data['tmin']))]  # 把空数据过滤掉

data = data[(data['date'] >= datetime(1951, 1, 1)) & (data['date'] <= datetime(2016, 1, 1))]
data.query("date.dt.day == 28 & date.dt.month == 6", inplace=True)  # 得到历史上每年 6.28 的数据
data.to_csv('maxmin.csv', index=None)
# result=[]
# result.append(1)
# d=0
# data_diff=data['tmax']
# #循环进行差分，直到通过单位根检验，单位根检验统计量<-3.47,即不平稳假设发生的概率小于1%
# while result[0]>-3.43:
#     d=d+1
#     data_diff = data_diff.diff(1);
#     data_diff = data_diff.fillna(0);
#     result=unitroot_adf(data_diff)
#     print(result[0],d)


