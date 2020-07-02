import pandas as pd
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import unitroot_adf





data_raw = pd.read_csv('2199262.csv', encoding='utf-8')
data_raw['date'] = data_raw['DATE'].apply(parser.parse)
data_raw['tmax'] = data_raw['TMAX'].astype(float)
data_raw['tmin'] = data_raw['TMIN'].astype(float)
data = data_raw.loc[:, ['date', 'tmax', 'tmin']]
data = data[(pd.Series.notnull(data['tmax'])) & (pd.Series.notnull(data['tmin']))]  # 把空数据过滤掉

data = data[(data['date'] >= datetime(1951, 1, 1)) & (data['date'] <= datetime(2016, 1, 1))]
data.query("date.dt.day == 28 & date.dt.month == 6", inplace=True)  # 得到历史上每年 6.28 的数据
data.to_csv('maxmin.csv', index=None)

data_diff = data['tmax'].diff(1);
data_diff = data_diff.fillna(0);
print(data_diff);
graph = plt.figure(figsize=(10, 4))
ax = graph.add_subplot(111)
ax.set(title='Total_Purchase_Amt', ylabel='Unit (Fahrenheit)', xlabel='Date')
plt.plot(data['date'], data_diff)
plt.show()
