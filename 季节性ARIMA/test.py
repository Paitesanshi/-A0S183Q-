import pandas as pd
from datetime import datetime
from dateutil import parser



for i in range(31):
    data_raw = pd.read_csv('乌鲁木齐.csv', encoding='utf-8')
    data_raw['date'] = data_raw['DATE'].apply(parser.parse)
    data_raw['tmax'] = data_raw['TMAX'].astype(float)
    data_raw['tmin'] = data_raw['TMIN'].astype(float)
    data = data_raw.loc[:, ['date', 'tmax', 'tmin']]
    data = data[(pd.Series.notnull(data['tmax'])) & (pd.Series.notnull(data['tmin']))]
    data = data[(data['date'] >= datetime(2000, 1, 1)) & (data['date'] <= datetime(2013, 1, 1))]
    data.query("date.dt.day == %d"%(i+1), inplace=True)
    data.to_csv('乌鲁木齐/%d.csv'%(i+1), index=None)
