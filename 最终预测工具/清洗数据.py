import pandas as pd
from datetime import datetime
from dateutil import parser
import time


for i in range(31):
    data_raw = pd.read_csv('wendu/银川.csv', encoding='utf-8')
    print(data_raw['DATE'])
    data_raw['date'] = data_raw['DATE'].apply(parser.parse)
    data_raw['tmax'] = data_raw['TMAX'].astype(float)
    data_raw['tmin'] = data_raw['TMIN'].astype(float)
    data = data_raw.loc[:, ['date', 'tmax', 'tmin']]
    data = data[(pd.Series.notnull(data['tmax'])) & (pd.Series.notnull(data['tmin']))]
    data = data[(data['date'] >= datetime(2000, 1, 1)) & (data['date'] <= datetime(2012, 12, 31))]
    data.query("date.dt.day == %d"%(i+1), inplace=True)
    data.to_csv('银川/%d.csv'%(i+1), index=None)
    work = pd.read_csv('银川/%d.csv' % (i + 1), encoding='utf-8')
    for j in range(len(work['date'])):
        print(work['date'][j])
        timeArray = time.strptime(work['date'][j], "%Y-%m-%d")
        work['date'][j] = time.strftime("%Y-%m", timeArray)

    work.to_csv('银川/%d.csv' % (i + 1), index=None)