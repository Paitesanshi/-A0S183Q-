from datetime import datetime, timedelta
import time
import csv
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt

features = ["DATE", "PRCP", "PRCP_ATTRIBUTES", "SNWD", "SNWD_ATTRIBUTES", "TAVG", "TAVG_ATTRIBUTES",
            "TMAX", "TMAX_ATTRIBUTES", "TMIN", "TMIN_ATTRIBUTES"]
DailySummary = namedtuple("DailySummary", features)
def read_weather_data(csvname):
    records = []
    csvFile = open(csvname, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        print(item[0])
        records[item[0]] = item[1]

    csvFile.close()
    return records
def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


df = pd.DataFrame(read_weather_data('2199262.csv'), columns=features).set_index('DATE')
tmp = df[['TMAX', 'TMIN']].head(10)
