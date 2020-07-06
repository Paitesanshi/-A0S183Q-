from datetime import datetime, timedelta
import time
import csv
from collections import namedtuple
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

features = ["DATE", "PRCP", "PRCP_ATTRIBUTES", "SNWD", "SNWD_ATTRIBUTES", "TAVG", "TAVG_ATTRIBUTES",
            "TMAX", "TMAX_ATTRIBUTES", "TMIN", "TMIN_ATTRIBUTES"]
DailySummary = namedtuple("DailySummary", features)
def read_weather_data(csvname):
    records = []
    csvFile = open(csvname, "r")
    reader = csv.reader(csvFile)
    n=0
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        print(item[n])
        print(n)
        records.append(item[n])
        n+=1

    csvFile.close()
    return records
def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements

trainFile = "E:/python/-A0S183Q-/2199262.csv"
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
trainData = pd.read_csv(os.path.basename(trainFile))
os.chdir(pwd)
print(trainData)#输出数据
data=trainData.iloc[0:892,0:12]#读取所有数据
print("------------------out",data)
#pandas数据格式为DataFrame,转化为numpy数组格式，方便处理
print (data.values(columns=None))
print(data.shape)

#df = pd.DataFrame(read_weather_data('2199262.csv'), columns=features).set_index('DATE')
#tmp = df[['TMAX', 'TMIN']].head(10)
