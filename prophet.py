import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
data_raw = pd.read_csv('2199262.csv')
data_raw['date'] = data_raw['DATE']
data_raw['tmax'] = data_raw['TMAX'].astype(float)
data_raw['tmin'] = data_raw['TMIN'].astype(float)
data = data_raw.loc[:, ['date', 'tmax', 'tmin']]
data = data[(pd.Series.notnull(data['tmax'])) & (pd.Series.notnull(data['tmin']))]  # 把空数据过滤掉
df = pd.DataFrame()
df['ds'] = data['date']
df['y'] = data['tmin']

# 3、训练集、测试集划分
trainsize = int(df.shape[0]*0.9)
train = df[0:trainsize]
test = df[trainsize:df.shape[0]]

# 4、模型训练
# changepoint_prior_scale默认为0.05,增大changepoint_prior_scale表示让模型拟合更多，可能发生过拟合风险
model = Prophet(changepoint_prior_scale=0.5)
model.fit(df);

#5、模型预测
# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
future = model.make_future_dataframe(periods=df.shape[0]-trainsize)
print("-----------------future.tail-----------------")
print(future.tail())

# 预测数据集
forecast = model.predict(future)
print("-----------------forcast tail-----------------")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
model.plot(forecast);

# 预测的成分分析绘图，展示预测中的趋势、周效应和年度效应
model.plot_components(forecast);
plt.show()
#6、模型评估
print("mse is",mean_squared_error(test['y'].values,forecast['yhat'].values[trainsize:df.shape[0]]))