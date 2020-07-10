# 观察数据集中数据
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from keras import models
from sklearn.metrics import mean_squared_error

data_dir = 'D:\\pythontest\\tset'
fname = os.path.join(data_dir, '2199.csv')
# data_dir = 'C:\\Users\\夜望君遥\\Desktop\\jena_climate_2009_2016.csv'
# fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# print(lines)

x = []
y = []
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    # print(values)
    float_data[i, :] = values
    (x_test, y_test, z_test) = values
    # print(float_data)
    x.append(z_test)
    y.append(x_test)

temp = float_data[:, 1].astype(float)
# # print(temp)
# plt.plot(range(len(temp)), temp, label='history')
# plt.plot(range(1440), temp[:1440],label='test')
# 数据标准化（减去平均数，除以标准差）
mean = float_data[:10000].mean(axis=0)
float_data -= mean
std = float_data[:10000].std(axis=0)
float_data /= std


# 生成时间序列样本及其目标的生成器
# data:浮点数数据组成的原始数组
# lookback：输入数据应该包括过去多少个时间步
# delay：目标应该在未来多少个时间步后
# min_index,max_index：数组中的索引，界定需要抽取哪些时间步，有助于保存数据用于验证和测试
# shuffle：是否打乱样本
# batch_size：每个批量的样本数
# step：数据采样周期，每个时间步是10min，设置为6，即每小时取一个数据点
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            # print(indices)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# 准备训练生成器，验证生成器，测试生成器
# 输入数据包括过去10天内的数据，每小时抽取一次数据点
# 目标为一天以后的天气，批次样本数为128
lookback = 365
step = 1
delay = 7
batch_size = 7

train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=10000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=10001, max_index=20000,
                    shuffle=True, step=step, batch_size=batch_size)
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=20001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)
pre_gen = generator(float_data, lookback=lookback, delay=delay, min_index=22500, max_index=22600,
                    shuffle=True, step=step, batch_size=batch_size)
val_steps = (20000 - 10001 - lookback) // batch_size
test_steps = (len(float_data) - 20001 - lookback) // batch_size

# 训练并评估一个使用dropout正则化的基于GRU的模型
# 不再过拟合，分数更稳定


model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mse')
history = model.fit_generator(train_gen, steps_per_epoch=50, epochs=1, validation_data=val_gen,
                              validation_steps=val_steps)
# print(pre_gen)
work = model.predict_generator(train_gen, batch_size)
print(work)

mm = MinMaxScaler()

pre = mm.inverse_transform(work)

plt.plot(range(len(pre)), pre, label='work')
# # print(train_gen)
# work1 = model.predict(val_gen, batch_size)
# print(work1)
# plt.plot(range(len(work1)), work1, label='work1')

# hist = model.predict(val_gen, batch_size)
# print(hist)
# plt.plot(range(len(hist)),hist)
# with open('answer.txt', 'w') as f:
#     f.write(str(work.history))

# def GRU_model():
# for t in range(len(temp)):
#     predictions = list()
#     model = Sequential()
#     model_fit = model.fit(disp=0)
#     yhat = model_fit.forecast(steps=1)[0]
#     predictions.append(yhat)
#     # history.append(temp[t+train_size])
#     # 计算样本的误差值
#     error = mean_squared_error(temp, predictions)

# epochs = len(history.history['loss'])
# plt.plot(range(epochs), history.history['loss'], label='history')
# plt.plot(range(epochs), history.history['val_loss'], label='prediction')
# 5、模型预测
# 构建待预测日期数据框，periods = 365 代表除历史数据的日期外再往后推 365 天
# future = model.make_future_dataframe(periods=365)
# print("-----------------future.tail-----------------")
# print(future.tail())
#
# # 预测数据集
# forecast = model.predict(future)
# print("-----------------forcast tail-----------------")
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# 展示预测结果
# model.plot(forecast);


plt.legend()
plt.show()
