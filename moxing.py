# # 观察数据集中数据
# import os
# data_dir = 'C:\\Users\\夜望君遥\\Desktop'
# fname = os.path.join(data_dir, 'maxmin.csv')
# f = open(fname)
# data = f.read()
# f.close()
# lines = data.split('\n')
# header = lines[0].split(',')
# lines = lines[1:]
# print(header)
# print(len(lines))
# import numpy as np
# float_data = np.zeros((len(lines), len(header) - 1))
# for i, line in enumerate(lines):
#     values = [float(x) for x in line.split(',')[1:]]
#     float_data[i, :] = values
# from matplotlib import pyplot as plt
# temp = float_data[:, 1]
# plt.plot(range(len(temp)), temp)
# plt.plot(range(1440), temp[:1440])
# # 数据标准化（减去平均数，除以标准差）
# mean = float_data[:200000].mean(axis=0)
# float_data -= mean
# std = float_data[:200000].std(axis=0)
# float_data /= std
# # 生成时间序列样本及其目标的生成器
# # data:浮点数数据组成的原始数组
# # lookback：输入数据应该包括过去多少个时间步
# # delay：目标应该在未来多少个时间步后
# # min_index,max_index：数组中的索引，界定需要抽取哪些时间步，有助于保存数据用于验证和测试
# # shuffle：是否打乱样本
# # batch_size：每个批量的样本数
# # step：数据采样周期，每个时间步是10min，设置为6，即每小时取一个数据点
# def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
#     if max_index is None:
#         max_index = len(data) - delay - 1
#     i = min_index + lookback
#     while 1:
#         if shuffle:
#             rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
#         else:
#             if i + batch_size >= max_index:
#                 i = min_index + lookback
#             rows = np.arange(i, min(i + batch_size, max_index))
#             i += len(rows)
#         samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
#         targets = np.zeros((len(rows),))
#         for j, row in enumerate(rows):
#             indices = range(rows[j] - lookback, rows[j], step)
#             samples[j] = data[indices]
#             targets[j] = data[rows[j] + delay][1]
#         yield samples, targets
#
# # 准备训练生成器，验证生成器，测试生成器
# # 输入数据包括过去10天内的数据，每小时抽取一次数据点
# # 目标为一天以后的天气，批次样本数为128
# lookback = 1440
# step = 6
# delay = 144
# batch_size = 128
# train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000,
#                       shuffle=True, step=step, batch_size=batch_size)
# val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
#                     shuffle=True, step=step, batch_size=batch_size)
# train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None,
#                       shuffle=True, step=step, batch_size=batch_size)
# val_steps = (300000 - 200001 - lookback) // batch_size
# test_steps = (len(float_data) - 300001 - lookback) // batch_size
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:08:46 2020
@author: Administrator
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# 下载天气数据集
# zip_path = tf.keras.utils.get_file(
#     origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
#     fname='jena_climate_2009_2016.csv.zip',
#     extract=True)
# csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv("219.csv")
print(df.head())  # 瞧一瞧数据集长啥样
# 每10分钟有一次观测数据。1小时有6次观测数据，1天有6x24=144次观测数据
print(df.shape)  # (420551, 15)，2920天（8年）的天气数据
'''
假设我们需要预测未来6小时的气温，为了做预测，我们可以选择5天的观测数据，这样我们就选择144x5 = 720个数据作为窗口来训练模型。
下面的函数就是返回类似这样的窗口。参数history_size是需要的历史数据个数，target_size 为需要预测的数据点个数。
'''
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)

# 头30万条数据作为训练集，剩下的作为验证集
TRAIN_SPLIT = 15000
tf.random.set_seed(13)
# Forecast a univariate time series 预测单变量（温度）
uni_data = df['TMAX']
uni_data.index = df['DATE']
print(uni_data.head())
# uni_data.plot(subplots=True) #绘制历史数据
uni_data = uni_data.values

# 数据标准化（减去均值，再除以标准差）
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data - uni_train_mean) / uni_train_std
univariate_past_history = 3650  # 用144个历史数据点
univariate_future_target = 30  # 预测接下来的6个数据点
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)
print('Single window of past history')
print(x_train_uni[0])
print('\n Target temperature to predict')
print(y_train_uni[0])

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']  # 红色叉死为真值，绿色圆点为预测值
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0
    plt.title(title)

    for i, x in enumerate(plot_data):
        if i == 0:  # 历史
            historyLine, = plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        else:  # 真值或预测值
            # historyLine.get_c()获取历史线的颜色，对应的数据保持颜色一致
            plt.plot(range(future), plot_data[i], marker[i], color=historyLine.get_c(),  #
                     markersize=10, label=labels[i])  #
    plt.legend()
    # plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

# show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
def baseline(history):  # baseline只是简单的将过去历史记录的均值作为预测值
    return np.mean(history)

# baseline 结果绘图

 # show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,'Baseline Prediction Example')


BATCH_SIZE = 256
BUFFER_SIZE = 10000
#训练集
train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()#打乱训练集
#验证集
val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat() #打乱验证集

#创建一个简单的LSTM网络模型
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=8, input_shape=x_train_uni.shape[-2:],activation="tanh"),#units：输出空间的维度
    tf.keras.layers.Dense(1)
])
simple_lstm_model.compile(optimizer='adam', loss='mae')#模型编译，设定优化器和损失类型
#做个简单的预测来检查模型的输出
for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

#因为数据集很大，为了节省时间，每个EPOCH仅跑300步，没有跑完所有训练数据
EVALUATION_INTERVAL = 200
EPOCHS = 1
simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)
for x, y in val_univariate.take(3):#做3次预测
  #plot = show_plot([x[0].numpy(), y[0].numpy(),simple_lstm_model.predict(x)[0]],
                   #delta=univariate_future_target,
                   #title= 'Simple LSTM model')
  plot = show_plot([x[0].numpy(), y[0:univariate_future_target].numpy(),simple_lstm_model.predict(x)[0:univariate_future_target]],
                   delta=univariate_future_target,
                   title= 'Simple LSTM model')
  plot.show()