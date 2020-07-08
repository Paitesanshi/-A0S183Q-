import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt

# df = pd.read_csv("219.csv")#读取出来为数值
# float_data = df['TMAX']
# print(df.head())
# print(float_data)

f = open("219.csv")#读取出来为字符
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(lines)
print(len(lines))
# float_data = np.zeros((len(lines), len(header) - 1))
# for i, line in enumerate(lines):
#     values = [float(x) for x in line.split(',')[1:]]
#     float_data[i, :] = values
# df = pd.read_csv("219.csv")#读取出来为数值
# float_data = df['TMAX']
df = pd.read_csv("219.csv")#读取出来为数值
float_data = df['TMAX']
print(float_data)

temp = float_data
plt.plot(range(len(temp)), temp)
plt.plot(range(365), temp[:365])

# 数据标准化（减去平均数，除以标准差）
mean = float_data[:5000].mean(axis=0)
float_data -= mean
std = float_data[:5000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=64, step=6):
    if max_index is None:
        max_index = len(float_data) - delay - 1
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
            indices = range(rows[j] - lookback-1000, rows[j]-1000, step)
            print(indices)
            print(data)
            print(len(data))
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# 准备训练生成器，验证生成器，测试生成器
# 输入数据包括过去10天内的数据，每小时抽取一次数据点
# 目标为一天以后的天气，批次样本数为128
lookback = 365
step = 1
delay = 72
batch_size = 64
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=11000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=11000, max_index=22000,
                    shuffle=True, step=step, batch_size=batch_size)
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=22001, max_index=None,
                      shuffle=True, step=step, batch_size=batch_size)
val_steps = (22000 - 11001 - lookback) // batch_size
test_steps = (len(float_data) - 22001 - lookback) // batch_size
# 训练并评估一个使用dropout正则化的基于GRU的模型
# 不再过拟合，分数更稳定

model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mse')
history = model.fit_generator(train_gen,steps_per_epoch=200, epochs=10, validation_data=val_gen,
                              validation_steps=val_steps)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()