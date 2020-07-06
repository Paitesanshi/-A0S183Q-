import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras import losses

#load data and set index
df = pd.read_csv('219.csv',infer_datetime_format=True)
df.set_index(inplace=True, keys=['DATE'])
# df = df.drop(['tmin'],axis=1)

#show the dataset
df.head(5)

print(df.shape)

#preparation for the dataset
def get_date_str(x, formate='%Y-%m-%d %H:%M:%S'):
    return x.strftime(formate)
start = datetime(1951, 1, 1)
start = get_date_str(start)
end = datetime(1952, 1, 1)
end = get_date_str(end)
# df = df.loc[start:end]
print(df.shape)

# #show the trend of "visibility"
# plt.figure(figsize=(20, 8))
# plt.plot(df['TMAX'])
# plt.show()


#preparation for the train dataset and test dataset
time_stamp = 24
train = df[0:365+time_stamp]
test = df[365-time_stamp:]
print(train.shape)
print(test.shape)


#set feature_range=(-1,1)
scaler = MinMaxScaler(feature_range=(-1,1))
#scale the train data
scaled_data = scaler.fit_transform(train)
x_train = []
y_train = []
print(scaled_data.shape)
print(scaled_data[0,1])

for i in range(time_stamp,len(train)):
    x_train.append(scaled_data[i-time_stamp:i])
    y_train.append(scaled_data[i,1])
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train.shape)

#scale the test data
scaled_data = scaler.fit_transform(test)
x_test = []
y_test = []
print(scaled_data.shape)
print(scaled_data[0,1])

for i in range(time_stamp,len(test)):
    x_test.append(scaled_data[i-time_stamp:i])
    y_test.append(scaled_data[i,1])

x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_test.shape)
print(y_test.shape)

#choose model to predict
cell_type = 'LSTM'

#set hyperparameters
epochs = 30
batch_size = 32

model = Sequential()
if cell_type == 'LSTM':
    model.add(LSTM(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    model.add(LSTM(units=50))
if cell_type == 'GRU':
    model.add(GRU(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    model.add(GRU(units=50))
if cell_type == 'RNN':
    model.add(SimpleRNN(units=100, return_sequences=True, input_dim=x_train.shape[-1], input_length=x_train.shape[1]))
    model.add(SimpleRNN(units=50))
#model.add(keras.layers.Dropout(rate=0.2)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

print("mean_squared_error",mean_squared_error)

#inverse_transform
vis = model.predict(x_test)

scaler.fit_transform(pd.DataFrame(test['TMAX'].values))

vis = scaler.inverse_transform(vis)
y_test = scaler.inverse_transform([y_test])

rmse = np.sqrt(np.mean(np.power((y_test - vis),2)))
print(rmse)