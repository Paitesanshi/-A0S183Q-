import numpy as np
xTrain = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]  # 训练数据（日期）
yTrain = np.array([33, 35, 28, 20, 26, 27, 23, 22, 22])  # 训练数据（最高气温）
# 查看天气统计数据
print("天气数据统计：")
print("最低：%.2f, 最高：%.2f, 平均：%.2f, 中位数：%.2f, 标准差：%.2f" %
      (np.min(yTrain), np.max(yTrain), np.mean(yTrain), np.median(yTrain), np.std(yTrain)))



''' 使用LinearRegression，进行归一化预处理 '''
import numpy as np
from sklearn.linear_model import LinearRegression

def normalizeData(X):
    # 每列(每个Feature)分别求出均值和标准差，然后与X的每个元素分别进行操作

    return (X - X.mean(axis=0)) / X.std(axis=0)

train_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[:, np.newaxis]  # 训练数据（日期）
train_temp = np.array([33, 35, 28, 20, 26, 27, 23, 22, 22])[:, np.newaxis]  # 训练数据（最高气温）
xTrain = np.array(train_data[:, 0:2])
yTrain = np.array(train_temp[:, -1])
xTrain = normalizeData(xTrain)
xTrain = np.c_[xTrain, np.ones(len(xTrain))]  # 归一化完成后再添加intercept item列
model = LinearRegression()
model.fit(xTrain, yTrain)
print("LinearRegression计算R方：", model.score(xTrain, yTrain))
