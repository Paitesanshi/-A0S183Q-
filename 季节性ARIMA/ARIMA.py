# coding='utf-8'
import numpy as np
from matplotlib import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

params = {'font.family': 'serif',
          'font.serif': 'FangSong',
          'font.style': 'italic',
          'font.weight': 'normal',  # or 'blod'
          'font.size': 12,  # 此处貌似不能用类似large、small、medium字符串
          'axes.unicode_minus': False
          }
rcParams.update(params)

# 未来pandas版本会要求显式注册matplotlib的转换器，所以添加了下面两行代码，否则会报警告


register_matplotlib_converters()


def load_data(i):
    from datetime import datetime
    date_parse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv('乌鲁木齐/%d.csv'%(i+1),
                       index_col='date',  # 指定索引列
                       parse_dates=['date'],  # 将指定列按照日期格式来解析
                       date_parser=date_parse  # 日期格式解析器
                       )
    ts = data['tmax']
    print(ts.head(10))
    return ts


def use_rolling_statistics(time_series_datas):
    roll_mean = time_series_datas.rolling(window=12).mean()
    roll_std = time_series_datas.rolling(window=12).std()


def use_df(time_series_datas):
    from statsmodels.tsa.stattools import adfuller
    dftest = adfuller(time_series_datas, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)


def use_moving_avg(ts_log):
    moving_avg_month = ts_log.rolling(window=12).mean()
    return moving_avg_month


def use_exponentially_weighted_moving_avg(ts_log):
    expweighted_avg = ts_log.ewm(halflife=12).mean()
    return expweighted_avg


def use_decomposition(ts_log):
    decomposition = seasonal_decompose(ts_log, freq=12)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # 衡量趋势强度
    r_var = residual.var()
    tr_var = (trend + residual).var()
    f_t = np.maximum(0, 1.0 - r_var / tr_var)
    # print(f_t)
    # 衡量季节性强度
    sr_var = (seasonal + residual).var()
    f_s = np.maximum(0, 1.0 - r_var / sr_var)
    # print(f"-------趋势强度:{f_t},季节性强度:{f_s}------")
    return residual


def transform_stationary(ts):
    # 利用log降低异方差性
    ts_log = np.log(ts)
    rs_log_diff = ts_log - ts_log.shift()  # 1阶差分
    rs_log_diff.dropna(inplace=True)
    use_df(rs_log_diff)
    return ts_log, rs_log_diff


def order_determination(ts_log_diff):
    lag_acf = acf(ts_log_diff, nlags=10, fft=False)
    lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')
    z = 1.96


def draw_rss_plot(ts_log_diff, orders, title, freq='MS'):
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(ts_log_diff, order=orders, freq=freq)
    results_fitted = model.fit(disp=-1)
    return results_fitted.fittedvalues


def draw_future_plot(ts_log_diff, orders, seasonal_order, title, freq='MS'):
    # 季节性ARIMA模型
    model = SARIMAX(ts_log_diff, order=orders, seasonal_order=seasonal_order)
    results_fitted = model.fit(disp=5)
    fit_values = results_fitted.fittedvalues
    # print(results_fitted.summary())
    fc = results_fitted.forecast(108)
    conf = None

    return fit_values, fc, conf, title


def build_arima(ts_log_diff):
    order = (3, 0, 2)  # 变量自回归拟合一定的波动 + 预测误差自回归拟合一定的波动
    seasonal_order = (0, 1, 0, 12)  # 季节性差分，季节窗口=12个月
    fittedvalues, fc, conf, title = draw_future_plot(ts_log_diff, order, seasonal_order,
                                                     '预测：%s,%s' % (str(order), str(seasonal_order)))
    return fittedvalues, fc, conf, title


def transform_back(ts, fittedvalues, fc, conf, title):
    # Make as pandas series
    future_index = pd.date_range(start=ts.index[-1], freq='MS', periods=108)
    fc_series = pd.Series(fc, index=future_index)
    # print(fc_series.head())
    # print(fittedvalues.head(24))
    lower_series, upper_series = None, None
    if conf is not None:
        lower_series = pd.Series(conf[:, 0], index=future_index)
        upper_series = pd.Series(conf[:, 1], index=future_index)

    current_ARIMA_log = pd.Series(fittedvalues, copy=True)
    future_ARIMA_log = pd.Series(fc_series, copy=True)

    # 逆log
    current_ARIMA = np.exp(current_ARIMA_log)
    future_ARIMA = np.exp(future_ARIMA_log)
    # Plot
    # plt.figure(figsize=(12, 5), dpi=100)
    # plt.plot(ts, label='current_actual')
    # plt.plot(current_ARIMA, label='current_fit')
    # plt.plot(future_ARIMA, label='forecast', marker='o', ms=2)

    future_ARIMA.to_csv("乌鲁木齐/乌鲁木齐_max.csv", mode='a+')

    # print(future_ARIMA)
    if lower_series is not None:
        pass
    # plt.title('Forecast vs Actuals %s' % title)
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()


def plot_lag(rs):
    from pandas.plotting import lag_plot
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:4]):
        lag_plot(rs, lag=i + 1, ax=ax, c='firebrick')
        ax.set_title('Lag ' + str(i + 1))

    fig.suptitle('Lag Plots of AirPassengers', y=1.15)
    # plt.show()


def SampEn(U, m, r):

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))]
        return sum(C)

    N = len(U)
    return -np.log(_phi(m + 1) / _phi(m))


if __name__ == '__main__':

    for i in range(31):
        # 加载时间序列数据
        _ts = load_data(i)
        # 使用样本熵评估可预测性
        # print(f'原序列样本熵:{SampEn(_ts.values, m=2, r=0.2 * np.std(_ts.values))}')
        # 检验平稳性
        use_rolling_statistics(_ts)  # rolling 肉眼
        use_df(_ts)  # Dickey-Fuller Test 量化
        # 平稳变换
        _ts_log, _rs_log_diff = transform_stationary(_ts)
        # 使用样本熵评估可预测性
        # print(f'平稳变换后的序列样本熵:{SampEn(_ts.values, m=2, r=0.2 * np.std(_ts.values))}')
        # acf,pacf定阶分析
        order_determination(_rs_log_diff)
        # 构建模型
        _fittedvalues, _fc, _conf, _title = build_arima(
            _ts_log)  # 这里只传取log后的序列是因为后面会通过指定ARIMA模型的参数d=1来做一阶差分，这样在预测的时候，就不需要手动做逆差分来还原序列，而是由ARIMA模型自动还原
        # 预测,并绘制预测结果图
        transform_back(_ts, _fittedvalues, _fc, _conf, _title)