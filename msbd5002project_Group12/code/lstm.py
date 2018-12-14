import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def data_prepro(data,inHours,outputHours,nLabel):

    nFeature = data.shape[1]
    inputNum = inHours * nFeature
    outputNum = outputHours * nLabel

    # 重shape成前n个小时feature + 当前小时的lable
    reframedData = series_to_supervised(data, inHours, outputHours)

    # 调整列顺序
    labelColList = [j + nFeature * (inHours + i)
                    for i in range(outputHours)
                    for j in range(nLabel)]

    labelColName = reframedData.columns[labelColList]
    # print(labelColName)
    allColName = reframedData.columns
    dealedColName = allColName.drop(labelColName).append(labelColName)
    data = reframedData[dealedColName]

    # 分为测试集和训练集
    values = data.values
    train_end_row = int(values.shape[0] * 0.8)
    train = values[:train_end_row, :]
    test = values[train_end_row:, :]

    # 分为feature和label
    train_X, train_y = train[:, :inputNum], train[:, -outputNum:]
    test_X, test_y = test[:, :inputNum], test[:, -outputNum:]
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # 重塑成3D形状 [样例, 时间步, 特征]
    train_X = train_X.reshape((train_X.shape[0], inHours, nFeature))
    test_X = test_X.reshape((test_X.shape[0], inHours, nFeature))
    # train_y = train_y.reshape((train_y.shape[0], outputHours, nLabel))
    # test_y = test_y.reshape((test_y.shape[0], outputHours, nLabel))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    return train_X,train_y,test_X, test_y

def createModel(train_X,train_Y):
    # LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(train_Y.shape[1]))
    model.compile(loss='mse', optimizer='adam')
    return model


ds = pd.read_csv('dealedData/combine/Train.csv', header=0, index_col=1)
# stats = set(ds['station_id'].unique())
stats = {'dongsi_aq', 'dongsihuan_aq', 'qianmen_aq', 'xizhimenbei_aq', 'gucheng_aq', 'yanqin_aq', 'dingling_aq', 'mentougou_aq', 'tongzhou_aq', 'yufa_aq', 'donggaocun_aq'}

ds_stats_dic = {}
for stat in stats:
    ds_stats_dic[stat] = ds[ds['station_id'] == stat]

for key in ds_stats_dic.keys():
    print(key)
    ds = ds_stats_dic[key]
    # 删除station列
    ds = ds.drop(columns=['station_id'])
    print(ds.shape)

    inHours = 24*3
    outputHours = 24
    batch_size = 128
    epochs = 5

    nFeature = ds.shape[1]
    nLabel = 6

    # 标准化
    values = ds.values
    scaler = MinMaxScaler(feature_range=(0, 100))
    values = scaler.fit_transform(values)
    print('标准化时的shape', values.shape)

    train_X, train_y, test_X, test_y = data_prepro(values,inHours,outputHours,nLabel)

    model = createModel(train_X,train_y)

    # 拟合神经网络模型
    weight_path = "model/{}_weights.best.hdf5".format(key)
    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', save_weights_only=True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10,
                                       verbose=0, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

    early = EarlyStopping(monitor="val_loss",
                          mode="min",
                          patience=10)

    callbacks_list = [checkpoint, early, reduceLROnPlat]
    history = model.fit(train_X, train_y, batch_size=batch_size,
              epochs=epochs, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, callbacks=callbacks_list)

    # 绘制历史数据
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # 做出预测
    yhat = model.predict(test_X)
    print(yhat.shape)

    test_X = test_X.reshape((test_X.shape[0] * inHours, nFeature))
    rowsNum = len(test_y) * outputHours

    # 反向转换预测值比例
    yhat = yhat.reshape((rowsNum, nLabel))
    inv_yhat = np.concatenate((yhat, test_X[:rowsNum, -(nFeature - nLabel):]), axis=1)
    # inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, :nLabel]

    # 反向转换实际值比例
    test_y = test_y.reshape((rowsNum, nLabel))
    inv_y = np.concatenate((test_y, test_X[:rowsNum, -(nFeature - nLabel):]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, :nLabel]

    # 计算RMSE
    print(inv_yhat[0])
    print(inv_y[0])
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    # rmse = sqrt(mean_squared_error(yhat, test_y))
    print('Test RMSE: %.3f' % rmse)

