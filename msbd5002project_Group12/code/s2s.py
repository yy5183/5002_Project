import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import regularizers
from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
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


def createModel(encoder_X, decoder_X, decoder_y, latent_dim, reg):

    encoder_inputs = Input(shape=(encoder_X.shape[1], encoder_X.shape[2]))
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(decoder_X.shape[1], decoder_X.shape[2]))
    decoder_gru = GRU(latent_dim, return_sequences=True, bias_regularizer=reg)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(decoder_y.shape[2],activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)

    return model

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def train(reg,inHours=24*3,outputHours=24,epochs=50,batch_size=128):
    ds = pd.read_csv('dealedData/combine/Train.csv', header=0, index_col=1)
    # stats = set(ds['station_id'].unique())
    # stats = {'dongsi_aq', 'dongsihuan_aq', 'qianmen_aq', 'xizhimenbei_aq', 'gucheng_aq', 'yanqin_aq', 'dingling_aq', 'mentougou_aq', 'tongzhou_aq', 'yufa_aq', 'donggaocun_aq'}

    stats = {'dongsi_aq'}

    ds_stats_dic = {}
    for stat in stats:
        ds_stats_dic[stat] = ds[ds['station_id'] == stat]

    for key in ds_stats_dic.keys():
        print(key)
        ds = ds_stats_dic[key]
        # 删除station列
        ds = ds.drop(columns=['station_id'])

        nFeature = ds.shape[1]
        nPollu = 6
        nLabel = 3
        inputNum = inHours * nFeature
        outputNum = outputHours * nLabel

        # 标准化
        # values = ds.values
        # scaler = MinMaxScaler(feature_range=(0, 100))
        # data = scaler.fit_transform(values)

        data = ds.values

        # 重shape成前n个小时feature + 当前小时的lable
        reframedData = series_to_supervised(data, inHours, outputHours)
        print(reframedData.head(3))
        dropList = [j + nFeature * (inHours + i)
                        for i in range(outputHours)
                        for j in [2,3,5]]

        print('drop:', dropList)

        reframedData.drop(reframedData.columns[dropList], axis=1, inplace=True)

        # 调整列顺序
        labelColList = [j + nFeature*inHours + (nFeature-nPollu+nLabel)*i
                        for i in range(outputHours)
                        for j in range(nLabel)]

        labelColName = reframedData.columns[labelColList]
        print('labelColName:',labelColName)

        allColName = reframedData.columns
        dealedColName = allColName.drop(labelColName).append(labelColName)
        data = reframedData[dealedColName]

        # 打乱顺序
        print('data\n', data.shape)
        # data = data.sample(random_state=10, frac=1)
        print('data\n', data.shape)

        # 分为测试集和训练集
        values = data.values
        # train = values
        test_end_row = int(values.shape[0] * 0.1)
        test = values[:test_end_row, :]
        train = values[test_end_row:, :]


        # 分为feature和label
        encoder_X, decoder_X, decoder_y = train[:, :inputNum], train[:, inputNum:-outputNum], train[:, -outputNum:]
        test_enc_X, test_dec_X, test_dec_y = test[:, :inputNum], test[:, inputNum:-outputNum], test[:, -outputNum:]
        print(encoder_X.shape, decoder_X.shape, decoder_y.shape)


        # 重塑成3D形状 [样例, 时间步, 特征]
        encoder_X = encoder_X.reshape((encoder_X.shape[0], inHours, nFeature))
        decoder_X = decoder_X.reshape((decoder_X.shape[0], outputHours, nFeature - nPollu))
        decoder_y = decoder_y.reshape((decoder_y.shape[0], outputHours, nLabel))

        test_enc_X = test_enc_X.reshape((test_enc_X.shape[0], inHours, nFeature))
        test_dec_X = test_dec_X.reshape((test_dec_X.shape[0], outputHours, nFeature - nPollu))
        test_dec_y = test_dec_y.reshape((test_dec_y.shape[0], outputHours, nLabel))

        print(encoder_X.shape, decoder_X.shape, decoder_y.shape)
        print(test_enc_X.shape, test_dec_X.shape, test_dec_y.shape)


        # 拟合神经网络模型
        weight_path = "model/{}_weights.best.hdf5".format(key)
        checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='min', save_weights_only=True)

        reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5,
                                           verbose=0, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

        early = EarlyStopping(monitor="val_loss",
                              mode="min",
                              patience=6)

        callbacks_list = [checkpoint, early, reduceLROnPlat]

        # Run training

        model = createModel(encoder_X, decoder_X, decoder_y,reg)
        model.compile(optimizer='adam', loss='mse')
        history = model.fit([encoder_X, decoder_X], decoder_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=([test_enc_X, test_dec_X], test_dec_y),
                  callbacks=callbacks_list
                            )


        # 绘制历史数据
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # 做出预测
        yhat = model.predict([test_enc_X, test_dec_X])
        print(yhat.shape)

        rowsNum = len(test_dec_X) * outputHours

        # 反向转换预测值比例
        yhat = yhat.reshape((rowsNum, nLabel))
        # test_dec_X = test_dec_X.reshape((rowsNum, nFeature-nLabel))
        # inv_yhat = np.concatenate((yhat, test_dec_X), axis=1)
        # inv_yhat = scaler.inverse_transform(inv_yhat)
        # inv_yhat = inv_yhat[:, :nLabel]

        # 反向转换实际值比例
        test_dec_y = test_dec_y.reshape((rowsNum, nLabel))
        # inv_y = np.concatenate((test_dec_y, test_dec_X), axis=1)
        # inv_y = scaler.inverse_transform(inv_y)
        # inv_y = inv_y[:, :nLabel]

        # 计算RMSE
        # print(inv_yhat[0])
        # print(inv_y[0])
        # rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        # print('Test RMSE: %.3f' % rmse)

        # 计算smape
        inv_yhat = yhat
        inv_y = test_dec_y
        print(inv_yhat.shape, inv_yhat[0])
        print(inv_y.shape, inv_y[0])

        print('PM2.5:', smape(inv_yhat[:, 0], inv_y[:, 0]))
        print('PM10:', smape(inv_yhat[:, 1], inv_y[:, 1]))
        print('O3:', smape(inv_yhat[:, 2], inv_y[:, 2]))




# reg = regularizers.l1_l2(l1=0.0, l2=0.01)
# reg = regularizers.l1_l2(l1=0.01, l2=0.01)
reg = regularizers.l1_l2(l1=0.03, l2=0.015)

train(reg,inHours=24*4,outputHours=48,epochs=20,batch_size=128)
# train(reg,inHours=3,outputHours=2,epochs=10,batch_size=256)