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
np.set_printoptions(precision=16)


def createModel(encoder_X, decoder_X, reg):

    latent_dim = 128

    encoder_inputs = Input(shape=(encoder_X.shape[1], encoder_X.shape[2]))
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(decoder_X.shape[1], decoder_X.shape[2]))
    decoder_gru = GRU(latent_dim, return_sequences=True, bias_regularizer=reg)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(6,activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)

    return model

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

ds = pd.read_csv('dealedData/combine/Test_feature.csv', header=0,index_col=1)
train = pd.read_csv('dealedData/combine/test_enc_feature.csv', header=0,index_col=1)
scTrain = pd.read_csv('dealedData/combine/Train.csv', header=0, index_col=1)

colname = ['test_id','PM2.5','PM10','NO2','CO','O3','SO2']
res = pd.DataFrame(columns = colname)

stats = set(scTrain['station_id'].unique())
# stats = {'zhiwuyuan_aq'}
# stats = {'gucheng_aq'}

ds_stats_dic = {}
train_stats_dic = {}
scTrain_stats_dic = {}
for stat in stats:
    ds_stats_dic[stat] = ds[ds['station_id'] == stat]
    train_stats_dic[stat] = train[train['station_id'] == stat]
    scTrain_stats_dic[stat] = scTrain[scTrain['station_id'] == stat]

for key in ds_stats_dic.keys():
    print(key)
    ds = ds_stats_dic[key]
    train = train_stats_dic[key]
    scTrain = scTrain_stats_dic[key]

    if key == 'zhiwuyuan_aq':
        print(train_stats_dic.keys())
        train = train_stats_dic['gucheng_aq']

    # print(ds)

    # 删除station列
    ds = ds.drop(columns=['station_id'])
    train = train.drop(columns=['station_id'])

    scTrain = scTrain.drop(columns=['station_id'])

    # 标准化
    a = train.iloc[:len(ds),:6]
    a = a.reset_index(drop=True)
    ds = ds.reset_index(drop=True)
    d = pd.concat([a,ds],axis=1)
    # print(d.shape)

    values = scTrain.values
    # print(scTrain.head(1))
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaler.fit_transform(values)

    tra = scaler.transform(train.values)
    data = scaler.transform(d.values)
    # print(data[0])
    data = data[:,6:]
    # print(data[0])


    # 重shape成前n个小时feature + 当前小时的lable
    decReframedData = series_to_supervised(data, 0, 48)
    # print(decReframedData.head(3))
    # print(decReframedData.shape)

    # 重shape成前n个小时feature + 当前小时的lable
    tra = train.values
    encReframedData = series_to_supervised(tra, 24*4, 0)
    # print(encReframedData.head(3))
    # print(encReframedData.shape)


    enc = encReframedData.iloc[-1:,:].values
    dec = decReframedData.values
    encoder_X = enc.reshape((enc.shape[0], 24*4, 27))
    decoder_X = dec.reshape((dec.shape[0], 24*2, 21))

    print(encoder_X.shape, decoder_X.shape)

    weight_path = "s2s_models/{}_weights.best.hdf5".format(key)
    reg = regularizers.l1_l2(l1=0.03, l2=0.015)
    model = createModel(encoder_X,decoder_X,reg)
    model.load_weights(weight_path)

    y = model.predict([encoder_X, decoder_X])
    # print(y.shape)
    # print(y[0])


    rowsNum = len(encoder_X) * 24 * 2

    # 反向转换预测值比例
    yp = y.reshape((rowsNum, 6))
    decoder_X = decoder_X.reshape((rowsNum, 21))
    inv_yhat = np.concatenate((yp, decoder_X), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    # print(inv_yhat[0])
    inv_yhat = inv_yhat[:, :6]
    # print(inv_yhat.shape)
    # print(inv_yhat[0])

    for i in range(48):
        sid = np.array([key +'#'+str(i)])
        row = np.concatenate((sid, inv_yhat[i]), axis=0)
        df_i = pd.DataFrame([row], columns=colname)
        # print(df_i)
        res = res.append(df_i)
# print(res)
res = res.drop(columns=['NO2','CO','SO2'])
res.to_csv('result.csv',index=None)

#
# train = pd.read_csv('dealedData/combine/Train.csv', header=0)
# t = train[(train['time'].str.contains('2018-04-30')) | (train['time'].str.contains('2018-04-29'))
#  | (train['time'].str.contains('2018-04-28')) | (train['time'].str.contains('2018-04-27')) |
#           (train['time'].str.contains('2018-04-26'))]
# t.to_csv('dealedData/combine/test_enc_feature.csv',index=None)