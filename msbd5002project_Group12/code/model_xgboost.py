import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
import os
os.environ ['KMP_DUPLICATE_LIB_OK'] ='True'


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


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def train(inHours=24*3,outputHours=24):
    ds = pd.read_csv('dealedData/combine/Train.csv', header=0, index_col=1)
    # stats = set(ds['station_id'].unique())
    # stats = {'dongsi_aq', 'dongsihuan_aq', 'qianmen_aq', 'xizhimenbei_aq', 'gucheng_aq', 'yanqin_aq', 'dingling_aq', 'mentougou_aq', 'tongzhou_aq', 'yufa_aq', 'donggaocun_aq'}

    ds['station_id'] = LabelEncoder().fit_transform(ds['station_id'])
    ds = ds.iloc[:10000]

    print(ds.shape)
    print(ds.head(4))


    # 删除station列
    # ds = ds.drop(columns=['station_id'])

    nFeature = ds.shape[1]
    nDellabel = 4
    # nPollu = 6
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
    dropList = [j + nFeature * (inHours + i)
                    for i in range(outputHours)
                    for j in [0,3,4,6]]

    print('drop:', reframedData.columns[dropList])

    reframedData.drop(reframedData.columns[dropList], axis=1, inplace=True)

    # 调整列顺序
    labelColList = [j + nFeature*inHours + (nFeature-nDellabel)*i
                    for i in range(outputHours)
                    for j in range(nLabel)]

    labelColName = reframedData.columns[labelColList]
    print('labelColName:',labelColName)

    allColName = reframedData.columns
    dealedColName = allColName.drop(labelColName).append(labelColName)
    data = reframedData[dealedColName]

    # 打乱顺序
    data = data.sample(random_state=10,frac=1)

    # 分为测试集和训练集
    values = data.values
    # train = values
    train_end_row = int(values.shape[0] * 0.8)
    train = values[:train_end_row, :]
    test = values[train_end_row:, :]


    # 分为feature和label
    train_X, train_y = train[:, :-outputNum], train[:, -outputNum:]
    test_X, test_y = test[:, :-outputNum], test[:, -outputNum:]
    print(train_X.shape, train_y.shape)

    # bst = xgb.XGBRegressor(learning_rate=0.1,
    #                      booster='gbtree',
    #                                n_estimators=300,
    #                                max_depth=8,
    #                                min_child_weight=1,
    #                                gamma=0,
    #                                subsample=0.8,
    #                                colsample_bytree=0.8,
    #                                objective='reg:linear',
    #                                nthread=2,
    #                                scale_pos_weight=1,
    #                                seed=10
    #                                )

    bst = xgb.XGBRegressor(learning_rate=0.1,
                           booster='gbtree',
                           subsample=0.8,
                           objective='reg:linear',
                           seed=10
                           )
    multioutputregressor = MultiOutputRegressor(bst).fit(train_X, train_y)

    y = multioutputregressor.predict(test_X)
    print(y[0])
    print(test_y[0])

    print('PM2.5:', smape(y[:, 0], test_y[:, 0]))
    print('PM10:', smape(y[:, 1], test_y[:, 1]))
    print('O3:', smape(y[:, 2], test_y[:, 2]))


# train(inHours=3*24,outputHours=48)
ds = pd.read_csv('dealedData/combine/Train.csv', header=0, index_col=1)
# stats = set(ds['station_id'].unique())
# stats = {'dongsi_aq', 'dongsihuan_aq', 'qianmen_aq', 'xizhimenbei_aq', 'gucheng_aq', 'yanqin_aq', 'dingling_aq', 'mentougou_aq', 'tongzhou_aq', 'yufa_aq', 'donggaocun_aq'}

ds['station_id'] = LabelEncoder().fit_transform(ds['station_id'])
ds = ds.iloc[:10000]
inHours=3*24
outputHours=48

print(ds.shape)

# 删除station列
# ds = ds.drop(columns=['station_id'])

nFeature = ds.shape[1]
nDellabel = 4
# nPollu = 6
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
dropList = [j + nFeature * (inHours + i)
                for i in range(outputHours)
                for j in [0,3,4,6]]

print('drop:', reframedData.columns[dropList])

reframedData.drop(reframedData.columns[dropList], axis=1, inplace=True)

# 调整列顺序
labelColList = [j + nFeature*inHours + (nFeature-nDellabel)*i
                for i in range(outputHours)
                for j in range(nLabel)]

labelColName = reframedData.columns[labelColList]
print('labelColName:',labelColName)

allColName = reframedData.columns
dealedColName = allColName.drop(labelColName).append(labelColName)
data = reframedData[dealedColName]

# 分为测试集和训练集
values = data.values
# train = values
train_end_row = int(values.shape[0] * 0.8)
train = values[:train_end_row, :]
test = values[train_end_row:, :]


# 分为feature和label
train_X, train_y = train[:, :-outputNum], train[:, -outputNum:]
test_X, test_y = test[:, :-outputNum], test[:, -outputNum:]
print(train_X.shape, train_y.shape)

# bst = xgb.XGBRegressor(learning_rate=0.1,
#                      booster='gbtree',
#                                n_estimators=300,
#                                max_depth=8,
#                                min_child_weight=1,
#                                gamma=0,
#                                subsample=0.8,
#                                colsample_bytree=0.8,
#                                objective='reg:linear',
#                                nthread=2,
#                                scale_pos_weight=1,
#                                seed=10
#                                )

bst = xgb.XGBRegressor(#learning_rate=0.1,
                       booster='gbtree',
                       subsample=0.8,
                       objective='reg:linear',
                       seed=10
                       )

multioutputregressor = MultiOutputRegressor(bst).fit(train_X, train_y)
y = multioutputregressor.predict(test_X)
print(y.shape,test_y.shape)
y = y.reshape((outputHours, nLabel))
test_y = test_y.reshape((outputHours, nLabel))
print(y[0])
print(test_y[0])

print('PM2.5:', smape(y[:, 0], test_y[:, 0]))
print('PM10:', smape(y[:, 1], test_y[:, 1]))
print('O3:', smape(y[:, 2], test_y[:, 2]))