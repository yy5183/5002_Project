import pandas as pd
import numpy as np
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

def grDataPreprocess():
    # gd1增加除1月份以外的weather
    gd = pd.read_csv('dealedData/gridWeather/gridWeather_201701-201803.csv',header=0)
    gd = gd.rename(columns={'stationName':'station_id','utc_time':'time','wind_speed/kph':'wind_speed'})
    gd = gd.drop(columns=['longitude','latitude'], axis=1)

    wr = pd.read_csv('dealedData/weather_relation.csv',header=0)
    ob = pd.read_csv('dealedData/obWeather.csv')

    gd['weather'] = None

    for i in range(len(gd)):
        print(i)
        gd_id = gd['station_id'][i]
        time = gd['time'][i]
        if "2017-01" in time:
            continue
        ob_id = wr[wr['id'] == gd_id]['observer'].to_string(index=False) + '_meo'

        we = ob[(ob['time'] == time) & (ob['station_id'] == ob_id)].weather
        we = we.to_string(index=False)

        gd['weather'][i] = we
    gd.to_csv('dealedGd1.csv',index=None)

    # 提取ob18年1月份的数据
    ob = pd.read_csv('OB_dealedWeather.csv', header=0)
    ob_Jan = pd.DataFrame()

    for i in range(len(ob)):
        time = ob['time'][i]
        if '2018-01' in time or '2017-01-30' in time or '2017-01-31' in time:
            a = ob.iloc[i]
            ob_Jan = pd.concat([ob_Jan,pd.DataFrame(a).T])
            # print(ob_Jan)
    ob_Jan.to_csv('OB_Jan.csv',index=None)


    # gd1增加1月份的weather(用ob的18年1月份)
    gr1 = pd.read_csv('GR_addWeahterPart.csv', header=0)
    ob_jan = pd.read_csv('OB_Jan.csv', header=0)
    wr = pd.read_csv('dealedData/relation_map/GR_OB_weathermap.csv',header=0)

    for i in range(len(gr1)):
        print(i)
        time = gr1['time'][i]
        if '2017-01' not in time:
            continue
        if '01-30' not in time and '01-31' not in time:
            time = time.replace('2017', '2018')

        gd_id = gr1['station_id'][i]
        ob_id = wr[wr['id'] == gd_id]['observer'].to_string(index=False) + '_meo'
        we = ob_jan[(ob_jan['time'] == time) & (ob_jan['station_id'] == ob_id)].weather
        we = we.to_string(index=False)
        gr1['weather'][i] = we
    gr1.to_csv('GR_addWeahter.csv',index=None)


    # 合并各个gr
    gr1 = pd.read_csv('dealedData/GR/GR_addWeahter.csv', header=0)
    gr2 = pd.read_csv('dealedData/GR/gridWeather_201804.csv', header=0)
    gr2 = gr2.drop(columns=['id'], axis=1)
    print(gr1.info())
    print(gr2.info())
    gr = pd.concat([gr1,gr2],axis=0)
    print(gr.info())
    gr.to_csv('dealedData/GR/GR_combine.csv', index=None)

    # gr数据处理
    gr = pd.read_csv('dealedData/GR/GR_combine.csv', header=0)
    a = gr.groupby(['weather'], as_index=False)['weather'].agg({'cnt': 'count'})
    print(a)
    gr.loc[gr['weather'] == 'Sunny/clear\nSunny/clear', 'weather'] = 'Sunny/clear'
    gr.loc[gr['weather'] == 'Hail\nHail', 'weather'] = 'Hail'
    gr.loc[gr['weather'] == 'Cloudy\nCloudy', 'weather'] = 'Cloudy'
    gr.loc[gr['weather'] == 'Overcast\nOvercast', 'weather'] = 'Overcast'
    gr.loc[gr['weather'] == 'Light Rain\nLight Rain', 'weather'] = 'Light Rain'
    gr.loc[gr['weather'] == 'Sleet\nSleet', 'weather'] = 'Sleet'

    gr.loc[gr['weather'] == 'PARTLY_CLOUDY_DAY', 'weather'] = 'CLOUDY'
    gr.loc[gr['weather'] == 'PARTLY_CLOUDY_NIGHT', 'weather'] = 'CLOUDY'
    gr.loc[gr['weather'] == 'CLEAR_DAY', 'weather'] = 'CLEAR'
    gr.loc[gr['weather'] == 'CLEAR_NIGHT', 'weather'] = 'CLEAR'
    gr.loc[gr['weather'] == 'WIND', 'weather'] = 'CLOUDY'

    gr.loc[gr['weather'] == 'Sunny/clear', 'weather'] = 'CLEAR'
    gr.loc[gr['weather'] == 'Dust', 'weather'] = 'SAND'
    gr.loc[gr['weather'] == 'Sleet', 'weather'] = 'SNOW'
    gr.loc[gr['weather'] == 'Rain/Snow with Hail', 'weather'] = 'HAIL'
    gr.loc[gr['weather'] == 'Rain with Hail', 'weather'] = 'HAIL'
    gr.loc[gr['weather'] == 'Light Rain', 'weather'] = 'RAIN'
    gr.loc[gr['weather'] == 'Thundershower', 'weather'] = 'RAIN'
    gr.loc[gr['weather'] == 'Overcast', 'weather'] = 'CLOUDY'
    gr['weather'] = gr['weather'].apply(lambda x: x.upper())
    a = gr.groupby(['weather'], as_index=False)['weather'].agg({'cnt': 'count'})
    print(a)
    gr.to_csv('dealedData/GR/GR_train.csv', index=None)

    # 处理风向为8个方向
    gr = pd.read_csv('dealedData/GR/GR_train.csv', header=0)
    for i in range(len(gr)):
        print(i)
        dr = gr['wind_direction'][i]
        spd = gr['wind_speed'][i]
        if spd <= 0.2:
            gr['wind_direction'][i] = 'dr0'
            # print(gr.iloc[i])
            continue

        if dr <= 45:
            gr['wind_direction'][i] = 'dr1'
        elif dr <= 90:
            gr['wind_direction'][i] = 'dr2'
        elif dr <= 135:
            gr['wind_direction'][i] = 'dr3'
        elif dr <= 180:
            gr['wind_direction'][i] = 'dr4'
        elif dr <= 225:
            gr['wind_direction'][i] = 'dr5'
        elif dr <= 270:
            gr['wind_direction'][i] = 'dr6'
        elif dr <= 315:
            gr['wind_direction'][i] = 'dr7'
        elif dr <= 360:
            gr['wind_direction'][i] = 'dr8'
        # print(gr.iloc[i])
    gr.to_csv('dealedData/GR/GR_train_winddrct.csv', index=None)

def obDataPreprocess():
    wr = pd.read_csv('dealedData/relation_map/AQ_GR_OB_MAP.csv',header=0)
    # 处理ob天气：全大写，更换类型
    ob = pd.read_csv('dealedData/OB/observedWeather_20180501-20180502.csv', header=0)
    ob.loc[ob['weather'] == 'Sunny/clear', 'weather'] = 'CLEAR'
    ob.loc[ob['weather'] == 'Dust', 'weather'] = 'SAND'
    ob.loc[ob['weather'] == 'Sleet', 'weather'] = 'SNOW'
    ob.loc[ob['weather'] == 'Rain/Snow with Hail', 'weather'] = 'HAIL'
    ob.loc[ob['weather'] == 'Rain with Hail', 'weather'] = 'HAIL'
    ob.loc[ob['weather'] == 'Light Rain', 'weather'] = 'RAIN'
    ob.loc[ob['weather'] == 'Thundershower', 'weather'] = 'RAIN'
    ob.loc[ob['weather'] == 'Overcast', 'weather'] = 'CLOUDY'
    ob['weather'] = ob['weather'].apply(lambda x:x.upper())
    print(ob['weather'].unique())
    ob.to_csv('OB_dealedWeather.csv',index=None)

    # 过滤OB数据
    ob = pd.read_csv('dealedData/OB/OB_dealedWeather.csv', header=0)
    OB_station_list = ['shijingshan_meo', 'huairou_meo', 'daxing_meo', 'fengtai_meo',
                        'fangshan_meo', 'tongzhou_meo', 'shunyi_meo', 'beijing_meo',
                        'yanqing_meo', 'chaoyang_meo', 'mentougou_meo', 'hadian_meo',
                        'miyun_meo', 'pinggu_meo', 'pingchang_meo']

    ob = ob[ob['station_id'].isin(OB_station_list)]
    a = ob['station_id'].unique()
    print(len(a), a)
    ob.to_csv('dealedData/OB/OB_dealedWeather_filterStation.csv', index=None)

    # 处理缺失数据，处理风向，处理无风999017
    ob = pd.read_csv('dealedData/OB/OB_dealedWeather_filterStation.csv', header=0)

    for i in range(len(ob)):
        dr = ob['wind_direction'][i]
        spd = ob['wind_speed'][i]
        if spd <= 0.2:
            ob['wind_direction'][i] = 'dr0'
            # print(ob.iloc[i])
            continue
        if dr <= 45:
            ob['wind_direction'][i] = 'dr1'
        elif dr <= 90:
            ob['wind_direction'][i] = 'dr2'
        elif dr <= 135:
            ob['wind_direction'][i] = 'dr3'
        elif dr <= 180:
            ob['wind_direction'][i] = 'dr4'
        elif dr <= 225:
            ob['wind_direction'][i] = 'dr5'
        elif dr <= 270:
            ob['wind_direction'][i] = 'dr6'
        elif dr <= 315:
            ob['wind_direction'][i] = 'dr7'
        elif dr <= 360:
            ob['wind_direction'][i] = 'dr8'

        row = ob.iloc[i]
        print('第', i, '次')
        nanCol = row[row.isna()].index.tolist()
        if len(nanCol) > 0:
            station = row['station_id']
            time = row['time']
            date = time.split(' ')[0]
            datas = ob[(ob['statio  n_id'] == station) & (ob.time.str.contains(date))]
            # print(datas)

            for col in nanCol:
                cols = datas[datas[col].notna()][col]
                if col == 'wind_direction':
                    col_fill = cols.mode().to_string(index=False)
                else:
                    col_fill = np.mean(cols)
                    col_fill = round(col_fill, 1)
                # print('avg: ', col_avg)
                ob[col][i] = col_fill

    ob.to_csv('dealedData/ob/ob_test.csv', index=None)

def combineGR_OB():
    # 处理三个站点对应关系文件
    map = pd.read_csv('dealedData/relation_map/AQ_GR_OB_MAP.csv', header=0)
    station = pd.read_csv('dealedData/relation_map/Station.csv', header=0)

    for i in range(len(map)):

        ows = map['ow_station'][i]
        grs = map['gw_station'][i]
        if pd.notna(ows):
            owsList = ows.split('/')
            owNameS = [o + '_meo' for o in owsList]
            map['ow_station'][i] = owNameS
        if pd.notna(grs):
            grslist = grs.split('/')
            grNameS = [(station[station['addr'] == g]['name']).to_string(index=False) for g in grslist]
            map['gw_station'][i] = grNameS
    map.to_csv('dealedData/combine/MAP.csv', index=None)


    map = pd.read_csv('dealedData/combine/MAP.csv', header=0)
    GR = pd.read_csv('dealedData/combine/GR.csv', header=0)
    OB = pd.read_csv('dealedData/combine/OB.csv', header=0)

    cols = ["station_id", "time", "weather", "humidity", "pressure", "temperature", "wind_direction", "wind_speed"]
    valCols = ["weather", "humidity", "pressure", "temperature", "wind_direction", "wind_speed"]
    Comb_Weather = pd.DataFrame(columns=cols)

    for i in range(len(map)):
    # for i in range(5,7,1):
        aq = map['aqs'][i]
        print(aq)
        grstr = map['grs'][i]
        obstr = map['obs'][i]
        grlist = []
        oblist = []
        if pd.notna(grstr):
            grlist = grstr.split('/')
        if pd.notna(obstr):
            oblist = obstr.split('/')

        if len(grlist) == 1 and len(oblist) == 0 :
            print('type 1')
            rows = GR[GR['station_id'] == grlist[0]]
            rows.loc[:,'station_id'] = aq
            # Comb_Weather.append(rows,ignore_index=True)
            Comb_Weather = pd.concat([Comb_Weather, rows],sort=False)
            print('len',len(Comb_Weather))
            continue
        if len(grlist) == 0 and len(oblist) == 1 :
            print('type 2')
            rows = OB[(OB['station_id'] == oblist[0])]
            rows.loc[:,'station_id'] = aq
            # Comb_Weather.append(rows,ignore_index=True)
            Comb_Weather = pd.concat([Comb_Weather, rows],sort=False)
            print('len',len(Comb_Weather))
            continue

        print('type 3')
        timeList = []
        for gr in grlist:
            time = GR[GR['station_id'] == gr].time.unique()
            timeList.extend(time)

        for ob in oblist:
            time = OB[OB['station_id'] == ob].time.unique()
            timeList.extend(time)

        timeSet = list(set(timeList))
        timeSet.sort(key=timeList.index)

        rowDf = pd.DataFrame(pd.DataFrame(columns=cols))
        for time in timeSet:
            print('     ',time)
            tempDf = pd.DataFrame(pd.DataFrame(columns=valCols))
            tempSr = pd.Series(index=cols)

            for gr in grlist:
                value = GR[(GR['station_id'] == gr) & (GR['time'] == time)]
                value = value.drop(columns=['station_id', 'time'])
                tempDf = pd.concat([tempDf, value])

            for ob in oblist:
                value = OB[(OB['station_id'] == ob) & (OB['time'] == time)]
                value = value.drop(columns=['station_id', 'time'])
                tempDf = pd.concat([tempDf, value])

            tempSr['station_id'] = aq
            tempSr['time'] = time
            tempSr['humidity'] = tempDf['humidity'].mean()
            tempSr['pressure'] = tempDf['pressure'].mean()
            tempSr['temperature'] = tempDf['temperature'].mean()
            tempSr['wind_speed'] = tempDf['wind_speed'].mean()

            tempSr['weather'] = tempDf['weather'].mode().values[0]
            tempSr['wind_direction'] = tempDf['wind_direction'].mode().values[0]
            # print(tempSr)
            # Comb_Weather.append(tempSr,ignore_index=True)
            rowDf.loc[len(rowDf)] = tempSr
        Comb_Weather = pd.concat([Comb_Weather, rowDf],sort=False)
        print('len', len(Comb_Weather))
        Comb_Weather.to_csv('dealedData/combine/Comb_Weather.csv', index=None)

    Comb_Weather.to_csv('dealedData/combine/Comb_Weather.csv', index=None)

    # 处理类型数据
    wt = pd.read_csv('dealedData/combine/Comb_Weather.csv', header=0)

    print(set(wt['wind_direction'].unique()))
    dum_attr = ['weather', 'wind_direction']
    for att in dum_attr:
        dum = pd.get_dummies(wt[att], prefix=att[:4])
        wt = pd.concat([wt, dum], axis=1)
        wt = wt.drop(att, axis=1)

    print(wt.head(3))
    wt.to_csv('dealedData/combine/feature_weather.csv', index=None)


map = pd.read_csv('dealedData/combine/MAP.csv', header=0)
GR = pd.read_csv('dealedData/combine/GR_test.csv', header=0)
OB = pd.read_csv('dealedData/combine/OB_test.csv', header=0)


cols = ["station_id", "time", "weather", "humidity", "pressure", "temperature", "wind_direction", "wind_speed"]
valCols = ["weather", "humidity", "pressure", "temperature", "wind_direction", "wind_speed"]
Comb_Weather = pd.DataFrame(columns=cols)

for i in range(len(map)):
    # for i in range(5,7,1):
    aq = map['aqs'][i]
    print(aq)
    grstr = map['grs'][i]
    obstr = map['obs'][i]
    grlist = []
    oblist = []
    if pd.notna(grstr):
        grlist = grstr.split('/')
    if pd.notna(obstr):
        oblist = obstr.split('/')

    if len(grlist) == 1 and len(oblist) == 0:
        rows = GR[GR['station_id'] == grlist[0]]
        rows.loc[:, 'station_id'] = aq
        # Comb_Weather.append(rows,ignore_index=True)
        Comb_Weather = pd.concat([Comb_Weather, rows], sort=False)
        print('len', len(Comb_Weather))
        continue
    if len(grlist) == 0 and len(oblist) == 1:
        rows = OB[(OB['station_id'] == oblist[0])]
        rows.loc[:, 'station_id'] = aq
        # Comb_Weather.append(rows,ignore_index=True)
        Comb_Weather = pd.concat([Comb_Weather, rows], sort=False)
        print('len', len(Comb_Weather))
        continue

    timeList = []
    for gr in grlist:
        time = GR[GR['station_id'] == gr].time.unique()
        timeList.extend(time)

    for ob in oblist:
        time = OB[OB['station_id'] == ob].time.unique()
        timeList.extend(time)

    timeSet = list(set(timeList))
    timeSet.sort(key=timeList.index)

    rowDf = pd.DataFrame(pd.DataFrame(columns=cols))
    for time in timeSet:
        print('     ', time)
        tempDf = pd.DataFrame(pd.DataFrame(columns=valCols))
        tempSr = pd.Series(index=cols)

        for gr in grlist:
            value = GR[(GR['station_id'] == gr) & (GR['time'] == time)]
            value = value.drop(columns=['station_id', 'time'])
            tempDf = pd.concat([tempDf, value])

        for ob in oblist:
            value = OB[(OB['station_id'] == ob) & (OB['time'] == time)]
            value = value.drop(columns=['station_id', 'time'])
            tempDf = pd.concat([tempDf, value])

        tempSr['station_id'] = aq
        tempSr['time'] = time
        tempSr['humidity'] = tempDf['humidity'].mean()
        tempSr['pressure'] = tempDf['pressure'].mean()
        tempSr['temperature'] = tempDf['temperature'].mean()
        tempSr['wind_speed'] = tempDf['wind_speed'].mean()

        tempSr['weather'] = tempDf['weather'].mode().values[0]
        tempSr['wind_direction'] = tempDf['wind_direction'].mode().values[0]
        # print(tempSr)
        # Comb_Weather.append(tempSr,ignore_index=True)
        rowDf.loc[len(rowDf)] = tempSr
    Comb_Weather = pd.concat([Comb_Weather, rowDf], sort=False)
    print('len', len(Comb_Weather))

Comb_Weather.to_csv('dealedData/combine/test.csv', index=None)

# 处理类型数据

print(set(Comb_Weather['wind_direction'].unique()))
dum_attr = ['weather', 'wind_direction']
for att in dum_attr:
    dum = pd.get_dummies(Comb_Weather[att], prefix=att[:4])
    Comb_Weather = pd.concat([Comb_Weather, dum], axis=1)
    Comb_Weather = Comb_Weather.drop(att, axis=1)

print(Comb_Weather.head(3))
Comb_Weather.to_csv('dealedData/combine/test.csv', index=None)