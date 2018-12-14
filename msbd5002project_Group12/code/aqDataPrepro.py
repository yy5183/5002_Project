import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.width', 5000)
pd.set_option('display.max_columns', None)

# a = ob.groupby(['weather'],as_index=False)['weather'].agg({'cnt':'count'})

def aqDataPreprocess():
    aq = pd.read_csv('dealedData/AQ/AQ_delete24hAllNan.csv', header=0)

    # 各污染物相关性图
    g = sns.heatmap(aq[["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]].corr(), annot=True, fmt=".2f",
                        cmap="coolwarm")
    plt.show()

    # 填充pm2.5
    fillNanByDaymean(aq, 'PM25', 'PM10', 'CO')
    aq.to_csv('dealedData/AQ/AQ_del1_fill1.csv', index=None)

    # 填充pm10
    aq = pd.read_csv('dealedData/AQ/AQ_del1_fill1.csv', header=0)
    index_NaN_PM10 = list(aq["PM10"][aq["PM10"].isnull()].index)
    for i in index_NaN_PM10:
        # age_pred=aq["Age"][((dataset["SibSp"]==dataset.iloc[i]["SibSp"])&(dataset["Parch"]==dataset.iloc[i]["Parch"])&(dataset["Pclass"]==dataset.iloc[i]["Pclass"]))].median()
        PM25 = aq["PM2.5"].iloc[i]
        if pd.notna(PM25):
            print("type 2")
            PM10_relation = aq["PM10"][aq['PM2.5'] == PM25]
            if len(PM10_relation) == 0:  # 如果等于的没有就改成一定范围里取值
                PM10_relation = aq["PM10"][(aq['PM2.5'] <= PM25 * 1.1)
                                           & (aq['PM2.5'] >= PM25 * 0.9)]

            if len(PM10_relation) > 0:
                pred = PM10_relation.mean()
                aq["PM10"].iloc[i] = pred
            continue

    print(aq['PM10'].isna().sum())
    aq.to_csv('dealedData/AQ/AQ_del1_fill2.csv', index=None)

    # 污染物全空的也清除掉
    aq = pd.read_csv('dealedData/AQ/AQ_delete24hAllNan.csv', header=0)
    print(aq.isna().sum())
    a = aq[(aq['PM25'].isna()) & (aq['PM10'].isna()) & (aq['NO2'].isna()) & (aq['CO'].isna()) & (aq['O3'].isna()) & (
    aq['SO2'].isna())]
    allNanIndex = list(a.index)
    aq = aq.drop(allNanIndex)
    aq.to_csv('dealedData/AQ/AQ_del2.csv', index=None)
    print(aq.isna().sum())

    # 按照关联性补值
    aq = pd.read_csv('dealedData/AQ/AQ_del2_fill10.csv', header=0)
    print(aq.isna().sum())
    sourceCol = 'PM25'
    recloList = ['PM10', 'CO', 'NO2']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    sourceCol = 'PM10'
    recloList = ['PM25', 'CO']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    sourceCol = 'NO2'
    recloList = ['CO', 'PM25', 'O3', 'SO2']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    sourceCol = 'CO'
    recloList = ['PM25', 'NO2', 'PM10']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    sourceCol = 'O3'
    recloList = ['NO2', 'CO']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    sourceCol = 'SO2'
    recloList = ['PM25', 'CO', 'NO2']
    fillNanByRelation(aq, sourceCol, recloList, 0.1)
    aq.to_csv('dealedData/AQ/AQ_del2_fill11.csv', index=None)

    # 剩下的按照每天的平均值补值


def fillNanByRelation(aq, sourceCol, reColList, ratio):
    oriNanNum = aq[sourceCol].isna().sum()
    index_NaN_sourceCol = list(aq[sourceCol][aq[sourceCol].isnull()].index)
    # index_NaN_sourceCol = index_NaN_sourceCol[0:1]

    for i in index_NaN_sourceCol:
        print(index_NaN_sourceCol.index(i))
        flag = len(aq)
        max_range = 1+ratio
        min_range = 1-ratio
        sc_relation = aq
        for reCol in reColList:
            source_reCol = aq[reCol].iloc[i]
            if pd.notna(source_reCol):
                # print(reCol)
                temp = sc_relation[sc_relation[reCol] == source_reCol]
                if len(temp) == 0 or temp[sourceCol].notna().sum() < 1:  # 如果等于的没有就改成一定范围里取值
                    temp = sc_relation[(sc_relation[reCol] <= source_reCol * max_range)
                                            & (sc_relation[reCol] >= source_reCol * min_range)]
                sc_relation = temp
        if len(sc_relation) < flag:
            # print(sc_relation.head(5))
            pred = sc_relation[sourceCol].mean()
            pred = round(pred, 1)
            aq[sourceCol].iloc[i] = pred

    aftNanNum = aq[sourceCol].isna().sum()
    print('now, Nan Count is ', aftNanNum, ' and filled: ', oriNanNum - aftNanNum)

def fillNanByDaymean(aq, sourceCol):
    index_NaN_sourceCol = list(aq[sourceCol][aq[sourceCol].isnull()].index)
    # for i in index_NaN_sourceCol:



#
# g = sns.heatmap(aq[["PM25", "PM10", "NO2", "CO", "O3", "SO2"]].corr(), annot=True, fmt=".2f",
#                         cmap="coolwarm")
# plt.show()
# print(aq.isna().sum())

# aq = pd.read_csv('dealedData/AQ/AQ_delete24hAllNan.csv', header=0)
# print(aq.isna().sum())
# a = aq[(aq['PM25'].isna()) & (aq['PM10'].isna()) & (aq['NO2'].isna()) & (aq['CO'].isna()) & (aq['O3'].isna()) & (aq['SO2'].isna())]
# allNanIndex = list(a.index)
# aq = aq.drop(allNanIndex)
# aq.to_csv('dealedData/AQ/AQ_del2.csv', index=None)
# print(aq.isna().sum())

# sourceColList = ["PM25", "PM10", "NO2", "CO", "O3", "SO2"]
# print(aq.isna().sum())
# for sourceCol in sourceColList:
#     index_NaN_sourceCol = list(aq[sourceCol][aq[sourceCol].isnull()].index)
#     for i in index_NaN_sourceCol:
#         time = aq['time'].iloc[i]
#         station = aq['station_id'].iloc[i]
#         date = time.split(' ')[0]
#         temp = aq[(aq['station_id'] == station) & (aq['time'].str.contains(date))]
#         mea = temp[sourceCol].mean()
#         aq[sourceCol].iloc[i] = round(mea,2)
#
# print(aq.isna().sum())


