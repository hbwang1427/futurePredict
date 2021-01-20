#load csv files
import pandas as pd
import os
from os import listdir
from os.path import isfile, join

#read IC
def load_future(mypath):
    #mypath = root_path + '/IC'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath,f).endswith('.csv')]
    Data = []
    for f in onlyfiles:
        prices = pd.read_csv(os.path.join(mypath,f),index_col=None, header=0)
        # preparing input features
        #prices = prices.drop(['symbol'], axis=1)
        #prices = prices.drop(['volume'], axis=1)
        prices = prices[['时间','最新','持仓','增仓','成交额','成交量','开仓','平仓','成交类型','方向']]
        Data.append(prices)
    Data = pd.concat(Data, axis=0, ignore_index=True)
    #Data = Data.sort_values('时间')
    return Data

def load_Daily(mypath):
    #mypath = root_path + '/IC'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath,f).endswith('.csv')]
    Data = []
    for f in onlyfiles:
        prices = pd.read_csv(os.path.join(mypath,f),index_col=None, header=0)
        # preparing input features
        #prices = prices.drop(['symbol'], axis=1)
        #prices = prices.drop(['volume'], axis=1)
        prices = prices[['代码','时间','开盘价','最高价','最低价','收盘价','成交量(股)','成交额(元)']]
        Data.append(prices)
    Data = pd.concat(Data, axis=0, ignore_index=True)
    #Data = Data.sort_values('时间')
    return Data

def load_future_2021(mypath):
    #mypath = root_path + '/IC'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath,f).endswith('.csv')]
    Data = []
    for f in onlyfiles:
        prices = pd.read_csv(os.path.join(mypath,f),index_col=None, header=0)
        Data.append(prices)
    Data = pd.concat(Data, axis=0, ignore_index=True)
    #Data = Data.sort_values('时间')
    return Data