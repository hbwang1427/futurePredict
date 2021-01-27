#load csv files
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#read IC
def load_future(mypath, n_sample):
    #mypath = root_path + '/IC'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath,f).endswith('.csv')]
    Data = []
    for f in sorted(onlyfiles):
        prices = pd.read_csv(os.path.join(mypath,f),index_col=None, header=0)
        print(os.path.join(mypath,f))
        # preparing input features
        #prices = prices[['最新','持仓','增仓','成交额','成交量','开仓','平仓','买一价','买二价','买三价','买四价','买五价','卖一价','卖二价','卖三价','卖四价','卖五价','买一量','买二量','买三量','买四量','买五量','卖一量','卖二量','卖三量','卖四量','卖五量']]
        #print(prices)
        #prices[['最新']] = prices[['最新']] - prices[['最新']].shift(1)
        prices = prices[['最新','持仓','成交额','成交量','买一价','买二价','买三价','买四价','买五价','卖一价','卖二价','卖三价','卖四价','卖五价']]
        prices = prices[::n_sample] #take every n_sample+1 ticks
        for c in list(prices): #take incrementals
            prices[c] = prices[c] - prices[c].shift(1)
        if len(prices) > 0:
            prices.drop(prices.index[0], axis=0, inplace=True)
        Data.append(prices)
    Data = pd.concat(Data, axis=0, ignore_index=True)
    #Data = Data.sort_values('时间')
    return Data

def load_Daily(mypath):
    #mypath = root_path + '/IC'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and join(mypath,f).endswith('.csv')]
    Data = []
    for f in sorted(onlyfiles):
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
    for f in sorted(onlyfiles):
        prices = pd.read_csv(os.path.join(mypath,f),index_col=None, header=0)
        Data.append(prices)
    Data = pd.concat(Data, axis=0, ignore_index=True)
    #Data = Data.sort_values('时间')
    return Data

if __name__ == '__main__':
    data = load_future('StockFutureData/IC',20)
    c = data.corr(method='spearman')
    c.to_csv('cor.csv',index=True)
    fig, ax = plt.subplots()
    ax.matshow(c, cmap=plt.cm.Blues)