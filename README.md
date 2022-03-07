# futurePredict

## Methodology

mainly time series analysis
1. autocorrelation / partial autocorrelation analysis to select time window
2. RNN/LSTM/Transformer modeling

## Data
1. 股指期货合约命名含义
IC开头的是中证500指数期货合约
IF开头的是沪深300指数期货合约
IH开头的是上证50指数期货合约

2. Train目录
IC、IF、IH   目录下每个文件含一个交易日的tick数据。 文件名格式为 [合约代码]_年月日.csv
FutureDaily 目录下包含股指期货合约的日线信息， 文 件名格式 [IC|IF|IH]年月.csv
StockDaily   目录下包含股票（含指数）的日线信息， 文件名格式 股票名.csv。 其中IC为sh000905.csv， IF为sh000300.csv,  IH为sh000016.csv， 上证综合指数为sh000001.csv

3。 Test数据格式
格式和训练的那个不一样， 下面是映射关系：
lastPrice 最新
openInterest 持仓
turnOver 成交额
totalVol 成交量
其中增仓需要自己计算一下： 增仓 = 当前tick的持仓 - 上一个tick的持仓

## Usage
python3 futures_predict.py --market IC --testMarket IC2021
