# futurePredict

High-frequency future price prediction (~20s) based on the following approaches

1 GRU 
2 LSTM

## TODO LIST

1. factor selection based on cor
2. period selection based on autocor
3. transformer based model

## Data

This dataset contains historical data for various Chinese stock index futures contracts. The data is organized into different directories based on data frequency (tick, daily) and asset type (futures, stocks). The naming conventions and data fields are clearly defined, providing a solid foundation for further analysis and modeling.

### Data Naming Conventions for Stock Index Futures Contracts
IC-prefixed contracts represent China Securities Index 500 Futures contracts.
IF-prefixed contracts represent SSE 300 Index Futures contracts.
IH-prefixed contracts represent Shanghai 50 Index Futures contracts.

### Data Directory Structure
Train directory:
Subdirectories named IC, IF, and IH.
Each file within these subdirectories contains tick data for a single trading day.
File naming format: [contract code]_yearmonthday.csv (e.g., IC_20230101.csv).

### FutureDaily directory:
Contains daily data for stock index futures contracts.
File naming format: [IC|IF|IH]yearmonth.csv (e.g., IC202301.csv).

### StockDaily directory:
Contains daily data for individual stocks (including indices).
File naming format: stock_name.csv (e.g., sh000905.csv).
Specific mappings:
IC corresponds to sh000905.csv.
IF corresponds to sh000300.csv.
IH corresponds to sh000016.csv.
Shanghai Composite Index corresponds to sh000001.csv.
Test Data Format
The format differs from the training data.

### Mapping between fields:
lastPrice: Last price
openInterest: Open interest
turnOver: Turnover
totalVol: Total volume
Note: The increase in open interest needs to be calculated manually using the formula:
Increase in open interest = Open interest of current tick - Open interest of previous tick

## Usage
python3 futures_predict.py --market IC --testMarket IC2021
