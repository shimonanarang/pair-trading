# Example 5.1: Pair Trading USD.AUD vs USD.CAD Using the Johansen Eigenvector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import statsmodels.formula.api as sm
# import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
from model import LSTM_train, prediction
from seq_norm import seq_and_norm
from evaluation import position_returns, eval_metrics
from check_mean_reversion import *

# change matplotlib.pyplot theme
plt.style.use('ggplot')


def generate_plot(xaxis, trace, trace_label, title, ylabel, xlabel):
    plt.figure()
    for idx in range(len(trace)):
        plt.plot(trace[idx], label=trace_label[idx])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


# If you trade currencies both must end in the same currency, here USD
df1 = pd.read_csv('CADUSD=X.csv')
df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
# df1 = df1[(df1['Date']>'2009-08-20')&(df1['Date']<'2014-11-10')]
df1.rename(columns={'Close': 'CAD'}, inplace=True)
df1.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

df2 = pd.read_csv('AUDUSD=X.csv')
df2['Date'] = pd.to_datetime(df2['Date'], infer_datetime_format=True)
# df2 = df2[(df2['Date']>'2009-08-20')&(df2['Date']<'2014-11-10')]
df2.rename(columns={'Close': 'AUD'}, inplace=True)
df2.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

df = pd.merge(df1, df2, how='inner', on='Date')
df.set_index('Date', inplace=True)
df.dropna(inplace=True)

df1 = pd.read_csv('CADUSD=X.csv')
df1['Date'] = pd.to_datetime(df1['Date'], infer_datetime_format=True)
# df1 = df1[(df1['Date']>'2009-08-20')&(df1['Date']<'2014-11-10')]
df1.rename(columns={'Open': 'CAD'}, inplace=True)
df1.drop(['Close', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

df2 = pd.read_csv('AUDUSD=X.csv')
df2['Date'] = pd.to_datetime(df2['Date'], infer_datetime_format=True)
# df2 = df2[(df2['Date']>'2009-08-20')&(df2['Date']<'2014-11-10')]
df2.rename(columns={'Open': 'AUD'}, inplace=True)
df2.drop(['Close', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

df_opens = pd.merge(df1, df2, how='inner', on='Date')

df_opens.set_index('Date', inplace=True)
df_opens.dropna(inplace=True)
print(df_opens.shape)
print(df.shape)

# check mean reversion
plot_scatter(df)
adf_test1(df)

generate_plot(df_opens.index, [df.loc[:, "AUD"], df.loc[:, "CAD"]], ["AUD-USD", "CAD-USD"],
              "Conversion Rate (Closing Price)", "Price", "Time Period")
generate_plot(df_opens.index, [df_opens.loc[:, "AUD"], df_opens.loc[:, "CAD"]], ["AUD-USD", "CAD-USD"],
              "Conversion Rate (Opening Price)", "Price", "Time Period")

trainlen = 30  # cointegration metric is calculated for train length 250
lookback = 30
print("Lookback", lookback)
print(df.head(5))

hedgeRatio = np.full(df.shape, np.NaN)
numUnits = np.full(df.shape[0], np.NaN)

target = pd.DataFrame()

for t in range(trainlen + 1, df.shape[0]):
    # Johansen test
    '''coint_johansen is a cointegration test to find possible 
    correlation between time series in the long term'''
    result = vm.coint_johansen(df.values[(t - trainlen - 1):t - 1], det_order=0, k_ar_diff=1)

    hedgeRatio[t, :] = result.evec[:, 0]
    # hedgeRatio[t,:]=result.evec[:,0]/result.evec[:, 0][0] #almost the same
    # Dot multiply (=multiply the 2 hedge ratios by the respective 2 closes and add up) to obtain the net market value of the portfolio (single column)
    yport = pd.DataFrame(np.dot(df.values[(t - lookback):t], result.evec[:, 0]), columns=['signal'])

    ma = yport.mean()
    mstd = yport.std()
    # calculate the number of units to invest in the portfolio (single column)
    # note the negative sign, mean reversion bet
    numUnits[t] = -(
                yport.iloc[-1, :] - ma) / mstd  # numUnits are number of units of unit portfolio of AUDUSD and CADUSD
    yport.index = df[(t - lookback):t].index
    # saving the latestyport value in target dataframe
    target = target.append(yport.iloc[-1, :])

# prepare data for LSTM model
# inputs to LSTM Model -- AUD/USD rate, CAD/USD rate, signal
# target for the model is the next day's signal

# merge targets with closing price
data_to_model = pd.merge(target, df, left_on=target.index, right_on=df.index, how="inner")
index = data_to_model.index
data_to_model.reset_index(inplace=True)
data_to_model = data_to_model[['signal', 'CAD', 'AUD']]

# split the data into train and test (train-split 75%)
train_size = int(len(data_to_model) * 0.80)
train = data_to_model.iloc[:train_size, :]
index_train = index[:train_size]
test = data_to_model.iloc[train_size:, :]
index_test = index[train_size:]

batch_size = 16
print("Batch Size", batch_size)
# converting the data into sequences of batch_size
# Each sequence/window is normalised
train_x, train_y, norm_y_train = seq_and_norm(train, batch_size)
test_x, test_y, norm_y_test = seq_and_norm(test, batch_size)

train_loss, val_loss, model = LSTM_train(train_x, train_y, batch_size)
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()

# predictions on train and test + inverse transformation
train_pred = prediction(train_x, model, norm_y_train)
test_pred = prediction(test_x, model, norm_y_test)

test_rmspe = eval_metrics(norm_y_test[:, 0], test_pred)
train_rmspe = eval_metrics(norm_y_train[:, 0], train_pred)

print("Test Set RMSPE: {rmspe}%".format(rmspe=test_rmspe))
print("Train Set RMSPE: {rmspe}%".format(rmspe=train_rmspe))

# plot trading signal prediction and actual
generate_plot(index_train, [train_pred, norm_y_train[:, 0]], ["Predicted Train", "Actual Train"],
              "Trading Signal RMSPE {}%".format(train_rmspe), "Trading Signal", "Time Period")
generate_plot(index_train, [test_pred, norm_y_test[:, 0]], ["Predicted Test", "Actual Test"],
              "Trading Signal RMSPE {}%".format(test_rmspe), "Trading Signal", "Time Period")

# postions, returns and evaluations

# Train Set
start = batch_size  # 270
end = train_size
print("+++++++++Results for Prediction on Train Set+++++++++")
positions_train, pnl_train, returns_train = position_returns(train_pred, hedgeRatio, df, df_opens, start, end)
print("+++++++++Results on Actual Train Set+++++++++")
positions_train_ac, pnl_train_ac, returns_train_ac = position_returns(np.array(train.loc[batch_size:, 'signal']),
                                                                      hedgeRatio, df, df_opens, start, end)

generate_plot(index_train, [positions_train, positions_train_ac], ["Predicted", "Actual"], "Positions on Train Set",
              "Positions", "Time Period")
generate_plot(index_train, [pnl_train, pnl_train_ac], ["Predicted", "Actual"], "Profit/Loss on Train Set", "Proft/Loss",
              "Time Period")
generate_plot(index_train, [returns_train, returns_train_ac], ['Predicted', "Actual"], "Return on Train Set",
              "Cumulative Returns", "Time Period")

# test set
start = df.shape[0] - test_pred.shape[0]  # 270
end = df.shape[0]
print("+++++++++Results for Prediction on Test Set+++++++++")
positions_test, pnl_test, returns_test = position_returns(test_pred, hedgeRatio, df, df_opens, start, end)
print("+++++++++Results on Actual Test Set+++++++++")
positions_test_ac, pnl_test_ac, returns_test_ac = position_returns(
    np.array(test.loc[train_size + batch_size:, 'signal']), hedgeRatio, df, df_opens, start, end)

generate_plot(index_test, [positions_test, positions_test_ac], ["Predicted", "Actual"], "Positions on Test Set",
              "Positions", "Time Period")
generate_plot(index_test, [pnl_test, pnl_test_ac], ["Predicted", "Actual"], "Proft/Loss on Test Set", "Proft/Loss",
              "Time Period")
generate_plot(index_test, [returns_test, returns_test_ac], ["Predicted", "Actual"], "Cumulative Returns on Test Set",
              "Cumulative Returns", "Time Period")
