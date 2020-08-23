# Example 5.1: Pair Trading USD.AUD vs USD.CAD Using the Johansen Eigenvector

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
from model import LSTM_train, prediction    
from seq_norm import seq_and_norm
from evaluation import position_returns

#change matplotlib.pyplot theme
plt.style.use('ggplot')

#If you trade currencies both must end in the same currency, here USD
df1=pd.read_csv('CADUSD=X_CADUSD.csv')
df1['Date']=pd.to_datetime(df1['Date'],  infer_datetime_format=True).dt.date 
df1.rename(columns={'Close': 'CAD'}, inplace=True)
df1.drop(['Open','High','Low','Adj Close','Volume'], axis=1, inplace=True)


df2=pd.read_csv('AUDUSD=X_AUDUSD.csv')
df2['Date']=pd.to_datetime(df1['Date'],  infer_datetime_format=True).dt.date 
df2.rename(columns={'Close': 'AUD'}, inplace=True)
df2.drop(['Open','High','Low','Adj Close','Volume'], axis=1, inplace=True)


df=pd.merge(df1, df2, how='inner', on='Date')

df.set_index('Date', inplace=True)
df.dropna(inplace=True)

df1=pd.read_csv('CADUSD=X_CADUSD.csv')
df1['Date']=pd.to_datetime(df1['Date'],  infer_datetime_format=True).dt.date 
df1.rename(columns={'Open': 'CAD'}, inplace=True)
df1.drop(['Close','High','Low','Adj Close','Volume'], axis=1, inplace=True)

df2=pd.read_csv('AUDUSD=X_AUDUSD.csv')
df2['Date']=pd.to_datetime(df1['Date'],  infer_datetime_format=True).dt.date 
df2.rename(columns={'Open': 'AUD'}, inplace=True)
df2.drop(['Close','High','Low','Adj Close','Volume'], axis=1, inplace=True)

df_opens=pd.merge(df1, df2, how='inner', on='Date')


df_opens.set_index('Date', inplace=True)
df_opens.dropna(inplace=True)

trainlen=250 #cointegration metric is calculated for train length 250
lookback=20 #for each batch of size 20

hedgeRatio=np.full(df.shape, np.NaN)
numUnits=np.full(df.shape[0], np.NaN)

target = pd.DataFrame()

for t in range(trainlen+1, df.shape[0]):
    # Johansen test
    '''coint_johansen is a cointegration test to find possible 
    correlation between time series in the long term'''
    result=vm.coint_johansen(df.values[(t-trainlen-1):t-1], det_order=0, k_ar_diff=1)
    
    hedgeRatio[t,:]=result.evec[:, 0]
    #hedgeRatio[t,:]=result.evec[:,0]/result.evec[:, 0][0] #almost the same
    #Dot multiply (=multiply the 2 hedge ratios by the respective 2 closes and add up) to obtain the net market value of the portfolio (single column)
    yport=pd.DataFrame(np.dot(df.values[(t-lookback):t], result.evec[:, 0]), columns = ['signal'])
         

    
    ma=yport.mean()
    mstd=yport.std()
    #calculate the number of units to invest in the portfolio (single column)
    #note the negative sign, mean reversion bet
    numUnits[t]=-(yport.iloc[-1,:]-ma)/mstd  #numUnits are number of units of unit portfolio of AUDUSD and CADUSD
    yport.index = df[(t-lookback):t].index
    #saving the latestyport value in target dataframe
    target = target.append(yport.iloc[-1,:])

#prepare data for LSTM model
#inputs to LSTM Model -- AUD/USD rate, CAD/USD rate, signal
#target for the model is the next day's signal

#merge targets with closing price
data_to_model = pd.merge(target, df, left_on = target.index, right_on = df.index, how = "inner")
data_to_model.reset_index(inplace = True)
data_to_model = data_to_model[['signal', 'CAD', 'AUD']]

#split the data into train and test (train-split 75%)
train_size = int(len(data_to_model)*0.75)
train = data_to_model.iloc[:train_size, :]
test = data_to_model.iloc[train_size:,:]

batch_size = 20
#converting the data into sequences of batch_size 
#Each sequence/window is normalised
train_x, train_y, norm_y_train = seq_and_norm(train, batch_size)
test_x, test_y, norm_y_test = seq_and_norm(test,batch_size)

train_loss,val_loss, model = LSTM_train(train_x,train_y,batch_size)
plt.plot(train_loss, label = "Training Loss")
plt.plot(val_loss, label = 'Validation Loss')
plt.legend()
plt.show()

#predictions on train and test + inverse transformation
train_pred = prediction(train_x, model, norm_y_train)
test_pred = prediction(test_x,model, norm_y_test)

#postions, returns and evaluations

#Train Set
start =  batch_size #270
end = train_size
print("+++++++++Results for Prediction on Train Set+++++++++")
positions_train,pnl_train, returns_train = position_returns(train_pred, hedgeRatio,df,df_opens,start,end)
print("+++++++++Results on Actual Train Set+++++++++")
positions_train_ac,pnl_train_ac, returns_train_ac = position_returns(np.array(train.loc[batch_size:,'signal']), hedgeRatio,df,df_opens,start,end)

plt.figure()
plt.plot(positions_train,label = "Predicted")
plt.plot(positions_train_ac, label = "Actual")
plt.title("Positions on Train Set")
plt.ylabel("Position")
plt.xlabel("Time Period")
plt.legend()
plt.show()

plt.figure()
plt.plot(pnl_train,label = "Predicted")
plt.plot(pnl_train_ac, label = "Actual")
plt.title("Profit/Loss on Train Set")
plt.ylabel("Profit/Loss")
plt.xlabel("Time Period")
plt.legend()
plt.show()

plt.figure()
plt.plot(returns_train,label = "Predicted")
plt.plot(returns_train_ac, label = "Actual")
plt.title("Return on Train Set")
plt.ylabel("Cumulative Returns")
plt.xlabel("Time Period")
plt.legend()
plt.show()
#test set
start =  df.shape[0]-test_pred.shape[0] #270
end = df.shape[0]
print("+++++++++Results for Prediction on Test Set+++++++++")
positions_test,pnl_test, returns_test = position_returns(test_pred, hedgeRatio,df,df_opens,start,end)
print("+++++++++Results on Actual Test Set+++++++++")
positions_test_ac,pnl_test_ac, returns_test_ac = position_returns(np.array(test.loc[train_size+batch_size:,'signal']), hedgeRatio,df,df_opens,start,end)

plt.figure()
plt.plot(positions_test,label = "Predicted")
plt.plot(positions_test_ac, label = "Actual")
plt.title("Predction on test Set")
plt.ylabel("Positions")
plt.xlabel("Time Period")
plt.legend()
plt.show()

plt.figure()
plt.plot(pnl_test,label = "Predicted")
plt.plot(np.array(pnl_train_ac), label = "Actual")
plt.title("Profit/Loss on test Set")
plt.ylabel("Profit/Loss")
plt.xlabel("Time Period")
plt.legend()
plt.show()

plt.figure()
plt.plot(returns_test,label = "Predicted")
plt.plot(returns_test_ac, label = "Actual")
plt.title("Returns on test Set")
plt.xlabel("Time Period")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()


'''
numUnits in line 60 is the main trading signal of this program and 
this line uses the latest, most up-to-date (net) market value of the portfolio (=yport.iloc[-1,:])
Use an LSTM to predict tomorrows (net) market value of the portfolio:
numUnits[t]=-(LSTMPREDICTIONOFYPORT-ma)/mstd
as input data for the LSTM, use the past values of yport, CADUSD and AUDUSD Open High Low Close Volume data perhaps other inputs,
normalized with window normalization.

"""
'''
'''

#multiply the number of units (one column) by the hedge ratio (two columns) and by the closes (two columns) 
#to obtain the positions (two columns).
#The positions are market values of AUDUSD and CADUSD in portfolio expressed in US$.
positions=pd.DataFrame(np.expand_dims(numUnits, axis=1)*hedgeRatio)*df.values # results.evec(:, 0)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
#multiply the 2 positions (both shifted 1 back) by the respective 2 price percent change and add up 
#to obtain the daily profit and loss (single column)
pnl=np.sum((positions.shift().values)*(df_opens.pct_change().values), axis=1)# daily P&L of the strategy, entering at the open
#calculate the returns and cummulative returns
ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
ret.fillna(value=0, inplace=True)
(np.cumprod(1+ret)-1).plot()
print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
# APR=0.064512 Sharpe=1.362926

"""
Ignore this comment
with integer numUnits the results are not as good
numUnitsdf = pd.DataFrame(numUnits.tolist(), columns=['nU1'])
numUnitsdf.fillna(value=0, inplace=True)
numUnitsdf['nU2'] = numUnitsdf.nU1
numUnitsdf.loc[np.abs(numUnitsdf.nU1) >= 0, "nU2"] = np.NaN # make all NANs
numUnitsdf.loc[numUnitsdf.nU1 > 1, "nU2"] = 1
numUnitsdf.loc[numUnitsdf.nU1 < -1, "nU2"] = -1
numUnitsdf['ss'] = np.sign(numUnitsdf.nU1) + np.sign(numUnitsdf.nU1.shift(1))
numUnitsdf.loc[ numUnitsdf['ss'] == 0, "nU2"] = 0 #when the sign changes (across zero), get out of the position
numUnitsdf.fillna(method='ffill', inplace=True)
numUnits = numUnitsdf.nU2.to_numpy().reshape(-1)
"""
'''