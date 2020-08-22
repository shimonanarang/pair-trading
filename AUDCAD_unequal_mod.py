# Example 5.1: Pair Trading USD.AUD vs USD.CAD Using the Johansen Eigenvector

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import statsmodels.formula.api as sm
#import statsmodels.tsa.stattools as ts
import statsmodels.tsa.vector_ar.vecm as vm
from model import train
from seq_norm import seq_and_norm

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
print(df.head(3))

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
print(len(df), len(df_opens))
trainlen=250 #cointegration metric is calculated for train length 250
lookback=20 #for each batch of size 20

hedgeRatio=np.full(df.shape, np.NaN)
numUnits=np.full(df.shape[0], np.NaN)

#for t in range(1):
for t in range(trainlen+1, df.shape[0]):
    # Johansen test
    '''coint_johansen is a cointegration test to find possible 
    correlation between time series in the long term'''
    result=vm.coint_johansen(df.values[(t-trainlen-1):t-1], det_order=0, k_ar_diff=1)
    #print(dir(result))
    #print(result.evec.shape) '''result.evec returns the eigenvector '''
    #input("coint_johansen")
    hedgeRatio[t,:]=result.evec[:, 0]
    #hedgeRatio[t,:]=result.evec[:,0]/result.evec[:, 0][0] #almost the same
    #Dot multiply (=multiply the 2 hedge ratios by the respective 2 closes and add up) to obtain the net market value of the portfolio (single column)
    yport=pd.DataFrame(np.dot(df.values[(t-lookback):t], result.evec[:, 0]), columns = ['signal'])
   #  (net) market value of portfolio     
       
    #data_to_model -- y_port for 20 days and corresponding closing values of currencies in USD
    #target y
    target = yport 
    #df['AuS], 'CAD'] as x
    data_to_model = df[(t-lookback):t] #[lookback,2] #each batch
    target.index =data_to_model.index
    data_to_model = pd.merge(target, data_to_model, left_on = target.index, right_on = data_to_model.index, how = "outer")
    data_to_model.reset_index(inplace = True)
    
    data_to_model.drop(['index','key_0'],axis = 1, inplace = True)
    x,y = seq_and_norm(data_to_model,3)
    
    train_loss_values, model = train(x, y)
    print( train_loss_values)
    input("wait")
    ma=yport.mean()
    mstd=yport.std()
    #calculate the number of units to invest in the portfolio (single column)
    #note the negative sign, mean reversion bet
    numUnits[t]=-(yport.iloc[-1,:]-ma)/mstd  #numUnits are number of units of unit portfolio of AUDUSD and CADUSD
    

"""
numUnits in line 60 is the main trading signal of this program and 
this line uses the latest, most up-to-date (net) market value of the portfolio (=yport.iloc[-1,:])
Use an LSTM to predict tomorrows (net) market value of the portfolio:
numUnits[t]=-(LSTMPREDICTIONOFYPORT-ma)/mstd
as input data for the LSTM, use the past values of yport, CADUSD and AUDUSD Open High Low Close Volume data perhaps other inputs,
normalized with window normalization.

"""
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