#check cointegration of pairs

from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
import pandas as pd
import itertools as it
from datetime import datetime


cointegrating_pairs = []


#pair is a df containing two ETF close price series
def check_johansen(symbList, cointegrating_pairs):

    global df1
    COINTEGRATION_CONFIDENCE_LEVEL = 90 # require cointegration at 90%, 95%, or 99% confidence 
    #symbList = ["GDX", "QID"] 
    #symbList = ["XLK", "UCO"] 
    #symbList = ["EWW", "MOO"] 
    #symbList = ["XLK", "GDX"]
    #symbList = ["XLK", "SKF"]
    #symbList = ["FXE", "DGBP"]
    
    start_date = "2013-01-01" 
    end_date = "2018-12-31"
    
    y = pd.read_csv(symbList[0] + '.csv', parse_dates=['Date'])
    y = y.sort_values(by='Date')
    y.set_index('Date', inplace = True)
    x = pd.read_csv(symbList[1] + '.csv', parse_dates=['Date'])
    x = x.sort_values(by='Date')
    x.set_index('Date', inplace = True)
     
    #doing an inner join to make sure dates coincide and there are no NaNs
    #inner join requires distinct column names
    y.rename(columns={'Open':'y_Open','High':'y_High','Low':'y_Low','Close':'y_Close','Adj Close':'y_Adj_Close','Volume':'y_Volume'}, inplace=True) 
    x.rename(columns={'Open':'x_Open','High':'x_High','Low':'x_Low','Close':'x_Close','Adj Close':'x_Adj_Close','Volume':'x_Volume'}, inplace=True) 
    df1 = pd.merge(y, x, left_index=True, right_index=True, how='inner') #inner join
    #get rid of extra columns but keep the date index
    df1.drop(['x_Open', 'x_High','x_Low','x_Adj_Close','x_Volume','y_Open', 'y_High','y_Low','y_Adj_Close','y_Volume'], axis=1, inplace=True)
    df1.rename(columns={'y_Close':'y','x_Close':'x'}, inplace=True) 
    
    # The second and third parameters indicate constant term, with a lag of 1. 
    result = coint_johansen(df1, 0, 1)
    
    # the 90%, 95%, and 99% confidence levels for the trace statistic and maximum 
    # eigenvalue statistic are stored in the first, second, and third column of 
    # cvt and cvm, respectively
    confidence_level_cols = {
        90: 0,
        95: 1,
        99: 2
    }
    confidence_level_col = confidence_level_cols[COINTEGRATION_CONFIDENCE_LEVEL]
    
    trace_crit_value = result.cvt[:, confidence_level_col]
    eigen_crit_value = result.cvm[:, confidence_level_col]
    print("trace_crit_value",trace_crit_value)
    print("eigen_crit_value",eigen_crit_value)
    print("lr1",result.lr1)
    print("lr2",result.lr2)
    # The trace statistic and maximum eigenvalue statistic are stored in lr1 and lr2;
    # see if they exceeded the confidence threshold
    if np.all(result.lr1 >= trace_crit_value) and np.all(result.lr2 >= eigen_crit_value):
        print("The two datasets "+symbList[0]+" and "+symbList[1]+" are cointegrated")
        # The first i.e. leftmost column of eigenvectors matrix, result.evec, contains the best weights.
        v1= result.evec[:,0:1]
        hr=v1/-v1[1] #to get the hedge ratio divide the best_eigenvector by the negative of the second component of best_eigenvector
        #the regression will be: close of symbList[1] = hr[0]*close of symbList[0] + error
        #where the beta of the regression is hr[0], also known as the hedge ratio, and
        #the error of the regression is the mean reverting residual signal that you need to predict, it is also known as the "spread"
        #the spread = close of symbList[1] - hr[0]*close of symbList[0] or alternatively (the same thing):
        #do a regression with close of symbList[0] as x and lose of symbList[1] as y, and take the residuals of the regression to be the spread.
        cointegrating_pairs.append(dict(
            sid_1=symbList[0],
            sid_2=symbList[1],
            hedge_ratio=v1
            ))

    return 0

symbList = ["CADUSD=X", "AUDUSD=X"] 

#get symbol pairs
symbPairs = list(it.combinations(symbList, 2))

    
for i in symbPairs:
    try:
        check_johansen(list(i),cointegrating_pairs)
    except Exception:
        continue

df = pd.DataFrame(cointegrating_pairs)
df.to_csv("cointegrating_pairs_2010_currency_asset_class.csv")



