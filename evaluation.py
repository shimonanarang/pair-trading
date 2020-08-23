import numpy as np
import pandas as pd
def position_returns(yport_pred, hedgeRatio,df,df_opens, start,end):
    df = df.iloc[start:end]
    df_opens = df_opens.iloc[start:end]
    hedgeRatio = hedgeRatio[start:end]
    print(df.shape,hedgeRatio.shape)
    
    mean = np.mean(yport_pred)
    std = np.std(yport_pred)
    numUnits =  (yport_pred - mean)/std
    print(numUnits.shape)
    positions= pd.DataFrame(np.expand_dims(numUnits, axis=1)*hedgeRatio)*df.values
     # results.evec(:, 0)' can be viewed as the capital allocation, while positions is the dollar capital in each ETF.
    pnl=np.sum((positions.shift().values)*(df_opens.pct_change().values), axis=1)# daily P&L of the strategy, entering at the open
    ret=pnl/np.sum(np.abs(positions.shift()), axis=1)
    ret.fillna(value=0, inplace=True)
    returns = np.cumprod(1+ret)-1
    print('APR=%f Sharpe=%f' % (np.prod(1+ret)**(252/len(ret))-1, np.sqrt(252)*np.mean(ret)/np.std(ret)))
    return positions,pnl, returns

