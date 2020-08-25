from statsmodels.api import OLS
import statsmodels.tsa.stattools as ts 
import pandas as pd 
import matplotlib.pyplot as plt

#plot scatter plot to check correlation of two time series
def plot_scatter(df):
    plt.scatter(df.iloc[:,0], df.iloc[:,1], s=0.8)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("Correlation b/w 2 time series {}".format(round(df.iloc[:,0].corr(df.iloc[:,1]),2)))
    plt.show()

#check mean reversion from the residual values
#pair trading works on the linear model representedby equations
#y(t) = bx(t) + c(t)
#where y and x are the two assets to be traded
#b is the linear regression coefficient
#c(t) are the residuals for a particular value of b
#in the function adf_test coefficient b is calculated using OLS
#residue is calculated for the best coefficient
#plot the residue with time, check the stationarity of residue
#Perform cointegrated Augmented Dickey Fuller (cADF test) to 
#mean reversion of two time series

def adf_test1(df):
    model = OLS(df.iloc[:,1], df.iloc[:,0]).fit() #OLS(y,x)
    beta = model.params[0] #b coefficient
    res = df.iloc[:,1] - df.iloc[:,0]*beta
    #plot residuals
    plt.plot(df.index, res)
    plt.xticks(rotation = 45)
    plt.xlabel("Month/Year")
    plt.ylabel("Residual")
    plt.show()
    #calculate adf test calue
    cADF = ts.adfuller(res)
    print("ADF Test Results \n",cADF)

    #if the series is mean-reverting in long term, the test result
    #should be lesser then 5% critical value
    #which means we can reject null hypothesis where H0 denotes that 
    #two time seriers are not mean reverting
