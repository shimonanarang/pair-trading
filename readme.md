# Pair Trading for Cointegrating Currencies
* Statistical arbitrage uses various financial statistics to find pricing inefficiencies in mean-reverting trading pairs. This project explores the statistical arbitrage of the Canadian and Australian dollars.
* A common type of statistical arbitrage is **pair-trading**. If two assets are cointegrated, then their price will converge to the mean price in the long term
* Pair-trading recognizes deviations in the price of a stock from the mean and either shorts or longs the stock. Once the price reverts to the mean, profit can be made.
* The trading signal is a statistical measure that indicates the moments of mean deviation and reversion
* Trading model for the next day is predicted using LSTMs and position of the portfolio is determined from hedge ratios

# Packages
* keras
* tensorflow
* statsmodels
* NumPy

# Files
* `check_johansen_loop.py`: checks cointegration with johansen test in loop
* `check_mean_reversion.py`: checks mean reversion of two time series using Augmented Dickey Fuller test
* `AUDCAD_unequal_mod.py`: main file
* `model.py`: Trains LSTM Model and Predicts the trading signal
* `evaluation.py`: evaluation using portfolio metrics and rmspe
* `seq_norm.py`: window normalisation and data prep for LSTM model

# Positions

Positions of securities are calculated by using the hedge ratios (hedge ratios determine capital allocation). Positions of two securities on training and test set:

! [image]('https://github.com/shimonanarang/pair-trading/blob/master/fig/positions_train.png')

! [image]('https://github.com/shimonanarang/pair-trading/blob/master/fig/positions_test.png')

When two cointegrating securities diverge due to market conditions, they tend to converge in few time periods

# Usage
1. Clone the repo

'''
git clone https://github.com/shimonanarang/pair-trading.git
'''

2. Install the requirements

'''
pip install -r requirements.txt
'''

3. Navigate and run the following command

'''
python AUDCAD_uneual_mod.py
'''


# References
1. Joao Frois Caldeira and Guilherme Moura. \Selection of a  portfolio of Pairs Based on Cointegration: A Statistical Arbitrage Strategy". In: Bras. Financas 11.1 (Mar. 2013)
2. Lu, Anran, et al. Cluster-Based Statistical Arbitrage Strategy. Stanford University, (June, 2018)


