# Importing dependencies
import math 
import numpy as np
import pandas as pd
import datetime
from pandas import Series, DataFrame
## Note: Install pandas_datareader
## pip install pandas-datareader
import pandas_datareader.data as web
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def calcDate(interval):
    
    if interval == 'month':
        time = 30
    elif interval == 'year':
        time = 365
    elif interval == 'fiveYear':
        time = 1825
    tod = datetime.datetime.now()
    d = datetime.timedelta(days = time)
    a = tod - d
    selectedDate = a.date().isoformat()
#     print(selectedDate)
    return selectedDate

## Loading Yahoo Finance data set form 2016
# Get start and end dates 

# times = 1 year 6 month 3 month  

# start_date = datetime.datetime(2019, 5, 9)
start_date = calcDate('year')
## Select today's date as end date
end_date = datetime.datetime.now().date().isoformat() 
print(end_date)
stocks_df = web.DataReader('AAPL', 'yahoo', start_date, end_date)

# Displaying letest 5 records
# stocks_df.tail()

## Getting Final Closing price
closing_price_df= stocks_df['Adj Close']

closing_price_df.index = pd.to_datetime(closing_price_df.index)

## Calculate 50 day Moving Average
ma_1= closing_price_df.rolling(window=50).mean()

ma_1.index = pd.to_datetime(ma_1.index)


ma_1.dropna(inplace=True)


ma_2= closing_price_df.rolling(window=200).mean()

ma_2.index = pd.to_datetime(ma_2.index)

ma_2.dropna(inplace=True)




#The size for our chart:
plt.figure(figsize = (12,6))
#Plotting price and SMA lines:
plt.plot(closing_price_df, label='Adj Closing Price', linewidth = 2)
plt.plot(ma_1, label='50 Day SMA', linewidth = 1.5)
plt.plot(ma_2, label='200 Day SMA', linewidth = 1.5)
#Adding title and labeles on the axes, making legend visible:
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price ($)')
plt.title('Simple Moving Average')
plt.legend()
plt.show()

## Analysing Multiple Stocks
tickers =['AMZN', 'GOOG', 'IBM', 'MSFT']
comp_stocks_df = web.DataReader(tickers,'yahoo',start_date,end_date)['Adj Close']
# comp_stocks_df.head()

retscomp =comp_stocks_df.pct_change()
corr = retscomp.corr()
# corr.head()

plt.scatter(retscomp.AMZN, retscomp.MSFT)
plt.xlabel("Returns MSFT")
plt.ylabel("Returns AMZN")

plt.title('Scatter Plot of AMZN and MSFT')
plt.legend()
plt.show()


plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);


plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


#high low percentage 
dfreg = stocks_df.loc[:,['Adj Close','Volume']]
dfreg['HL_PCT'] = (stocks_df['High'] - stocks_df['Low']) / stocks_df['Close'] *100.0
#percentage change 
dfreg['PCT_change'] = (stocks_df['Close'] - stocks_df['Open']) / stocks_df['Open']  * 100.0
dfreg


#drop missing value 
dfreg.fillna(value=99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(dfreg)))

forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

#linear regression
#X = preprocessing.scale(X)

#train for model generation and evaluation 
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
#quadratic regression
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)


#KNN Regression 
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)


#evaluation 
conf_reg = clfreg.score(X_test, y_test)
confpoly2 = clfpoly2.score(X_test, y_test)
confpoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)
# hmmmm
confidenceknn

forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
forecast_set


#plotting prediciton 
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()