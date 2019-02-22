import pandas as pd
import quandl,math,datetime
import numpy as np
from sklearn import preprocessing,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Open','High','Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['High']/2 - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Open']/2)/(df['Open']/2) * 100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999999,inplace=True)

forcast_out = int(math.ceil(0.01*len(df)))

df['Label'] = df[forecast_col].shift(-forcast_out)

x = np.array(df.drop(['Label'],axis = 1))
x = preprocessing.scale(x)
x_lately = x[-forcast_out:]
x = x[:-forcast_out]

df.dropna(inplace=True)
y = np.array(df['Label'])
y = np.array(df['Label'])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# clf = LinearRegression(n_jobs=-1)
# clf.fit(x_train,y_train)
# with open('linearregression.pickle','wb') as f:
# 	pickle.dump(clf,f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(x_test,y_test)
forecast_set = clf.predict(x_lately)

print(forecast_set,accuracy,forcast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix +one_day

for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


print()