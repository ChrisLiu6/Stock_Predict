import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt

style.use('ggplot')

pd.set_option('display.max_columns', 20)

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. Close', 'Adj. High', 'Adj. Low']]
pretest = 'Adj. Close'

df.fillna(-999999, inplace=True)

shiftp = 0.001
shift = int(math.ceil(shiftp*len(df)))

df['Pretest'] = df[pretest].shift(-shift)
df.dropna(inplace=True)

x = np.array(df['Adj. Close'])
y = np.array(df['Pretest'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size = 0.2)
x_train = x_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
y_test = y_test.reshape(-1,1)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train,y_train)
print('Training accuracy: ', clf.score(x_test,y_test))

df.dropna(inplace=False)

recent_cls = np.array(df[pretest]).reshape(-1,1)
recent_cls = recent_cls[-shift:]
predict = clf.predict(recent_cls)

print('\nResult: ', predict)

