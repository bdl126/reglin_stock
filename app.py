import sys
import pandas as pd
import datetime
import pandas_datareader.data as web
import math
import numpy as np
from sklearn import preprocessing, model_selection
from pandas import Series, DataFrame
sys.path.insert(0, "function/")
from plot import *

start = datetime.datetime(2010, 1, 1)

end = datetime.date.today()

df = web.DataReader("AAPL", 'yahoo', start, end)
dfreg=df
#dfreg = df.loc[:,["Adj Close","Volume"]]
dfreg["HL_PCT"] = (df["High"] - df["Low"]) / df["Close"] * 100.0
dfreg["PCT_change"] = (df["Close"] - df["Open"]) / df["Open"] * 100.0


# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
#forecast_out = int(math.ceil(0.01 * len(dfreg)))
#forecast_col = 'Adj Close'
#dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)

X_train, X_test, y_train, y_test = model_selection.train_test_split(dfreg.drop(['Adj Close'],1),
                                                                    dfreg["Adj Close"], test_size=0.01)

X = np.array(dfreg.drop(['label'], 1))
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_test = X[-forecast_out:]
X_train = X[:-forecast_out]

y = np.array(dfreg['label'])
y = y[:-forecast_out]

print(df.tail())
