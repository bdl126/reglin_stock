import sys

import datetime
import pandas_datareader.data as web

import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
sys.path.insert(0, "function/")


class extract():

    def __init__(self):
        self.dfreg=0

    def StockHistory(self):

        start = datetime.datetime(2010, 1, 1)

        end = datetime.date.today()

        self.dfreg = web.DataReader("AAPL", 'yahoo', start, end)

        #preprocessing of data as well

        self.dfreg["HL_PCT"] = (self.dfreg["High"] - self.dfreg["Low"]) / self.dfreg["Close"] * 100.0
        self.dfreg["PCT_change"] = (self.dfreg["Close"] - self.dfreg["Open"]) / self.dfreg["Open"] * 100.0


        X_train, X_test, y_train, y_test = model_selection.train_test_split(self.dfreg.drop(['Adj Close'],1),
                                                                            self.dfreg["Adj Close"], test_size=0.1)
        X_train_np = np.array(preprocessing.scale(X_train))
        X_test_np = np.array(preprocessing.scale(X_test))
        y_train_np = np.array(preprocessing.scale(y_train))
        y_test_np = np.array(preprocessing.scale(y_test))

        return X_train_np,X_test_np,y_train_np,y_test_np

    def getRawExtract(self):
        return self.dfreg