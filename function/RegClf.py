import sys
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sys.path.insert(0, "function/")
from plot import *

class RegClf():

    def __init__(self,X_train, X_test, Y_train, Y_test):

        self.X_train    = X_train
        self.X_test     = X_test
        self.Y_train    = Y_train
        self.Y_test     = Y_test

        self.clf = []
        self.Y_pred = []
        self.confidencereg  = []


    def clftype(self,degre_ask):
        self.clf.append(LinearRegression(n_jobs=-1))
        for degree in degre_ask:
            self.clf.append(make_pipeline(PolynomialFeatures(degree), Ridge()))



    def predict(self):
        for regression in self.clf:
            self.Y_pred.append(regression.predict(self.X_test))

    def fit(self):
        for regression in self.clf:
            regression.fit(self.X_train, self.Y_train)

    def score(self):
         for regression in self.clf:
            self.confidencereg.append(regression.score(self.X_test, self.Y_test))

    def mostAccurateScore (self):
        most_reliable=0;
        for pos in range(0,len(self.confidencereg)):
            if self.confidencereg[pos] > self.confidencereg[most_reliable]:
                most_reliable=pos

        return self.confidencereg[most_reliable]


    def mostAccurateRegression (self):
        most_reliable=0;
        for pos in range(0,len(self.confidencereg)):
            if self.confidencereg[pos] > self.confidencereg[most_reliable]:
                most_reliable=pos

        return self.clf[most_reliable]
