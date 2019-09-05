import sys
from sklearn.linear_model import LinearRegression,Ridge,LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from pylab import text
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import re
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
        self.accuracy= []
        self.posBestClf=0;


    def clftype(self,degre_ask):
        self.clf.append(LinearRegression(n_jobs=-1))
        self.clf.append(LassoLars(alpha=0.1))
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
        self.posBestClf=most_reliable
        return self.clf[most_reliable]


    def plotGraph(self, dataset):

        fig=figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        #figure.suptitle('Adj Close vs Date', fontsize=16)
        fig.canvas.set_window_title('Adj Close vs Date')
        dataset = dataset.reset_index()
        for regression in range(0,len(self.clf)):
            plt.subplot(3, 3, regression + 1)

            plt.plot(dataset['Date'][-self.Y_pred[regression].size:], self.Y_pred[regression], color='red', linewidth=2, label='Prediction')
            plt.scatter(dataset['Date'][-self.Y_pred[regression].size:], dataset['Adj Close'][-self.Y_pred[regression].size:], color='blue', s=2,
                        label='Data samples')
            plt.text('2018-09', 220, 'accuracy:'+str(self.confidencereg[regression]),backgroundcolor="#D3D3D3")
            plt.legend()

            #if it's polynomyal, get it's degree
            if hasattr(self.clf[regression], 'steps'):
                plt.title('polynomial ' + str(self.clf[regression].steps[0][1].degree))
            else:

                title=self.clf[regression].__class__.__name__
                plt.title(title)

            plt.ylabel('Adj Close')
            plt.xlabel('Date')
        plt.show()
