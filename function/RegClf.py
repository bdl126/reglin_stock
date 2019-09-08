import sys
from sklearn.linear_model import LinearRegression,Ridge,LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
sys.path.insert(0, "function/")

class RegressionClassifier():

    def __init__(self,X_train, X_test, Y_train, Y_test):

        self.X_train    = X_train
        self.X_test     = X_test
        self.Y_train    = Y_train
        self.Y_test     = Y_test

        self.classifiers = []
        self.Y_preds = []
        self.confidenceRegressions  = []
        self.positionBestClassifier=0


    def defineClassifierType(self, degre_ask):
        self.classifiers.append(LinearRegression(n_jobs=-1))
        self.classifiers.append(LassoLars(alpha=0.1))
        for degree in degre_ask:
            self.classifiers.append(make_pipeline(PolynomialFeatures(degree), Ridge()))



    def predict(self):
        for regression in self.classifiers:
            self.Y_preds.append(regression.predict(self.X_test))

    def fit(self):
        for regression in self.classifiers:
            regression.fit(self.X_train, self.Y_train)

    def score(self):
         for regression in self.classifiers:
            self.confidenceRegressions.append(regression.score(self.X_test, self.Y_test))


    def mostAccurateScore (self):
        most_reliable=0;
        for pos in range(0, len(self.confidenceRegressions)):
            if self.confidenceRegressions[pos] > self.confidenceRegressions[most_reliable]:
                most_reliable=pos

        return self.confidenceRegressions[most_reliable]


    def mostAccurateRegression (self):
        most_reliable=0;
        for pos in range(0, len(self.confidenceRegressions)):
            if self.confidenceRegressions[pos] > self.confidenceRegressions[most_reliable]:
                most_reliable=pos
        self.positionBestClassifier=most_reliable
        return self.classifiers[most_reliable]


    def plotGraph(self, dataset):

        fig=figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
        #figure.suptitle('Adj Close vs Date', fontsize=16)
        fig.canvas.set_window_title('Adj Close vs Date')
        dataset = dataset.reset_index()
        for regression in range(0, len(self.classifiers)):
            plt.subplot(3, 3, regression + 1)

            plt.plot(dataset['Date'][-self.Y_preds[regression].size:], self.Y_preds[regression], color='red', linewidth=2, label='Prediction')
            plt.scatter(dataset['Date'][-self.Y_preds[regression].size:], dataset['Adj Close'][-self.Y_preds[regression].size:], color='blue', s=2,
                        label='Data samples')
            plt.text('2018-09', 220, 'Confidence Level:' + str(self.confidenceRegressions[regression]), backgroundcolor="#D3D3D3")
            plt.legend()

            #if it's polynomyal, get it's degree
            if hasattr(self.classifiers[regression], 'steps'):
                plt.title('polynomial ' + str(self.classifiers[regression].steps[0][1].degree))
            else:

                title=self.classifiers[regression].__class__.__name__
                plt.title(title)

            plt.ylabel('Adj Close')
            plt.xlabel('Date')
        plt.show()
