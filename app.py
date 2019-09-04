import sys


from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sys.path.insert(0, "function/")


from DataExtractor import *
from plot import *

data = extract()

X_train, X_test, Y_train, Y_test=data.StockHistory()

plotGraph(data.getRawExtract())







#
# clfreg = LinearRegression(n_jobs=-1)
# clfreg.fit(X_train, Y_train)
#
#
#
# # Quadratic Regression 2
# clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
# clfpoly2.fit(X_train, Y_train)
#
# # Quadratic Regression 3
# clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
# clfpoly3.fit(X_train, Y_train)
#
#
#
# confidencereg = clfreg.score(X_test, Y_test)
# confidencepoly2 = clfpoly2.score(X_test, Y_test)
# confidencepoly3 = clfpoly3.score(X_test,Y_test)