import sys
import pandas as pd


from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
sys.path.insert(0, "function/")


from DataExtractor import *
from plot import *

data = extract()

X_train, X_test, Y_train, Y_test=data.StockHistory()









#
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, Y_train)
dfreg=data.getRawExtract()
y_pred = clfreg .predict(X_test)

#To retrieve the intercept:
print(clfreg.intercept_)#For retrieving the slope:
print(clfreg.coef_)


plotGraph(data.getRawExtract(),X_test,y_pred)

df = pd.DataFrame({'Actual': Y_test.flatten(), 'Predicted': y_pred.flatten()})

print(df)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, Y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, Y_train)

#
#
confidencereg = clfreg.score(X_test, Y_test)
confidencepoly2 = clfpoly2.score(X_test, Y_test)
confidencepoly3 = clfpoly3.score(X_test,Y_test)

print(confidencereg)
print(confidencepoly2)
print(confidencepoly3)