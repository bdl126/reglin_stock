import matplotlib.pyplot as plt
from pandas import DataFrame

def plotGraph(dataset,X_test,y_pred):

    dataset = dataset.reset_index()
    # df = DataFrame(data, columns=['Date', 'Adj Close'])
    dataset[-y_pred.size:].plot(x='Date',y='Adj Close', style='o',ms=2)
    #plt.scatter(dataset['Date'][-y_pred.size:],dataset['Adj Close'][-y_pred.size:],s=2)
    plt.plot(dataset['Date'][-y_pred.size:], y_pred, color='red', linewidth=2)
    plt.title('Adj Close vs Date')
    plt.ylabel('Adj Close')
    plt.xlabel('Date')
    plt.show()