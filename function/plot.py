import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pandas import DataFrame



def plotGraph(dataset,X_test,y_pred):
    figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 2, 1)
    dataset = dataset.reset_index()
    # df = DataFrame(data, columns=['Date', 'Adj Close'])
    #dataset[-y_pred.size:].plot(x='Date',y='Adj Close', style='o',ms=2)
    #plt.scatter(dataset['Date'][-y_pred.size:],dataset['Adj Close'][-y_pred.size:],s=2)
    plt.plot(dataset['Date'][-y_pred.size:], y_pred, color='red', linewidth=2,label='Prediction')
    plt.scatter(dataset['Date'][-y_pred.size:], dataset['Adj Close'][-y_pred.size:], color='blue',s=2,label='Data samples')
    plt.legend()
    plt.title('Adj Close vs Date')
    plt.ylabel('Adj Close')
    plt.xlabel('Date')


    plt.show()