import matplotlib as plt
from pandas import DataFrame

def plotGraph(dataset):
    # data={'Date': list(dataset.index),
    #     'Adj Close': dataset['Adj Close']
    #    }
    # df = DataFrame(data, columns=['Date', 'Adj Close'])

    dataset.plot(y='Adj Close',x='DatetimeIndex', kind ='scatter')
    plt.title('MinTemp vs MaxTemp')
    plt.xlabel('MinTemp')
    plt.ylabel('MaxTemp')
    plt.show()