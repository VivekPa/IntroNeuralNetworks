import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
import numpy as np
fix.pdr_override()


def back_test(strategy, seq_len, ticker, start_date, end_date, dim):
    """
    A simple back test for a given date period
    :param strategy: the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param seq_len: length of the days used for prediction
    :param ticker: company ticker
    :param start_date: starting date
    :type start_date: "YYYY-mm-dd"
    :param end_date: ending date
    :type end_date: "YYYY-mm-dd"
    :param dim: dimension required for strategy: 3dim for LSTM and 2dim for MLP
    :type dim: tuple
    :return: Percentage errors array that gives the errors for every test in the given date range
    """
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    stock_data = data["Adj Close"]
    errors = []
    for i in range((len(stock_data)//10)*10 - seq_len - 1):
        x = np.array(stock_data.iloc[i: i + seq_len, 1]).reshape(dim) / 200
        y = np.array(stock_data.iloc[i + seq_len + 1, 1]) / 200
        predict = strategy.predict(x)
        while predict == 0:
            predict = strategy.predict(x)
        error = (predict - y) / 100
        errors.append(error)
        total_error = np.array(errors)
    print(f"Average error = {total_error.mean()}")
    # If you want to see the full error list then print the following statement
    # print(errors)
