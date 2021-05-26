import numpy as np
import pandas as pd
from math import *


def get_LKN(np_time_series):
    n = len(np_time_series)
    l = floor(n / 2)
    k = n - l + 1
    return l, k, n


def read_graph_data_snp500(path):
    graph_data = pd.read_csv(path, delimiter=",", names=['Date', 'S&P500', 'Dividend', 'Earnings', 'CPI', 'LIR', 'RP', 'RD', 'RE', 'PE10'])
    graph_data = graph_data.drop(['Date', 'Dividend', 'Earnings', 'CPI', 'LIR', 'RP', 'RD', 'RE', 'PE10'], 1)
    return graph_data.to_numpy().T[0]


def read_some_graph_data(path):
    pass  # TODO: if implementing a universal files' parsing, define such an auxiliary function.

### Checks ###


def main():
    arr = read_graph_data_snp500(r"C:\Users\olegm\PycharmProjects\RP_SSA_DATA\data_csv.csv")
    print("(data-set is: ", arr, "dtype=", arr.dtype, ")\n")


if __name__ == "__main__":
    main()
