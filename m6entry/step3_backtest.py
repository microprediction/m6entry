
import pandas as pd
import os
from m6entry.whereami import TOP, COV_FILE, RANK_FILE, SUBMISSION_FILE, BETS_FILE, M6_DATA
from precise.skatertools.m6.covarianceforecasting import m6_data
import numpy as np
import matplotlib.pyplot as plt



if __name__=='__main__':
    submission = pd.read_csv(SUBMISSION_FILE)
    print(submission)
    tickers = list(submission.index)
    weights = submission['Decision'].values
    try:
        data = pd.read_csv(M6_DATA)
        data.drop(axis=1, labels=['Unnamed: 0'],inplace=True)
    except FileNotFoundError:
        data = m6_data(interval='d', n_dim=100, n_obs=30)
        data.to_csv(M6_DATA)
    col_tickers = list(data.columns)
    w = np.array([ weights[ col_tickers.index(ticker)] for ticker in col_tickers ])
    portfolio_returns = np.squeeze(np.dot(data.values,w))
    portfolio_mean = np.mean(portfolio_returns)
    portfolio_std = np.std(portfolio_returns)
    asset_std = np.std(data, axis=0)
    portfolio_info = portfolio_mean/portfolio_std
    print(portfolio_std)
    plt.plot(portfolio_returns)
    plt.title('info = '+str(portfolio_info))
    plt.show()
    pass


