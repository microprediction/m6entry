import numpy as np
import pandas as pd
from precise.skatertools.m6.covarianceforecasting import m6_corr
from precise.skatertools.m6.quintileprobabilities import mvn_quintile_probabilities
from m6entry.volatiitydata import all_vols



def m6_probabilities(interval='d',n_dim=100, n_samples=5000, n_obs=200):
    corrdf = m6_corr(interval=interval, n_dim=n_dim, n_obs=n_obs)
    tickers = list(corrdf.columns)
    vols = [ all_vols.get(t) for t in tickers ]
    nan_vols = [ np.nan if v is None else v for v in vols ]
    mean_vol = np.nanmean(nan_vols)
    clean_vols = [ mean_vol if np.isnan(v) else v for v in nan_vols ]
    normalized_vols = [ v/mean_vol for v in clean_vols ]
    D = np.diag(normalized_vols)
    sgma = np.matmul(np.matmul(D,corrdf.values), D)
    print('Starting simulation')
    p = mvn_quintile_probabilities(sgma=sgma, n_samples=n_samples)
    df = pd.DataFrame(columns=tickers, data=p).transpose()
    df_cov = pd.DataFrame(columns=tickers, index=tickers, data=sgma)
    return df, df_cov


if __name__=='__main__':

    df, df_cov = m6_probabilities(interval='d',n_dim=100, n_obs=300, n_samples=200)
    print(df[:7])
    df.to_csv('probabilities1.csv')
    df_cov.to_csv('covariance1.csv')
    df, df_cov = m6_probabilities(interval='d',n_dim=100, n_obs=300, n_samples=200)
    print(df[:7])
    df.to_csv('probabilities2.csv')
    df_cov.to_csv('covariance2.csv')




