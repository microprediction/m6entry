import numpy as np
import pandas as pd
from precise.skatertools.m6.covarianceforecasting import m6_corr
from precise.skaters.covariance.allcovskaters import ALL_D0_SKATERS
from precise.skatertools.m6.quintileprobabilities import mvn_quintile_probabilities
from m6entry.inputdata.volatiitydata import all_vols
from m6entry.inputdata.earnings import TICKERS_WITH_EARNINGS_SOON
from m6entry.whereami import RANK_FILE, COV_FILE, VOL_FILE, M6_DATA


def m6_probabilities(interval='d',n_dim=100, n_samples=200000, n_obs=200):
    from precise.skaters.covariance.ewapm import ewa_pm_emp_scov_r01_n100_t0 as f
    corrdf = m6_corr(f=f,interval=interval, n_dim=n_dim, n_obs=n_obs)

    # Combine with volatility data
    tickers = list(corrdf.columns)

    vols = [ all_vols.get(t) for t in tickers ]
    nan_vols = [ np.nan if v is None else v for v in vols ]
    mean_vol = np.nanmean(nan_vols)
    clean_vols = [ mean_vol if np.isnan(v) else v for v in nan_vols ]
    try:
        vol_df = pd.DataFrame(index=tickers, columns={'iv30'}, data=clean_vols)
    except Exception as e:
        vol_df = pd.DataFrame(index=tickers, columns={'iv30'}, data=[[v] for v in clean_vols])
    vol_df.to_csv(VOL_FILE)

    # Shrink vols?
    lmbd = 0.5
    shrunk_vols = [v * (1 - lmbd) + (lmbd) * mean_vol for v in clean_vols]
    normalized_vols = [ v/mean_vol for v in shrunk_vols ]

    # Rank probabilities
    D = np.diag(normalized_vols)
    sgma = np.matmul(np.matmul(D,corrdf.values), D)
    print('Starting simulation')
    p = mvn_quintile_probabilities(sgma=sgma, n_samples=n_samples)
    df = pd.DataFrame(columns=tickers, data=p).transpose()
    df_cov = pd.DataFrame(columns=tickers, index=tickers, data=sgma)
    return df, df_cov


if __name__=='__main__':
    df, df_cov = m6_probabilities(interval='d',n_dim=100, n_obs=300, n_samples=20000)
    print(df[:7])
    df.to_csv(RANK_FILE)
    df_cov.to_csv(COV_FILE)





