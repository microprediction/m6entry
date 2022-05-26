
import pandas as pd
import os
from m6entry.whereami import TOP, COV_FILE, RANK_FILE, SUBMISSION_FILE, BETS_FILE
from pprint import pprint
import numpy as np
from precise.skaters.locationutil.vectorfunctions import scatter, normalize
from m6entry.inputdata.earnings import TICKERS_WITH_EARNINGS_SOON
from m6entry.inputdata.inclusion import TICKERS_TO_INCLUDE
from precise.skatertools.m6.competition import m6_dump

from pypfopt.efficient_frontier import EfficientFrontier
from precise.skaters.portfoliostatic.schurport import schur_weak_weak_s5_g100_long_port as port
from precise.skaters.portfoliostatic.weakport import weak_h125_long_port as port

if __name__=='__main__':
    df_cov = pd.read_csv(COV_FILE, index_col=0)
    df_prob = pd.read_csv(RANK_FILE, index_col=0)

    # Fix rounding
    df_prob = df_prob.round(decimals=4)
    excess = [v-1.0 for v in df_prob.sum(axis=1).values]
    df_prob['2'] = df_prob['2'] - excess
    df_prob = df_prob.round(decimals=4)
    excess = [v - 1.0 for v in df_prob.sum(axis=1).values]
    assert( sum(excess)<1e-8 )



    pprint(df_cov[:4])
    tickers = list(df_cov.columns)
    n_dim = len(df_cov.index)
    cov = df_cov.values
    from precise.skaters.covarianceutil.covfunctions import affine_shrink, nearest_pos_def, cov_to_corrcoef
    phi = 1.1
    lbmda = 0.03*np.random.randn()
    cov = affine_shrink(cov,phi=phi, lmbd=lbmda )
    cov = nearest_pos_def(cov)

    # Work with correlations and assume that volatility premium is roughly accurate

    mu = np.diag(cov)
    risk_exponent = 0.1 # If >0 we make higher vol look relatively riskier
    earnings_penalty = 1.1  # If >1 we make those with earnings look riskier, otherwise safer

    LS = False
    if LS:
        exclusion_penalty = 10.0  # If >1, shy away from anything not explicitly listed
    else:
        exclusion_penalty = 1.02

    risk_penalties = np.exp(normalize(np.log(np.power(mu, risk_exponent))))
    exclusion_penalties  = np.array([exclusion_penalty if tck not in TICKERS_TO_INCLUDE else 1.0 for tck in tickers])
    earnings_penalties = np.array([earnings_penalty if tck in TICKERS_WITH_EARNINGS_SOON else 1.0 for tck in tickers])

    penalties = exclusion_penalties * earnings_penalties * risk_penalties
    S = scatter(penalties)
    penalized_cov = cov * S
    penalized_cov = nearest_pos_def(penalized_cov)
    long_w = port(cov=penalized_cov)

    if LS:
        w_ivv = np.array([1.0 if tck == 'IVV' else 0 for tck in tickers])
        w_submit = 0.499*np.array(long_w) -0.499*w_ivv
    else:
        w_submit = [0.999*wi for wi in long_w]

    print('Done portfolio opt')
    entry = df_prob.copy()
    entry['Decision'] = [round(w_scaled_i,4) for w_scaled_i in w_submit]
    entry.rename(inplace=True,columns={'0':'Rank1','1':'Rank2','2':'Rank3','3':'Rank4','4':'Rank5'})






    entry.to_csv(SUBMISSION_FILE)
    m6_dump(entry, SUBMISSION_FILE)


    entry['abs'] = entry['Decision'].abs()
    bets = entry.sort_values(by='abs', ascending=False)[:10]
    bets.to_csv(BETS_FILE)
    pprint(bets)

