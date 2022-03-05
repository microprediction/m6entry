
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


if __name__=='__main__':
    df_cov = pd.read_csv(COV_FILE, index_col=0)
    df_prob = pd.read_csv(RANK_FILE, index_col=0)
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
    risk_exponent = 0.05 # If >0 we make higher vol look relatively riskier
    earnings_penalty = 1.1  # If >1 we make those with earnings look riskier, otherwise safer
    exclusion_penalty = 1.  # If >1, shy away from anything not explicitly listed

    risk_penalties = np.exp(normalize(np.log(np.power(mu, risk_exponent))))
    exclusion_penalties  = np.array([exclusion_penalty if tck not in TICKERS_TO_INCLUDE else 1.0 for tck in tickers])
    earnings_penalties = np.array([earnings_penalty if tck in TICKERS_WITH_EARNINGS_SOON else 1.0 for tck in tickers])

    penalties = exclusion_penalties * earnings_penalties * risk_penalties
    S = scatter(penalties)
    penalized_cov = cov * S
    penalized_cov = nearest_pos_def(penalized_cov)
    long_w = port(cov=penalized_cov)


    w_ivv = np.array([1.0 if tck == 'IVV' else 0 for tck in tickers])
    w_submit = 0.499*np.array(long_w) -0.499*w_ivv

    CURVE_TRADE = False
    if CURVE_TRADE:
        # Completely override this and do a term structure trade for fun
        w_shy = np.array([ 1/3.5 if tck=='SHY' else 0 for tck in tickers ])
        w_ief = np.array([ -2/8.3 if tck == 'IEF' else 0 for tck in tickers])
        w_tlt = np.array([ 1/14.6 if tck=='TLT' else 0 for tck in tickers ])
        w_sum = sum(np.abs(w_shy))+sum(np.abs(w_tlt))+sum(np.abs(w_ief))
        w_submit = (w_shy+w_tlt+w_ief)/w_sum



    print('Done portfolio opt')
    entry = df_prob.copy()
    entry['Decision'] = [round(w_scaled_i,5) for w_scaled_i in w_submit]
    entry.rename(inplace=True,columns={'0':'Rank1','1':'Rank2','2':'Rank3','3':'Rank4','4':'Rank5'})
    entry.to_csv(SUBMISSION_FILE)
    m6_dump(entry, SUBMISSION_FILE)


    entry['abs'] = entry['Decision'].abs()
    bets = entry.sort_values(by='abs', ascending=False)[:10]
    bets.to_csv(BETS_FILE)
    pprint(bets)

