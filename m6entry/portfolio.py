
import pandas as pd
import os
from m6entry.whereami import TOP
from pprint import pprint
import numpy as np


from pypfopt.efficient_frontier import EfficientFrontier

if __name__=='__main__':
    for tag in ['','1','2']:
        df_cov = pd.read_csv(os.path.join(TOP, 'covariance'+tag+'.csv'), index_col=0)
        df_prob = pd.read_csv(os.path.join(TOP, 'probabilities'+tag+'.csv'), index_col=0)
        pprint(df_cov[:4])
        tickers = list(df_cov.columns)
        n_dim = len(df_cov.index)
        mu = np.zeros(n_dim)
        cov = df_cov.values
        from precise.skaters.covarianceutil.covfunctions import affine_shrink, nearest_pos_def
        phi = 1.1+0.1*np.random.rand()
        lbmda = 0.03*np.random.randn()
        cov = affine_shrink(cov,phi=phi, lmbd=lbmda )
        cov = nearest_pos_def(cov)
        ef = EfficientFrontier(mu, cov)
        weights = ef.max_quadratic_utility()
        print('Done portfolio opt')
        entry = df_prob.copy()
        entry['Decision'] = [ round(weights[i],5) for i in range(n_dim)]
        entry.rename(inplace=True,columns={'0':'Rank1','1':'Rank2','2':'Rank3','3':'Rank4','4':'Rank5'})
        entry.to_csv('entry'+tag+'.csv')
        print(entry[:10])
        print(entry.abs().sum())
