import os
from pprint import pprint
from datetime import  datetime

TOP = os.path.dirname(os.path.abspath(__file__))
TODAY_STR = datetime.today().strftime('%Y-%m-%d')
RANK_FILE = os.path.join(TOP, 'outputdata/rankprobabilities/', 'rp_' + TODAY_STR + '.csv')
COV_FILE = os.path.join(TOP, 'outputdata/covariances/', 'cov_' + TODAY_STR + '.csv')
PORT_FILE = os.path.join(TOP, 'outputdata/portfolios/', 'port_' + TODAY_STR + '.csv')
VOL_FILE = os.path.join(TOP, 'outputdata/impliedvols/', 'iv_' + TODAY_STR + '.csv')
SUBMISSION_FILE = os.path.join(TOP, 'outputdata/submissions/', 'submission_' + TODAY_STR + '.csv')
BETS_FILE = os.path.join(TOP, 'outputdata/bets/', 'bets_' + TODAY_STR + '.csv')

if __name__=='__main__':
    print(TOP)
    print(RANK_FILE)

