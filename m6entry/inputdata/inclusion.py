from m6entry.inputdata.volatiitydata import etf_vols

TICKERS_TO_INCLUDE = [k for k,v in etf_vols.items() if 'XL' in k]

if __name__=='__main__':
    print(TICKERS_TO_INCLUDE)