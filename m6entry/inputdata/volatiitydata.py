
stock_vols = {'ABBV':20.7,
           'ACN':23.6,
           'AEP':17.4,
           'AIZ':20.5,
           'ALLE':27.4,
           'AMAT':39.7,
           'AMP':31.7,
           'AMZN':35.5,
           'AVB':19.6,
           'AVY':25.3,
           'AXP':31.7,
           'BDX':19.1,
           'BMY':19.2,
            'BF-B':17.8,
           'BR':23.9,
           'CARR':28.6,
           'CDW':27.5,
           'CE':28.5,
           'CHTR':30.9,
           'CNC':33.3,
           'CNP':18.7,
           'COP':34.5,
           'CTAS':24.2,
           'CZR':49.9,
           'DG':20.8,
           'DPZ':38.6,
           'DRE':19.3,
           'DXC':40.0,
           'FB':45.8,
           'FTV':26.6,
           'GOOG':30.5,
           'GPC':24.2,
           'HIG':27.3,
           'HST':31.0,
           'JPM':28.0,
           'KR':28.2,
           'OGN':27.4,
           'PG':19.3,
           'PPL':15.9,
           'PYPL':41.4,
           'RE':23.9,
           'ROL':26.0,
           'ROST':28.9,
           'UNH':24.2,
           'URI':37.8,
           'V':29.0,
           'VRSK':24.8,
           'PRU':25.6,
           'WRK':30.3,
           'XOM':30.8}

etf_vols={'EWA':18.2,
              'EWC':16.3,
              'EWG':26.8,
              'EWH':20.1,
              'EWJ':17.9,
              'EWL':18.0,
              'EWQ':29.0,
              'EWT':22.6,
              'EWU':18.3,
              'EWY':24.1,
              'EWZ':27.8,
              'GSG':40.9,
              'HYG':9.3,
              'IAU':15.9,
              'ICLN':31.7,
              'IEF':9.3,
              'IEMG':18.8,
              'IEUS':22.0,
              'IGF':14.3,
              'INDA':19.2,
              'IVV':16.1,
              'IWM':22.4,
              'IXN':24.9,
              'LQD':10.9,
              'MCHI':33.3,
              'REET':16.4,
              'SHY':3.6,
              'SLV':25.7,
              'TLT':19.3,
              'VXX':121.6,
              'XLB':18.3,
              'XLC':19.7,
              'XLE':30.3,
              'XLF':21.3,
              'XLI':17.4,
              'XLK':23.1,
              'XLP':13.1,
              'XLU':15.4,
              'XLV':14.1,
              'XLY':25.7}

all_vols = stock_vols.copy()
all_vols.update(etf_vols)

import pandas as pd
VOL = pd.DataFrame(index=all_vols.keys(), columns=['vol'], data=all_vols.values())
