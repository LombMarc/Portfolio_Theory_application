import pandas_datareader as pd
import pandas
import numpy as np
import datetime as dt

stocks = ['atl.mi','race.mi','ENI.MI']

def Log_return(stk,Yframe):
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    rtrn = []
    for x in stk:
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn['r'] = atl['Close'].resample('Q').ffill().pct_change()
        QRtrn = Qrtrn['r'].to_list()
        QRtrn.sort()
        del QRtrn[0:int(len(QRtrn) * 0.05)]
        del QRtrn[(len(QRtrn) - int(len(QRtrn) * 0.05)):-1]
        Average_return = np.average(QRtrn)
        rtrn.append(Average_return)
    return rtrn
Average_Q_Return = Log_return(stocks,5)

def Covariance(stk,Yframe):
    Stock_MTRX = pandas.DataFrame()
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    for x in stk:
        filler = 0
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn['r'] = atl['Close'].resample('Q').ffill().pct_change()
        QRtrn = Qrtrn['r'].to_list()
        QRtrn.sort()
        del QRtrn[0:int(len(QRtrn) * 0.05)]
        del QRtrn[(len(QRtrn) - int(len(QRtrn) * 0.05)):-1]
        Stock_MTRX[x] = np.asarray(QRtrn)
    cova = Stock_MTRX.cov()
    return cova
Covar = Covariance(stocks,5)
st = np.dot(weight.T,np.dot(Covar,weight))**0.5


print("Stock Quarterly return average: "+Average_Q_Return)
print("Covariance matrix of the stock return: ")
print(Covar)
print(weight)
print(st)
