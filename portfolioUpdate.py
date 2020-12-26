import pandas_datareader as pd
import pandas
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

stocks = ["CPR.MI", "RACE.MI", "PST.MI", "G.MI", "LDO.MI", "PRY.MI", "SPM.MI", "MB.MI", "UNI.MI", "REC.MI", "FCA.MI",
          "BPE.MI", "MONC.MI",
          "ENI.MI", "BZU.MI", "IP.MI", "TIT.MI", "SRG.MI", "STM.MI", "TRN.MI", "BMED.MI", "AZM.MI", "BAMI.MI", "TEN.MI",
          "IG.MI", "FBK.MI",
          "DIA.MI", "ATL.MI", "CNHI.MI"]
# Including "PIRC.MI" will leed to an error in calculating the covariance matrix
amnt = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100]
'''
i=True
stocks=[]
amnt=[]
while i==True:
    inpt=input("Insert Stock Code (UPPER CASE): ")
    if len(inpt)>=1:
        stocks.append(str(inpt))
    else:
        i=False        
    inp=input("Insert quantities: ")
    if len(inp)>=1:
        amnt.append(int(inp))
    else:
        i=False
'''

kinp=input("Choose Target Expected Return: ")
if len(kinp)<1:
    k=0.0036
        
SUM =sum(amnt)
weight =[]
for w in amnt:
    weight.append((w/SUM))
weight=np.asarray(weight)

def Log_return(stk,Yframe):
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    rtrn = []
    for x in stk:
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn['r'] = atl['Close'].resample('M').ffill().pct_change()
        QRtrn = Qrtrn['r'].to_list()
        QRtrn.sort()
        del QRtrn[0:int(len(QRtrn) * 0.05)]
        del QRtrn[(len(QRtrn) - int(len(QRtrn) * 0.05)):-1]
        Average_return = np.average(QRtrn)
        rtrn.append(Average_return)
    return rtrn
Average_Q_Return = Log_return(stocks,5)

def Covariance(stk, Yframe):
    Stock_MTRX = pandas.DataFrame()
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    Stock_MTRX = pandas.DataFrame()
    for x in stk:
        filler = 0
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn['r'] = atl['Close'].resample('M').ffill().pct_change()
        QRtrn = Qrtrn['r'].to_list()
        QRtrn.sort()
        QRtrn = QRtrn[9:50]
        QRT = pandas.Series(QRtrn)
        Stock_MTRX[str(x)] = QRT.values
    return Stock_MTRX.cov()


Covar = Covariance(stocks, 5)
st = np.dot(weight.T, np.dot(Covar, weight)) ** 0.5

averg_rtr = []
for i in np.arange(0,len(stocks)):
    averg_rtr.append((stocks[i],Average_Q_Return[i]))

for i in np.arange(0,len(stocks)):
    portfolio_return = Average_Q_Return[i]*weight[i]

def mvpx(): 
    mu=[]
    for i in averg_rtr:
        mu.append(i[1])
    mut=np.transpose(mu)
    e=[]
    for j in range(len(stocks)):
        e.append(1)
    et=np.transpose(e)
    Vinv=np.linalg.inv(Covar)
    a=np.dot(np.dot(mut,Vinv),mu)
    b=np.dot(np.dot(et,Vinv),mu)
    c=np.dot(np.dot(et, Vinv), e)
    
    l1=(k*c-b)/(a*c-b**2)
    l2=(a-b*k)/(a*c-b**2)
    
    x1=np.dot(np.dot(l1,Vinv), mu)
    x2=np.dot(np.dot(l2, Vinv), e)
    x=x1+x2
    return x    
x = mvpx()

def mvpxret(): 
    mu=[]
    for i in averg_rtr:
        mu.append(i[1])
    mut=np.transpose(mu)
    e=[]
    for j in range(len(stocks)):
        e.append(1)
    et=np.transpose(e)
    Vinv=np.linalg.inv(Covar)
    a=np.dot(np.dot(mut,Vinv),mu)
    b=np.dot(np.dot(et,Vinv),mu)
    c=np.dot(np.dot(et, Vinv), e)
    
    l1=(k*c-b)/(a*c-b**2)
    l2=(a-b*k)/(a*c-b**2)
    
    x1=np.dot(np.dot(l1,Vinv), mu)
    x2=np.dot(np.dot(l2, Vinv), e)
    x=x1+x2
    xt=np.transpose(x)
    xret=np.dot(mut, xt)
    return xret   
xret = mvpxret()

def mvpxvar(): 
    mu=[]
    for i in averg_rtr:
        mu.append(i[1])
    mut=np.transpose(mu)
    e=[]
    for j in range(len(stocks)):
        e.append(1)
    et=np.transpose(e)
    Vinv=np.linalg.inv(Covar)
    a=np.dot(np.dot(mut,Vinv),mu)
    b=np.dot(np.dot(et,Vinv),mu)
    c=np.dot(np.dot(et, Vinv), e)
    
    l1=(k*c-b)/(a*c-b**2)
    l2=(a-b*k)/(a*c-b**2)
    
    x1=np.dot(np.dot(l1,Vinv), mu)
    x2=np.dot(np.dot(l2, Vinv), e)
    x=x1+x2
    xt=np.transpose(x)
    xvar=np.dot(np.dot(x,Covar), xt)**0.5
    return xvar
xvar = mvpxvar()

def Hmodf():
    H=0
    for n in weight:
        H=H+n**2
    oon=1/(int(len(stocks)))
    Hmod=(H-oon)/(1-oon)
    return Hmod
Hmod=Hmodf()
    
def ewret():
    ewp=[]
    mu=[]
    for i in averg_rtr:
        mu.append(i[1])
    mut=np.transpose(mu)
    oon=1/(int(len(stocks)))
    for l in range(len(stocks)):
        ewp.append(float(oon))
    ewreturn=np.dot(mut, ewp)
    return ewreturn
ewreturn=ewret()
    
def ewst():
    ewp=[]
    oon=1/(int(len(stocks)))
    for l in range(len(stocks)):
        ewp.append(float(oon))
    ewpt=np.transpose(ewp)
    ewstd=(np.dot(np.dot(ewpt, Covar), ewp))**0.5 
    return ewstd
ewstd=ewst()

def ewc():
    ewp=[]
    oon=1/(int(len(stocks)))
    for l in range(len(stocks)):
        ewp.append(float(oon))
    ewpt=np.transpose(ewp)
    return ewpt
ewpt=ewc()


def historical_port_return(stk, Yframe):
    df = pandas.DataFrame()
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    for x in stk:
        wg = weight[stocks.index(x)]
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn[str(x)] = atl['Close'].resample('M').ffill().pct_change()
        Qrtrn[str(x)] = wg*Qrtrn[str(x)]
        PRT = Qrtrn[str(x)].to_list()
        df[str(x)] = Qrtrn[str(x)]
    PR = pandas.DataFrame()
    PR['Portfolio Retrun'] = df.sum(axis =1)
    return PR
Portfolio_return = historical_port_return(stocks,5)

def historical_port_value(stk, Yframe):
    df = pandas.DataFrame()
    Edt = dt.date.today()
    Sdt = Edt - dt.timedelta(days=365 * Yframe)
    for x in stk:
        am = amnt[stocks.index(x)]
        atl = pd.DataReader(x, data_source='yahoo', start=Sdt, end=Edt)
        Qrtrn = pandas.DataFrame()
        Qrtrn[str(x)] = atl['Close'].resample('M').ffill()
        Qrtrn[str(x)] = am * Qrtrn[str(x)]
        PRT = Qrtrn[str(x)].to_list()
        df[str(x)] = Qrtrn[str(x)]
    PR = pandas.DataFrame()
    PR['Portfolio Value'] = df.sum(axis=1)
    return PR

Portfolio_value = historical_port_value(stocks, 5)

print(f"""\
stock monthly return average: {averg_rtr}.\n
 Covariance matrix:\n
{Covar}\n
Chosen weight: {weight}. \n
Expected portfolio return: {round(portfolio_return, 4) * 100}%.\n
Portfolio's standard deviation: {round(st, 4) * 100} % \n
\n
Minimum Variance Portfolio, given target expected return: {k}. \n
Composition: {x} \n
Expected Return: {round(xret, 5) * 100}% \n
Standard Deviation: {round(xvar, 5) * 100}% \n
\n
        Diversification Analysis:   \n
Modified Herfindal Index: {Hmod} \n
    Composition {ewpt} \n
    Average Return: {round(ewreturn, 4) * 100}%\n
    Standard Deviation: {round(ewstd, 4) * 100}%""")

plt.title("Historical portfolio return")
plt.plot(Portfolio_return)
plt.show()


print(Portfolio_value)
plt.title("Historical portfolio value")
plt.plot(Portfolio_value)
plt.show()
