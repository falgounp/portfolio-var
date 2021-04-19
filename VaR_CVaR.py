#implement historical method
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from scipy.stats import norm,t
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
pd.options.plotting.backend = "plotly"


##Streamlit page setup downloading daily...
st.set_page_config(page_title = 'VaR-WebApp', layout = 'wide')
st.title('VaR and CVaR Calculation for Stocks')

expander_bar = st.beta_expander("About")

expander_bar.markdown("""**This app calculates VaR and CVaR of any porfolio of 
                      stocks listed under Nasdaq using three different methods 
                      namely historical, parametric and monte carlo stimulations.**""", 
            unsafe_allow_html = True)

expander_bar.markdown("""
* ** Python Libraries:**Streamlit,base64,pandas,numpy,plotly, datetime
* ** Data Source:** Daily Data from Yahoo Pandas DataReader
""")
    
@st.cache
#import data
def getData():
    stocks = ['MSFT', 'AAPL', 'AMZN', 'ACN', 'TSLA', 'KRMD', 'SEEL', 'BIOC', 'MBIO']
    endDate = dt.datetime.today()
    startDate = endDate - dt.timedelta(days = 800)
    stockData = pdr.get_data_yahoo(stocks, start = startDate, end = endDate)
        
    stockData = stockData['Close']
    
    returns = stockData.pct_change()

    return returns, stockData, stocks

returns, stockData, stock_list = getData()
endDate = dt.datetime.today()
startDate = endDate - dt.timedelta(days = 800)



#Sidebar - Options
col1 = st.sidebar


selected_portfolio = col1.multiselect('Stocks Selection for portfolio', stock_list, stock_list)
print(selected_portfolio)
col1.subheader('End Datetime')
col1.write(endDate)
col1.subheader('Start Datetime')
col1.write(startDate)
col1.subheader('Holding Period')
Time = col1.selectbox('T', [1, 10, 50, 100, 200], index = 1)
col1.subheader('Initial Investment in $')
InitialInvestment = col1.selectbox('Total Value', [100, 1000, 10000, 100000], index = 2)
col1.subheader('Select the level of confidence')
alpha = col1.selectbox('Alpha', ['90%', '95%', '99%'], index = 1)
if alpha == '90%':
    alpha = 10
elif alpha == '95%':
    alpha = 5
elif alpha == '99%':
    alpha = 1
col1.subheader('Number of Monte Carlo Stimulations')
    
mc_sims = col1.selectbox('MC_SIMS', [1,10,100,1000,10000,100000], index = 0)
###display selected portfolio returns --- filtering data
returns = returns[selected_portfolio] #filtering data
returns = returns.dropna()

def covMatrix(returns):
    return returns.cov()

covMatrix = covMatrix(returns)


def meansReturn(returns):
    return returns.mean()

meansReturn = meansReturn(returns)

weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)
returns['portfolio'] = returns.dot(weights).copy()









#sidebar




#Portfolio Performance
def portfolioPerformance(weights, meansReturn, covMatrix, Time):
    
    returns = np.sum(meansReturn * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(Time)
    
    
    return returns, std



expander_bar2 = st.beta_expander('Portfolio Details')

expander_bar2.write("Selected Stocks in the potfolio - ")

expander_bar2.write(f"""

                    
**Stocks in the portfolio -- {selected_portfolio} **
""")
expander_bar2.write('Stock Price Charts')
fig = stockData[selected_portfolio].plot()
expander_bar2.plotly_chart(fig)

with expander_bar2:
    col2,col3, col4 = st.beta_columns((1, 1, 1))
    col2.subheader('Returns of the selected portfolio')
    col2.write(returns['portfolio'])
    
    col3.subheader('Covariance Matrix')
    col3.write(covMatrix)

    col4.subheader('Randomly assigned weights')
    col4.write(weights)




#Var calculations

def historicalVaR(returns, alpha = alpha):
    """
    Read in a pandas series of returns or pandas dataframe
    
    Output the percentile of the distribution at the given alpha level 
    confidence
    
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha = 5)
    
    else:
        raise TypeError('Expected Return should be dataframe or Series')


def var_parametric(portfolioReturn, portfolioStd, distribution = 'normal', alpha = 5, dof = 6):
    #Calculate the var of the portfolio, given the distribution with known parameters
    
    if distribution == 'normal':
        VaR = norm.ppf(1-alpha/100) * portfolioStd - portfolioReturn
    
    
    elif distribution == 't-distribution':
        nu = dof
        VaR = np.sqrt((nu - 2) / nu) * t.ppf(1-alpha/100, nu) * portfolioStd - portfolioReturn
    
    else:
        raise TypeError('Distriubtion is normal or t')
    
    return VaR

pRet, pStd = portfolioPerformance(weights, meansReturn, covMatrix, Time)    

T = Time 


meanM = np.full(shape = (T, len(weights)), fill_value = meansReturn)
meanM = meanM.T
portfoliosims = np.full(shape = (T, mc_sims), fill_value = 0.0)

initialPortfolioValue = InitialInvestment
#assume cholesky decompostion to return daily returns
for m in range(0, mc_sims):
    #Mc loops
    
    Z = np.random.normal(size = (T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfoliosims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1) * initialPortfolioValue

portfolio_MonteC_Results = pd.Series(portfoliosims[-1, :])


def MonteCarloVaR(returns, alpha = 5):
    """
    Read in a pandas series of returns or pandas dataframe
    
    Output the percentile of the distribution at the given alpha level 
    confidence
    
    """
    if isinstance(returns, pd.Series):
        
        return np.percentile(returns, alpha)
    

    else:
        
        raise TypeError('Expected pandas series')


VARs = []
st.subheader('VaR calculations')
VaR_methods = ['Historical VaR', 'Parametric VaR', 'MonteCarlo VaR']
VaR_button = st.radio('VaR', VaR_methods, index = 0)
if VaR_button == 'Historical VaR':
    VaR = historicalVaR(returns['portfolio'], alpha = alpha) * np.sqrt(Time)
    VARs.append(['Historical VaR', VaR])
    st.write(f"Value at Risk at {100 - alpha} % confidence interval is  : ${round(-VaR * InitialInvestment, 2)}")

elif VaR_button == 'Parametric VaR':
    distribution = st.selectbox('Select Distribution: ', ['normal', 't-distribution'], index = 0)
    if distribution == 'normal':
        normVaR = var_parametric(pRet, pStd, distribution = 'normal', alpha = alpha, dof = 6)
        VARs.append(['Parametric Normal VaR', normVaR])
        st.write(f"Normal Value at Risk at {100 - alpha} % confidence interval is  : ${round(normVaR * InitialInvestment, 2)}")
    else:
        tVaR = var_parametric(pRet, pStd, distribution = 't-distribution', alpha = 5, dof = 6)
        VARs.append(['Parametric T VaR', tVaR])
        st.write(f" T Value at Risk at {100 - alpha} % confidence interval  : ${round(tVaR * InitialInvestment, 2)}")
elif VaR_button == 'MonteCarlo VaR':
    mVaR = initialPortfolioValue - MonteCarloVaR(portfolio_MonteC_Results, alpha = alpha)
    st.write(f" Value at Risk using monte carlo at {100 - alpha} % confidence interval  : ${round(mVaR, 2)}")
    VARs.append(['MonteCarlo VaR', mVaR])
    df_p = pd.DataFrame(portfoliosims)
    fig = df_p.plot(title = "Stock Portfolio Stimulations", template="simple_white", labels=dict(index="Time Period", value=" Portfolio Value"))
    st.plotly_chart(fig)

#CVaR calculations:--
    
    
    
def historicalCVaR(returns, alpha = alpha):
    """
    Read in a pandas series of returns or pandas dataframe
    
    Output the CVaR for dataframe/series 
    
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha = alpha)
        return returns[belowVaR].mean() #expectation of loss
    
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha = alpha)
    
    else:
        raise TypeError('Expected Return should be dataframe or Series')


def cvar_parametric(portfolioReturn, portfolioStd, distribution = 'normal', alpha = alpha, dof = 6):
    #Calculate the cvar of the portfolio, given the distribution with known parameters
    
    if distribution == 'normal':
        CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(1-alpha/100)) * portfolioStd - portfolioReturn
    
    
    elif distribution == 't-distribution':
        nu = dof
        x_anu = t.ppf(alpha/100, nu)
        CVaR = -1/(alpha/100) * (1-nu) ** -1 * (nu - 2 + x_anu ** 2) * t.pdf(x_anu, nu) * portfolioStd - portfolioReturn
    
    else:
        raise TypeError('Distriubtion is normal or t')
    
    return CVaR


def MonteCarloCVaR(returns, alpha = alpha):
    """
    Read in a pandas series of returns or pandas dataframe
    
    Output the CVaR for dataframe/series 
    
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha = alpha)
        
        return returns[belowVaR].mean() #expectation of loss
    
    else:
        
        raise TypeError('Expected pandas series')


CVARs = []
st.subheader('CVaR calculations')
CVaR_methods = ['Historical CVaR', 'Parametric CVaR', 'MonteCarlo CVaR']
CVaR_button = st.radio('CVaR', CVaR_methods, index = 0)
if CVaR_button == 'Historical CVaR':
    CVaR = historicalCVaR(returns['portfolio'], alpha = alpha) * np.sqrt(Time)
    CVARs.append(['Historical CVaR', CVaR])
    st.write(f"Conditional Value at Risk at {100 - alpha} % confidence interval is  : ${round(-CVaR * InitialInvestment, 2)}")

elif CVaR_button == 'Parametric CVaR':
    distribution = st.selectbox('Select Distribution: ', ['normal', 't-distribution'], index = 0)
    if distribution == 'normal':
        normCVaR = cvar_parametric(pRet, pStd, distribution = 'normal', alpha = alpha, dof = 6)
        CVARs.append(['Parametric Normal CVaR', normCVaR])
        st.write(f"Normal Conditional Value at Risk at {100 - alpha} % confidence interval is  : ${round(normCVaR * InitialInvestment, 2)}")
    else:
        tCVaR = cvar_parametric(pRet, pStd, distribution = 't-distribution', alpha = alpha, dof = 6)
        CVARs.append(['Parametric T CVaR', tCVaR])
        st.write(f" T Conditional Value at Risk at {100 - alpha} % confidence interval  : ${round(tCVaR * InitialInvestment, 2)}")
elif CVaR_button == 'MonteCarlo VaR':
    mCVaR = initialPortfolioValue - MonteCarloCVaR(portfolio_MonteC_Results, alpha = alpha)
    CVARs.append(['Monte Carlo CVaR', mCVaR])
    st.write(f" Conditional Value at Risk using monte carlo at {100 - alpha} % confidence interval  : ${round(mVaR, 2)}")
    df_p = pd.DataFrame(portfoliosims)
    fig = df_p.plot(title = "Stock Portfolio Stimulations", template="simple_white", labels=dict(index="Time Period", value=" Portfolio Value"))
    st.plotly_chart(fig)


df_c = pd.DataFrame(VARs)

















    


    
    



        
        
