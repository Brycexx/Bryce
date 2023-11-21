# Be more ambitious, be more diligent
import pandas as pd
import numpy as np
import statsmodels.api as sm


df_price=pd.read_csv("adjusted.csv")
df_price['date'] = pd.to_datetime(df_price['Date'], format='%Y%m%d')
df_price.set_index('date', inplace=True)
df_price = df_price.drop('Date', axis=1)

# Calculate the volatility
log_price=np.log(df_price)
log_ret=log_price.diff()
log_ret.fillna(0,inplace=True)
daily_volatility = log_ret.rolling(window=21).std()
daily_volatility[daily_volatility < 0.005] = 0.005
daily_volatility.fillna(0,inplace=True)
daily_volatility.to_csv('daily_volatility.csv')

# Calculate the 10-day return
log_10d_ret=np.log(df_price/df_price.shift(10))
log_10d_ret.fillna(0,inplace=True)

# normalize the variable
standardized_returns = log_10d_ret/daily_volatility
standardized_returns.fillna(0,inplace=True)

# Calculate Market Return
univ=pd.read_csv('univ_h.csv',index_col=0)
univ.index=pd.to_datetime(univ.index,format='%Y')
univ_daily=univ.reindex(log_10d_ret.index,method='ffill')
average_returns=((univ_daily*log_10d_ret).sum(axis=1)/univ_daily.sum(axis=1)).to_frame(name='average_Return')
factor_v=log_10d_ret.sub(average_returns['average_Return'],axis=0)

# Construct factor M
average_1d_returns=((univ_daily*log_ret).sum(axis=1)/univ_daily.sum(axis=1)).to_frame(name='average_1d_Return')
ex_1d_returns=log_ret.sub(average_returns['average_Return'],axis=0)
ex_1d_returns_abs=ex_1d_returns.abs()
factor_m=ex_1d_returns_abs.rolling(window=21).max()
average_m=((univ_daily*factor_m).sum(axis=1)/univ_daily.sum(axis=1)).to_frame(name='average_m')
factor_M=factor_m.sub(average_m['average_m'],axis=0)

# clean the table
factor_M_shifted=factor_M.shift(1)
factor_v_shifted=factor_v.shift(1)
factor_M_shifted=factor_M_shifted[factor_M_shifted.index>='20040101']
factor_v_shifted=factor_v_shifted[factor_v_shifted.index>='20040101']
ex_1d_returns=ex_1d_returns[ex_1d_returns.index>='20040101']
univ_daily=univ_daily[univ_daily.index>='20040101']

# multi OLS
coefficients=pd.DataFrame(columns=['Date','constant','factor_M','factor_v'])
factor_M_shifted.to_csv('factor_M_shifted.csv')
factor_v_shifted.to_csv('factor_v_shifted.csv')
ex_1d_returns.to_csv('ex_1d_returns.csv')

for date in ex_1d_returns.index:
    universe = univ_daily.loc[date] == 1
    factors=pd.concat([factor_M_shifted.loc[date], factor_v_shifted.loc[date]], axis=1)
    returns=ex_1d_returns.loc[date]
    factors = factors[universe]
    returns = returns[universe]
    factors=sm.add_constant(factors)
    model=sm.OLS(returns,factors).fit()
    coefficients=coefficients._append({'Date':date,'constant':model.params.iloc[0],'factor_M':model.params.iloc[1],'factor_v':model.params.iloc[2]},ignore_index=True)
coefficients.to_csv('Coefficients.csv',index=False)
print(coefficients['factor_M'].mean())



