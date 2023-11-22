# Be more ambitious, be more diligent
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df_price=pd.read_csv("adjusted.csv")
df_price['date'] = pd.to_datetime(df_price['Date'], format='%Y%m%d')
df_price.set_index('date', inplace=True)
df_price = df_price.drop('Date', axis=1)

#将univ以外的股票都剔除
df_factor=pd.read_csv('results.csv')
df_factor.index=pd.to_datetime(df_factor['Year'],format='%Y')
df_factor=df_factor.drop('Year',axis=1)
df_factor.index=df_factor.index.year


df_univ=pd.read_csv('univ_h.csv',index_col=0)
df_univ.index=pd.to_datetime(df_univ.index,format='%Y')
df_univ_daily=df_univ.reindex(df_price.index,method='ffill')
df_price=df_univ_daily*df_price

#calculate expected return
df_exp=pd.DataFrame(index=df_price.index,columns=df_price.columns)
beta_v_df = pd.DataFrame(index=df_price.index, columns=df_price.columns)
beta_m_df = pd.DataFrame(index=df_price.index, columns=df_price.columns)
for year in range(2006, 2024):
    # 获取该年的 beta 值
    beta_v = df_factor.loc[year - 1, 'Avg_Beta_v']
    beta_m = df_factor.loc[year - 1, 'Avg_Beta_m']

    # 将 beta 值填充到对应的年份
    beta_v_df[df_price.index.year == year] = beta_v
    beta_m_df[df_price.index.year == year] = beta_m

# 计算预期回报,导入参数
df_factor_v=pd.read_csv('factor_v.csv')
df_factor_m=pd.read_csv('factor_M.csv')
df_factor_v.index=beta_v_df.index
df_factor_m.index=beta_v_df.index
df_factor_v=df_factor_v.shift(1)
df_factor_m=df_factor_m.shift(1)
df_exp=df_factor_v*beta_v_df+df_factor_m*beta_m_df
df_univ_daily.index=df_exp.index
#把所有的0值，即universe之外的值，改写为NaN，方便后续排序
df_exp=df_exp.replace(0,np.nan)
portfolio_decisions = pd.DataFrame(index=df_exp.index,columns=df_exp.columns)

for date in df_exp.index:
    daily_returns=df_exp.loc[date]
    # 排序股票
    sorted_returns = daily_returns.sort_values(ascending=False)
    # 选择顶部20%进行买入
    long_stocks = sorted_returns.head(int(0.2*((df_univ_daily.loc[date]==1).sum())))
    # 选择底部20%进行卖空
    short_stocks = sorted_returns.tail(int(0.2*((df_univ_daily.loc[date]==1).sum())))
    portfolio_decisions.loc[date] = [1 if stock in long_stocks else -1 if stock in short_stocks else 0 for stock in portfolio_decisions.columns]
date_begin = pd.to_datetime('20060101', format='%Y%m%d')
portfolio_decisions=portfolio_decisions[portfolio_decisions.index>=date_begin]

#结合价格计算收益
df_price=df_price[df_price.index>=date_begin]
df_in=pd.DataFrame(index=df_price.index,columns=df_price.columns)
df_out=pd.DataFrame(index=df_price.index,columns=df_price.columns)
df_return=pd.DataFrame(index=df_price.index,columns=df_price.columns)
df_port_return=pd.DataFrame(index=df_price.index)
df_port_return['return']=pd.NA

df_in=df_price*portfolio_decisions
df_out=df_price.shift(1)*portfolio_decisions
df_in = df_in.replace(0, pd.NA)
df_out = df_out.replace(0, pd.NA)
df_in.to_csv('df_in.csv')
df_out.to_csv('df_out.csv')
df_return=(df_out/df_in)-1
for date in df_return.index:
    df_port_return.loc[date]=(df_return.loc[date].sum())/int(0.2*((df_univ_daily.loc[date]==1).sum()))
df_port_return.to_csv('df_port_return.csv')
df_port_return['return'] = pd.to_numeric(df_port_return['return'], errors='coerce')
df_port_return = df_port_return.dropna()
df_port_return['cumulative_net_value'] = df_port_return['return'].add(1).groupby(df_port_return.index.year).cumprod()
annual_returns=df_port_return['cumulative_net_value'].sub(1).resample('A').last()

annual_vol = df_port_return['return'].groupby(df_port_return.index.year).std()
annual_returns.to_csv('annual_returns.csv')
annual_vol.to_csv('annual_vol')

# 将收益率转化为百分数形式
def to_percent(y, position):
    return "{:.1%}".format(y)

# plot
plt.figure(figsize=(10, 6))
plt.plot(annual_returns, label='Annual Returns')
formatter = mticker.FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('Date')
plt.ylabel('Return')
plt.title('Annual Returns Over Time')
plt.legend()
plt.grid(True)
plt.ylim(annual_returns.min() * 1.1, annual_returns.max() * 1.1)
plt.show()
plt.savefig('Annual_returns.png')