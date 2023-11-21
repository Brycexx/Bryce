# Be more ambitious, be more diligent
import pandas as pd
import numpy as np
import statsmodels.api as sm

df=pd.read_csv("Coefficients.csv")
df['date'] = pd.to_datetime(df['Date'])
df.set_index('date', inplace=True)
df=df.drop('Date',axis=1)
print(df.head())

# create the output sheet
results=pd.DataFrame(columns=['Year','Avg_Beta_v', 'Avg_Beta_m', 'T_Stat_Beta_v', 'T_Stat_Beta_m'])
for year in range(2005,2024):
    Last_day=df[df.index.year==year].index.max()
    First_day=df[df.index.year==year-1].index.min()
    window=df.loc[First_day:Last_day]
    print(window)
    avg_beta_v = window['factor_v'].mean()
    avg_beta_m = window['factor_M'].mean()
    T = window.shape[0]
    std_beta_v = window['factor_v'].std()
    std_beta_m = window['factor_M'].std()
    t_stat_beta_v = (avg_beta_v / std_beta_v) * (T ** 0.5)
    t_stat_beta_m = (avg_beta_m / std_beta_m) * (T ** 0.5)
    results = results._append({
        'Year': year,
        'Avg_Beta_v': avg_beta_v,
        'Avg_Beta_m': avg_beta_m,
        'T_Stat_Beta_v': t_stat_beta_v,
        'T_Stat_Beta_m': t_stat_beta_m
    }, ignore_index=True)
results.to_csv('results.csv',index=False)


