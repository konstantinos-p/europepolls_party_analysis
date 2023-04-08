from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def plot_df(df, all_parties):
    plt.figure()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Date_delta'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
    df['Date_delta'] = df['Date_delta']/df['Date_delta'].max()

    for party in all_parties:

        plt.plot(df['Date_delta'], df[party+'_mean'])
        plt.fill_between(df['Date_delta'], df[party+'_mean']-2*df[party+'_std'], df[party+'_mean']+2*df[party+'_std'],
                         alpha=0.2)

    plt.show()
    return
