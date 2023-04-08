from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import numpy as np
import pandas as pd


def fit_gp(df_partial, party):

    df_partial['Date'] = pd.to_datetime(df_partial['Date'])
    df_partial['Date_delta'] = (df_partial['Date'] - df_partial['Date'].min()) / np.timedelta64(1, 'D')
    df_partial['Date_delta'] = df_partial['Date_delta']/df_partial['Date_delta'].max()
    df_partial = df_partial.dropna()

    kernel = Matern(nu=0.5) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(df_partial[['Date_delta']], df_partial[party])
    pred_mean, pred_std = gpr.predict(df_partial[['Date_delta']], return_std=True)

    pred_mean = pd.Series(pred_mean, index=df_partial[['Date_delta']].index)
    pred_std = pd.Series(pred_std, index=df_partial[['Date_delta']].index)

    df_partial[party + '_mean'] = pred_mean
    df_partial[party + '_std'] = pred_std

    return df_partial[[party + '_mean', party + '_std']]


def get_regression(df_partial, party, start_date=None, end_date=None):

    df_partial['Date'] = pd.to_datetime(df_partial['Date'])
    df_partial['Date_delta'] = (df_partial['Date'] - df_partial['Date'].min()) / np.timedelta64(1, 'D')
    norm = df_partial['Date_delta'].max()
    df_partial['Date_delta'] = df_partial['Date_delta']/norm

    if start_date is None and end_date is None:
        detailed_dates = pd.date_range(df_partial['Date'].min(), df_partial['Date'].max(), freq='d')
    else:
        detailed_dates = pd.date_range(start_date, end_date, freq='d')
    normalized_detailed_dates = ((detailed_dates - df_partial['Date'].min()) / np.timedelta64(1, 'D'))/norm

    df_partial = df_partial.dropna()

    kernel = Matern() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(df_partial[['Date_delta']], df_partial[party])

    pred_mean, pred_std = gpr.predict(normalized_detailed_dates.to_numpy().reshape(-1, 1), return_std=True)

    d = {
        'Date': detailed_dates,
        party + '_mean': pred_mean,
        party + '_std': pred_std
    }

    return pd.DataFrame(d)


def remove_defunct_parties(df):

    all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])

    for party in all_parties:

        if df[party].iloc[:3].isnull().all():
            df = df.drop(party, axis='columns')

    return df


def compute_all_bias(df, agent='Polling Firm'):

    all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])

    for party in all_parties:

        df[party] = df[party]/100

        df = df.join(fit_gp(df[[party, 'Date']], party=party))

        df[party] = df[party]-df[party + '_mean']

        df[party] = df[party] * 100

    bias_matrix = pd.DataFrame(index=df[agent].unique(), columns=list(all_parties))

    for pollster in bias_matrix.index:
        for party in bias_matrix.columns:

            bias_matrix[party][pollster] = df.loc[df[agent] == pollster][party].dropna().mean()

    bias_matrix = bias_matrix.astype('float').round(2)

    return bias_matrix


def debias(df, bias_matrix, agent='Polling Firm'):

    for pollster in bias_matrix.index:
        for party in bias_matrix.columns:

            df.loc[df[agent] == pollster, party] = df.loc[df[agent] == pollster, party] - bias_matrix.loc[pollster, party]

    return df


def compute_regression(df, start_date=None, end_date=None):

    all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])

    regressions = []

    for party in all_parties:

        df[party] = df[party]/100

        regressions.append(get_regression(df[[party, 'Date']], party=party, start_date=start_date, end_date=end_date))

    regressions = [df.set_index('Date') for df in regressions]
    regressed = pd.concat(regressions, axis=1)*100
    regressed.reset_index(inplace=True)
    regressed = regressed.rename(columns={'index': 'Date'})
    return regressed





