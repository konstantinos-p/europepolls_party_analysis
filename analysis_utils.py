from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import numpy as np
import pandas as pd


def fit_gp(df_partial, party):
    """
    Fits a Gaussian Process on the polling data of a single party.

    Parameters
    ----------
    df_partial: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    party: str
        The name of the party for which to fit the GP.

    Returns
    -------
            : pandas.dataframe
        A pandas dataframe with two columns party_mean | party_std containing the regressed mean and standard deviation
        respectively.
    """

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
    """
    Fits a Gaussian Process on the polling data of a single party. The difference with fit_gp is that regressed values
    are returned for every day withing the specified range, as opposed to only returning values when polls are
    available.

    Parameters
    ----------
    df_partial: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    party: str
        The name of the party for which to fit the GP.
    start_date: str
        The starting date to compute the regression. Should be of the form 'YYYY-MM-DD'.
    end_date: str
        The ending date to compute the regression. Should be of the form 'YYYY-MM-DD'.

    Returns
    -------
    d: pandas.dataframe
        A pandas dataframe with two columns party_mean | party_std containing the regressed mean and standard deviation
        respectively.
    """

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
    """
    Looks for parties that haven't been polled recently and removes them from the dataframe. Useful for countries where
    not much information is readily available about whether a party has recently ceased operations.

    Parameters
    ----------
    df: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.

    Returns
    -------
    df: pandas.dataframe
        A pandas dataframe with defunct parties removed. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    """

    all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])

    for party in all_parties:

        if df[party].iloc[:3].isnull().all():
            df = df.drop(party, axis='columns')

    return df


def compute_all_bias(df, agent='Polling Firm'):
    """
    Computes the bias of either Polling Firms or Commissioners.

    Parameters
    ----------
    df: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    agent: str
        The type of agent for which to compute the bias this is either 'Polling Firm' or 'Commissioner'.

    Returns
    -------
    bias_matrix: pandas.dataframe
        A pandas.dataframe object that containes all the biases. Rows are different agents, columns are the political
        parties.
    """

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
    """
    Debiases the polling estimates.

    Parameters
    ----------
    df: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    bias_matrix: pandas.dataframe
        A pandas.dataframe object that containes all the biases. Rows are different agents, columns are the political
        parties.
    agent: str
        The type of agent for which to compute the bias this is either 'Polling Firm' or 'Commissioner'.

    Returns
    -------
    df: pandas.dataframe
        A pandas dataframe containing the debiased polling data for the different political parties. The dataframe
        structure has to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    """

    for pollster in bias_matrix.index:
        for party in bias_matrix.columns:

            df.loc[df[agent] == pollster, party] = df.loc[df[agent] == pollster, party] - bias_matrix.loc[pollster, party]

    return df


def compute_regression(df, start_date=None, end_date=None):
    """
    Computes a regression for all days within the specified time interval and for all available parties.

    Parameters
    ----------
    df: pandas.dataframe
        A pandas dataframe containing the polling data for the different political parties. The dataframe structure has
        to be Date | Polling Firm | Commissioner | Sample Size | {Party Names} | Others.
    start_date: str
        The starting date to compute the regression. Should be of the form 'YYYY-MM-DD'.
    end_date: str
        The ending date to compute the regression. Should be of the form 'YYYY-MM-DD'.

    Returns
    -------
    regressed: pandas.dataframe
        A pandas dataframe with columns {party}_mean | {party}_std containing the regressed mean and standard deviation
        respectively for all days within the specified interval, and for all available parties.
    """

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





