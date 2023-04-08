from unittest import TestCase
from plot_utils import plot_df
import pandas as pd
from analysis_utils import compute_all_bias, remove_defunct_parties, compute_regression, debias


class Test(TestCase):
    def test_plot_df(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        df = remove_defunct_parties(df)
        all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])
        df_reg = compute_regression(df)
        plot_df(df_reg, all_parties)

    def test_plot_df_debiased(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        df = remove_defunct_parties(df)
        all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])
        bias_matrix = compute_all_bias(df.copy())
        debiased_df = debias(df.copy(), bias_matrix)
        df_reg = compute_regression(debiased_df)
        plot_df(df_reg, all_parties)

    def test_plot_df_debiased_commissioner(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        df = remove_defunct_parties(df)
        all_parties = set(df.columns) - set(['Date', 'Polling Firm', 'Commissioner', 'Sample Size', 'Others'])
        bias_matrix = compute_all_bias(df.copy(), agent='Commissioner')
        debiased_df = debias(df.copy(), bias_matrix, agent='Commissioner')
        df_reg = compute_regression(debiased_df)
        plot_df(df_reg, all_parties)

