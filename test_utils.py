from unittest import TestCase
import pandas as pd
from analysis_utils import compute_all_bias, remove_defunct_parties, compute_regression, debias


class Test(TestCase):
    def test_compute_all_bias(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        compute_all_bias(df)
        self.assertTrue(True)

    def test_remove_defunct_parties(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        remove_defunct_parties(df)
        self.assertTrue(True)

    def test_compute_regression(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        compute_regression(df)
        self.assertTrue(True)

    def test_debias(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        df = remove_defunct_parties(df)
        bias_matrix = compute_all_bias(df.copy())
        debiased_df = debias(df.copy(), bias_matrix)
        self.assertTrue(True)

    def test_debias_commissioner(self):
        df = pd.read_csv('greece_cleaned.csv', index_col=0)
        df = remove_defunct_parties(df)
        bias_matrix = compute_all_bias(df.copy(), agent='Commissioner')
        debiased_df = debias(df.copy(), bias_matrix)
        self.assertTrue(True)





