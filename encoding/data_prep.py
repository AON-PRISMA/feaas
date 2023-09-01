import os
import re

import pandas as pd
from recordlinkage.datasets import load_febrl4

from dataset import preprocess


def data_prep(dataset_name):
    if dataset_name == 'ag':
        L_DATA_ID = 'idAmazon'
        R_DATA_ID = 'idGoogleBase'

        amazon_data = pd.read_csv('Amazon-GoogleProducts/Amazon.csv', encoding='unicode_escape')
        google_data = pd.read_csv('Amazon-GoogleProducts/GoogleProducts.csv', encoding='unicode_escape')
        ground_truth = pd.read_csv('Amazon-GoogleProducts/Amzon_GoogleProducts_perfectMapping.csv',
                                   encoding='unicode_escape')

        # temporarily replace nan with empty str
        amazon_data.fillna('', inplace=True)
        google_data.fillna('', inplace=True)

        amazon_data = amazon_data.rename(columns={'id': L_DATA_ID})
        google_data = google_data.rename(columns={'id': R_DATA_ID})
        google_data = google_data.rename(columns={'name': 'title'})
        if os.path.exists('Amazon-GoogleProducts/match_id.csv'):
            new_match = pd.read_csv('Amazon-GoogleProducts/match_id.csv', encoding='unicode_escape')
        else:
            new_match = preprocess(amazon_data, google_data, ground_truth, 'idAmazon', 'idGoogleBase')
            new_match.to_csv('Amazon-GoogleProducts/match_id.csv', index=False)

        amazon_data = amazon_data.drop(L_DATA_ID, axis=1)
        google_data = google_data.drop(R_DATA_ID, axis=1)

        return amazon_data, google_data, new_match

    elif dataset_name == 'febrl4':
        data = load_febrl4(return_links=True)

        data_l = data[0].rename(mapper=lambda x: int(re.search('-(.*?)-', x).group(1)), axis='index')
        data_r = data[1].rename(mapper=lambda x: int(re.search('-(.*?)-', x).group(1)), axis='index')
        data_match = data[2].to_frame()
        data_match = data_match.applymap(lambda x: int(re.search('-(.*?)-', x).group(1)))
        data_match.reset_index(drop=True, inplace=True)

        return data_l, data_r, data_match

    elif dataset_name == 'febrl_gen':
        df = pd.read_csv('../dsgen/febrl_gen.csv', skipinitialspace=True)
        df = df.drop(columns=['age', 'phone_number', 'blocking_number'])  # remove additional columns

        data_l = df.loc[df['rec_id'].str.contains('-org')]
        data_r = df.loc[df['rec_id'].str.contains('-dup')]

        data_l.set_index('rec_id', inplace=True)
        data_r.set_index('rec_id', inplace=True)

        data_l = data_l.rename(mapper=lambda x: int(re.search('-(.*?)-', x).group(1)), axis='index')
        data_r = data_r.rename(mapper=lambda x: int(re.search('-(.*?)-', x).group(1)), axis='index')

        ind1 = sorted(list(data_l.index.values))
        ind2 = sorted(list(data_r.index.values))

        data_match = pd.DataFrame({0: ind1, 1: ind2})
        data_match.reset_index(drop=True, inplace=True)

        return data_l, data_r, data_match

    elif dataset_name == 'abt_buy':
        abt_data = pd.read_csv('Abt-Buy/Abt.csv', encoding='unicode_escape')
        buy_data = pd.read_csv('Abt-Buy/Buy.csv', encoding='unicode_escape')
        ground_truth = pd.read_csv('Abt-Buy/abt_buy_perfectMapping.csv', encoding='unicode_escape')

        # temporarily replace nan with empty str
        abt_data.fillna('', inplace=True)
        buy_data.fillna('', inplace=True)

        abt_data.set_index('id', inplace=True, verify_integrity=True)
        buy_data.set_index('id', inplace=True, verify_integrity=True)
        return abt_data, buy_data, ground_truth


if __name__ == "__main__":
    data_prep('abt_buy')
    print('end')
