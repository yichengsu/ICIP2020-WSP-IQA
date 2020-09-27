import argparse
import os
import random

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To generate a new order file. '
                                     'This procedure is not necessary because we '
                                     'have provided a default one.')
    parser.add_argument('-m', '--mos_file', default='koniq10k_scores_and_distributions.csv',
                        type=str, help='path to mos file in datasets '
                        '(default: koniq10k_scores_and_distributions.csv)')
    parser.add_argument('-n', default=10, type=int,
                        help='the number of new order (default: 10)')
    args = parser.parse_args()

    assert os.path.exists(args.mos_file), 'MOS file not exists'

    len_mos_value = len(pd.read_csv(args.mos_file))
    orders = [random.sample(range(len_mos_value), len_mos_value) for _ in range(args.n)]

    pd.DataFrame(orders).to_csv('orders.csv', index=False)
