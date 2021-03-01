import os
import argparse
import sys
import numpy as np
import pyfpgrowth
import timeit


def run_apriori(transactions, min_sup, min_conf):


def main(args):
    transactions = get_data(args.input_file, args.rows_to_use)
    min_sup = int(transactions * args.min_sup)

    start_apriori = timeit.timeit()
    run_apriori(transactions, min_sup, min_conf)
    end_apriori = timeit.timeit()
    if args.run_perf
        start_fp = timeit.timeit()
        patterns = pyfpgrowth.find_frequent_patterns(transactions, min_sup) 
        rules = pyfpgrowth.generate_association_rules(patterns, args.min_conf)
        end_fp = timeit.timeit()
        print("Time for fpgrowth: {}".format(end_fp-start_fp))
        print("Time for apriori: {}".format(end_apriori - start_apriori))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apriori on grocery data')
    parser.add_argument('--input_file', default='HW1_dataset/groceries.csv', type=str, help='input file with grocery data.')
    parser.add_argument('--min_sup', default = 0.15, type=float, help='Minimum support value')
    parser.add_argument('--min_conf', default = 0.5, type=float, help='Minimum confidence values')
    parser.add_argument('--rows_to_use', default = 9835, type=int, help='How many rows to use. Used to compare algo speed'
    parser.add_argument('--run_perf', action='store_true', default=False, help='Run FP growth to compare speed')
    args = parser.parse_args()
    main(args)    