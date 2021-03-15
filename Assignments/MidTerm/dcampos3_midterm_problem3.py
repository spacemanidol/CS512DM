import os
import argparse
import sys
import numpy as np
import pyfpgrowth
import timeit
from itertools import chain, combinations
from collections import defaultdict

def subsets(a):
    return chain(*[combinations(a, i + 1) for i, _ in enumerate(a)])

def join_set(itemset, length):
    return set([i.union(j) for i in itemset for j in itemset if len(i.union(j)) == length])

def get_data(filename, rows_to_use):
    items = {}
    itemset = set()
    transaction_size = []
    transactions = []
    rows_read = 0
    with open(filename, 'r') as f:
        for l in f:
            if rows_read >= rows_to_use:
                break
            rows_read += 1
            l = l.strip()
            transaction_size.append(len(l))
            record = frozenset(l.split(","))
            transactions.append(frozenset(record))
            for item in record:
                itemset.add(frozenset([item]))
                if item not in items:
                    items[item] = 0
                items[item] += 1
    print("There are {} items and {} transactions".format(len(items), len(transaction_size)))
    print("Each transaction averages {} products".format(np.mean(transaction_size)))
    print("Top 5 most common products:{}".format(sorted(items.items(), key=lambda item: item[1], reverse=True)[:5]))
    return itemset, transactions

def get_items_with_min_support(itemset, transactions, min_support, frequentset):
    tmp_set = {}
    for item in itemset:
        for transaction in transactions:
            if item.issubset(transaction):
                if item not in frequentset:
                    frequentset[item] = 0
                if item not in tmp_set:
                    tmp_set[item] = 0
                frequentset[item] += 1
                tmp_set[item] += 1
    supported_set = set()
    for item, count in tmp_set.items():
        if count >= min_support:
            supported_set.add(item)
    return supported_set, frequentset

def run_apriori(items, transactions, min_support, min_conf):
    frequentset = defaultdict(int)
    fullset = {}
    rules = {}
    cset, frequentset = get_items_with_min_support(items, transactions, min_support, frequentset)
    lset = cset
    k=2
    num_transactions = len(transactions)
    while lset != set([]):
        fullset[k-1] = lset
        lset = join_set(lset, k)
        cset, frequentset = get_items_with_min_support(lset, transactions, min_support, frequentset)
        lset = cset
        k += 1
    for key in fullset:
        for item in fullset[key]:
            print("Item: {} ---> Support: {}".format(item, frequentset[item]/num_transactions))
    for k, v in list(fullset.items())[1:]:
        for item in v:
            for e in map(frozenset, [x for x in subsets(item)]):
                wo = item.difference(e)
                if len(wo) > 0:
                    support_element = frequentset[e]/num_transactions
                    support_item = frequentset[item]/num_transactions
                    if support_item/support_element >= min_conf:
                        print("Itemset {} -> Itemset {} with confidence:{}".format(e, wo, support_item/support_element))

def main(args):
    items, transactions = get_data(args.input_file, args.rows_to_use)
    min_sup = int(len(transactions) * args.min_sup)
    start_apriori = timeit.timeit()
    run_apriori(items, transactions, min_sup, args.min_conf)
    end_apriori = timeit.timeit()
    if args.run_perf:
        start_fp = timeit.timeit()
        patterns = pyfpgrowth.find_frequent_patterns(transactions, min_sup) 
        rules = pyfpgrowth.generate_association_rules(patterns, args.min_conf)
        print(patterns)
        print(rules)
        end_fp = timeit.timeit()
        print("Time for fpgrowth: {}".format(end_fp-start_fp))
        print("Time for apriori: {}".format(end_apriori - start_apriori))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apriori on grocery data')
    parser.add_argument('--input_file', default='HW1_dataset/groceries.csv', type=str, help='input file with grocery data.')
    parser.add_argument('--min_sup', default = 0.05, type=float, help='Minimum support value')
    parser.add_argument('--min_conf', default = 0.25, type=float, help='Minimum confidence values')
    parser.add_argument('--rows_to_use', default = 9835, type=int, help='How many rows to use. Used to compare algo speed')
    parser.add_argument('--run_perf', action='store_true', default=False, help='Run FP growth to compare speed')
    args = parser.parse_args()
    main(args)    