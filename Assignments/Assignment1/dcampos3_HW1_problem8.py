import argparse
import math
import numpy as np
import pandas as pd
from numpy.linalg import norm

def load_data(filename):
    edges = []
    vertexes = set()
    largest = 0
    with open(filename) as f:
        for l in f:
            l = l.strip().split(' ')
            l[0] = int(l[0])
            l[1] = int(l[1])
            if l[0] > largest:
                largest = l[0]
            if l[1] > largest:
                largest = l[1]
            vertexes.add(l[0])
            vertexes.add(l[1])
            edges.append((l[0],l[1]))
    return edges, vertexes, largest+1

def create_adjancecy(edges, size):
    a = np.zeros((size, size))
    for i,j in edges:
        a[i][j] = 1
        a[j][i] = 1
    return a 

def normalize(a):
    d = np.zeros(a.shape)
    for i in range(a.shape[0]):
        d_cur = 0
        for j in range(a.shape[0]):
            d_cur += a[i][j]
        d[i][i] = d_cur
    for i in range(a.shape[0]):
        d[i][i] = 1/(math.sqrt(d[i][i]))
    return d*a*d

def run_rwr(args, a, largest):
    e = np.zeros(largest)
    e[args.seed] = 1
    r = e
    r1 = e
    scores = np.zeros(args.max_runs)
    for i in range(args.max_runs):
        r = args.c * (a.dot(r1)) + (1- args.c) * e
        scores[i] = norm(r - r1, 1)
        if scores[i] <= args.epsilon:
            break
        r1 = r
    return r

def print_results(ids, x):
    df = pd.DataFrame()
    df['Node'] = ids
    df['Score'] = x
    df = df.sort_values(by=['Score'], ascending=False)
    df = df.reset_index(drop=True)
    print(df[0:10])

def main(args):
    edges, vertexes, largest = load_data(args.input_file)
    a = create_adjancecy(edges, largest)
    if args.normalize:
        a = normalize(a)
    a = a.T
    x = run_rwr(args, a, largest)
    print_results(np.arange(0, largest), x)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RWR on email data')
    parser.add_argument('--input_file', default='HW1_dataset/email-Eu-core.txt', type=str, help='input graph file')
    parser.add_argument('--c', default=0.9, type=float, help='dampening factor')
    parser.add_argument('--seed', default=42, type=int, help='seed node to start rwr on')
    parser.add_argument('--max_runs', default=100, type=int, help='number of times to run the matix multiply')
    parser.add_argument('--epsilon', default=1e-7, type=float, help='Keep iterating until the normilized difference is less than this value')
    parser.add_argument('--normalize', action='store_true', help='Normalize the adjacency matrix')
    args = parser.parse_args()
    main(args)    