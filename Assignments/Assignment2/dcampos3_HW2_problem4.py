import os
import argparse
import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import random

def load_social(filename):
    g = nx.Graph()
    with open(filename,'r') as f:
        for l in f:
            l = l.strip().split(',')
            if l[0] not in g.nodes:
                g.add_node(l[0])
            if l[1] not in g.nodes:
                g.add_node(l[1])
            g.add_edge(l[0],l[1])
            g.add_edge(l[1],l[0])
    return g 

def main(args):
    g = load_social(args.filename)
    susceptible, infected = [], []
    for i in range(args.initial_infected):
        infected.append(str(i))
    for node in g.nodes:
        if node not in infected:
            susceptible.append(node)
    if args.vaccinate:
        vaccinated = []
        if args.vaccination_strategy == 'random':
            random.shuffle(susceptible)
            vaccinated = susceptible[:args.vaccination_size]
        elif args.vaccination_strategy == 'degree-high':
            node2degree = {}
            for node in susceptible:
                node2degree[node] = len(g[node])
            vaccinated = list(dict(sorted(node2degree.items(), key=lambda item: item[1], reverse=True)).keys())[:args.vaccination_size]
        elif args.vaccination_strategy == 'infection_proximity':
            node2infection = {}
            for node in susceptible:
                n = 0 
                for edge in g.adj[node]:
                    if edge in infected:
                        n += 1
                node2infection[node] = n
            vaccinated = list(dict(sorted(node2infection.items(), key=lambda item: item[1], reverse=True)).keys())[:args.vaccination_size]
        elif args.vaccination_strategy  == 'two-hop':
            node2degree = {}
            for node in susceptible:
                n = 0
                for edge in g.adj[node]:
                    n += len(g.adj[edge])
                node2degree[node] = n
            vaccinated = list(dict(sorted(node2degree.items(), key=lambda item: item[1], reverse=True)).keys())[:args.vaccination_size]
        elif args.vaccination_strategy == 'even-random':
            random.shuffle(susceptible)
            node = susceptible[0]
            while len(vaccinated) < 200:
                old_node = node
                vaccinated.append(node)
                new_nodes = list(g.adj[node])
                random.shuffle(new_nodes)
                for edge in new_nodes:
                    if edge not in infected:
                        node = edge
                        break
                    else:
                        pass
                while old_node == node:
                    random.shuffle(susceptible)
                    node = susceptible[0]
                    if node in infected or node in vaccinated:
                        node = old_node
    x, num_s, num_i = [0], [len(susceptible)], [len(infected)]
    for step in range(args.iterations):
        print("step {}".format(step))
        new_infected, new_susceptible = [], []
        for node in susceptible:
            n = 0 
            for edge in g.adj[node]:
                if edge in infected:
                    n += 1
            probs_infected = 1-(1-args.infection_rate)**n
            if random.random() <= probs_infected and node not in vaccinated:
                new_infected.append(node)
            else:
                new_susceptible.append(node)
        for node in infected:
            if random.random() <= args.recovery_rate:
                new_susceptible.append(node)
            else:
                new_infected.append(node)
        susceptible = new_susceptible
        infected = new_infected
        num_s.append(len(susceptible))
        num_i.append(len(infected))
        x.append(step+1)
    print("After convergence there are an average of {} infected nodes".format(np.mean(num_i[-100:])))
    plt.plot(x, num_s, label='Number Susceptible')
    plt.plot(x, num_i, label='Number Infected')
    plt.xlabel("Iterations")
    plt.ylabel("Number of nodes each class")
    plt.legend()
    plt.savefig(args.output_img)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIS Virus')
    parser.add_argument('--infection_rate', default=0.08, type=float, help='Beta value for infection')
    parser.add_argument('--vaccinate', action='store_true', help="Vaccinate some portion of nodes")
    parser.add_argument('--vaccination_size', default=200, type=int, help="Amount of nodes to vaccinate")
    parser.add_argument('--vaccination_strategy', default='random', choices=['random', 'degree-high','infection_proximity','two-hop','even-random'])
    parser.add_argument('--recovery_rate', default=0.02, type=float, help='Recovery odds')
    parser.add_argument('--initial_infected', default=300, type=int, help='Amount of initial nodes infected')
    parser.add_argument('--iterations', default=300, type=int, help='Amount of transmision phases to run')
    parser.add_argument('--output_img', default='sis.png', type=str, help='file name to save sis curves')
    parser.add_argument('--filename', default='CS512_HW2_dataset/social_network_edge_list.txt', type=str, help='MAT file to run Outlier Detection on')
    args = parser.parse_args()
    main(args)    