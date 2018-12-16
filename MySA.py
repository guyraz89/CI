# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def ImprovedSimulatedAnnealing(graph, T_init = 40.0, max_iter = 10**5, T_min = 1e-5, alpha = 0.999, max_internal_iter = 50):
    n = len(graph)
    T = T_init # Starting temprature value
    iter = 0 # Iteration counter
    current_perm = np.random.permutation(n) # first permutation initlize
    current_fitness = computeTourLength(current_perm, graph) # compute first permutation route length
    best_fitness = current_fitness # default best
    best_perm = current_perm # default best
    history = []

    while (T > T_min) and (iter < max_iter):
        for _ in range(max_internal_iter):
            # Set an optional permutation of the edges of graph
            optional_perm = list(current_perm)
            # Randomize two integers in range of the list of edges
            range1 = random.randint(2, n - 1)
            range2 = random.randint(0, n - range1)
            # Swap whole range by it's reverse
            optional_perm[range1 : range1 + range2] = reversed(optional_perm[range1 : range1 + range2])
            # Compute the route length for the new permutation (Target function)
            optional_fitness = computeTourLength(optional_perm, graph)
            # Accept any rout that better then the current rout permutation
            if optional_fitness < current_fitness :
                current_fitness, current_perm = optional_fitness, optional_perm
            # Accept with random probability
            if np.exp(-abs(optional_fitness - current_fitness) / T) > np.random.uniform(size=1) :
                current_fitness, current_perm = optional_fitness, optional_perm
            # Update the best result if needed
            if current_fitness < best_fitness :
                best_fitness, best_perm = current_fitness, current_perm

            iter += 1
            history.append(current_fitness)
        # Cooling temprature
        T *= alpha

    return best_perm, best_fitness, history


# Target function
def computeTourLength(perm, graph) :
    tlen = 0.0
    for i in range(len(perm)) :
        tlen += graph[perm[i], perm[np.mod(i+1, len(perm))]]
    return tlen


# Reference Method
def HillClimber(graph, evals = 10**5) :
    history = []
    n = len(graph)
    xmin = np.random.permutation(n)
    fmin = computeTourLength(xmin, graph)
    history.append(fmin)
    for _ in range(evals) :
        x = np.random.permutation(n)
        f_x = computeTourLength(x, graph)
        if f_x < fmin :
            xmin = x
            fmin = f_x
        history.append(fmin)
    return xmin,fmin,history


# Main
if __name__ == "__main__":
    dirname = ""
    fname = os.path.join(dirname, "hachula130.dat")
    data = []
    NTrials = 10**6
    with open(fname) as f:
        for line in f:
            data.append(line.split())

    n = len(data)
    G = np.empty([n, n])
    for i in range(n):
        for j in range(i, n):
            G[i, j] = np.linalg.norm(np.array([float(data[i][1]), float(data[i][2])]) - np.array([float(data[j][1]), float(data[j][2])]))
            G[j, i] = G[i, j]

    print(('#' * 40) + '\t' + 'Simulation started' + '\t' + ('#' * 40))
    start_time = time.time()
    #Simulated Annealing implementation
    perm, fitness, history = ImprovedSimulatedAnnealing(G)
    #HillClimber implementation as refernce method
    #perm, fitness, history = HillClimber(G)
    plt.semilogx(history)
    plt.ylabel("Fitness")
    plt.ylim(5000, 50000)
    plt.xlabel("Iterations")
    plt.show()
    print("The best route length is:" + str(fitness))
    print("The best permutation is :" + str(perm))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(('#' * 40) + '\t' + 'Simulation ended' + '\t' + ('#' * 40))