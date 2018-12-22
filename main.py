import numpy as np
import random
import matplotlib.pyplot as plt


def main(N, T, P, G, F):

    print("N : "+str(N))
    print("T : "+str(T))
    print("P : "+str(P))
    print("G : "+str(G))
    print("F : "+str(F))
    print("\n")

    """Inialization"""
    neighbors_num = int(np.round(T*N))
    w_vectors = unitary_vectors(N)
    poblation = np.random.uniform(P[0], P[1], N)

    dist_matrix = calculate_distance_matrix(w_vectors, neighbors_num)

    """Iteration"""

    for gen in range(G):

        fitness = evaluate(N, poblation)
        weight_population = map_weight_poblation(w_vectors, poblation)
        z = calculate_z(fitness)
        y = reproduction(dist_matrix, weight_population, F, P)

        childen_fitness = evaluate(N, y)
        children_z = calculate_z(childen_fitness)

        z = update_z(z, children_z)
        poblation = update_poblation(fitness, childen_fitness, dist_matrix, z, poblation, y)


""" Selection """


def update_poblation(fitness, children_fitness, dist_matrix, z, poblation, y):

    for elem in dist_matrix:
        for index, x_dist in enumerate(elem):
            parent = poblation[index]
            children = y[index]

            agg_fit_children = agg_func(children_fitness[children], x_dist[0], z)
            agg_fit_parent = agg_func(fitness[parent], x_dist[0], z)

            if agg_fit_children <= agg_fit_parent:
                poblation[index] = children

    return poblation


def agg_func(fitness, weights, z):

    ge1 = weights[0]*(fitness[0]-z[0])
    ge2 = weights[1]*(fitness[1]-z[1])

    agg_fitness = np.maximum(ge1, ge2)

    return agg_fitness


def update_z(z, children_z):

    if z[0] > children_z[0]:
        z[0]=children_z[0]
    if z[1] > children_z[1]:
        z[1]=children_z[1]

    return z


"""Reproduction"""


def map_weight_poblation(w_vectors, poblation):
    weight_population = {}

    for i, w in enumerate(w_vectors):

        weight_population[w] = poblation[i]

    return weight_population


def reproduction(dist_matrix, weight_poblation, F, P):
    childens = []

    for elem in dist_matrix:
        selected_elem = random.sample(elem, 3)
        children = weight_poblation[selected_elem[0]]+F*(weight_poblation[selected_elem[1]]-weight_poblation[selected_elem[2]])
        children = check_space(children, P)
        childens.append(children)

    return childens


def check_space(children, P):

    if children < P[0]:
        children = P[0]

    elif children > P[1]:
        children = P[1]

    return children


"""Evaluation"""


def calculate_z(fitness):

    min_f1 = float('inf')
    min_f2 = float('inf')
    for x, fitn in fitness.items():
        if fitn[0] < min_f1:
            min_f1 = fitn[0]
        if fitn[1] < min_f2:
            min_f2 = fitn[1]

    return [min_f1, min_f2]


def evaluate(N, poblation):

    f1 = []
    f2 = []
    fitness = {}
    sum_x = sum_poblation(poblation)
    for x in poblation:
        fitness[x] = zdt3_func(x, sum_x, N)
        f1.append(fitness[x][0])
        f2.append(fitness[x][2])

    plt.scatter(f1, f2)
    plt.show()

    return fitness


"""Vector Space"""


def unitary_vectors(N):
    vectors = []
    for n in np.linspace(0, 1, N):
        vectors.append((n, 1-n))

    return vectors


"""Distance"""


def calculate_distance_matrix(vector, neighbor_num):

    matrix = []

    for index, p1 in enumerate(vector):
        matrix.append({})
        for p2 in vector:
            matrix[index][p2] = euclidean_distance(p1,p2)

        matrix[index] = sorted(matrix[index].items(), key=lambda kv: kv[1])
        matrix[index] = matrix[index][:neighbor_num]

    return matrix


def euclidean_distance(p1, p2):

    p = [p1[0]-p2[0], p1[1]-p2[1]]

    return np.sqrt(p[0]**2+p[1]**2)


"""ZDT3 Function"""


def zdt3_func(x, sum_x, n):

    f1 = x
    g = g_func(x, sum_x, n)

    f2 = g*h(f1, g)


    return [f1, f2]


def g_func(x, sum_x, n):

    return 1+(9/(n-1))*sum_x


def h(f1, g):

    return 1 - np.sqrt(f1/g)-(f1/g)*np.sin(10*np.pi*f1)


def sum_poblation(poblation):
    sum_x = 0
    for x in poblation[2:]:
        sum_x += x

    return sum_x


if __name__=='__main__':
    main(20, 0.2, (0, 1), 100, 0.5)