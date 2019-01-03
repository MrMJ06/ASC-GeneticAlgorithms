import numpy as np
import random
import matplotlib.pyplot as plt
import os


def main(N, T, P, G, F, type):

    print("N : "+str(N))
    print("T : "+str(T))
    print("P : "+str(P))
    print("G : "+str(G))
    print("F : "+str(F))
    print("\n")

    """Inialization"""
    neighbors_num = int(np.round(T*N))
    w_vectors = unitary_vectors(N)
    poblation = np.random.uniform(P[0], P[1], N-1)
    poblation = np.insert(poblation, 0, np.random.random())

    fitness = evaluate(N, poblation, type=type)
    dist_matrix = calculate_distance_matrix(w_vectors, neighbors_num)
    weight_population = map_weight_poblation(w_vectors, poblation)
    # front = [[],[]]
    front = update_dominance(fitness)
    """Iteration"""

    for gen in range(G):
        z = calculate_z(fitness)
        y = reproduction(dist_matrix, weight_population, F, P, type=type)

        weight_children = map_weight_poblation(w_vectors, y)

        childen_fitness = evaluate(N, y, type=type)
        children_z = calculate_z(childen_fitness)

        z = update_z(z, children_z)
        poblation, fitness, weight_population = update_poblation(fitness, childen_fitness, dist_matrix, z, poblation, y, weight_population, weight_children)
        front = update_dominance(fitness)
        #print(front)
    print_front(front)

    return poblation


"""Dominance"""


def update_dominance(fitness):

    front = [[],[]]

    # if len(front[0]) == 0:
    for x1, fitness1 in fitness.items():
        x1_dominated = False
        for x2, fitness2 in fitness.items():
            if (float(fitness2[0]) < float(fitness1[0]) and float(fitness2[1])<=float(fitness1[1])) or \
                    (float(fitness2[0]) <= float(fitness1[0]) and float(fitness2[1])<float(fitness1[1])):
                x1_dominated = True
                break
        if not x1_dominated:
            front[0].append(fitness1[0])
            front[1].append(fitness1[1])
    # else:
    #     for x, fitness_x in fitness.items():
    #         x_dominated = False
    #         for i in range(len(front[0])):
    #             if (float(front[0][i]) > float(fitness_x[0]) and float(front[1][i]) >= float(fitness_x[1])) or (
    #                      float(front[0][i]) >= float(fitness_x[0]) and float(front[1][i]) > float(fitness_x[1])):
    #                 front[0][i]=fitness_x[0]
    #                 front[1][i]=fitness_x[1]

    return front


""" Selection """


def update_poblation(fitness, children_fitness, dist_matrix, z, poblation, y, weight_poblation, weight_children):

    for elem in dist_matrix:
        for x_dist in elem:
            parent = weight_poblation[x_dist[0]]
            children = weight_children[x_dist[0]]

            agg_fit_children = agg_func(children_fitness[children], x_dist[0], z)
            agg_fit_parent = agg_func(fitness[parent], x_dist[0], z)

            if agg_fit_children <= agg_fit_parent:
                poblation = [p if p != parent else children for p in poblation]
                weight_poblation[x_dist[0]] = children
                #del fitness[parent]
                fitness[children] = children_fitness[children]

    return [poblation, fitness, weight_poblation]


def agg_func(fitness, weights, z):

    ge1 = weights[0]*(fitness[0]-z[0])
    ge2 = weights[1]*(fitness[1]-z[1])

    agg_fitness = np.maximum(ge1, ge2)

    return agg_fitness


def update_z(z, children_z):

    if z[0] > children_z[0]:
        z[0] = children_z[0]
    if z[1] > children_z[1]:
        z[1] = children_z[1]

    return z


"""Reproduction"""


def map_weight_poblation(w_vectors, poblation):
    weight_population = {}

    for i, w in enumerate(w_vectors):

        weight_population[w] = poblation[i]

    return weight_population


def reproduction(dist_matrix, weight_poblation, F, P, type=0):
    childens = []

    for i, elem in enumerate(dist_matrix):
        selected_elem = [random.choice(elem) for i in elem]
        children = weight_poblation[selected_elem[0][0]]+F*(weight_poblation[selected_elem[1][0]]-weight_poblation[selected_elem[2][0]])
        if i == 0 and type != 0:
            children = check_space(children, (0, 1))
        else:
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


def evaluate(N, poblation, type=0):

    f1 = []
    f2 = []
    fitness = {}
    sum_x = sum_poblation(poblation)
    for index, x in enumerate(poblation):
        if type == 0:
            fitness[x] = zdt3_func(x, sum_x, N)
        else:
            fitness[x] = cf6_func(index, poblation, N)

        f1.append(fitness[x][0])
        f2.append(fitness[x][1])

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
    g = g_func(sum_x, n)

    f2 = g*h(f1, g)

    return [f1, f2]


def g_func(sum_x, n):

    return 1+(9/(n-1))*sum_x


def h(f1, g):

    return 1 - np.sqrt(f1/g)-(f1/g)*np.sin(10*np.pi*f1)


def sum_poblation(poblation):
    sum_x = 0
    for x in poblation[2:]:
        sum_x += x

    return sum_x


def print_front(front):

    plt.scatter(front[0], front[1], c='r')
    plt.show()


"""CF6 Function"""


def cf6_func(index, poblation, N):
    x = poblation[index]
    y1, y2 = calculate_j(x, poblation, N)

    f1 = x+sum(np.power(y1, 2))
    f2 = ((1-x)**2)+sum(np.power(y2, 2))
    penalization = check_restrictions(poblation, N)
    if index == 0 or index == 1 or index == 3:
        f1 += penalization
        f2 += penalization

    return f1, f2


def check_restrictions(poblation, N):
    x1 = poblation[0]
    x2 = poblation[1]
    x4 = poblation[3]
    penalization = 0
    r1 = x2-0.8*np.sin(6*np.pi*x1+2*np.pi/N)-np.sign(0.5*(1-x1)-(1-x1)**2)*np.sqrt(np.abs(0.5*(1-x1)-(1-x1)**2))
    r2 = x4-0.8*np.sin(6*np.pi*x1+4*np.pi/N)-np.sign(0.25*np.sqrt(1-x1)-0.5*(1-x1))*np.sqrt(np.abs(0.25*np.sqrt(1-x1)-0.5*(1-x1)))
    if r1 >= 0:
        penalization += r1+1
    if r2 >= 0:
        penalization += r2+1
    return penalization


def calculate_j(x1, poblation, N):
    y1 = []
    y2 = []

    for j, x in enumerate(poblation):

        if j % 2 != 0 and 2 <= j <= N:
            y = x-0.8*x1*np.cos(6*np.pi+(j*np.pi/N))
            y1.append(y)
        elif j % 2 == 0 and 2 <= j <= N:
            y = x - 0.8 * x1 * np.cos(6 * np.pi + (j * np.pi / N))
            y2.append(y)

    return y1, y2


if __name__=='__main__':
    main(200, 0.3, (-2, 2), 50, 0.5, 1)
    #print(zdt3_func(0, 10, 100))
