import numpy as np
import matplotlib.pyplot as plt
from hv import HyperVolume


def main(N, T, P, G, F, type, dim):

    print("N : "+str(N))
    print("T : "+str(T))
    print("P : "+str(P))
    print("G : "+str(G))
    print("F : "+str(F))
    print("\n")

    """Inialization"""
    neighbors_num = int(np.round(T*N))
    w_vectors = unitary_vectors(N)
    poblation = np.random.uniform(P[0], P[1], (N, dim-1))
    if type !=0:
        new_poblation = []
        for i, pob in enumerate(poblation):
            new_poblation.append(np.insert(pob, 0, np.random.uniform(0, 1, 1)))
        poblation = new_poblation
    if type == 0:
        ideal = get_points()
    else:
        ideal = get_points(path_f="PF_CF6_4.dat")
    print(poblation)
    fitness = evaluate(N, poblation, ideal, type=type)
    dist_matrix = calculate_distance_matrix(w_vectors, neighbors_num)
    weight_population = map_weight_poblation(w_vectors, poblation)
    front = [[],[]]
    front = update_dominance(fitness)
    """Iteration"""
    z = calculate_z(fitness)
    for gen in range(G):

        y = reproduction(dist_matrix, weight_population, F, P, type=type)
        weight_children = map_weight_poblation(w_vectors, y)

        childen_fitness = evaluate(N, y, ideal, type=type)
        children_z = calculate_z(childen_fitness)

        z = update_z(z, children_z)
        poblation, fitness, weight_population = update_poblation(fitness, childen_fitness, dist_matrix, z, poblation, y, weight_population, weight_children, type=type)
        front = update_dominance(fitness)
    print_front(front, ideal, True, type=type)

    if type == 0:
        folder = "ZDT3"
    else:
        folder = "CF6_"+str(dim)

    with open(folder+"/results_"+str(N)+"_"+str(T)+"_"+str(P)+"_"+str(G)+"_"+str(G), "w+") as r:
        for i in range(len(front[0])):
            r.write(str(front[0][i])+" "+str(front[1][i])+" 0"+"\n")

    r.close()

    return poblation


def get_points(path_f="PF.dat"):
    points = [[],[]]
    with open(path_f,"r") as pf:
        while True:
            x = pf.readline()
            x = x.rstrip()
            if not x: break
            points[0].append(float(x.split()[0]))
            points[1].append(float(x.split()[1]))
    return points


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

    return front


""" Selection """


def update_poblation(fitness, children_fitness, dist_matrix, z, poblation, y, weight_poblation, weight_children, type=0):

    for i, elem in enumerate(dist_matrix):
        # for index, y_dist in enumerate(elem):
        children = y[i]
        if type != 0:
            children_restrictions = check_restrictions(children)

        for x_dist in elem:
            agg_fit_children = agg_func(children_fitness[i], x_dist[0], z)
            parent_pos = list(weight_poblation.keys()).index(x_dist[0])
            # print(str(parent_pos))
            agg_fit_parent = agg_func(fitness[parent_pos], x_dist[0], z)

            if type != 0:
                parent_restrictions = check_restrictions(poblation[parent_pos])

            if agg_fit_children <= agg_fit_parent and type == 0:

                poblation[parent_pos] = children
                weight_poblation[x_dist[0]] = children
                fitness[parent_pos] = children_fitness[i]

            elif type!= 0 and children_restrictions != 0 and parent_restrictions!= 0 and children_restrictions<=parent_restrictions :

                poblation[parent_pos] = children
                weight_poblation[x_dist[0]] = children
                fitness[parent_pos] = children_fitness[i]

            elif type != 0 and children_restrictions == 0 and parent_restrictions != 0 :

                poblation[parent_pos] = children
                weight_poblation[x_dist[0]] = children
                fitness[parent_pos] = children_fitness[i]

            elif type != 0 and children_restrictions == 0 and parent_restrictions == 0 and agg_fit_children <= agg_fit_parent:

                poblation[parent_pos] = children
                weight_poblation[x_dist[0]] = children
                fitness[parent_pos] = children_fitness[i]

    return [poblation, fitness, weight_poblation]


def agg_func(fitness, weights, z):

    ge1 = weights[0]*(np.abs(fitness[0]-z[0]))
    ge2 = weights[1]*(np.abs(fitness[1]-z[1]))

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
        original = weight_poblation[elem[0][0]]
        np.random.shuffle(elem)
        selected_elem = elem[0:3]

        children = weight_poblation[selected_elem[0][0]]+F*(weight_poblation[selected_elem[1][0]]-weight_poblation[selected_elem[2][0]])
        for j, e in enumerate(original):
            ran = np.random.randint(0, 2)
            # print(ran)
            if ran >= 1:
                # print("changed")
                children[j] = original[j]
        # sigma = (np.max(weight_poblation[elem[0][0]])-np.min(weight_poblation[elem[0][0]]))/20
        # children = weight_poblation[elem[0][0]]+np.random.normal(0, sigma, 30)
        children = check_space(children, P)

        childens.append(children)

    return childens


def check_space(children, P):

    fixed_children = children
    for i, x in enumerate(children):
        if i == 0:
            if children[0] < 0:
                fixed_children[0] = 0
            elif children[0] > 1:
                fixed_children[0] = 1
        else:
            if x < P[0]:
                fixed_children[i] = P[0]

            elif x > P[1]:
                fixed_children[i] = P[1]
    return fixed_children


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


def evaluate(N, poblation, ideal, type=0):

    f1 = []
    f2 = []
    fitness = {}
    for index, x in enumerate(poblation):
        if type == 0:
            fitness[index] = zdt3_func(x)
        else:
            fitness[index] = cf6_func(x)

        f1.append(fitness[index][0])
        f2.append(fitness[index][1])

    plt.scatter(f1, f2)
    plt.scatter(ideal[0], ideal[1], c="r")
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


def zdt3_func(x):

    f1 = x[0]
    g = g_func(np.sum(x)-x[0])

    f2 = g*h(f1, g)

    return [f1, f2]


def g_func(sum_x):

    return 1+((9/29)*sum_x)


def h(f1, g):

    return 1 - np.sqrt(f1/g)-(f1/g)*np.sin(10*np.pi*f1)


def sum_poblation(poblation):
    sum_x = 0
    for x in poblation[2:]:
        sum_x += x

    return sum_x


def print_front(front, ideal, final=False, type=0):

    plt.scatter(ideal[0], ideal[1], c='r', alpha=0.1, linewidths=0.01)
    plt.scatter(front[0], front[1], c='b', linewidths=0.01)
    front = transform_points(front)
    if final:
        if type != 0:
            nsga_points = get_points("best_pop_cf6_4.out")
        else:
            nsga_points = get_points("best_pop.out")

        plt.scatter(nsga_points[0], nsga_points[1], c="y", alpha=0.8, linewidth=0.01)
        referencePoint = [1, 1]
        hyperVolume = HyperVolume(referencePoint)
        nsga_points = transform_points(nsga_points)
        result = hyperVolume.compute(front)
        result_nsga = hyperVolume.compute(nsga_points)
        coverage_f2_f1 = calculate_coverage(front, nsga_points)
        coverage_f1_f2 = calculate_coverage(nsga_points, front)
        print("Coverage my front over other front = "+str(coverage_f2_f1))
        print("Coverage other front over my front = "+str(coverage_f1_f2))
        print("Hypervolume my solution = "+str(result))
        print("Hypervolume other solution = "+str(result_nsga))
    plt.show()


def calculate_coverage(front1, front2):

    points_front_2_dominated = 0
    for point2 in front2:
        for point1 in front1:
            if (-float(point1[0]) < -float(point2[0]) and -float(point1[1]) <= -float(point2[1])) or \
                    (-float(point1[0]) <= -float(point2[0]) and -float(point1[1]) < -float(point2[1])):
                points_front_2_dominated += 1
                break
        else:
            continue
        break

    return points_front_2_dominated/len(front2)


def transform_points(front):
    points = []
    for i, x in enumerate(front[0]):
     points.append([-x, -front[1][i]])

    return points


"""CF6 Function"""


def cf6_func(indv):
    y1, y2 = calculate_j(indv)

    f1 = indv[0]+sum(np.power(y1, 2))
    f2 = ((1-indv[0])**2)+sum(np.power(y2, 2))

    return f1, f2


def check_restrictions(indv):
    x1 = indv[0]
    x2 = indv[1]
    x4 = indv[3]
    penalization = 0
    r1 = x2-0.8*x1*np.sin(6*np.pi*x1+2*np.pi/len(indv))-np.sign(0.5*(1-x1)-(1-x1)**2)*np.sqrt(np.abs(0.5*(1-x1)-(1-x1)**2))
    r2 = x4-0.8*x1*np.sin(6*np.pi*x1+4*np.pi/len(indv))-np.sign(0.25*np.sqrt(1-x1)-0.5*(1-x1))*np.sqrt(np.abs(0.25*np.sqrt(1-x1)-0.5*(1-x1)))
    if r1 < 0:
        penalization += -r1
    if r2 < 0:
        penalization += -r2
    return penalization


def calculate_j(indv):
    y1 = []
    y2 = []

    for j, x in enumerate(indv):

        if (j+1) % 2 != 0 and 2 <= (j+1) <= len(indv):
            y = x-0.8*indv[0]*np.cos(6 * np.pi * indv[0]+((j+1) * np.pi /len(indv)))
            y1.append(y)
        elif (j+1) % 2 == 0 and 1 <= (j+1) <= len(indv):
            y = x - 0.8 * indv[0] * np.cos(6 * np.pi * indv[0] + ((j+1) * np.pi / len(indv)))
            y2.append(y)

    return y1, y2


if __name__ == '__main__':
    main(100, 0.15, (0, 1), 100, 0.5, 0, 4)

