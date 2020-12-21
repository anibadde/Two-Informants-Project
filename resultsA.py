#Code to create heatmaps

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas
# from scipy.special import softmaxscipy

import matplotlib.pyplot as plt

import datetime

def printGraph(X, Y, name):

	plt.figure('Draw')
	# plt.plot(X, Y)
	plt.scatter(X, Y, color = 'r', marker='.')
	plt.draw()
	plt.savefig(name + ".svg")
	plt.close()
	print("print figure finish: " + name)

def lp_calc(pa, ra, pd, rd, uc, uu, p, r, t):
    ilp_model = gp.Model('utility_optimization')

    #target, t1, t2, positive/negative
    x = ilp_model.addVars(len(pa), len(pa)+1, len(p), len(pa)+1, len(p), name='x')

#maximize defender utility
    ilp_model.setObjective(gp.quicksum(p[j] * (gp.quicksum(
        (p[i] * ((x[t, t, i, t, j] * rd[t] + (1 - x[t, t, i, t, j]) * pd[t]))) 
        for i in range(len(p)))) for j in range(len(p))), GRB.MAXIMIZE)

#attacker utility constraint
    ilp_model.addConstrs(gp.quicksum(p[j] * gp.quicksum(
        (p[i] * ((x[t, t, i, t, j] * pa[t] + (1 - x[t, t, i, t, j]) * ra[t]))) 
        for i in range(len(p))) for j in range(len(p)))
                         >= gp.quicksum(p[j] * gp.quicksum(
        (p[i] * ((x[t1, t1, i, t1, j] * pa[t1] + (1 - x[t1, t1, i, t1,j]) * ra[t1]))) 
        for i in range(len(p))) for j in range(len(p))) for t1 in range(len(pa)))
    
    #Assuming the two informants are both aligned in type, these constraints ensure truthful reporting of 
    #the target observed (as type reporting is the same when done truthfully)
    ilp_model.addConstrs(
        (gp.quicksum((p[j] * (x[t, t, i, t, j])*(uc[t,i]-uu[t,i])) for j in range(len(p))))
        >= 
        (gp.quicksum((p[j] * (x[t, t1, i1, t, j])*(uc[t,i]-uu[t,i])) for j in range(len(p))))
        for t1 in range(len(pa)+1) for i in range(len(p)) for i1 in range(len(p)))

    ilp_model.addConstrs(
        (gp.quicksum((p[j] * (x[t, t, j, t, i])*(uc[t,i]-uu[t,i])) for j in range(len(p))))
        >= 
        (gp.quicksum((p[j] * (x[t, t, j, t1, i1])*(uc[t,i]-uu[t,i])) for j in range(len(p))))
        for t1 in range(len(pa)+1) for i in range(len(p)) for i1 in range(len(p)))
    
    #ilp_model.addConstrs(x[t1, i, len(pa)+1] <= x[t1, i, j] for t1 in range(len(pa)) for i in range(len(pa)+1) for j in range(len(pa)+1))

    ilp_model.addConstrs(x[ty, t1, i1, t2, i2] >= 0 for ty in range(len(pa)) 
    for t1 in range(len(pa)+1) for i1 in range(len(p)) for t2 in range(len(pa)+1) for i2 in range(len(p)))

    #blank message: x[t, len(pa), 0, t', i'] - dealing with invalid blank
    ilp_model.addConstrs(x[ty, t1, i1, t2, i2] == 0 for ty in range(len(pa)) 
    for t1 in range(len(pa)+1) for i1 in range(len(p)) for t2 in range(len(pa),len(pa)+1) for i2 in range(1,len(p)))

    #blank message: x[t, t', i', len(pa), 0] - dealing with invalid blank
    ilp_model.addConstrs(x[ty, t1, i1, t2, i2] == 0 for ty in range(len(pa)) 
    for t1 in range(len(pa),len(pa)+1) for i1 in range(1,len(p)) for t2 in range(len(pa)+1) for i2 in range(len(p)))

    ilp_model.addConstrs(x[ty, t1, i1, t2, i2] <= 1 for ty in range(len(pa)) 
    for t1 in range(len(pa)+1) for i1 in range(len(p)) for t2 in range(len(pa)+1) for i2 in range(len(p)))

    ilp_model.addConstrs(gp.quicksum(x[t, t1, i1, t2, i2] for t in range(len(pa))) <= r
    for t1 in range(len(pa)+1) for i1 in range(len(p)) for t2 in range(len(pa)+1) for i2 in range(len(p)))

    ilp_model.optimize()

    try:
        print(ilp_model.objVal)
    except Exception as e:
        print(e)
        return -np.inf, -np.inf
    x1 = ilp_model.getAttr('X', x)
    attacker_val = sum(sum(
        (p[i] * p[j] * ((x1[t, t, i, t, j] * pa[t] + (1 - x1[t, t, i, t,j]) * ra[t]))) 
        for i in range(len(p))) for j in range(len(p)))

    print(attacker_val)

    '''
    for ty in range(len(pa)):
                for t1 in range(len(pa)+1):
                    for i1 in range(len(p)):
                        for t2 in range(len(pa)+1):
                            for i2 in range(len(p)):
                                if x1[ty, t1, i1, t2, i2] > 0.0:
                                    print("I")
                                    print(i)
                                    print(r, ty, t1, i1, t2, i2)
                                    print(p, x1[ty, t1, i1, t2, i2])
    '''

    return ilp_model.objVal, attacker_val

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    type_number = 3
    target_number = 7
    alpha = 0.3

    # data need for the linear program

    r = 2
    pw = 0.3
    maxv = -np.inf

    list_a = []
    list_result = []

    pa = -np.random.rand(target_number)
    ra = np.random.rand(target_number)
    uc = np.random.rand(target_number, type_number)

    pd = -ra + alpha * np.random.normal(size=target_number)
    rd = -pa + alpha * np.random.normal(size=target_number)

    uu = np.zeros_like(uc)
    uu[:, 0] = uc[:, 0] - 0.2
    uu[:, 1] = uc[:, 1] + 0.2
    uu[:, 2] = np.random.rand(target_number)

    heat_results = np.empty((20, 20))
    heat_results[:, :] = np.nan

    attv_res = np.copy(heat_results)

    for beta1 in np.arange(0, 1, 0.05):
        for beta2 in np.arange(0, 1 - beta1, 0.05):
            p = np.array([beta1, beta2, 1 - beta1 - beta2])
            mean_maxv = 0
            mean_attv = 0

            for _ in range(3):
                maxv = -np.inf
                attv = 0
                tmp_random = np.random.rand()

                for i in range(target_number):
                    def_v, att_v = lp_calc(pa, ra, pd, rd, uc, uu, p, r, i)
                    if def_v > maxv:
                        maxv = def_v
                        attv = att_v
                mean_maxv += maxv
                mean_attv += attv

            mean_maxv /= 3
            mean_attv /= 3

            print("AAA")
            print(mean_maxv)
            print(mean_attv)
            print("BBB")

            heat_results[int(beta1 / 0.05), int(beta2 / 0.05)] = mean_maxv
            attv_res[int(beta1 / 0.05), int(beta2 / 0.05)] = mean_attv
    
    # Save the model for painting convenience

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save(heat_results, 'defender_v_{}.pickle'.format(current_time))
    save(attv_res, 'attacker_v_{}.pickle'.format(current_time))

    # The following part is to draw a draft with the data, not exactly painting the figure in the paper

    '''
    with open('/Users/Anirudh/defender_v_20201218-213540.pickle', 'rb') as f:
        heat_results = pickle.load(f)
    '''

    minv = np.nanmin(heat_results)
    maxv = np.nanmax(heat_results)

    print('minv, maxv:')

    print(minv)
    print(maxv)

    heat_results = (heat_results - minv) / (maxv - minv)

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(heat_results)

    cbar = ax.figure.colorbar(im, ax=ax)

    plt.xticks(np.arange(0, 20, 10), np.arange(0, 1, 0.50))
    plt.yticks(np.arange(0, 20, 10), np.arange(0, 1, 0.50))

    plt.xlabel('p(attacker_align)')
    plt.ylabel('p(defender_align)')


    plt.savefig('heat_def_utility.pdf')
    plt.close()


    minv = np.nanmin(attv_res)
    maxv = np.nanmax(attv_res)

    print('minv, maxv:')

    print(minv)
    print(maxv)

    heat_results = (attv_res - minv) / (maxv - minv)

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(attv_res)

    cbar = ax.figure.colorbar(im, ax=ax)

    plt.xticks(np.arange(0, 20, 10), np.arange(0, 1, 0.50))
    plt.yticks(np.arange(0, 20, 10), np.arange(0, 1, 0.50))

    plt.xlabel('p(attacker_align)')
    plt.ylabel('p(defender_align)')


    plt.savefig('heat_atk_utility.pdf')
    plt.close() 
