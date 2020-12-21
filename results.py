import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas
# from scipy.special import softmaxscipy

import matplotlib.pyplot as plt
#import seaborn as sns

import datetime

def printGraph(X, Y, name):
    plt.figure('Draw')
    plt.plot(X, Y)
    # plt.scatter(X, Y, color = 'r', marker='.')
    plt.xlabel('p_defender_align')
    plt.ylabel('defender resource')
    plt.draw()
    plt.savefig(name + ".pdf")
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
        (p[i] * p[j] * ((x[t, t, i, t, j] * pa[t] + (1 - x[t, t, i, t,j]) * ra[t]))) 
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
    return ilp_model.objVal, x1, attacker_val

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

import pickle

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    type_number = 3
    #target_numbers = [5*(i+1) for i in range(10)]
    #target_numbers = [5, 10, 20]
    target_numbers = [10]
    target_number = 10
    alpha = 0.3
    eps = 1e-8
    num_informants = 2
    cur_res = []

    res = dict()
    for target_number in target_numbers:
        res[target_number] = []

    min_p0 = 0

    for num_instances in range(10):
        r = 1
        maxv = -np.inf

        list_a = []
        plans = []
        total_u = 0

        for target_number in target_numbers:
            r = 1
            list_result = []
            pa = -np.random.rand(target_number)
            ra = np.random.rand(target_number)
            uc = np.random.rand(target_number, type_number)

            uu = np.zeros_like(uc)
            uu[:, 0] = uc[:, 0] - 0.1
            uu[:, 1] = uc[:, 1] + 0.1
            uu[:, 2] = np.random.rand(target_number)

            pd = -np.random.rand(target_number)
            rd = np.random.rand(target_number)

            p = np.array([1.0, 0.0])

            target_u = -np.inf
            val = None
            plan = None

            for i in range(0, target_number):
                cur_utility, x, _ = lp_calc(pa, ra, pd, rd, uc, uu, p, r, i)
                if cur_utility >= target_u:
                    target_u = max(target_u, cur_utility)

                #target_u = max(target_u, cur_utility)
            print("THIS IS TARGETU")
            print(target_u, target_number)
            #cur_res.append(target_u)

            #plans.append(plan)

            for p0 in np.arange(1.0, min_p0, -0.05):
                p = np.array([p0, 1-p0])
                while True:
                    maxv = -np.inf
                    for i in range(target_number):
                        def_v, x, att_v = lp_calc(pa, ra, pd, rd, uc, uu, p, r, i)
                        print('current instance:')
                        print(p, num_instances, target_number)
                        #print(def_v)
                        if def_v > maxv:
                            maxv = def_v
                            attv = att_v
                    if target_u - maxv > eps:
                        print("AAAAA")
                        print(str(r) + "IS THE VALUE OF R in loop")
                        r = r + 0.1
                        if r > 20.0:
                            min_p0 = p0
                            break
                    else:
                        break
                list_result.append(r)
                print("THIS IS THE VALUE OF R")
                print(r)
                print("END LINE")
            res[target_number].append(list_result)
    save(res, 'resource_request_n.pickle')
    
    
    
    for target_number in target_numbers:
        print_x = np.arange(0.0, 1.0, 0.01)
        #print("This is print_x")
        #print(print_x)
        for i in range(len(res[target_number])):
            res[target_number][i] = res[target_number][i][:len(print_x)]
        cur_res = np.array(res[target_number])
    
    print(res)

    '''
        for ty in range(len(pa)):
                for t1 in range(len(pa)+1):
                    for i1 in range(len(p)):
                        for t2 in range(len(pa)+1):
                            for i2 in range(len(p)):
                                if plan[ty, t1, i1, t2, i2] > 0.0:
                                    print(r, ty, t1, i1, t2, i2)
                                    print(p, plan[ty, t1, i1, t2, i2])
    '''
        #print(plans)

        #save(cur_res, 'resource_result_atk_n_{}.pickle'.format(str(target_number)))
    