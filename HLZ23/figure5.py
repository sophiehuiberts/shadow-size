#!/usr/bin/env python3
# coding: utf-8
import math
import random

# main code used to verify the construction works as intended, and to measure
# the actual lower bound that could be obtained from our construction
#code by Sophie Huiberts, 2022

# major rewrite on april 6, 2022
# this time i knew what i was doing, unlike last time :)
#
# if we want to include measurements in the paper,
# we should implement a shadow simplex method to base
# plotCircle() and esimate() on. that will make for faster computation
# and more accurate results.



# let's use gurobi for formulating our constraint matrix.
from gurobipy import GRB, Model, quicksum, LinExpr
# this is to get gurobi's little output out of the way so that it wont mix with our own stdout
Model()



# given k, return a gurobi model in k+5 variables and 4k+5 constraints
# whose feasible region is P-(bar x, bar y, bar p_0, bar t, bar s)
def shiftedP(k):
    # first, construct a Gurobi model with k+5 variables and 4k+5 constraints
    # with feasible region P as defined in \eqref{eq:extension}
    P = Model()
    P.Params.OutputFlag = 0

    # this is x in the paper
    x = P.addVar(lb=-GRB.INFINITY,name='x')
    # this is y in the paper
    y = P.addVar(lb=-GRB.INFINITY,name='y')

    # these two together form p_0 in the paper
    pnaughtx = P.addVar(lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='p0x')
    pnaughty = P.addVar(lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='p0y')

    # this is t in the paper. Also (7)
    t = P.addVars([i for i in range(1,k+1)],lb=0, ub=1,vtype=GRB.CONTINUOUS,name='t')
    # this is s in the paper. Also (8)
    s = P.addVar(lb=0, ub=1,vtype=GRB.CONTINUOUS,name='s')

    # (3)
    P.addConstr(pnaughtx >= x)
    P.addConstr(pnaughtx >= -x)
    P.addConstr(pnaughty >= y)
    P.addConstr(pnaughty >= -y)

    # set up p_{i-1} for the loop defining (5)
    piminusonex = pnaughtx
    piminusoney = pnaughty

    # as in the paper, we have i \in [k]
    for i in range(1,k+1):
        # this is w_i^\T p_{i-1} in the paper
        witransposepiminusone = math.cos(math.pi/2/2**i) * piminusonex + math.sin(math.pi/2/2**i) * piminusoney
        # this is v_i^\T p_{i-1} in the paper
        vitransposepiminusone = - math.sin(math.pi/2/2**i) * piminusonex + math.cos(math.pi/2/2**i) * piminusoney

        # these two together form p_i together. note that these are LinExpr's, not variables.
        # note that this definition uses (4), that is, it uses w_i^\T p_{i-1} = w_i^\T p_i
        pix = math.cos(math.pi/2/2**i) * witransposepiminusone + math.sin(math.pi/2/2**i) * (t[i] + i * s)
        piy = math.sin(math.pi/2/2**i) * witransposepiminusone - math.cos(math.pi/2/2**i) * (t[i] + i * s)

        # (5)
        P.addConstr(t[i] + i*s >= vitransposepiminusone,name='5ub')
        P.addConstr(t[i] + i*s >= - vitransposepiminusone,name='5lb')

        # set up the value of p_{i-1} to be used in the next iteration
        piminusonex = pix
        piminusoney = piy

    # (6)
    P.addConstr(pix <= 1,name='death')

    P.update()

    # now that P has been constructed, we modify its right-hand side vector
    # and variable bounds
    # such that the Gurobi model represents P - (\bar x, \bar y, \bar p_0, \bar t, \bar s)

    for a in P.getConstrs():
        # for the row a, compute its inner product with (\bar x, \bar y, \bar p_0, \bar t, \bar s) as defined in \ref{lem:innerball}
        innerproduct = P.getCoeff(a, pnaughtx) * 1/6  \
                + P.getCoeff(a, pnaughty) * 1/6  \
                + sum(P.getCoeff(a, t[i]) * 1/30 for i in range(1,k+1)) \
                + P.getCoeff(a, s) * 1/3
        a.setAttr('RHS', a.getAttr('RHS') - innerproduct)
    s.setAttr('LB', -1/3)
    s.setAttr('UB', 2/3)

    P.setAttr('LB', t, [-1/30]*(len(t)+1))
    P.setAttr('UB', t, [29/30]*(len(t)+1))

    P.update()
    return P

# given a model for which the origin is strictly feasible,
# return the matrix A such that the feasible region is given
# by {x : Ax <= 1, lb <= x <= ub}
def A(model):
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    unscaledmatrix = model.getA()

    # by strict feasibility, none of these is 0
    scales = sparse.diags([1/s for s in model.getAttr('RHS', model.getConstrs())])

    return scales @ unscaledmatrix

# given a model for which the origin is strictly feasible,
# compute a perturbed matrix \tilde{A}
# such that the original feasible set is given by
# {x : \E[A]x <= 1}
# and st the rows of A are independently normally distributed
# with independent coordinates. each standard deviation is
# sigma * max_i \|\E[A_i]\|, where A_i is the i'th row of the matrix A.
#
# returns the max row norm of E[A] and the perturbed matrix A
def perturb(model,sigma):
    model.update()
    import scipy.sparse as sparse
    import scipy.sparse.linalg as linalg

    oldvars = model.getVars()
    oldconstrs = model.getConstrs()

    # first we compute the max row norm for \E[A]
    mat = A(model)
    maxrownorm = max(
            math.sqrt(max(linalg.norm(mat.multiply(mat), math.inf, 1))),
            max([1/ub for ub in model.getAttr('UB', oldvars)]),
            max([-1/lb for lb in model.getAttr('LB', oldvars)]))

    # define the distribution to draw perturbations from
    from scipy import stats
    rvs = stats.norm(scale=sigma*maxrownorm).rvs

    # now, perturb the constraints arising from model's matrix
    matrixconstraints = mat + \
            sparse.random(len(oldconstrs), len(oldvars), density=1, data_rvs = rvs)

    # next, the constraints in the variable bounds
    ubconstraints = sparse.diags([1/ub for ub in model.getAttr('UB', oldvars)]) + \
            sparse.random(len(oldvars), len(oldvars), density=1, data_rvs = rvs)
    lbconstraints = sparse.diags([1/lb for lb in model.getAttr('LB', oldvars)]) + \
            sparse.random(len(oldvars), len(oldvars), density=1, data_rvs = rvs)

    return maxrownorm, sparse.vstack([matrixconstraints, ubconstraints, lbconstraints])

# given a model, estimate the number of shadow vertices and the minimum&maximum
#of the radial support function
# in time exponential in its number of variables
# NOTE: this function will change the model's objective function
# TODO: replace with a shadow simplex method for extra precision and speed
from tqdm import tqdm

def estimate(model,x,y,k=0):
    model.update()
    minimum = 1000
    maximum = -1000
    model.Params.OutputFlag = 0
    model.setAttr('ModelSense', -1)
    prevx = 0
    prevy = 0
    vtexcount = 0
    # here, d = k+2.
    # considering that the object is a 2^{k+1}-gon,
    # this should typically count most vertices
    # nonetheless, this is absolutely just an estimate
    d = len(model.getVars()) + 6
    if k != 0:
        d = k+5
    for i in tqdm(range(2**d)):
        model.setObjective( math.cos(math.pi * (i+0.3)/2**(d-1)) * x + math.sin(math.pi * (i+0.3) / 2**(d-1)) * y)
        model.optimize()
        if model.getAttr('status') != GRB.OPTIMAL:
            model.Params.DualReductions = 0
            model.optimize()
        value = model.getAttr('ObjVal')
        if value > maximum:
            maximum = value
        if value < minimum:
            minimum = value
        # might have to verify that this is valid,
        # wrt gurobi accuracy and stuff
        # because we want our measurement to be a lower bound ideally?
        # note that gurobi's optimalitytol is 1-e6,
        # and its feasibilitytol is also 1e-6.
        if abs(x.X - prevx) + abs(y.X - prevy) > 1e-11:
            vtexcount = vtexcount + 1
        prevx = x.X
        prevy = y.X
    print("new vertex ratio",vtexcount/(2**d))
    return vtexcount, minimum, maximum


# for our favorite value of k,
# we can use our eyes to check that shadow looks kind of circular,
#plotCircle(shiftedP(k))
# estimate the number of vertices in the shadow,
#shadowsize, minimum, maximum = estimate(shiftedP(k))
# or write the model to file
#shiftedP(k).write('construction-' + str(k) + '.lp')

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.title('measured quality of lower bound')
plt.ylabel('shadow size')
plt.xlabel('sigma')
ax1.set_yscale('log')
ax1.set_xscale('log')


# estimated 64*11min=12 hours of calculation here :')
krange = [10,15,20]
biggestsigma = 0.01
stepcount = 20

print('krange',krange)
print('stepcount', stepcount)

smallestsigma = 0.01* (2**(-max(krange)))
stepsize = (biggestsigma/smallestsigma)**(1/(stepcount-1))

# reference line
refxs = []
refys = []
for i in range(stepcount):
    sigma = smallestsigma * (stepsize**i)
    refxs.append(sigma)
    refys.append(sigma**(-3/4))
ax1.plot(refxs,refys,label="y=x^(-3/4)")

# actual data
for k in krange:
    print("k = {}".format(k))
    # in my tests this was enough to show the flattening off
    smallestsigma = 0.0001* (2**(-k))
    stepsize = (biggestsigma/smallestsigma)**(1/(stepcount-1))

    originalmodel = shiftedP(k)

    xs = []
    ys = []
    for i in range(stepcount):
        print("step {}".format(i))
        sigma = smallestsigma * (stepsize**i)

        norm, perturbedmatrix = perturb(originalmodel, sigma)

        # with the pertubed matrix, we now formulate new gurobi model
        perturbedmodel = Model()
        variables = perturbedmodel.addVars(range(len(originalmodel.getVars())),
                name=originalmodel.getAttr('VarName',originalmodel.getVars()),
                lb=-GRB.INFINITY)
        perturbedmodel.addMConstr(perturbedmatrix, None, '<',
                [1]*(len(originalmodel.getConstrs()) + 2*len(originalmodel.getVars())))

        # and compute some things
        shadowsize, minimum, maximum = estimate(perturbedmodel, variables[0], variables[1],k)
        xs.append(sigma)
        ys.append(shadowsize)
    label = "measurements for k={}".format(k)
    ax1.scatter(xs,ys,label=label)
    with open("measurements-k{}.csv".format(k), 'w') as f:
        for i in range(len(xs)):
            f.write("{},{}\n".format(xs[i],ys[i]))
plt.legend(loc='upper right')
plt.savefig('coolplot.pdf')
#plt.show()
