#!/usr/bin/env python3
# coding: utf-8

# Code for Upper and Lower Bounds on the Smoothed Complexity of the Simplex Method
# by Sophie Huiberts, Yin Tat Lee and Xinzhi Zhang, STOC 2023
# https://arxiv.org/abs/2211.11860
# Running this file as-is outputs 4 pdf figures, which together form Figure 4 in the paper.


# forked from extended.py, this code will produce a few plots.
# at the bottom of this file, a number k is set. the code will construct
# the polytope described by cons:absvalue, cons:w, cons:v, cons:deathbarrier.
# note that no constraints on s and t are included, which means that
# the resulting polytope will be a superset of our actual set
#
# the code then produces k+1 plots:
# a) the polytope projected onto the space given by (x,y)
# b) the k projections of the polytope onto p_1,\dots,p_k
# the purpose is to verify intuitions about what these projections look like,
# in the hope of obtaining a proof that our projection onto (x,y)
# is indeed the regular 2^{k+1}-gon
#
# note that the plots use different scales for their axes!
#
# code by Sophie Huiberts, 2022



import math
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# let's use gurobi for formulating our constraint matrix and for finding vertices.
from gurobipy import GRB, Model, quicksum, LinExpr
# this is to get gurobi's little output out of the way so that it wont mix with our own stdout
Model()



# given k, return a gurobi model in k+5 variables and 4k+5 constraints
# whose feasible region is P-(bar x, bar y, bar p_0, bar t, bar s)
def construction(k,shifted=True,unboundedts=False):
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
    pixs=[pnaughtx]
    piys=[pnaughty]

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
        pixs.append(pix)
        piys.append(piy)
    P.addConstr(pixs[-1] >= 0)
    P.addConstr(piys[-1] >= 0)

    # (6)
    P.addConstr(pix <= 1,name='death')

    P.update()

    # now that P has been constructed, we modify its right-hand side vector
    # and variable bounds
    # such that the Gurobi model represents P - (\bar x, \bar y, \bar p_0, \bar t, \bar s)
    if shifted:
        for a in P.getConstrs():
            # for the row a, compute its inner product with (\bar x, \bar y, \bar p_0, \bar t, \bar s) as defined in \ref{lem:innerball}
            innerproduct = P.getCoeff(a, pnaughtx) * 1/6  \
                    + P.getCoeff(a, pnaughty) * 1/6  \
                    + sum(P.getCoeff(a, t[i]) * 1/30 for i in range(1,k+1)) \
                    + P.getCoeff(a, s) * 1/3
            a.setAttr('RHS', a.getAttr('RHS') - innerproduct)
        s.setAttr('LB', -1/3)
        s.setAttr('UB', 2/3)

        # i forgor why the +1 is here
        P.setAttr('LB', t, [-1/30]*(len(t)+1))
        P.setAttr('UB', t, [29/30]*(len(t)+1))
    if unboundedts:
        P.setAttr('LB', t, [-GRB.INFINITY]*(len(t)+1))
        P.setAttr('UB', t, [GRB.INFINITY]*(len(t)+1))
        s.setAttr('LB', -GRB.INFINITY)
        s.setAttr('UB', GRB.INFINITY)

    P.update()
    return P, pixs, piys




# given a model, plot a good number of shadow vertices on the two given objectives
# in time exponential in its number of variables
# NOTE: this function will change the model's objective function
# TODO: replace with a shadow simplex method for extra precision and speed
def plotCircle(model, x,y, title="", filename='fig.png', circleoverlayradii=[-1]):
    model.update()
    model.Params.OutputFlag = 0
    hor = []
    ver = []
    d = max(14,len(model.getVars()))
    for i in range(2**d):
        model.setObjective( math.cos(math.pi * (i+0.3)/2**(d-1)) * x + math.sin(math.pi * (i+0.3) / 2**(d-1)) * y)
        model.optimize()
        if model.getAttr('status') != GRB.OPTIMAL:
            model.Params.DualReductions = 0
            model.optimize()
            print(':( with status code ' + str(model.getAttr('status')))
            break
        verpt = LinExpr(y).getValue()
        horpt = LinExpr(x).getValue()
        # deduplicate points otherwise the pdf viewer hates us
        if len(ver) >= 1:
            if abs(verpt - ver[-1]) + abs(horpt - hor[-1]) > 1e-5:
                hor.append(horpt)
                ver.append(verpt)
        else:
            hor.append(horpt)
            ver.append(verpt)
    fig, ax = plt.subplots()
    limit = max(max(circleoverlayradii), 1.1, max(ver), max(hor), -min(ver), -min(hor))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(hor,ver,'.')
    plt.title(title)
    for r in circleoverlayradii:
        if r > 0:
            t = [2*math.pi * i/100 for i in range(101)]
            c = [r*math.cos(theta) for theta in t]
            s = [r*math.sin(theta) for theta in t]
            plt.plot(c,s,linewidth=1)
    points = [[hor[i], ver[i]] for i in range(len(hor))]
    hull = ConvexHull( points )
    for simplex in hull.simplices:
        segmentxs = [hor[simplex[0]], hor[simplex[1]]]
        segmentys = [ver[simplex[0]], ver[simplex[1]]]
        plt.plot(segmentxs, segmentys, 'c')
    plt.savefig(filename)



# for our favorite value of k,
# we can use our eyes to check that shadow looks kind of circular,
#plotCircle(shiftedP(k))
# estimate the number of vertices in the shadow,
#shadowsize, minimum, maximum = estimate(shiftedP(k))
# or write the model to file
#shiftedP(k).write('construction-' + str(k) + '.lp')

k = 2
plt.rcParams.update({
    "text.usetex": True,
})

originalmodel, pixs, piys = construction(k, shifted=False, unboundedts=True)
plotCircle(originalmodel, LinExpr(originalmodel.getVars()[0]), LinExpr(originalmodel.getVars()[1]),
           title="Vertices of $\pi_W(R)$".format(k),
           filename="k{}-xy.pdf".format(k))

for i in range(len(pixs)):
    plotCircle(originalmodel, pixs[i], piys[i],
               title="Vertices of $R_{}$".format(i),
               filename="k{}-p{}.pdf".format(k,i))
