import numpy as np
from pygam import LinearGAM
from pygam.terms import TermList, SplineTerm

def train_gam(X, y, pars = {'numBasisFcts':10}):
    if 'numBasisFcts' not in pars:
        pars['numBasisFcts'] = 10

    p = X.shape
    if p[0]/p[1] < 3*pars['numBasisFcts']:
        pars['numBasisFcts'] = int(np.ceil(p[0]/(3*p[1])))
        print(f"Changed number of basis functions to {pars['numBasisFcts']} in order to have enough samples per basis function")
    terms = TermList()
    for i in range(X.shape[1]):
        terms += SplineTerm(i, n_splines=pars['numBasisFcts'])
    try:
        mod_gam = LinearGAM(terms).gridsearch(X,y)
    except:
        print("There was some error with gam. The smoothing parameter is set to zero.")
        terms = TermList()
        for i in range(X.shape[1]):
            terms += SplineTerm(i, n_splines=pars['numBasisFcts'], lam=0)
        mod_gam = LinearGAM(terms).fit(X,y)

    result = {
        'Yfit': mod_gam.predict(X),
        'residuals': (mod_gam.predict(X)-y.squeeze()),
        'model': mod_gam,
        'deviance': mod_gam.statistics_['deviance'],
        'edf': mod_gam.statistics_['edof'],
        'edf_per_coef': mod_gam.statistics_['edof_per_coef'],
        'p_values': mod_gam.statistics_['p_values'],
    }

    return result


def selGam(X, k, pars={'cutOffPVal':0.001, 'numBasisFcts':10}, output = False):
    p = X.shape
    if p[1] > 1: 
        selVec = [False] * p[1]
        mod_gam = train_gam(X[:,:k], X[:, k].reshape(-1, 1), pars)
        pValVec = np.array(mod_gam['p_values'])[:k]

        if output:
            print(f"vector of p-values:{pValVec}")
        if len(pValVec) != len(selVec) - 1: 
            print("This should never happen (function selGam).")
        selVec[:k] = pValVec < pars['cutOffPVal']
    else:
        selVec = []
    return selVec


def pruning(X, G, output=False, pruneMethod=selGam, pruneMethodPars={'cutOffPVal': 0.001, 'numBasisFcts': 10}):
    p = G.shape[0] 
    finalG = np.zeros((p, p))

    for i in range(p):
        parents = np.where(G[:, i] == 1)[0] 
        lenpa = len(parents)
        if output:
            print(f"Pruning variable: {i}")
            print(f"Considered parents: {parents}")
        if lenpa > 0:
            Xtmp = np.column_stack((X[:, parents], X[:, i]))
            selectedPar = pruneMethod(Xtmp, k=lenpa, pars=pruneMethodPars, output=output)
            finalParents = parents[np.array(selectedPar)[:-1]]
            finalG[finalParents, i] = 1
    return finalG


def cam_pruning(A, X, cutoff):
    np.random.seed(42)
    dag = pruning(X, A, output=False, pruneMethod=selGam, pruneMethodPars={'cutOffPVal': cutoff, 'numBasisFcts': 10})
    return dag

