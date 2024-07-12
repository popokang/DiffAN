import numpy as np
from pygam import LinearGAM
from pygam.terms import TermList, SplineTerm

def train_gam(X, y, numBasisFcts=10):

    p = X.shape
    if p[0]/p[1] < 3*numBasisFcts:
        numBasisFcts = int(np.ceil(p[0]/(3*p[1])))
        print(f"Changed number of basis functions to {numBasisFcts} in order to have enough samples per basis function")
    terms = TermList()
    for i in range(p[1]):
        terms += SplineTerm(i, n_splines=numBasisFcts)
    try: # FIXME
        mod_gam = LinearGAM(terms).gridsearch(X,y)
    except:
        print("There was some error with gam. The smoothing parameter is set to zero.")
        terms = TermList()
        for i in range(p[1]):
            terms += SplineTerm(i, n_splines=numBasisFcts, lam=0)
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


def selGam(X, y, k, cutOffPVal=0.001, numBasisFcts=10, output=False):
    p = X.shape
    if p[1] > 0: 
        mod_gam = train_gam(X, y, numBasisFcts=numBasisFcts)
        pValVec = np.array(mod_gam['p_values'])

        if output:
            print(f"vector of p-values:{pValVec}")
        if len(pValVec) - 1 != p[1]: 
            print("This should never happen (function selGam).")
        selVec = pValVec[:k] < cutOffPVal
    else:
        selVec = np.array([])
    return selVec


def pruning(X, G, output=False, pruneMethod=selGam, cutOffPVal=0.001, numBasisFcts=10):
    p = G.shape[0] 
    finalG = np.zeros((p, p))

    for i in range(p):
        parents = np.where(G[:, i] == 1)[0] 
        lenpa = len(parents)
        if output:
            print(f"Pruning variable: {i}")
            print(f"Considered parents: {parents}")
        if lenpa > 0:
            selectedPar = pruneMethod(X[:, parents], X[:, i].reshape(-1,1), k=lenpa, cutOffPVal=cutOffPVal, numBasisFcts=numBasisFcts, output=output)
            finalParents = parents[selectedPar] 
            finalG[finalParents, i] = 1
    return finalG


def cam_pruning(A, X, cutoff):
    dag = pruning(X, A, output=False, pruneMethod=selGam, cutOffPVal=cutoff, numBasisFcts=10)
    return dag
