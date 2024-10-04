import objective_func as objFunc #nf
import BFGS as BFGS
import numpy as np
from utility_functions import *
from numpy import linalg as LA

def main():

    type = "A"
    relerrorsBFGS = []
    gradientsBFGS = []

    A = readMatrix(type, 1)
    f = objFunc.ObjectiveFunc(A)
    initial_vector = f.init_x()

    matrix_norm = LA.norm(A, 2)

    ls_method = 'W'
    c1=1e-4
    c2=0.9

    if ls_method == 'W':
            line_search_args = {'c1': c1, 'c2': c2}
    else:
            line_search_args = {'c1': c1}
    
    line_search_method = f.line_search_methods[ls_method]

    H0 = np.identity(A.shape[1])
    alg_method='O'

    # Optimizer BFGS
    optimizerBFGS = BFGS.BFGS(matrix_norm, f, line_search_method, initial_vector, H0, 1e-5, alg_method, line_search_args, False)
    gradientBFGS, normsBFGS = optimizerBFGS.bfgs()

    # Norm numpy
    norm = np.linalg.norm(A, ord=2) ** 2

    # Norm and errors BFGS
    normsBFGS = np.array(normsBFGS)
    gradientsBFGS.insert(0,np.array(gradientBFGS))
    size1 = normsBFGS.size

    normvec = np.ones(size1) * norm
    relerrorsBFGS.insert(0, (abs(normsBFGS - normvec) / abs(normvec)))

    printPlot2(relerrorsBFGS, gradientsBFGS, None, None, A, type, "1")


if __name__ == "__main__":
    main()