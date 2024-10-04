import numpy as np
import BFGS as BFGS
import steepestGradientDescent as SGD
from numpy import linalg as LA
import objective_func as objFunc #nf
from utility_functions import *


# numberOfMatrix = 10
# typeMatrix = ['A', 'B', 'C', 'D', 'E']
numberOfMatrix = 1
typeMatrix = ['A']

def main():
    global A

    for i in range(0, len(typeMatrix)):

        relerrorsSGD = []
        gradientsSGD = []
        relerrorsBFGS = []
        gradientBFGS = []

        type = typeMatrix[i]

        for j in range(1, numberOfMatrix + 1):
            A = readMatrix(type, j)
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

            # Optimizer SGD and CG
            optimizerSGD = SGD.steepestGradientDescent(function=f, x=initial_vector, verbose=False)
            optimizerBFGS = BFGS.BFGS(matrix_norm, f, line_search_method, initial_vector, H0, 1e-5, alg_method, line_search_args, False)
            gradientBFGS, normsBFGS = optimizerBFGS.bfgs()
            gradientSGD, normsSGD = optimizerSGD.steepestGradientDescent()

            # Norm numpy
            norm = LA.norm(A, ord=2) ** 2

            # Norm and errors SGD
            normsSGD = np.array(normsSGD)
            gradientsSGD.insert(j - 1, np.array(gradientSGD))
            size1 = normsSGD.size

            normvec = np.ones(size1) * norm
            relerrorsSGD.insert(j - 1, (abs(normsSGD - normvec) / abs(normvec)))

            # Norm and errors CG
            normsBFGS = np.array(normsBFGS)
            gradientBFGS.insert(j - 1, np.array(gradientBFGS))

            size2 = normsBFGS.size
            normvec = np.ones(size2) * norm
            relerrorsBFGS.insert(j - 1, abs(normsBFGS - normvec) / abs(normvec))

        # printPlot(None, relerrorsSGD, gradientsSGD, None, relerrorsCG, gradientsCG, A, type, str(i))

        printPlot2(relerrorsSGD, gradientsSGD, relerrorsBFGS, gradientBFGS, A, type, str(i+1))
if __name__ == "__main__":
    main()
