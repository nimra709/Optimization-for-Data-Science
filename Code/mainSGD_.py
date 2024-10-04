import objective_func as objFunc #nf
import steepestGradientDescent as SGD
import numpy as np
from utility_functions import *

def main():

    type = "A"
    relerrorsSGD = []
    gradientsSGD = []

    A = readMatrix(type, 1)
    f = objFunc.ObjectiveFunc(A)
    initial_vector = f.init_x()

    # Optimizer SGD
    optimizerSGD = SGD.steepestGradientDescent(function=f, x=initial_vector, verbose=True)
    gradientSGD, normsSGD = optimizerSGD.steepestGradientDescent()

    # Norm numpy
    norm = np.linalg.norm(A, ord=2) ** 2

    # Norm and errors SGD
    normsSGD = np.array(normsSGD)
    gradientsSGD.insert(0,np.array(gradientSGD))
    size1 = normsSGD.size

    normvec = np.ones(size1) * norm
    relerrorsSGD.insert(0, (abs(normsSGD - normvec) / abs(normvec)))

    printPlot2(relerrorsSGD, gradientsSGD, None, None, A, type, "1")


if __name__ == "__main__":
    main()