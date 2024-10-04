import steepestGradientDescent as SGD
import numpy as np
import pandas as pd
import objective_func as objFunc #nf
import matplotlib.pyplot as plt
import matplotlib
import os
import seaborn as sns

def readMatrix(type, number):
    # Set the working directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("New working directory:", os.getcwd())

    raw = []
    try:
        f = open(os.getcwd() + '/Matrices/Matrix' + type + '/matrix' + type + str(number) + '.txt', 'r')
    except:
        f = open(os.getcwd() + '/Matrices/Matrix' + type + '/matrix' + type + str(number) + '.txt', 'r')
    for line in f:
        raw.append(line.split())
    A = np.array(raw, dtype=float)
    return A

def normSDG(A):
    f = objFunc.ObjectiveFunc(A)
    SGDoptimizer = SGD.steepestGradientDescent(function=f)
    return SGDoptimizer.steepestGradientDescentTIME()

# map the type of matrix with the density
def density(type):
    if type == 'A':
        return 1
    elif type == 'B':
        return 0.5
    elif type == 'C':
        return 0.25
    elif type == 'D':
        return 0.01
    elif type == 'E':
        return 1

# (old) method to plot different errors and gradients with different lines
def printPlot(relerrorsSGD=None, gradientsSGD=None, relerrorsCG=None, gradientsCG=None,
              A=None, type=None, num=None, ):
    yLabel2 = 'Relative Error'
    yLabel3 = 'Norms Gradient'
    xLabel1 = 'Iterations'

    fig = plt.figure()
    relErrPlotSGD = fig.add_subplot(2, 2, 1)
    relErrPlotCG = fig.add_subplot(2, 2, 2, sharey=relErrPlotSGD)
    gradPlotSGD = fig.add_subplot(2, 2, 3)
    gradPlotCG = fig.add_subplot(2, 2, 4, sharey=gradPlotSGD)

    fig.set_size_inches(18.5, 10.5)
    m, n = np.shape(A)

    for i in range(len(relerrorsSGD)):
        relerrorsSGD[i] = [max(err, 1e-20) for err in relerrorsSGD[i]]
    for i in range(len(relerrorsCG)):
        relerrorsCG[i] = [max(err, 1e-20) for err in relerrorsCG[i]]

    plt.ylim(10e-16, 10e0)
    relErrPlotSGD.set_title(
        'Steepest Gradient Descent \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))

    relErrPlotSGD.set_yscale('log')
    relErrPlotSGD.set(ylabel=yLabel2)

    for relSGD in relerrorsSGD:
        relErrPlotSGD.plot(relSGD)

    gradPlotSGD.set(ylabel=yLabel3)
    gradPlotSGD.set(xlabel=xLabel1)
    gradPlotSGD.set_yscale('log')

    relErrPlotCG.set_title(
        'Conjugate Gradient \n Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))

    relErrPlotCG.set_yscale('log')
    for relCG in relerrorsCG:
        relErrPlotCG.plot(relCG)

    gradPlotCG.set(xlabel=xLabel1)
    gradPlotCG.set_yscale('log')

    plt.ylim(10e-10, 10e5)

    for gradCG in gradientsCG:
        gradPlotCG.plot(gradCG)
    for gradSGD in gradientsSGD:
        gradPlotSGD.plot(gradSGD)
    plt.show()
    savePlot(type, num, fig)


# current method to plot errors and gradients
def printPlot2(relerrorsSGD=None, gradientsSGD=None, relerrorsCG=None, gradientsCG=None,
               A=None, type=None, num=None, ):
    font = {'family': 'DejaVu Sans',
            'weight': 'light',
            'size': 18}

    matplotlib.rc('font', **font)
    plt.rcParams.update({'font.size': 18})
    # Strings

    yLabel2 = 'Relative Error'
    yLabel3 = 'Norms Gradient'
    xLabel1 = 'Iterations'

    # Figures
    fig = plt.figure()
    relErrPlot = fig.add_subplot(2, 1, 1)
    gradPlot = fig.add_subplot(2, 1, 2)
    relErrPlot.set_yscale('log')
    gradPlot.set_yscale('log')
    fig.set_size_inches(13.5, 10.5)
    m, n = np.shape(A)

    # Title
    relErrPlot.set_title(
        'Type ' + type + '     Density =  ' + str(density(type)) + '    M = ' + str(
            m) + ' N = ' + str(n))

    if gradientsCG is not None and relerrorsCG is not None:
        for i in range(len(relerrorsCG)):
            relerrorsCG[i] = [max(err, 1e-20) for err in relerrorsCG[i]]

        relCG = (pd.DataFrame((pair_vector(relerrorsCG)))).melt()
        gradCG = (pd.DataFrame((pair_vector(gradientsCG)))).melt()

        # Plot relative errors and gradient of CG
        sns.lineplot(x="variable", y="value", data=relCG, estimator=geo_mean, ax=relErrPlot)
        sns.lineplot(x="variable", y="value", data=gradCG, estimator=geo_mean, ax=gradPlot)
    else:
        print("Relative errors and Norms of Gradient must be both None")

    if gradientsSGD is not None and relerrorsSGD is not None:
        for i in range(len(relerrorsSGD)):
            relerrorsSGD[i] = [max(err, 1e-20) for err in relerrorsSGD[i]]

        relSGD = (pd.DataFrame((pair_vector(relerrorsSGD)))).melt()
        gradSGD = (pd.DataFrame((pair_vector(gradientsSGD)))).melt()

        # Plot relative errors and gradient of SGD
        sns.lineplot(x="variable", y="value", data=relSGD, estimator=geo_mean, ax=relErrPlot)
        sns.lineplot(x="variable", y="value", data=gradSGD, estimator=geo_mean, ax=gradPlot)
    else:
        print("Relative errors and Norms of Gradient must be both None")

    # Set label x and y
    relErrPlot.set(xlabel=xLabel1)
    relErrPlot.set(ylabel=yLabel2)
    gradPlot.set(xlabel=xLabel1)
    gradPlot.set(ylabel=yLabel3)

    # Set legend of subplots
    if gradientsSGD is None and relerrorsSGD is None:
        relErrPlot.legend(('Conjugate Gradient', 'Steepest Descent Gradient'), loc='upper right')
        gradPlot.legend(('Conjugate Gradient', 'Steepest Descent Gradient'), loc='upper right')
    else:
        relErrPlot.legend(('Steepest Descent Gradient', 'Conjugate Gradient'), loc='upper right')
        gradPlot.legend(('Steepest Descent Gradient', 'Conjugate Gradient'), loc='upper right')

    plt.show()
    savePlot(type, num, fig)


def savePlot(type, num, fig):

    directory = "../Plot/"
    file = type + num + ".png"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + file)


def fromCSVToLatexTable(nome1, nome2):
    df = pd.read_csv("CSVresult/" + nome1 + ".csv")
    a = df.values
    a = a[:, 1:]
    np.savetxt("CSVresult/Latextable" + nome2 + ".csv", a, delimiter=' & ', fmt='%2.2e', newline=' \\\\\n')


def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


def pair_vector(vector_list):
    maxlen = len(max(vector_list, key=len))
    for i in range(len(vector_list)):
        lastEl = len(vector_list[i]) - 1
        tmpArray = np.full((maxlen - lastEl - 1, 1), vector_list[i][lastEl])
        vector_list[i] = np.array(np.concatenate((vector_list[i], tmpArray), axis=None))
    return vector_list