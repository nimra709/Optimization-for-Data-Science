#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import sparse
from numpy import linalg as la

"""
Authors: Hafiz Muhammad Umer
         Nimra Nawaz
"""

homedirectory = '/Users/mumer/My Data/Unipi - Study Material/Semester - 3rd/Optimization for Data Science/project/Umer/Code/Matrices'
if not os.path.exists(homedirectory):
    os.makedirs(homedirectory)

def create_x0(type, i, n):
    x0 = np.round(np.random.randn(n), decimals = 3)
    filename = os.path.join(homedirectory, 'Matrix' + type + '/x0_' + type + '_' + str(i) + '.txt')
    print(filename)
    np.savetxt(filename, x0)

def create_matrices(**kwargs):
    """

    Generate random matrices with the following dimensions and properties

    A = Matrix 1000x50, x E [-range,range], density = 1
    B = Matrix 1000x50, x E [-range,range], density = 0.5
    C = Matrix 1000x5, x E [-range,range], density = 0.25
    D = Matrix 1000x5, x E [-range,range], density = 0.01
    E = Matrix 1000x1000, 0<x<1, density = 1, ill conditioned

    Parameters:
    - type (required): A required parameter to generate a type of random matrices with the specified properties. If not provided no matrix will be generated.
    - m (optional): An optional parameter m for the total number of rows in a matrix. Default value is 1000.
    - n (optional): An optional parameter n for the total number of columns in a matrix. Default value is 50.
    - n1 (optional): An optional parameter n1 for the total number of columns in a matrix used for type C and D matrices. Default value is 5.

    Returns:
    None

    """

    type = 'G' if "type" not in kwargs.keys() else kwargs["type"]
    m = 1000 if "m" not in kwargs.keys() else kwargs["m"]
    n = 50 if "n" not in kwargs.keys() else kwargs["n"]

    directory = os.path.join(homedirectory, 'Matrix' + type)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(1, 11): #Run a loop to create 10 matrices for each type
        if type == 'A' or type == 'B' or type == 'C' or type == 'D':
            r = np.random.randn(m, n)
            create_x0(type, i, n)
        elif type == 'E':
            r = sparse.random(m, n, density=0.3, data_rvs=np.random.randn)
            r = np.squeeze(np.asarray(r.todense()))
            create_x0(type, i, n)
        elif type == 'F':
            r = golub(n)
            create_x0(type, i, n)
        else:
            print('Invalid Type')

        filename = os.path.join(directory, 'matrix' + type + str(i) + '.txt')
        print(filename)
        np.savetxt(filename, r)

def golub(n):
    """

    GOLUB  Badly conditioned integer test matrices.
        GOLUB(n) is the product of two random integer n-by-n matrices,
        one of them unit lower triangular and one unit upper triangular.
        LU factorization without pivoting fails to reveal that such
        matrices are badly conditioned.
        See also LUGUI.

    Copyright 2014 Cleve Moler
    Copyright 2014 The MathWorks, Inc.

    Parameters:
    - n (required): number of rows and columns in the required matrix.

    Returns:
    - A: an ill conditioned nxn Matrix with density 1

    """
    cond_P = 2e15
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P/4., log_cond_P * (n )/(4 * (n - 1)), log_cond_P/(2.*(n-1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    V, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    A = U.dot(S).dot(V.T)
    A = A.dot(A.T) / 1e7
    return A

# create_matrices(type='A', m=1000, n=1000)
# create_matrices(type='B', m=1000, n=100)
create_matrices(type='C', m=100, n=1000)
# create_matrices(type='D', m=100, n=100)
# create_matrices(type='E', m=1000, n=100)
# create_matrices(type='F', m=1000, n=1000)
