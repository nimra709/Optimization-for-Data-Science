#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import const_ as const
import numpy as np

"""
Authors: Hafiz Muhammad Umer
         Nimra Nawaz
"""



"""

This file implements the optimization of a function with the gradient method. 
The class steepestGradientDescent needs a function object whith this three methods: 
1) init_x() -> return the starting point (could be usefull for some fucntions)
2) minimizefx_(ponitX) -> returns f(X) and gradient in X 
3) stepSizeCalculation_(direction) -> return a point that satisfy at least the Wolfe condition along the gradient direction

"""

class steepestGradientDescent():

    def __init__(self, **kwargs):
        """

        Constructor

        parameters:
        - verbose: A boolean flag to enable or disable verbose output. Defaults to True.
        - x: An optional initial vector. If not provided, it will be initialized within the function.
        - function (required): A callable object or an object that has a calculate method.

        """

        self.verbose = True if "verbose" not in kwargs.keys() else kwargs["verbose"]
        self.function = kwargs["function"]
        self.status = ''
        self.feval = 1
        self.x = kwargs["x"] if kwargs["x"] is not None else self.function.init_x()

        self.v = self.function.minimizefx_(self.x)
        self.g = self.function.grad_func_(self.x)
        self.ng = np.linalg.norm(self.g)
        # Absolute error or relative error?
        if const.eps < 0:
            self.ng0 = - self.ng
        else:
            self.ng0 = 1

    def steepestGradientDescent(self):
        
        self.historyNorm = []
        self.historyValue = []

        while True:
            self.historyNorm.append(self.ng.item())
            self.historyValue.append(self.v.item())
            if self.verbose: 
                self.print()

            # Norm of the gradient lower or equal of the epsilon
            if self.ng <= const.eps * self.ng0:
                self.status = 'optimal'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue


            # Man number of iteration?
            if self.feval > const.MaxFeval:
                self.status = 'stopped'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            # calculate step along direction
            alpha = self.function.stepSizeUsingExactSearch_()

            # step too short
            if alpha <= const.mina:
                self.status = 'error'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            lastx = self.x
            self.x = self.x - alpha * self.g
            self.v = self.function.minimizefx_(self.x)
            self.g = self.function.grad_func_(self.x)
            self.feval = self.feval + 1

            if self.v <= const.MInf:
                self.status = 'unbounded'
                if self.verbose:
                    self.print()
                return self.historyNorm, self.historyValue

            self.ng = np.linalg.norm(self.g)

        print('\n x = ' + str(self.x) + '\nvalue = %0.4f' % self.v)

    def print(self):
        print("Iterations number %d, -f(x) = %0.4f, gradientNorm = %f - " % (self.feval, self.v, self.ng) + self.status)
    
    # same function as the previus one but it returns also the time and
    # we avoid print and other operation which slow down the algorithm
    def steepestGradientDescentTIME(self):
        while True:
            if self.ng <= const.eps * self.ng0:
                self.status = 'optimal'
                return self.ng.item(), self.v.item()

            if self.feval > const.MaxFeval:
                self.status = 'stopped'
                #print(self.status)
                break

            alpha = self.function.stepSizeUsingExactSearch_()

            # step too short
            if alpha <= const.mina:
                self.status = 'error'
                #print(self.status)
                break

            self.x = self.x - alpha * self.g
            self.v = self.function.minimizefx_(self.x)
            self.g = self.function.grad_func_(self.x)
            self.feval = self.feval + 1

            if self.v <= const.MInf:
                self.status = 'unbounded'
                #print(self.status)
                break

            self.ng = np.linalg.norm(self.g)
        return self.ng.item(), self.v.item()
