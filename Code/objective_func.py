#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn

class LineSearchWarning(RuntimeWarning):
    pass

"""
Authors: Hafiz Muhammad Umer
         Nimra Nawaz
"""

"""
Objective function to optimize the norm of a Matrix
"""

class ObjectiveFunc():
    def __init__(self, A) -> None:
        self.M = np.matmul(np.transpose(A), A) # A'A
        self.dim = self.M.shape[0]

    def minimizefx_(self, x):

        """

        f(x) = x'A'Ax / x'x

        parameters:
        - x (required): a non-zero vector

        returns:
        The negative value computed for the objective function with respect to the vector x

        """

        self.x = x
        self.xT_ = np.transpose(x) # x'
        self.Mx_ = np.matmul(self.M, x) # A'Ax
        self.xMx = np.matmul(self.xT_, self.Mx_) # x'A'Ax

        self.xTx_ = np.matmul(self.xT_, x) # x'x
        
        f_x = self.xMx / self.xTx_

        self.f_x = f_x

        return f_x*(-1)
    
    def grad_func_(self, x):

        """

        Gradient of func_. f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))

        parameters:
        - x (required): a non-zero vector

        returns:
        Gradient of the objective function

        """

        grad = (2 * x * self.f_x) / self.xTx_ - (2 * self.Mx_) / self.xTx_ # Where f_x = x^T M x / x^T x
        self.d = grad
        self.dT_ = np.transpose(self.d)

        return grad
    
    # def func_(self, x): # Function to be minimized. f(x) = (x^T M x) / x^T x , where M = A^T A
        return (np.matmul(np.matmul(x.T, self.M), x) / np.matmul(x.T, x))*(-1)
    
    # def func_grad_(self, x): # Gradient of func_. f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
        return ((2 * x * np.matmul(x.T, np.matmul(self.M, x))) / (np.matmul(x.T, x))**2 - (2 * np.matmul(self.M, x)) / np.matmul(x.T, x))
    
    # funciton that return the step size of the algorithm using exact line search along the 
    # graient direction. It uses the closed formula of the derivative along the direction
    # def stepSizeUsingExactSearch_(self):

        """

        The value of α along the direction d is given as the root of the equation below:

        α^2 a + αb + c = 0


        """

        #calculate a = (d^T M x)*(d^Td)

        dTMx_ = np.matmul(self.dT_, self.Mx_)
        dTd_ = np.matmul(self.dT_, self.d)

        a = float(dTMx_ * dTd_)

        #calculate b = -((x^T M x) d^T d - (d^T M d) (x^T x))

        xTMx_ = np.matmul(self.xT_, self.Mx_)
        Md_ = np.matmul(self.M, self.d)
        dTMd_ = np.matmul(self.dT_, Md_)

        b = float(-((xTMx_ * dTd_) - (dTMd_ * self.xTx_)))

        #calculate c = (x^T M x) x^T d - (d^T M x) (x^T x)

        xTd_ = np.matmul(self.xT_, self.d)

        c = float((xTMx_ * xTd_) - (dTMx_ * self.xTx_))

        coef = np.array([a, b, c])
        roots = np.roots(coef)

        if(roots[0] < 0 and roots[1] < 0):
            return 0
        elif(roots[0] < 0):
            return roots[1]
        elif(roots[1] < 0):
            return roots[0]
        else:
            return roots[1] if roots[1] > roots[0] else roots[0]
    
    def stepSizeUsingExactSearch_(self):


        """

        The value of α along the direction d is given as the root of the equation below:

        α^2 a + αb + c = 0


        """

        #calculate a = (d^T M x)*(d^Td)

        dTMx_ = np.matmul(self.dT_, self.Mx_)
        dTd_ = np.matmul(self.dT_, self.d)

        # a = float(dTMx_ * dTd_)

        #calculate b = -((x^T M x) d^T d - (d^T M d) (x^T x))

        xTMx_ = np.matmul(self.xT_, self.Mx_)
        Md_ = np.matmul(self.M, self.d)
        dTMd_ = np.matmul(self.dT_, Md_)

        # b = float(-((xTMx_ * dTd_) - (dTMd_ * self.xTx_)))

        #calculate c = (x^T M x) x^T d - (d^T M x) (x^T x)

        xTd_ = np.matmul(self.xT_, self.d)

        # c = float((xTMx_ * xTd_) - (dTMx_ * self.xTx_))

        xMd_ = np.matmul(self.xT_, Md_)
        dMd = np.matmul(self.dT_, Md_)

        a = float(dTd_ * xMd_ - dMd * xTd_)
        # b = (xTx)(dQd) - (dTd)(xQx)
        b = float(self.xTx_ * dMd - dTd_ * self.xMx)
        # c = (xTd)(xQx) - (xTx)(xQd)
        c = float(xTd_ * self.xMx - self.xTx_ * xMd_)

        coef = np.array([a, b, c])
        roots = np.roots(coef)

        if roots[0] < 0 and roots[1] < 0:
            return 0
        elif roots[0] < 0:
            return roots[1]
        elif roots[1] < 0:
            return roots[0]
        return np.min([roots])
    
    def wolfe_line_search(func, xk, pk, c1=1e-4, c2=0.9, amax=None, maxiter=10):
        
        gval = [None]
        gval_alpha = [None]
        
        def phi(alpha):
            return func.func_(xk + alpha * pk)
        
        def derphi(alpha):
            gval[0] = func.func_grad_(xk + alpha * pk)  # store for later use
            gval_alpha[0] = alpha
            return np.dot(gval[0].T, pk)
        
        def zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2):
            
            def cubicmin(a, fa, fpa, b, fb, c, fc):
                with np.errstate(divide='raise', over='raise', invalid='raise'):
                    try:
                        C = fpa
                        db = b - a
                        dc = c - a
                        denom = (db * dc) ** 2 * (db - dc)
                        d1 = np.empty((2, 2))
                        d1[0, 0] = dc ** 2
                        d1[0, 1] = -db ** 2
                        d1[1, 0] = -dc ** 3
                        d1[1, 1] = db ** 3
                        [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                                        fc - fa - C * dc]).flatten())
                        A /= denom
                        B /= denom
                        radical = B * B - 3 * A * C
                        xmin = a + (-B + np.sqrt(radical)) / (3 * A)
                    except ArithmeticError:
                        return None
                if not np.isfinite(xmin):
                    return None
                return xmin

            def quadmin(a, fa, fpa, b, fb):
                with np.errstate(divide='raise', over='raise', invalid='raise'):
                    try:
                        D = fa
                        C = fpa
                        db = b - a * 1.0
                        B = (fb - D - C * db) / (db * db)
                        xmin = a - C / (2.0 * B)
                    except ArithmeticError:
                        return None
                if not np.isfinite(xmin):
                    return None
                return xmin
            
            maxiter = 10
            i = 0
            delta1 = 0.2  # cubic interpolant check
            delta2 = 0.1  # quadratic interpolant check
            phi_rec = phi0
            a_rec = 0
            
            while True:   # Interpolate to find a trial step length between a_lo and a_hi 
                dalpha = a_hi - a_lo
                if dalpha < 0:
                    a, b = a_hi, a_lo
                else:
                    a, b = a_lo, a_hi

                if (i > 0):
                    cchk = delta1 * dalpha
                    a_j = cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec) # cubic interpolation
                if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):      # if the result is within delta * dalpha or outside of the interval bounded by a_lo or a_hi
                    qchk = delta2 * dalpha
                    a_j = quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)                  # quadratic interpolation
                    if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):                  # if the result is still too close
                        a_j = a_lo + 0.5*dalpha                                            # bisection

                # Check new value of a_j

                phi_aj = phi(a_j)
                if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_j
                    phi_hi = phi_aj
                else:
                    derphi_aj = derphi(a_j)
                    
                    if abs(derphi_aj) <= -c2*derphi0:
                        a_star = a_j
                        break
                        
                    if derphi_aj*(a_hi - a_lo) >= 0:
                        phi_rec = phi_hi
                        a_rec = a_hi
                        a_hi = a_lo
                        phi_hi = phi_lo
                    else:
                        phi_rec = phi_lo
                        a_rec = a_lo
                        
                    a_lo = a_j
                    phi_lo = phi_aj
                    derphi_lo = derphi_aj
                    
                i += 1
                
                if (i > maxiter):
                    # Failed to find a conforming step size
                    a_star = None
                    break
                    
            return a_star
            
        gfk = func.func_grad_(xk)
            
        alpha0 = 0
        phi0 = phi(alpha0)
        derphi0 = np.dot(gfk.T, pk)
        
        if amax is not None:
            alpha1 = min(1.0, amax)
        else:
            alpha1 = 1.0   

        phi_a1 = phi(alpha1)

        phi_a0 = phi0
        derphi_a0 = derphi0
        
        for i in range(maxiter):
            if amax is not None and alpha0 == amax:
                alpha_star = None
                phi_star = phi0
                # phi0 = old_phi0
                derphi_star = None

                if alpha1 == 0:
                    msg = 'Rounding errors prevent the line search from converging'
                else:
                    msg = "The line search algorithm could not find a solution " + \
                          "less than or equal to amax: %s" % amax

                warn(msg, LineSearchWarning)
                break

            not_first_iteration = i > 0
            if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or ((phi_a1 >= phi_a0) and not_first_iteration):
                alpha_star = zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0, c1, c2)
                break

            derphi_a1 = derphi(alpha1)
            if (abs(derphi_a1) <= -c2*derphi0):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

            if (derphi_a1 >= 0):
                alpha_star = zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0, c1, c2)
                break

            alpha2 = 2 * alpha1  # increase by factor of two on each iteration
            
            if amax is not None:
                alpha2 = min(alpha2, amax)
            
            # update everything 
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi(alpha1)
            derphi_a0 = derphi_a1
            
        else:
            # stopping test maxiter reached
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = None
            warn('The line search algorithm did not converge', LineSearchWarning)

        return alpha_star
    
    def armijo_line_search(self, func, xk, pk, c1=1e-4, alpha0=1, amin=0):
        
        gfk = self.grad_func_(xk)
        xk = np.atleast_1d(xk)

        def phi(alpha1):
            return func(xk + alpha1*pk)

        phi0 = phi(0.)

        derphi0 = np.dot(gfk, pk)

        phi_a0 = phi(alpha0)
        
        if phi_a0 <= phi0 + c1*alpha0*derphi0:
            return alpha0

        # Otherwise, compute the minimizer of a quadratic interpolant:

        alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
        phi_a1 = phi(alpha1)

        if (phi_a1 <= phi0 + c1*alpha1*derphi0):
            return alpha1

        while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
            factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
            a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
                alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
            a = a / factor
            b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
                alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
            b = b / factor

            alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
            phi_a2 = phi(alpha2)

            if (phi_a2 <= phi0 + c1*alpha2*derphi0):
                return alpha2

            if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
                alpha2 = alpha1 / 2.0

            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = phi_a2

        # Failed to find a suitable step length
        return None

    def init_x(self):
        return np.random.rand(self.dim, 1)
    
    line_search_methods = {'W' : wolfe_line_search, 'A' : armijo_line_search}