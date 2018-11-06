import numpy as np
from sympy import Array
from sympy import derive_by_array


def steepest_descent(f, alpha, max_iter, threshold, X0, *func_vars):
    """The steepest descent algorithm for numerically finding the minimum value of a function,
    based on the gradient of that function. It uses the gradient function (or the scalar derivative,
    if the function is single-valued) to determine the direction in which a function is decreasing most rapidly.
    Each successive iteration of the algorithm moves along this direction for a specified step size,
    and recomputes the gradient to determine the new direction to travel.

    Args:
        f (function): function to minimize.
        alpha (float): learning rate.
        max_iter (int): the maximum number of iterations.
        threshold (float): whenever the computed "NEW POINT" [x1, x2, x3, ...] changes is
            less than threshold (sum of changes of all components), the function stops.
        X0 (array_like): an array like of initial point. For example if the function f is
            a two variable function, then initial point could be a tuple or list or array
            with length 2. e.g. X0 = [0, 0]
        *func_vars (sympy.Symbol): a list of function variables. the derivative of the function
            f will be computed with respect to these parameters.

    Returns:
        list: The history of X0, X1, X2, ..., xn. The first item is the initial point (X0)
            and the last one is the best point that minimizes the function f.

    Examples:
        >>> from sympy import symbols
        >>> def f1(x1, x2): return x1**2 + 25 * x2**2
        >>> x1, x2 = symbols('x1 x2')
        >>> alpha = 0.01
        >>> threshold = 0.001
        >>> X0 = [0.5, 0.5]
        >>> max_iter = 300
        >>> X = steepest_descent(f1, alpha, max_iter, threshold, X0, x1, x2)
        >>> print('First 5 points:\n', np.array(X[:5], dtype=np.float))
        First 5 points:
         [[0.5        0.5       ]
         [0.49       0.25      ]
         [0.4802     0.125     ]
         [0.470596   0.0625    ]
         [0.46118408 0.03125   ]]
        >>> print('Minimum point:\n', np.round(np.array(X[-1], dtype=np.float)))
        Minimum point:
         [0. 0.]
    """
    grad = derive_by_array(f(*func_vars), func_vars)
    X0 = Array(X0)  # X0 should be an array to support subtraction (old_X - alpha * grad(f, old_X))
    X = [X0]  # list of history
    old_X = X0
    new_X = old_X - alpha * grad.xreplace(dict(zip(func_vars, old_X)))

    i = 1
    while np.sum(np.abs(new_X - old_X) > threshold) and i < max_iter:
        X.append(new_X)
        old_X = new_X
        new_X = old_X - alpha * grad.xreplace(dict(zip(func_vars, old_X)))
        i += 1

    return X


def f1(x1, x2):
    """Example function with two variables.

    Args:
        x1 (number or sympy.Symbol): x1 could be an integer or float number or a sympy.Symbol.
        x2 (number or sympy.Symbol): x2 could be an integer or float number or a sympy.Symbol.

    Returns:
        A number, array or symbolic formula: this function computes and returns
            fX = x1 ** 2 + 25 * x2 ** 2

        if x1 and x2 are (int/float):
            returns the value of fX (a number)
        if x1 and x2 are numpy arrays with same shape:
            returns a numpy array
            (dont use arrays with different shapes if you don't know about numpy broadcasting)
        if x1 and/or x2 are sympy Symbol:
            returns a symbolic formula of fX with type sympy.core.add.Add

    Examples:
        >>> x1, x2 = 2, 1
        >>> print(f1(x1, x2))
        29
        >>>
        >>>
        >>> from sympy import symbols
        >>> x1, x2 = symbols('x1 x2')
        >>> f1X = f1(x1, x2)
        >>> print(f1X)
        x1**2 + 25*x2**2
        >>> print(type(f1X))
        <class 'sympy.core.add.Add'>
        >>> print(f1X.xreplace({x1:2, x2:1}))
        29
        >>>
        >>>
        >>> x1 = np.arange(4)
        >>> x2 = np.arange(4)
        >>> print(f2(x1, x2))
        [  0  26 104 234]
    """
    fX = x1 ** 2 + 25 * x2 ** 2
    return fX


def f2(x1, x2):
    """Example function with two variables.

    Args:
        x1 (number or sympy.Symbol): x1 could be an integer or float number or a sympy.Symbol.
        x2 (number or sympy.Symbol): x2 could be an integer or float number or a sympy.Symbol.

    Returns:
        A number, array or symbolic formula: this function computes and returns
            fX = x1 ** 2 + 2 * x1 * x2 + 2 * x2 ** 2 + x1

        if x1 and x2 are (int/float):
            returns the value of fX (a number)
        if x1 and x2 are numpy arrays with same shape:
            returns a numpy array
            (dont use arrays with different shapes if you don't know about numpy broadcasting)
        if x1 and/or x2 are sympy Symbol:
            returns a symbolic formula of fX with type sympy.core.add.Add

    Examples:
        >>> x1, x2 = 2, 1
        >>> print(f2(x1, x2))
        12
        >>>
        >>>
        >>> from sympy import symbols
        >>> x1, x2 = symbols('x1 x2')
        >>> f2X = f2(x1, x2)
        >>> print(f2X)
        x1**2 + 2*x1*x2 + x1 + 2*x2**2
        >>> print(type(f2X))
        <class 'sympy.core.add.Add'>
        >>> print(f2X.xreplace({x1:2, x2:1}))
        12
        >>>
        >>>
        >>> x1 = np.arange(4)
        >>> x2 = np.arange(4)
        >>> print(f2(x1, x2))
        [ 0  6 22 48]
    """
    fX = x1 ** 2 + 2 * x1 * x2 + 2 * x2 ** 2 + x1
    return fX
