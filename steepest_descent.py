from sympy import Array
from sympy import derive_by_array


def f1(x1, x2):
    """Example function with two variables.

    Args:
        x1 (number or sympy.Symbol): x1 could be an integer or float number or a sympy.Symbol.
        x2 (number or sympy.Symbol): x2 could be an integer or float number or a sympy.Symbol.

    Returns:
        number or symbolic formula: if the two parameters are (int/float), it returns the value of
            of x1 ** 3 + 25 * x2 ** 2 otherwise it returns a symbolic formula of type sympy.core.add.Add.

    Examples:
        >>> x1, x2 = 2, 1
        >>> print(f1(x1, x2))
        33
        >>>
        >>>
        >>> from sympy import symbols
        >>> x1, x2 = symbols('x1 x2')
        >>> f1X = f1(x1, x2)
        >>> print(f1X)
        x1**3 + 25*x2**2
        >>> print(type(f1X))
        <class 'sympy.core.add.Add'>
        >>> print(f1X.xreplace({x1:2, x2:1}))
        33
    """
    return x1 ** 3 + 25 * x2 ** 2


def steepest_descent(f, alpha, max_iter, x0, *args):
    """The steepest descent algorithm for numerically finding the minimum value of a function,
    based on the gradient of that function. It uses the gradient function (or the scalar derivative,
    if the function is single-valued) to determine the direction in which a function is decreasing most rapidly.
    Each successive iteration of the algorithm moves along this direction for a specified step size,
    and recomputes the gradient to determine the new direction to travel.

    Args:
        f (function): function to minimize.
        alpha (float): learning rate.
        max_iter (int): the maximum number of iterations.
        x0 (array_like): an array like of initial point. For example if the function f is
            a two variable function, then initial point could be a tuple or list or array
            with length 2. e.g. x0 = [0, 0]
        *args (sympy.Symbol): a list of function variables. the derivative of the function
            f will be computed with respect to these parameters.

    Returns:
        list: The history of x0, x1, x2, ..., xn. The first item is the initial point (x0)
            and the last one is the best point that minimizes the function f.

    Examples:
        >>> from sympy import symbols
        >>> def f1(x1, x2): return x1**2 + 25 * x2**2
        >>> x1, x2 = symbols('x1 x2')
        >>> alpha = 0.01
        >>> x0 = [0.5, 0.5]
        >>> max_iter = 300
        >>> X = steepest_descent(f1, alpha, max_iter, x0, x1, x2)
        >>> print('First 5 points:\n', np.array(X[:5], dtype=np.float))
        First 5 points:
         [[0.5        0.5       ]
         [0.49       0.25      ]
         [0.4802     0.125     ]
         [0.470596   0.0625    ]
         [0.46118408 0.03125   ]]
        >>> print('Minimum point:\n', np.array(X[-1], dtype=np.float))
        Minimum point:
         [2.77580750e-006 2.40991987e-181]
    """
    grad = derive_by_array(f(*args), args)
    x0 = Array(x0)
    X = [x0]
    old_x = x0
    new_x = old_x - alpha * grad.xreplace(dict(zip(args, old_x)))
    i = 1
    while new_x != old_x and i < max_iter:
        X.append(new_x)
        old_x = new_x
        new_x = old_x - alpha * grad.xreplace(dict(zip(args, old_x)))
        i += 1

    return X
