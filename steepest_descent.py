from sympy import Array
from sympy import derive_by_array


def f1(x1, x2):
    '''Example function with two variables.

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
    '''
    return x1 ** 3 + 25 * x2 ** 2


def steepest_descent(f, alpha, x0, max_iter, *params):
    x0 = Array(x0)
    grad = derive_by_array(f(*params), params)
    X = [x0]
    old_x = x0
    compute_x = lambda old_x: old_x - alpha * grad.xreplace(dict(zip(params, old_x)))
    new_x = old_x - alpha * grad.xreplace(dict(zip(params, old_x)))
    i = 1
    while new_x != old_x and i < max_iter:
        X.append(new_x)
        old_x = new_x
        new_x = compute_x(old_x)
        i += 1

    return X
