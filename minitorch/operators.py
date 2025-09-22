"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Product of x and y
    """
    return x * y


def id(x: float) -> float:
    """
    Return the input unchanged.

    Args:
        x: Input number

    Returns:
        Same as input x
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Sum of x and y
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate a number.

    Args:
        x: Input number

    Returns:
        Negated value of x
    """
    return -x

def lt(x: float, y: float) -> bool:
    """
    Check if one number is less than another.

    Args:
        x: First number
        y: Second number

    Returns:
        True if x < y, False otherwise
    """
    return x < y


def eq(x: float, y: float) -> bool:
    """
    Check if two numbers are equal.

    Args:
        x: First number
        y: Second number

    Returns:
        True if x == y, False otherwise
    """
    return x == y


def max(x: float, y: float) -> float:
    """
    Return the larger of two numbers.

    Args:
        x: First number
        y: Second number

    Returns:
        Maximum of x and y
    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """
    Check if two numbers are close in value.

    Args:
        x: First number
        y: Second number

    Returns:
        True if |x - y| < 1e-2, False otherwise
    """
    return abs(x - y) < 1e-2

def sigmoid(x: float) -> float:
    """
    Calculate the sigmoid function.

    Uses stable computation: 
    f(x) = 1.0 / (1.0 + exp(-x)) if x >= 0 else exp(x) / (1.0 + exp(x))

    Args:
        x: Input number

    Returns:
        Sigmoid of x
    """
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """
    Apply the ReLU activation function.

    Args:
        x: Input number

    Returns:
        max(0, x)
    """
    return max(0.0, x)


def log(x: float) -> float:
    """
    Calculate the natural logarithm.

    Args:
        x: Input number (must be positive)

    Returns:
        Natural logarithm of x
    """
    import math
    return math.log(x)


def exp(x: float) -> float:
    """
    Calculate the exponential function.

    Args:
        x: Input number

    Returns:
        e raised to the power of x
    """
    import math
    return math.exp(x)


def inv(x: float) -> float:
    """
    Calculate the reciprocal.

    Args:
        x: Input number (cannot be zero)

    Returns:
        Reciprocal of x: 1 / x
    """
    return 1.0 / x

def log_back(x: float, d: float) -> float:
    """
    Compute the derivative of log times a second arg.

    Args:
        x: Input to log function (must be positive)
        d: Second argument for multiplication

    Returns:
        Derivative of log(x) * d: (1 / x) * d
    """
    return (1.0 / x) * d


def inv_back(x: float, d: float) -> float:
    """
    Compute the derivative of reciprocal times a second arg.

    Args:
        x: Input to reciprocal function (cannot be zero)
        d: Second argument for multiplication

    Returns:
        Derivative of inv(x) * d: (-1 / x**2) * d
    """
    return (-1.0 / (x * x)) * d


def relu_back(x: float, d: float) -> float:
    """
    Compute the derivative of ReLU times a second arg.

    Args:
        x: Input to ReLU function
        d: Second argument for multiplication

    Returns:
        Derivative of relu(x) * d: d if x > 0, else 0
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """
    Apply a function to each element of a list.

    Args:
        fn: Function to apply
        ls: List of numbers

    Returns:
        List with fn applied to each element
    """
    return [fn(x) for x in ls]


def zipWith(fn: Callable[[float, float], float], 
            ls1: Iterable[float], 
            ls2: Iterable[float]) -> Iterable[float]:
    """
    Combine two lists element-wise using a function.

    Args:
        fn: Function to combine elements
        ls1: First list of numbers
        ls2: Second list of numbers

    Returns:
        List of fn applied to each pair of elements
    """
    return [fn(a, b) for a, b in zip(ls1, ls2)]


def reduce(fn: Callable[[float, float], float], 
           ls: Iterable[float], 
           initial: float = 0.0) -> float:
    """
    Reduce a list using a function.

    Args:
        fn: Function to combine elements
        ls: List of numbers
        initial: Initial value

    Returns:
        Result of reducing the list with fn
    """
    result = initial
    for x in ls:
        result = fn(result, x)
    return result

def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate a list of numbers.

    Args:
        ls: List of numbers

    Returns:
        List with each element negated
    """
    return map(neg, ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add two lists of numbers element-wise.

    Args:
        ls1: First list of numbers
        ls2: Second list of numbers

    Returns:
        List of element-wise sums
    """
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """
    Sum a list of numbers.

    Args:
        ls: List of numbers

    Returns:
        Sum of all elements
    """
    return reduce(add, ls, 0.0)


def prod(ls: Iterable[float]) -> float:
    """
    Take the product of a list of numbers.

    Args:
        ls: List of numbers

    Returns:
        Product of all elements
    """
    return reduce(mul, ls, 1.0)
