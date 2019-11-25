from __future__ import division
from math import log
import numpy as np
from collections import Counter
from scipy.stats import entropy
import math

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def info_gain(Ex, a, nan=True):
    """ Compute the information gain of an attribute a for given examples.
        Parameters
        ----------
        Ex : list of hashable
            A list of hashable objects (examples)
            corresponding to the given attributes a.
            I.e. a[i] <--> Ex[i].
        a : list of hashable
            A list of hashable objects (attributes)
            corresponding to the given examples Ex.
            I.e. a[i] <--> Ex[i].

        nan : boolean, default=True
            Boolean indicating how nan==nan should be evaluated.
            Default == True to avoid division by 0 errors.
        Returns
        -------
        result : float
            Information gain by knowing given attributes.
        """
    # Check whether examples and attributes have the same lengths.
    if len(Ex) != len(a):
        raise ValueError("Ex and a must be of the same size.")

    # Compute the entropy of examples
    H_Ex = entropy(list(Counter(Ex).values()))

    # If nan is True, replace all nan values in a by the string "__nan__"
    if nan:
        a = ['__nan__' if isinstance(x, float) and math.isnan(x) else x for x in a]

    # Compute the sum of all values v in a
    sum_v = 0
    for v in set(a):
        Ex_a_v = [x for x, t in zip(Ex, a) if t == v]
        sum_v += (len(Ex_a_v) / len(Ex)) *\
                 (entropy(list(Counter(Ex_a_v).values())))

    # Return result
    return H_Ex - sum_v


def intrinsic_value(Ex, a, nan=True):
    """ Compute the intrinsic value of an attribute a for given examples.
        Parameters
        ----------
        Ex : list of hashable
            A list of hashable objects (examples)
            corresponding to the given attributes a.
            I.e. a[i] <--> Ex[i].
        a : list of hashable
            A list of hashable objects (attributes)
            corresponding to the given examples Ex.
            I.e. a[i] <--> Ex[i].

        nan : boolean, default=True
            Boolean indicating how nan==nan should be evaluated.
            Default == True to avoid division by 0 errors.
        Returns
        -------
        result : float
            Intrinsic value of attribute a for samples Ex.
        """
    # Check whether examples and attributes have the same lengths.
    if len(Ex) != len(a):
        raise ValueError("Ex and a must be of the same size.")

    # If nan is True, replace all nan values in a by the string "__nan__"
    if nan:
        a = ['__nan__' if isinstance(x, float) and math.isnan(x) else x for x in a]

    # Compute the sum of all values v in a
    sum_v = 0
    for v in set(a):
        Ex_a_v = [x for x, t in zip(Ex, a) if t == v]
        sum_v += (len(Ex_a_v) / len(Ex)) * math.log(len(Ex_a_v) / len(Ex), 2)

    # Return result
    return -sum_v


def info_gain_ratio(Ex, a, nan=True):
    """ Compute the information gain ratio of an attribute a for given examples.
        Parameters
        ----------
        Ex : list of hashable
            A list of hashable objects (examples)
            corresponding to the given attributes a.
            I.e. a[i] <--> Ex[i].
        a : list of hashable
            A list of hashable objects (attributes)
            corresponding to the given examples Ex.
            I.e. a[i] <--> Ex[i].

        nan : boolean, default=True
            Boolean indicating how nan==nan should be evaluated.
            Default == True to avoid division by 0 errors.
        Returns
        -------
        result : float
            Information gain ratio by knowing given attributes.
            I.e. information gain normalised with intrinsic value calculation.
        """
    # Check whether examples and attributes have the same lengths.
    if len(Ex) != len(a):
        raise ValueError("Ex and a must be of the same size.")

    # Compute information gain ratio as IG/IV
    return info_gain(Ex, a, nan) / intrinsic_value(Ex, a, nan)
