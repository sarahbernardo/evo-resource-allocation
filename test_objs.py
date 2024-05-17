from sorting import overallocation, conflicts, undersupport, unwilling, unpreferred
import pytest
import pandas as pd
import numpy as np


test1 = pd.read_csv("tests/test1.csv", header=None).to_numpy()
test2 = pd.read_csv("tests/test2.csv", header=None).to_numpy()
test3 = pd.read_csv("tests/test3.csv", header=None).to_numpy()


def test_overallocation():
    zeros = np.zeros((43, 17))

    assert overallocation(zeros) == 0, 'Overallocation value is wrong'
    assert overallocation(test1) == 37, 'Overallocation value is wrong'
    assert overallocation(test2) == 41, 'Overallocation value is wrong'
    assert overallocation(test3) == 23, 'Overallocation value is wrong'


def test_conflicts():
    assert conflicts(test1) == 8, 'Conflict value is wrong'
    assert conflicts(test2) == 5, 'Conflict value is wrong'
    assert conflicts(test3) == 2, 'Conflict value is wrong'


def test_undersupport():
    assert undersupport(test1) == 1, 'undersupport value is wrong'
    assert undersupport(test2) == 0, 'undersupport value is wrong'
    assert undersupport(test3) == 7, 'undersupport value is wrong'


def test_unwilling():
    assert unwilling(test1) == 53, 'Value of tas unwilling'
    assert unwilling(test2) == 58, 'Value of tas unwilling'
    assert unwilling(test3) == 43, 'Value of tas unwilling'


def test_unpreferred():
    assert unpreferred(test1) == 15, 'Value of tas unpreferred'
    assert unpreferred(test2) == 19, 'Value of tas unpreferred'
    assert unpreferred(test3) == 10, 'Value of tas unpreferred'
