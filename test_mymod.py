#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import mymod


def test_add_two_int():
    a = 5
    b = 7
    #Check for types to make sure that the test is ok
    assert type(a) is int
    assert type(b) is int

    #This is actually what we want to test
    assert mymod.add(a,b) == 12

def test_add_int_float():
    a = 723
    b = 4.98
    assert type(a) is int
    assert type(b) is float
    assert mymod.add(a,b) == pytest.approx(727.98)

def test_add_two_float():
    a = 309.123
    b = 4.98
    assert type(a) is float
    assert type(b) is float
    assert mymod.add(a,b) == pytest.approx(314.103)

def test_add_two_arrays():
    a = np.array((3,4,5))
    b = np.array((9,1,3))
    c = mymod.add(a,b) 
    for aa,bb,cc in zip(a,b,c):
        assert cc == aa+bb

def test_throws_on_string():
    with pytest.raises(TypeError):
        mymod.add('a', 7)

def test_throws_when_dims_disagree():
    with pytest.raises(ValueError):
        mymod.add(3, (4,5))

def test_throws_when_dims_disagree():
    a = np.arange(10).reshape(2,5)
    b = np.arange(10).reshape(5,2)
    with pytest.raises(ValueError):
        mymod.add(a, b)