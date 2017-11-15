#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:11:10 2017

@author: l_frojdh
"""
import sys
import _example


a = 5
b = 7

res = _example.add(a,b)
print(res)
if res == 12:
    print("Good")
    sys.exit(0)
else:
    sys.exit(1)
