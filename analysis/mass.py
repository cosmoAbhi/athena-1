#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 13:39:55 2022

@author: ajd6518
"""

# Python standard modules
import argparse
import warnings

# Other Python modules
import numpy as np

# Athena++ modules
import athena_read

def vol_func(rm, rp, thetam, thetap, phim, phip):
    return ((rp**3-rm**3) * abs(np.cos(thetam)-np.cos(thetap))
                                * (phip-phim))

mass = 0.0
data = athena_read.athdf('/Users/ajd6518/Documents/Research/athena_tests/local_tests_new/neoneo_latest/gr_tov.prim.00000.athdf', raw=True)
mb = data['NumMeshBlocks']
x1f = data['x1f']
x2f = data['x2f']
x3f = data['x3f']
nghost = 4
x1l = len(data['x1f'][0])
x1range = x1l - nghost - 1
x2l = len(data['x2f'][0])
x2range = x2l - nghost - 1
x3l = len(data['x3f'][0])
x3range = x3l - nghost - 1
rho = data['rho']

for i in range(0, mb):
    for j in range(nghost, x3range):
        for k in range(nghost, x2range):
            for l in range(nghost, x1range):
                vol = vol_func(x1f[i][l], x1f[i][l+1], x2f[i][k], x2f[i][k+1], x3f[i][j], x3f[i][j+1])
                mass +=rho[i][j][k][l]*vol
                
print("Mass of the star is %d", mass)