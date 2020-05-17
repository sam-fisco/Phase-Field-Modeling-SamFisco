# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 18:29:14 2020

@author: sam10
"""
################################################################################
# 2D Heat Equation Solution 
# Foward in Time, Central in Space Finite Difference Method (FTCS FDM)
# Source: https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
# Script that solves the 2D Heat Equation using FTCS FDM
# Created by Samuel Fisco
################################################################################
#%% Importing Packages & Modules
import numpy as np
import matplotlib.pyplot as plt
import time
time0 = time.time()
#%% Functions
################################################################################
# do_Timestep: Function to modify the Matrix, representing the nodes of the    #
# material, by slicing and using the FTCS FDM                                  #
#   Takes Original Matrices (T0 & T_rc), Modifies T_rc by using values from T0,#
#   and finally takes the final T0 becomes a copy of the modified T_rc         #
#                                                                              # 
#   input:                                                                     #
#       Initial Zero Matrices that have the boundary conditions added          #
#       T0: Modified Matrix for Reiteration                                    #
#       T_rc: Modified Matrix for Reiteration                                  #
#                                                                              #
#   output:                                                                    #
#       T0: Modified Matrix with Gradient                                      #
#       T_rc: Modfied Matrix with Gradient                                     #
################################################################################

def do_timestep(T0, T_rc):
    T0[:,0] = T_HOT
    T0[:, -1] = T_COOL
    Txx = (T0[2:, 1:-1] - 2*T0[1:-1,1:-1] + T0[:-2, 1:-1]) / dx2
    Tyy = (T0[1:-1, 2:] - 2*T0[1:-1,1:-1] + T0[1:-1, :-2]) / dy2
    T_rc[1:-1, 1:-1] = T0[1:-1, 1:-1] + D * dt * (Txx + Tyy)
    Txx_u = (T0[0, 2:] - 2 * T0[0, 1:-1] + T0[0, :-2]) / dx2
    Tyy_u = (T0[1, 1:-1] - T0[0, 1:-1]) / dy2 + (T0[0,1:-1]) / dy
    T_rc[0,1:-1] = T0[0, 1:-1] + D * dt * (Txx_u + Tyy_u)
    Txx_l = (T0[-1, 2:] - 2 * T0[-1, 1:-1] + T0[-1, :-2]) / dx2
    Tyy_l = (T0[-2, 1:-1] - T0[-1, 1:-1]) / dy2 + (T0[-1,1:-1]) / dy 
    T_rc[-1, 1:-1] = T0[-1,1:-1] + D * dt * (Txx_l + Tyy_l)
    T0 = T_rc.copy()
    T0[:,0] = T_HOT
    T0[:, -1] = T_COOL
    return T0, T_rc

#%% Constants
x = y = 1
# intervals in x-, y- directions, mm
dx = dy = 0.02
# Thermal diffusivity of steel, mm2.s-1
D = 1.

T_COOL, T_HOT = 0, 1

nx, ny = int(x/dx), int(y/dy)

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2)) 

T0 = np.zeros((nx, ny))
T_rc = np.zeros((nx, ny))
# Number of timesteps
nsteps = 10**4
# Output 4 figures at these timesteps
mfig = [500, 1000, 5000, 9000]
fignum = 0
fig = plt.figure()
T_HOT1 = 1.5 # Set for the graph to add more depth to the color
#################################################################################
# The below section is primarily to plot. The for loop is to itterate over the  #
# number of timesteps. Following the for loop, the if statement is simply asking#
# if the timestep is one of four values from mfig, then to take that matrix and #
# use the values at that time to plot one of the graphs                         #
################################################################################# 
for m in range(nsteps):
    T0, T_rc = do_timestep(T0, T_rc)
    print('m:', m, '\nT0',T0)
    if m in mfig:
        fignum += 1
        print(m, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(T_rc.copy(), cmap=plt.get_cmap('hot'), vmin=T_COOL,vmax=T_HOT1)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*dt*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()
time1 = time.time()
print(f'Total Time To Run = {time1 - time0}')