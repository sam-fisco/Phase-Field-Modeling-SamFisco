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
#%% Functions
################################################################################
# do_Timestep: Function to modify the Matrix, representing the nodes of the    #
# material, by slicing and using the FTCS FDM                                  #
#   Takes Original Matrices (temp_0 & temp_rc), Modifies temp_rc by using values from temp_0,#
#   and finally takes the final temp_0 becomes a copy of the modified temp_rc  #
#                                                                              # 
#   input:                                                                     #
#       Initial Zero Matrices that have the boundary conditions added          #
#       temp_0: Modified Matrix for Reiteration                                #
#       temp_rc: Modified Matrix for Reiteration                               #
#                                                                              #
#   output:                                                                    #
#       temp_0: Modified Matrix with Gradient                                  #
#       temp_rc: Modfied Matrix with Gradient                                  #
################################################################################
# To explain the function: 1st the left and right boundaries are set
# 2nd: The center is sliced using the FTCS FDM 
# 3rd: The upper and lower boundaries are sliced 
# 4th: The combination of the slices are impossed on temp_rc
# 5th: temp_0 becomes a copy of temp_rc
# 6th: resets the boundaries of temp_rc
# 7th: end of function returing our matrices for iteration
def do_timestep(temp_0, temp_rc):
    temp_0[:,0] = T_HOT
    temp_0[:, -1] = T_COOL
    temp_xx = (temp_0[2:, 1:-1] - 2*temp_0[1:-1,1:-1] + temp_0[:-2, 1:-1]) / DX2
    temp_yy = (temp_0[1:-1, 2:] - 2*temp_0[1:-1,1:-1] + temp_0[1:-1, :-2]) / DY2
    temp_rc[1:-1, 1:-1] = temp_0[1:-1, 1:-1] + D * D_TIME * (temp_xx + temp_yy)
    temp_xx_u = (temp_0[0, 2:] - 2 * temp_0[0, 1:-1] + temp_0[0, :-2]) / DX2
    temp_yy_u = (temp_0[1, 1:-1] - temp_0[0, 1:-1]) / DY2 + (temp_0[0,1:-1]) / DY
    temp_rc[0,1:-1] = temp_0[0, 1:-1] + D * D_TIME * (temp_xx_u + temp_yy_u)
    temp_xx_l = (temp_0[-1, 2:] - 2 * temp_0[-1, 1:-1] + temp_0[-1, :-2]) / DX2
    temp_yy_l = (temp_0[-2, 1:-1] - temp_0[-1, 1:-1]) / DY2 + (temp_0[-1,1:-1]) / DY 
    temp_rc[-1, 1:-1] = temp_0[-1,1:-1] + D * D_TIME * (temp_xx_l + temp_yy_l)
    temp_0 = temp_rc.copy()
    temp_0[:,0] = T_HOT
    temp_0[:, -1] = T_COOL
    return temp_0, temp_rc

################################################################################
#%% Constants
################################################################################
x = y = 1                         # x and y dimensions

DX = DY = 0.02                    # intervals in x-, y- directions

D = 1.                            # Diffusion Coefficient

T_COOL, T_HOT = 0, 1              # Right and Left Dirichlet BC

NX, NY = int(x/DX), int(y/DY)     # Number of grid points in x and y directions

DX2, DY2 = DX*DX, DY*DY           # Grid Spacing squared

D_TIME = DX2 * DY2 / (2 * D * (DX2 + DY2))  # Value of the Time Step

N_STEPS = 10**4                   # Number of Timesteps
################################################################################
#%% Initial Variable Matrices
################################################################################
temp_0 = np.zeros((NX, NY))       # Initial Zero Matrix

temp_rc = np.zeros((NX, NY))      # Initial Zero Matrix
################################################################################
#%%
################################################################################
# Start of Main Program Now:                                                   #
################################################################################
# This section primarily is for visuals 
# The graph and temperature scale are the majority while the for loop is just 
# to iterate the function above over the total number of time steps
mfig = [500, 1000, 5000, 9000]
fignum = 0
fig = plt.figure()
T_HOT1 = 1.5
for m in range(N_STEPS):
    temp_0, temp_rc = do_timestep(temp_0, temp_rc)
    print('m:', m, '\ntemp_0',temp_0)
    if m in mfig:
        fignum += 1
        print(m, fignum)
        ax = fig.add_subplot(220 + fignum)
        im = ax.imshow(temp_rc.copy(), cmap=plt.get_cmap('hot'), vmin=T_COOL,vmax=T_HOT1)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*D_TIME*1000))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)
plt.show()