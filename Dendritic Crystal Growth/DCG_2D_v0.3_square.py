# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:23:59 2020

@author: sam10
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:37:24 2020

@author: sam10
"""

###############################################################################
# Dendritic Crystal Growth, in 2-Dimensions, Using link                       #
# https://www.ctcms.nist.gov/fipy/examples/phase/generated/examples.phase.anisotropy.html#module-examples.phase.anisotropy #
# Sam Fisco                                                                   #
# 4/26/2020                                                                   #
###############################################################################

#%% Module Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fipy import Variable, CellVariable, Grid2D, TransientTerm, DiffusionTerm
from fipy import ImplicitSourceTerm, Viewer, Matplotlib2DGridViewer
from fipy import MultiViewer, MatplotlibViewer
from fipy.tools import numerix
#%% Initial grid set up
if __name__ == '__main__':
    NX = NY = 300
else:
        NX = NY = 30
X_SIZE = Y_SIZE = 9
DX = X_SIZE / NX
DY = Y_SIZE / NY
mesh = Grid2D(dx = DX, dy = DY, nx = NX, ny = NY)
D_time = 0.0005
#%% Constants and Other Parameters
TAU = 0.0003 # Tau CONSTANT
ALPHA = 0.9 # Constant
GAMMA = 10.0 # Nondimensionalized 
A_noise = 0.01 # Amplitude of the Noise
EPSILON_BAR = 0.01 # Compute Param, Thickness of layer at interface
#%% Anisotropy Functions
def m_temp(alpha, gamma, temp_eq, temp):
    """
    

    Parameters
    ----------
    alpha : Constant Value, Positive
        Positive Constant.
    gamma : Nondimensional Constant
        Nondimensional Constant.
    temp_eq : Temperature Constant
        Equilibrium Temperature.
    temp : Temperature Variable
        Current Temperature.

    Returns
    -------
    m(T)
        Thermodynamic Driving Force

    """
    return (alpha / np.pi) * np.arctan(gamma * (temp_eq - temp))

def sigma_ani(strength_delta, mode_num, theta_0, theta):
    """
    

    Parameters
    ----------
    strength_delta : Variable
        Strength of Anisotropy, lower case sigma
    mode_num : Constant
        Mode Number of Anisotropy, j
    theta_0 : Constant
        Initial Angle
    theta : Variable
        Current Angle

    Returns
    -------
    TYPE
        Anisotropy as a function of theta .

    """
    return 1 + strength_delta * np.cos(mode_num * (theta - theta_0))

#%% Temperature Derivative, NO FLUX
def nabla_T(temp_i, dx, dy):
    temp_f = temp_i.copy()
    temp_xx = (temp_i[2:, 1:-1] - 2*temp_i[1:-1, 1:-1] + temp_i[:-2, 1:-1])/ (dx**2)  
    temp_yL = (2 * temp_i[2:, 1:-1] - 2*temp_i[1:-1, 1:-1]) / (dx**2)
    temp_yR = (2 * temp_i[:-2, 1:-1] - 2*temp_i[1:-1, 1:-1]) / (dx**2)
    temp_yy = (temp_i[1:-1, 2:] - 2*temp_i[1:-1, 1:-1] + temp_i[:2, 1:-1]) / (dy**2)
    temp_xL = (2 * temp_i[1:-1, 2:] - 2 * temp_i[1:-1, 1:-1]) / (dy**2)
    temp_xR = (2 * temp_i[1:-1, :-2] - 2 * temp_i[1:-1, 1:-1]) / (dy**2)
    temp_f[1:-1, 1:-1] = temp_xx + temp_yy
    temp_f[-1, 1:-1] = temp_xR
    temp_f[0, 1:-1] = temp_xL
    temp_f[1:-1, 0] = temp_yL
    return temp_f

#%% Phase Field Derivative










#%% Phase Field Variable
phase = CellVariable(name= r'$\phi$', mesh=mesh, hasOld = True)
D_temp = CellVariable(name= r'$\Delta T$', mesh=mesh, hasOld = True)
CHANGE_TEMP = 2.25
#%% Heat Equation
heatEQ = (TransientTerm() == DiffusionTerm(CHANGE_TEMP) 
          + (phase - phase.old) / D_time)

#%% Parameter Setup
# ALPHA = 0.015 # Alpha CONSTANT 
C_ani = 0.02 # Component of (D) the anisotropic diffusion tensor in 2D
N = 6. # Symmetry
THETA  = np.pi / 8 # Orientation

psi = THETA + np.arctan2(phase.faceGrad[1], phase.faceGrad[0])

PHI = np.tan(N * psi / 2)
PHI_SQ = PHI ** 2
BETA = (1. - PHI_SQ) / (1. + PHI_SQ)
D_BETA_D_PSI = -N * 2 * PHI / (1 + PHI_SQ)
D_DIAG = (1 + C_ani * BETA)
D_OFF = C_ani * D_BETA_D_PSI
I0 = Variable(value = ((1, 0), (0, 1)))
I1 = Variable(value = ((0, -1), (1, 0)))
DIF_COEF = ALPHA ** 2 * (1. + C_ani * BETA) * (D_DIAG * I0 + D_OFF * I1)

TAU = 0.0003
KAPPA_1 = 0.9
KAPPA_2 = 20.

phase_EQ = (TransientTerm(TAU) == DiffusionTerm(DIF_COEF) 
            + ImplicitSourceTerm((phase - 0.5 - KAPPA_1 / np.pi
                                  * np.arctan(KAPPA_2 * D_temp)) * (1 - phase)))

#%% Circular Solidified Region in the Center
radius = DX * 5.0
C_circ = (NX * DX / 2, NY * DY / 2)
X, Y = mesh.cellCenters
phase.setValue(1., where=((X - C_circ[0])**2 + (Y - C_circ[1])**2) < radius**2)
D_temp.setValue(-0.5)

#%% Plotting

if __name__ == "__main__": 
    try: 
        import pylab
        class DendriteViewer(Matplotlib2DGridViewer):
            def __init__(self, phase, D_temp, title = None,
                         limits = {}, **kwlimits):
                self.phase = phase
                self.contour = None
                Matplotlib2DGridViewer.__init__(self, vars=(D_temp,),
                                                title=title, 
                                                cmap=pylab.cm.hot,
                                                limits=limits, **kwlimits)
            def _plot(self):
                Matplotlib2DGridViewer._plot(self)
                    
                if self.contour is not None:
                    for C_ani in self.contour.collections:
                        C_ani.remove()
                            
                mesh = self.phase.mesh
                shape = mesh.shape
                x, y = mesh.cellCenters
                z = self.phase.value
                x, y, z = [a.reshape(shape, order='F') for a in (x, y, z)]
                    
                # self.contour = self.axes.contour(X, Y, Z, (0.5,))
                    
        viewer = DendriteViewer(phase=phase, D_temp=D_temp, 
                                title = r'%s & %s' % (phase.name, D_temp.name),
                                datamin=-0.1, datamax=0.05)
    except ImportError:
        viewer = DendriteViewer(viewers=(Viewer(vars=phase),
                                      Viewer(vars=D_temp,
                                             datamin=-0.5,
                                             datamax=0.5)))
if __name__ == "__main__":
    steps = 500
else: 
    steps = 10
from builtins import range
for i in range(steps):
    phase.updateOld()
    D_temp.updateOld()
    phase_EQ.solve(phase, dt = D_time)
    heatEQ.solve(D_temp, dt = D_time)
    if __name__ == "__main__" and (i % 10 == 0):
        viewer.plot()
        
























