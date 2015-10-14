import math
import numpy as np
from openmdao_sellar import openmdao_sellar_converged
"""
Manual calculation of total derivatives for the coupled Sellar problem:

http://openmdao.readthedocs.org/en/1.1.1/usr-guide/tutorials/sellar.html
"""

# variable values at a converged state
z1= 1.97763876909
z2= 0.0
x1= 0.0
y1= 3.16000036155
y2= 3.75527869722

"""
Objective:
F1 = x1**2 + z2 + y1 + math.exp(-y2)

Constraints:
F2 = 3.16 < y1   -->   3.16 - y1
F3 = y2 < 24     -->   y2 - 24
"""

# Derivative of objectives and constraints w.r.t design vars
dF1_dz1 = 0.0
dF1_dz2 = 1.0
dF1_dx1 = 2*x1

dF2_dz1 = 0.
dF2_dz2 = 0.
dF2_dx1 = 0.

dF3_dz1 = 0.
dF3_dz2 = 0.
dF3_dx1 = 0.

# Derivative of objectives and constraints w.r.t states
dF1_dy1 = 1.0
dF1_dy2 = -math.exp(-y2)

dF2_dy1 = -1.
dF2_dy2 = 0.

dF3_dy1 = 0.
dF3_dy2 = 1.0

"""
Residuals of coupling conditions
R1 = (z1**2 + z2 + x1 - 0.2*y2) - y1
R2 = y1**(.5) + z1 + z2 - y2
"""

# derivatives of residuals w.r.t states
dR1_dy1 = -1.
dR1_dy2 = -0.2
dR2_dy1 = 0.5*y1**(-0.5)
dR2_dy2 = -1.

# derivative of residuals w.r.t design vars
dR1_z1 = 2*z1
dR1_z2 = 1.
dR1_x1 = 1.
dR2_z1 = 1.
dR2_z2 = 1.
dR2_x1 = 0.


# -------------
"""
Total derivs = dF/dX - dF/dy[(dR/dy)^-1 (dR/dx)] 
"""
# Jacobian of residuals w.r.t design vars
dR_dx = np.array([[dR1_z1, dR1_z2, dR1_x1],
                  [dR2_z1, dR1_z2, dR2_x1]])

# Jacobian of residuals w.r.t. state vars
dR_dy = np.array([[dR1_dy1, dR1_dy2],
                  [dR2_dy1, dR2_dy2]])

# Jacobian of objective & constraints w.r.t design vars
dF_dx = np.array([[dF1_dz1, dF1_dz2, dF1_dx1],
                  [dF2_dz1, dF2_dz2, dF2_dx1],
                  [dF3_dz1, dF3_dz2, dF3_dx1]])

# Jacobian of objective & constraints w.r.t. state vars
dF_dy = np.array([[dF1_dy1, dF1_dy2],
                  [dF2_dy1, dF2_dy2],
                  [dF3_dy1, dF3_dy2]])


print "forward (one design var at a time):"
print "dF_dz1:", dF_dx[:,0] - dF_dy.dot(np.linalg.solve(dR_dy, dR_dx[:,0]))
print "dF_dz2:", dF_dx[:,1] - dF_dy.dot(np.linalg.solve(dR_dy, dR_dx[:,1]))
print "dF_dx1:", dF_dx[:,2] - dF_dy.dot(np.linalg.solve(dR_dy, dR_dx[:,2]))

print
print "adjoint (one objective/constraint at a time):"
print "dF1_dx: ", dF_dx[0] - np.linalg.solve(dR_dy.T, dF_dy.T[:,0]).T.dot(dR_dx)
print "dF2_dx: ", dF_dx[1] - np.linalg.solve(dR_dy.T, dF_dy.T[:,1]).T.dot(dR_dx)
print "dF3_dx: ", dF_dx[2] - np.linalg.solve(dR_dy.T, dF_dy.T[:,2]).T.dot(dR_dx)
print
print "forward mode, one solve (for LAPACK _gesv):"
print dF_dx - dF_dy.dot(np.linalg.solve(dR_dy, dR_dx))
print
print "adjoint mode, one solve (FOR LAPACK _gesv):"
print dF_dx - dR_dx.T.dot(np.linalg.solve(dR_dy.T, dF_dy.T)).T

# compare to OpenMDAO total derivs
top = openmdao_sellar_converged()
derivs = top.check_total_derivatives(out_stream=None)
print "check against OpenMDAO fwd:"
print "dF_dz1:", derivs[('obj', 'z')]['J_rev'][0][0], derivs[('con1', 'z')]['J_rev'][0][0], derivs[('con2', 'z')]['J_rev'][0][0]
print "dF_dz2:", derivs[('obj', 'z')]['J_rev'][0][1], derivs[('con1', 'z')]['J_rev'][0][1], derivs[('con2', 'z')]['J_rev'][0][1]
print "dF_dx1:", derivs[('obj', 'x')]['J_rev'][0][0], derivs[('con1', 'x')]['J_rev'][0][0], derivs[('con2', 'x')]['J_rev'][0][0]


