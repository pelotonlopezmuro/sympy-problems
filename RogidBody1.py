import sympy as sp
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t')    # time

g = sp.symbols('g')    # gravity
m = sp.symbols('m')    # mass
l = sp.symbols('l')    # mass center

psi, theta, phi = me.dynamicsymbols('psi, theta, phi')      # 3-2-1 euler angles
k_psi, k_theta, k_phi = sp.symbols('k_psi, k_theta, k_phi') # springs
b_psi, b_theta, b_phi = sp.symbols('b_psi, b_theta, b_phi') # dampers

E = me.ReferenceFrame('E')  # earth
D = me.ReferenceFrame('D')  # body-z
C = me.ReferenceFrame('C')  # body-z-y
B = me.ReferenceFrame('B')  # body-z-y-x
D.orient(E, 'Axis', (psi,  E.z))
C.orient(D, 'Axis', (theta,D.y))
B.orient(C, 'Axis', (phi,  C.x))

o  = me.Point('o')      # origin point
mc = me.Point('mc')     # mass point
o.set_vel(E, 0)
mc.set_pos(o, l*B.z)

Ixx, Iyy, Izz = sp.symbols('Ixx, Iyy, Izz') # inertias
I = me.inertia(B, Ixx, Iyy, Izz)

# Applied Forces:
FL = []
# Due to the stiffness
FL.append((D, -k_psi  *psi  *D.z))
FL.append((C, -k_theta*theta*C.y))
FL.append((B, -k_phi  *phi  *B.x))
# Due to the damping
FL.append((D, -b_psi  *psi.diff(t)  *D.z))
FL.append((C, -b_theta*theta.diff(t)*C.y))
FL.append((B, -b_phi  *phi.diff(t)  *B.x))
# Due to gravity
FL.append((mc, -m  *g  *E.z))

# Rotational kinetic energy
T = (1/2*m*mc.vel(E).dot(mc.vel(E))) + 1/2*B.ang_vel_in(E).dot(I.dot(B.ang_vel_in(E))) 

# Solve for eom with Lagrange's method
q = sp.Matrix([psi, theta, phi])
qd = q.diff(t)
qdd = qd.diff(t)
LM = me.LagrangesMethod(T, q, forcelist=FL, frame=E)
lag_eqs = LM.form_lagranges_equations()

# Equations of motion: mass matrix and forcing term
p = sp.Matrix([Ixx, Iyy, Izz, b_phi, b_psi, b_theta, g, k_phi, k_psi, k_theta, l, m]) # parameters
eom = sp.lambdify((sp.Matrix([*q, *q.diff(t)]), p), (LM.mass_matrix_full, LM.forcing_full))

# test
# qv = np.array([0, 0, 1, 0, 0, 0])
# pv = np.array([1, 1, 1, 1, 1, 1, 10, 1, 1, 1, 2, 1])
# Mass, Force = eom(qv, pv)
# qdd = np.squeeze(np.linalg.solve(Mass, Force))

# Solve equations of motion
def rhs(t,x,p): return np.squeeze(np.linalg.solve(*eom(x, p)))
x0 = np.array([0, 0.1, 0.095, 0, 0, 0])
t0 = 0
tf = 25
t_span = (t0,  tf)
p0 = np.array([1, 1, 1, 1, 1, 1, 10, 2, 2, 2, 0.25, 1])
xsol = solve_ivp(rhs, t_span, x0, args=(p0, ), t_eval=np.linspace(t0, tf, num=tf*10))

# Plot
plt.plot(xsol.t, np.transpose(xsol.y))
plt.xlabel('Time, s')
plt.legend([*q, *q.diff(t)])
plt.show()