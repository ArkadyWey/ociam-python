import numpy
import matplotlib.pyplot as plt
import scipy.integrate

"""
Solve dy/dt = 1 for y(0)=0
"""

# Option 1:
def f(t,y,c):
    dydt = c
    return dydt

# Option 2:
def rhs_wrap(c):
    def rhs(t,y):
        dydt = c # must be 1-d array
        return dydt
    return rhs

# Initial value 
y_init = numpy.array([0])

# Constants
c = numpy.array([1])

# Solution times 
times  = numpy.linspace(0,10,100)
t_init = times[0]
t_end  = times[-1]


# Solve
# -------
# Option 1 -- use lambda:
#sol = scipy.integrate.solve_ivp(fun=lambda t, y: f(t, y, c), t_span=[t_init,t_end], y0=y_init, method="RK45", t_eval=times, rtol=1e-5)

# Option 2 -- use wrapper:
# Get rhs 
rhs = rhs_wrap(c)

sol = scipy.integrate.solve_ivp(fun=rhs, t_span=[t_init,t_end], y0=y_init, method="RK45", t_eval=times, rtol=1e-5) #args=c,

# Use solution
times = sol.t
y     = sol.y # y[other_index,time_index]

plt.plot(times,y[0,:])
plt.show()