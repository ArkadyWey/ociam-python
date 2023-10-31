import numpy
import scipy.integrate 
import matplotlib.pyplot as plt

"""
Solve higher order system

Example - constant jerk equation (third order system)

d**3/dt**3(x) = alpha

Write higher equation as system of first order equations:

Define 

y[0] = x
dy[0]/dt = dx/dt = y[1]        i.e., dx/dt = v
dy[1]/dt = d**2x/dt**2 = y[2]  i.e., dv/dt = a
dy[2]/dt = d**3x/dt**3 = alpha i.e., da/dt = alpha

Hence, rewrite third order equation as 

dydt[0] = y[1]
dydt[1] = y[2]
dydt[2] = c[0]
"""

# Define rhs function 
def rhs_wrap(c):
    def rhs(t,y):
        # Make return placeholder 
        dydt = numpy.zeros_like(y)

        # Make indices readable
        i_x = 0
        i_v = 1
        i_a = 2

        i_alpha = 0

        # Make rhs
        dydt[i_x] = y[i_v]
        dydt[i_v] = y[i_a]
        dydt[i_a] = c[i_alpha]
        return dydt
    return rhs

# Parameters 
num_vars = 3
num_pars = 1

# Make indices readable
i_x = 0
i_v = 1
i_a = 2

i_alpha = 0

# Initial conditions
y_init = numpy.zeros(shape=num_vars) # Create placeholder
y_init[i_x] = 6
y_init[i_v] = 2 
y_init[i_a] = -4 

# Constants
c = numpy.zeros(shape=num_pars) # Create placeholder
c[i_alpha] = 1.3 

# Solution times 
times  = numpy.linspace(0,8,100)
t_init = times[0]
t_end  = times[-1]


# Solve
# Get rhs 
rhs = rhs_wrap(c)

sol = scipy.integrate.solve_ivp(fun=rhs, t_span=[t_init,t_end], y0=y_init, method="RK45", t_eval=times, rtol=1e-5)

# Use solution/3.1.1/api/_as_gen/matplotlib.pyplot.xlabel.html
times = sol.t
y     = sol.y # y[other_index,time_index]

plt.plot(times,y[i_x,:], label=r"$x$")
plt.plot(times,y[i_v,:], label=r"$v$")
plt.plot(times,y[i_a,:], label=r"$a$")
plt.xlabel(r"$t$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()
