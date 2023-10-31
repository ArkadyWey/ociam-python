import scipy.integrate
import matplotlib.pyplot as plt
import numpy
import math

"""
Solve two variable system 

da/dt = \alpha*\cos(\beta*t)
db/dt = \gamma*a + \delta*t
"""
def rhs_wrap(c):
    def rhs(t,y):
        # Make empty return
        dydt = numpy.zeros_like(y)
        
        # Make indices readable
        i_a = 0
        i_b = 1

        i_alpha = 0 
        i_beta  = 1
        i_gamma = 2
        i_delta = 3

        # Make rhs
        dydt[i_a] = c[i_alpha]*numpy.cos(c[i_beta]*t)
        dydt[i_b] = c[i_gamma]*y[i_a] + c[i_delta]*t
        return dydt
    return rhs

# Parameters 
num_vars = 2
num_pars = 4

# Make indices readable
i_a = 0
i_b = 1

i_alpha = 0 
i_beta  = 1
i_gamma = 2
i_delta = 3

# Initial conditions
y_init = numpy.zeros(shape=num_vars) # Create placeholder
y_init[i_a] = 0 
y_init[i_b] = -3 

# Constants
c = numpy.zeros(shape=num_pars) # Create placeholder
c[i_alpha] = 4 
c[i_beta] = 3
c[i_gamma] = -2
c[i_delta] = 0.5

# Solution times 
times  = numpy.linspace(0,5,100)
t_init = times[0]
t_end  = times[-1]


# Solve
# Get rhs 
rhs = rhs_wrap(c)

sol = scipy.integrate.solve_ivp(fun=rhs, t_span=[t_init,t_end], y0=y_init, method="RK45", t_eval=times, rtol=1e-5)

# Use solution/3.1.1/api/_as_gen/matplotlib.pyplot.xlabel.html
times = sol.t
y     = sol.y # y[other_index,time_index]

plt.plot(times,y[i_a,:], label=r"$a$")
plt.plot(times,y[i_b,:], label=r"$b$")
plt.xlabel(r"$t$")
plt.ylabel(r"$y$")
plt.legend()
plt.show()