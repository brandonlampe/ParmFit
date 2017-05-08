import numpy as np
from matplotlib import pyplot as plt

ep = np.linspace(0.00, 2, 50)
phi = np.linspace(0.2, 0.001, 5)
h = np.zeros((50))
a = 8
for i in xrange(len(phi)):
    h = 0.1 + 1*ep ** (a * (1. - phi[i]))
    # h = (1 - phi[i]) * ep ** (a * (1. - phi[i]))
    # h = (1. - phi[i]) * a ** ep
    plt.plot(ep, h, '-o', label='phi:' + str(phi[i]))

plt.ylabel('h')
plt.xlabel('ep')
plt.legend(loc=0)
plt.grid()
plt.show()
