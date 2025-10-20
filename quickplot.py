import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x=[10,12,14,16,18,20,26,28,30,32,34]
y=[7.579,7.599,7.604,7.569,7.574,7.585,7.583,7.582,7.585,7.582,7.644]

plt.plot(x,y,lw=3)
plt.xlabel("k-grid size")
plt.ylabel("band gap (eV)")
plt.show()
