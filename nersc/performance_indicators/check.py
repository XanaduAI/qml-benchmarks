import numpy as np

array = np.random.rand(10,10)

np.savetxt("check1.txt", array)
np.savetxt("WORKDIR/check2.txt", array)
