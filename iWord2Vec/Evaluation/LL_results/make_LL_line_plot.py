import numpy as np
import matplotlib.pyplot as plt

w2v_x = [100, 300, 500]
iw2v_x = [275, 355, 509]
w2v_y = [-1.570390,-1.567323,-1.557550]
iw2v_y = [-1.494027,-1.489864,-1.47823]
plt.figure(figsize=(6.2,5))
plt.plot(w2v_x, w2v_y, 'rs-', linewidth=4, markersize=15, mec='r', label='Skip-Gram')
plt.plot(iw2v_x, iw2v_y, 'b^-', linewidth=4, markersize=15, mec='b', label='infinite Skip-Gram')
plt.xlim([75, 525])
plt.ylabel("Average Test Log Likelihood")
plt.xlabel("Vector Dimensionality")
plt.legend(loc=2)
#plt.show()
plt.savefig("prediction_plt.png")
