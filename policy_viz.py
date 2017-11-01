import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

data_set = 'out_policy_1.txt'
data = genfromtxt(data_set, dtype='int64', delimiter=',',skip_header=True)
def write_policy(estimator):
	with open('medium.policy', 'w') as f:
		state = 1 + data[i,0] + data[i,1]*500
		f.write('{}\n'.format(state))

sample_size = np.size(data[:,0])

plt.show()