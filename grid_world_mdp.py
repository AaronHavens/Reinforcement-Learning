import matplotlib.pyplot as plt
from math import *
from numpy import genfromtxt
import numpy as np
import matplotlib.patches as patches
import pylab as pylab

#plt.ion()

data_set = 'data/small.csv'
data = genfromtxt(data_set, dtype='int64', delimiter=',',skip_header=True)

def Q_max(Q,s):
	max_r = 0
	for reward in Q[s-1,:]:
		if reward > max_r:
			max_r = reward
	return max_r

def plot_stage(ax,Q):
	grid = np.zeros((10,10))
	index = 0
	reds = plt.get_cmap('Greys')
	max_q = np.max(Q)
	print(max_q)
	min_q = np.min(Q)
	for j in range(10):
		for i in range(10):
			r_m= np.max(Q[index,:])
			ax.add_patch(patches.Rectangle((i-.5, j-.5),1,1,facecolor = reds(r_m/100)))
			grid[i,j] = r_m
			index += 1
	print(grid)
	index = 1
	for j in range(10):
		for i in range(10):
			a = np.argmax(Q[index-1,:])
			ax.text(i+.25, j+.25, "{:.1f}".format(grid[i,j]), ha="center", va="center",size=6,color='g')
			ax.text(i-.25, j+.25, "{:d}".format(index), ha="center", va="center",size=6,color='b')
			if(a == 0):
				ax.arrow(i, j, -.25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
			elif(a == 1):
				ax.arrow(i, j, .25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
			elif(a == 2):
				ax.arrow(i, j, 0, .25, head_width=0.1, head_length=0.1, fc='k', ec='k')
			elif(a == 3):
				ax.arrow(i, j, 0, -.25, head_width=0.1, head_length=0.1, fc='k', ec='k')
			else:
				ax.arrow(i, j, 0, -.25, head_width=0.1, head_length=0.1, fc='k', ec='k')
				ax.arrow(i, j, 0, .25, head_width=0.1, head_length=0.1, fc='k', ec='k')
				ax.arrow(i, j, .25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
				ax.arrow(i, j, -.25, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
			index += 1
	ax.set_xlim(-1,10)
	ax.set_ylim(-1,10)

def write_policy(q):
	with open('small.policy', 'w') as f:
		for i in range(100):
			f.write('{}\n'.format(np.argmax(q[i,:])))



gamma = .95
alpha = .01
n_states = 100
n_actions = 4
n_samples = len(data[:,0])
t = 0

s = data[0,0]
Q = np.zeros((n_states,n_actions))
ax = plt.axes()
for k in range(200):
	for i in range(n_samples):
		Q[data[i,0]-1,data[i,1]-1] += alpha*(data[i,2]+gamma*Q_max(Q,data[i,3])-Q[data[i,0]-1,data[i,1]-1])
	print(k)
write_policy(Q)
plot_stage(ax,Q)
plt.axis('off')
plt.show()