import matplotlib.pyplot as plt
from math import *
from numpy import genfromtxt
import numpy as np
import matplotlib.patches as patches
import pylab as pylab

plt.ion()

data_set = 'data/small.csv'
data = genfromtxt(data_set, dtype='int64', delimiter=',',skip_header=True)

def Q_max(Q,s):
	max_r = 0
	for reward in Q[s-1,:]:
		if reward > max_r:
			max_r = reward
	return max_r

def Q_neighbors(grid,i,j):
	max_r = 0
	max_a = -1
	if(i - 1 >= 0):
		if(grid[i-1,j] > max_r):
			max_r = grid[i-1,j]
			max_a = 0
	if(i + 1 < 10):
		if(grid[i+1,j] > max_r):
			max_a = 1
			max_r = grid[i+1,j]
	if(j + 1 < 10):
		if(grid[i,j+1] > max_r):
			max_a = 2
			max_r = grid[i,j+1]
	if(j - 1 >= 0):
		if(grid[i,j-1] > max_r):
			max_a = 3

	return max_a

def plot_stage(ax,Q):
	grid = np.zeros((10,10))
	index = 0
	reds = plt.get_cmap('inferno')
	max_q = np.max(Q)
	print(max_q)
	min_q = np.min(Q)
	for i in range(10):
		for j in range(10):
			r_sum = np.sum(Q[index,:])
			ax.add_patch(patches.Rectangle((i-.5, j-.5),1,1,facecolor = reds(r_sum/300)))
			grid[i,j] = r_sum
			index += 1
	for i in range(10):
		for j in range(10):
			a = Q_neighbors(grid,i,j)
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

	ax.set_xlim(-1,10)
	ax.set_ylim(-1,10)

	


gamma = .95
alpha = .5
n_states = 100
n_actions = 4
n_samples = len(data[:,0])
t = 0

s = data[0,0]
Q = np.zeros((100,4))
ax = plt.axes()
for i in range(n_samples):
	Q[data[i,0]-1,data[i,1]-1] += alpha*(data[i,2]+gamma*Q_max(Q,data[i,3])-Q[data[i,0]-1,data[i,1]-1])
	if((i+1)%5000 == 0):
		plot_stage(ax,Q)
		if(i != 49999):
			plt.pause(.15)
			plt.cla()
		else:
			plt.pause(100)
		print(i)


