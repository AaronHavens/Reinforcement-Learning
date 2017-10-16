import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sys
import sklearn.pipeline
import sklearn.preprocessing

def decode_state(state):
	vel = state//500
	pos = state%500 - 1
	return [pos, vel]
data_set = 'data/medium.csv'
data = genfromtxt(data_set, dtype='int64', delimiter=',',skip_header=True)
#500*100 possible states, 7 actions for each state.
sample_size = np.size(data[:,0])
episodes = 1
gamma = .5

examples = np.zeros((10000,2))
for i in range(10000):
	examples[i,:] = decode_state(data[i,0])
observation_examples = np.array(examples)
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))
# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
class Estimator():

	def __init__(self,data):

		self.models=[]
		for _ in range(7):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(decode_state(data[0,0]))], [0])
			self.models.append(model)

	def featurize_state(self,state):
		scaled = scaler.transform([state])
		featurized = featurizer.transform(scaled)
		return featurized[0]

	def predict(self,s,a=None):
		features = self.featurize_state(s)
		if not a:
			return np.array([m.predict([features])[0] for m in self.models])
		else:
			return self.models[a].predict([features])[0]

	def update(self,s,a,y):
		features = self.featurize_state(s)
		self.models[a].partial_fit([features],[y])

def q_learn(data, estimator,episodes,gamma):
	for i in range(episodes):
		for k in range(sample_size):
			state,action,reward,next_state = data[k,:]
			state = decode_state(state)
			next_state = decode_state(next_state)
			q_values_next = estimator.predict(next_state)
			td_step = gamma*np.max(q_values_next)
			td_target = reward + td_step
			estimator.update(state,action-1,td_target)
			print(td_step)

def write_policy(estimator):
	with open('out_policy.txt', 'w') as f:
		for pos in range(500):
			for vel in range(100):
				f.write('{},{},{}\n'.format(pos,vel,np.argmax(estimator.predict([pos,vel]))))


estimator = Estimator(data)
#print(estimator.featurize_state(decode_state(data[0,0])))
q_learn(data,estimator,1,.5)
write_policy(estimator)