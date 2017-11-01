import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sys
import sklearn.pipeline
import sklearn.preprocessing


data_set = 'data/medium.csv'
data = genfromtxt(data_set, dtype='int64', delimiter=',',skip_header=True)
#500*100 possible states, 7 actions for each state.
sample_size = np.size(data[:,0])
episodes = 1
gamma = .5

examples = np.zeros((10000,1))
for i in range(10000):
	examples[i,0] = data[i,0]
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
		for _ in range(8):
			model = SGDRegressor(learning_rate = "constant")
			model.partial_fit([self.featurize_state(data[0,0])], [0])
			self.models.append(model)
#state = 1 + pos +vel*500
	def featurize_state(self,state):
		scaled = scaler.transform([[state]])
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
			q_values_next = estimator.predict(next_state)
			td_step = gamma*np.max(q_values_next)
			td_target = reward + td_step
			estimator.update(state,action-1,td_target)
			print(k)

def write_policy(estimator):
	with open('large.policy', 'w') as f:
		for i in range(1757600):
				f.write('{}\n'.format(np.argmax(estimator.predict(i))))


estimator = Estimator(data)
#print(estimator.featurize_state(decode_state(data[0,0])))
q_learn(data,estimator,1,.99)
write_policy(estimator)