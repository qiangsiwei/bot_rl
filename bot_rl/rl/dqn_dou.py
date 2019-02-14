# -*- coding: utf-8 -*-

from dqn import DQN
import random, numpy as np, tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)

class DouDQN(DQN):

	def __init__(self, s_size):
		super(DouDQN,self).__init__(s_size)

	def train(self):
		n_batch = len(self.memory)/self.b_size
		for b in range(n_batch):
			batch = random.sample(self.memory,self.b_size)
			states,actions,rewards,nstates,dones = zip(*batch)
			q_beh,q_next = self.sess.run([self.q_beh,self.q_next],feed_dict={
				self.state:np.array(states),self.nstate:np.array(nstates)})
			q_eval = self.sess.run(self.q_beh,feed_dict={self.state:np.array(nstates)})
			q_tar = q_beh.copy()
			batch_index = np.arange(self.b_size,dtype=np.int32)
			q_tar[batch_index,np.array(actions)] = np.array(rewards)+\
				self.gamma*q_next[batch_index,np.argmax(q_eval,axis=1)]*(1-np.array(dones))
			_,cost = self.sess.run([self.train_op,self.loss],feed_dict={
				self.state:np.array(states),self.q_tar:q_tar})

if __name__ == '__main__':
	pass
