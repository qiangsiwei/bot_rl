# -*- coding: utf-8 -*-

from base import RL
import random, numpy as np, tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)

class DQN(RL):

	def __init__(self, s_size):
		super(DQN,self).__init__(s_size)
		tar_params = tf.get_collection('tar_params')
		beh_params = tf.get_collection('beh_params')
		self.copy_op = [tf.assign(t,b) for t,b in zip(tar_params,beh_params)]
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def build_model(self):
		self.state = tf.placeholder(tf.float32,[None,self.s_size],name='state')
		self.q_tar = tf.placeholder(tf.float32,[None,self.act_num],name='q_tar')
		self.nstate = tf.placeholder(tf.float32,[None,self.s_size],name='nstate')
		w_init,b_init = tf.random_normal_initializer(0.,0.1),tf.constant_initializer(0.1)
		def build_layer(s, c_names):
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.s_size,self.h_dim],initializer=w_init,collections=c_names)
				b1 = tf.get_variable('b1',[1,self.h_dim],initializer=b_init,collections=c_names)
				l1 = tf.nn.relu(tf.matmul(s,w1)+b1)
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[self.h_dim,self.act_num],initializer=w_init,collections=c_names)
				b2 = tf.get_variable('b2',[1,self.act_num],initializer=b_init,collections=c_names)
				out = tf.matmul(l1,w2)+b2
			return out
		with tf.variable_scope('beh_net'):
			self.q_beh = build_layer(self.state,['beh_params',tf.GraphKeys.GLOBAL_VARIABLES])
		with tf.variable_scope('loss'):
			self.loss = tf.losses.mean_squared_error(self.q_tar,self.q_beh)
		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		with tf.variable_scope('tar_net'):
			self.q_next = build_layer(self.nstate,['tar_params',tf.GraphKeys.GLOBAL_VARIABLES])

	def dqn_action(self, state):
		state = state[np.newaxis,:]
		if np.random.uniform() < self.epsilon:
			return np.random.randint(0,self.n_act)
		ind = np.argmax(self.sess.run(self.q_beh,feed_dict={self.state:state}))
		return ind, self.ind_to_act(ind)

	def add_exp(self, state, action, reward, nstate, done):
		if len(self.memory) < self.max_mem: self.memory.append(None)
		self.memory[self.mem_ind] = state,action,reward,nstate,done
		self.mem_ind = (self.mem_ind+1)%self.max_mem

	def copy(self):
		self.sess.run(self.copy_op)

	def train(self):
		n_batch = len(self.memory)/self.b_size
		for b in range(n_batch):
			batch = random.sample(self.memory,self.b_size)
			states,actions,rewards,nstates,dones = zip(*batch)
			q_beh,q_next = self.sess.run([self.q_beh,self.q_next],feed_dict={
				self.state:np.array(states),self.nstate:np.array(nstates)})
			q_tar = q_beh.copy()
			batch_index = np.arange(self.b_size,dtype=np.int32)
			q_tar[batch_index,np.array(actions)] = np.array(rewards)+\
				self.gamma*np.max(q_next,axis=1)*(1-np.array(dones))
			_,cost = self.sess.run([self.train_op,self.loss],feed_dict={
				self.state:np.array(states),self.q_tar:q_tar})

if __name__ == '__main__':
	pass
