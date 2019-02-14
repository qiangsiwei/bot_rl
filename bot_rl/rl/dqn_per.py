# -*- coding: utf-8 -*-

from dqn import DQN
import random, numpy as np, tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
	pp = 0
	def __init__(self, size):
		self.size = size
		self.tree = np.zeros(2*size-1)
		self.data = []
	def add(self, p, data):
		if len(self.data) < self.size:
			self.data.append(None)
		self.data[self.pp] = data
		self.update(p,self.pp+self.size-1)
		self.pp += 1
		if self.pp>=self.size: self.pp = 0
	def update(self, p, trid):
		change = p-self.tree[trid]
		self.tree[trid] = p
		while trid != 0:
			trid = (trid-1)/2
			self.tree[trid] += change
	def get_leaf(self, v, pid=0):
		while True:
			cl_idx, cr_idx = 2*pid+1, 2*pid+2
			if cl_idx >= len(self.tree):
				leaf_idx = pid; break
			if v <= self.tree[cl_idx]:
				pid = cl_idx
			else:
				v -= self.tree[cl_idx]
				pid = cr_idx
		data_idx = leaf_idx-self.size+1
		return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
	@property
	def total_p(self):
		return self.tree[0]

class Memory(object):
	epsilon = 0.01; alpha = 0.6; beta = 0.4
	beta_increment_per_sampling = 0.001
	abs_err_upper = 1.
	def __init__(self, size):
		self.tree = SumTree(size)
	def store(self, trans):
		max_p = np.max(self.tree.tree[-self.tree.size:])
		if max_p == 0: max_p = self.abs_err_upper
		self.tree.add(max_p,trans)
	def sample(self, n):
		self.beta = np.min([1.,self.beta+self.beta_increment_per_sampling])
		ids, mem, is_wg = [],[],np.empty((n,1))
		p_seg, min_prob = self.tree.total_p/n,np.min(self.tree.tree[\
			-self.tree.size:-self.tree.size+len(self.tree.data)])/self.tree.total_p
		for i in xrange(n):
			trid,p,data = self.tree.get_leaf(np.random.uniform(p_seg*i,p_seg*(i+1)))
			is_wg[i,0] = np.power(p/self.tree.total_p/min_prob,-self.beta)
			ids.append(trid); mem.append(data)
		return np.array(ids), mem, is_wg
	def batch_update(self, trids, abs_err):
		ps = np.power(np.minimum(abs_err+self.epsilon,self.abs_err_upper),self.alpha)
		for trid,p in zip(trids,ps): self.tree.update(p,trid)

class PerDQN(DQN):

	def __init__(self, s_size):
		super(PerDQN,self).__init__(s_size)
		self.memory = Memory(self.max_mem)

	def build_model(self):
		self.state = tf.placeholder(tf.float32,[None,self.s_size],name='state')
		self.q_tar = tf.placeholder(tf.float32,[None,self.act_num],name='q_tar')
		self.is_wg = tf.placeholder(tf.float32,[None,1],name='is_wg')
		self.nstate = tf.placeholder(tf.float32,[None,self.s_size],name='nstate')
		w_init,b_init = tf.random_normal_initializer(0.,0.1),tf.constant_initializer(0.1)
		def build_layer(s, c_names, trainable):
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.s_size,self.h_dim],initializer=w_init,collections=c_names,trainable=trainable)
				b1 = tf.get_variable('b1',[1,self.h_dim],initializer=b_init,collections=c_names,trainable=trainable)
				l1 = tf.nn.relu(tf.matmul(s,w1)+b1)
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[self.h_dim,self.act_num],initializer=w_init,collections=c_names,trainable=trainable)
				b2 = tf.get_variable('b2',[1,self.act_num],initializer=b_init,collections=c_names,trainable=trainable)
				out = tf.matmul(l1,w2)+b2
			return out
		with tf.variable_scope('beh_net'):
			self.q_beh = build_layer(self.state,['beh_params',tf.GraphKeys.GLOBAL_VARIABLES],True)
		with tf.variable_scope('loss'):
			self.abs_err = tf.reduce_sum(tf.abs(self.q_tar-self.q_beh),axis=1)
			self.loss = tf.reduce_mean(self.is_wg*tf.squared_difference(self.q_tar,self.q_beh))
		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		with tf.variable_scope('tar_net'):
			self.q_next = build_layer(self.nstate,['tar_params',tf.GraphKeys.GLOBAL_VARIABLES],False)

	def add_exp(self, state, action, reward, nstate, done):
		self.memory.store((state,action,reward,nstate,done))

	def empty_mem(self):
		self.memory = Memory(self.max_mem)

	def is_mem_full(self):
		return len(self.memory.tree.data) == self.max_mem

	def train(self):
		n_batch = len(self.memory.tree.data)/self.b_size
		for b in range(n_batch):
			trids,batch,is_wg = self.memory.sample(self.b_size)
			states,actions,rewards,nstates,dones = zip(*batch)
			q_beh,q_next = self.sess.run([self.q_beh,self.q_next],feed_dict={
				self.state:np.array(states),self.nstate:np.array(nstates)})
			q_tar = q_beh.copy()
			batch_index = np.arange(self.b_size,dtype=np.int32)
			q_tar[batch_index,np.array(actions)] = np.array(rewards)+\
				self.gamma*np.max(q_next,axis=1)*(1-np.array(dones))
			_,abs_err,cost = self.sess.run([self.train_op,self.abs_err,self.loss],feed_dict={
				self.state:np.array(states),self.q_tar:q_tar,self.is_wg:is_wg})
			self.memory.batch_update(trids,abs_err)

if __name__ == '__main__':
	pass
