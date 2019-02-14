# -*- coding: utf-8 -*-

from dqn import DQN
import random, numpy as np, tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)

class DueDQN(DQN):

	def __init__(self, s_size):
		super(DueDQN,self).__init__(s_size)

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
			with tf.variable_scope('v'):
				w2 = tf.get_variable('w2',[self.h_dim,1],initializer=w_init,collections=c_names)
				b2 = tf.get_variable('b2',[1,1],initializer=b_init,collections=c_names)
				V = tf.matmul(l1,w2)+b2
			with tf.variable_scope('a'):
				w2 = tf.get_variable('w2',[self.h_dim,self.act_num],initializer=w_init,collections=c_names)
				b2 = tf.get_variable('b2',[1,self.act_num],initializer=b_init,collections=c_names)
				A = tf.matmul(l1,w2)+b2
			with tf.variable_scope('Q'):
				out = V+(A-tf.reduce_mean(A,axis=1,keep_dims=True)) # Q = V(s) + A(s,a)
			return out
		with tf.variable_scope('beh_net'):
			self.q_beh = build_layer(self.state,['beh_params',tf.GraphKeys.GLOBAL_VARIABLES])
		with tf.variable_scope('loss'):
			self.loss = tf.losses.mean_squared_error(self.q_tar,self.q_beh)
		with tf.variable_scope('train'):
			self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		with tf.variable_scope('tar_net'):
			self.q_next = build_layer(self.nstate,['tar_params',tf.GraphKeys.GLOBAL_VARIABLES])

if __name__ == '__main__':
	pass
