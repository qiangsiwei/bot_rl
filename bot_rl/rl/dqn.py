# -*- coding: utf-8 -*-

from base import RL

class DQN(RL):

	def __init__(self, s_size):
		super(RL,self).__init__()

	def build_model(self):
		pass

	def dqn_action(self, state):
		pass

	def add_exp(self, state, action, reward, nstate, done):
		if len(self.memory) < self.max_mem: self.memory.append(None)
		self.memory[self.mem_ind] = state,action,reward,nstate,done
		self.mem_ind = (self.mem_ind+1)%self.max_mem

	def train(self):
		pass

if __name__ == '__main__':
	pass
