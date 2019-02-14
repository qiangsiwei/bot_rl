# -*- coding: utf-8 -*-

import copy, random
from ..config import config as c

class RL(object):

	def __init__(self, s_size):
		self.memory = []
		self.mem_ind = 0
		for k,v in c.c['agent'].iteritems():
			setattr(self,k,v)
		assert self.max_mem >= self.b_size
		self.s_size = s_size
		self.actions = c.agent_actions
		self.act_num = len(self.actions)
		self.rule_requests = c.rule_requests
		self.build_model()
		self.load_weights()
		self.reset()

	def build_model(self):
		pass

	def save_weights(self):
		pass

	def load_weights(self):
		pass

	def reset(self):
		self.rule_slot_ind = 0
		self.rule_phase = 'not done'

	def ind_to_act(self, ind):
		for i,a in enumerate(self.actions):
			if ind == i: return copy.deepcopy(a)

	def act_to_ind(self, act):
		for i,a in enumerate(self.actions):
			if act == a: return i

	def rule_action(self):
		resp = {'intent':'','inform_slots':{},'request_slots':{}}
		if self.rule_slot_ind < len(self.rule_requests):
			resp['intent'],resp['request_slots'] = 'request',\
				{self.rule_requests[self.rule_slot_ind]:'UNK'}
			self.rule_slot_ind += 1
		elif self.rule_phase == 'not done':
			resp['intent'],self.rule_phase = 'match_found','done'
		elif self.rule_phase == 'done':
			resp['intent'] = 'done'
		return self.act_to_ind(resp), resp

	def dqn_action(self, state):
		pass

	def get_action(self, state, use_rule=False):
		if self.epsilon > random.random():
			ind = random.randint(0,self.act_num-1)
			return ind, self.ind_to_act(ind)
		return self.rule_action() if use_rule else self.dqn_action(state)

	def add_exp(self, state, action, reward, nstate, done):
		pass

	def empty_mem(self):
		self.memory, self.mem_ind = [], 0

	def is_mem_full(self):
		return len(self.memory) == self.max_mem

	def copy(self):
		pass
		
	def train(self):
		pass

if __name__ == '__main__':
	pass
