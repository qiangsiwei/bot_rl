# -*- coding: utf-8 -*-

import re, copy, random, numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from config import config as c

class DQN(object):

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
		self.beh_model = self.build_model()
		self.tar_model = self.build_model()
		self.load_weights()
		self.reset()

	def build_model(self):
		model = Sequential()
		model.add(Dense(self.h_dim,input_dim=self.s_size,activation='relu'))
		model.add(Dense(self.act_num,activation='linear'))
		model.compile(loss='mse',optimizer=Adam(lr=self.lr))
		return model

	def copy(self):
		self.tar_model.set_weights(self.beh_model.get_weights())

	def save_weights(self):
		if not self.save_path: return
		self.beh_model.save_weights(re.sub(ur'(?=.h5$)','_beh',self.save_path))
		self.tar_model.save_weights(re.sub(ur'(?=.h5$)','_tar',self.save_path))

	def load_weights(self):
		if not self.load_path: return
		self.beh_model.load_weights(re.sub(ur'(?=.h5$)','_beh',self.load_path))
		self.tar_model.load_weights(re.sub(ur'(?=.h5$)','_tar',self.load_path))

	def reset(self):
		self.rule_slot_ind = 0
		self.rule_phase = 'not done'

	def dqn_predict(self, states, target=False):
		return self.tar_model.predict(states) if target else\
			   self.beh_model.predict(states)

	def dqn_predict1(self, state, target=False):
		return self.dqn_predict(state.reshape(1,self.s_size),target).flatten()

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
		ind = np.argmax(self.dqn_predict1(state))
		return ind, self.ind_to_act(ind)

	def get_action(self, state, use_rule=False):
		if self.epsilon > random.random():
			ind = random.randint(0,self.act_num-1)
			return ind, self.ind_to_act(ind)
		return self.rule_action() if use_rule else self.dqn_action(state)

	def add_exp(self, state, action, reward, nstate, done):
		if len(self.memory) < self.max_mem: self.memory.append(None)
		self.memory[self.mem_ind] = state,action,reward,nstate,done
		self.mem_ind = (self.mem_ind+1)%self.max_mem

	def empty_mem(self):
		self.memory, self.mem_ind = [], 0

	def is_mem_full(self):
		return len(self.memory) == self.max_mem

	def train(self):
		n_batch = len(self.memory)/self.b_size
		for b in range(n_batch):
			batch = random.sample(self.memory,self.b_size)
			states,nstates = [np.array([s[i] for s in batch]) for i in (0,3)]
			beh_preds = self.dqn_predict(states)
			beh_next_preds = self.dqn_predict(nstates)
			tar_next_preds = self.dqn_predict(nstates,target=True)
			x = np.zeros((self.b_size,self.s_size))
			y = np.zeros((self.b_size,self.act_num))
			for i,(s,a,r,s_,d) in enumerate(batch):
				t = beh_preds[i]
				t[a] = r+self.gamma*np.amax(tar_next_preds[i])*(not d) if self.vanilla else\
					   r+self.gamma*tar_next_preds[i][np.argmax(beh_next_preds[i])]*(not d)
				x[i],y[i] = s,t
			self.beh_model.fit(x,y,epochs=1,verbose=0)

if __name__ == '__main__':
	pass
