# -*- coding: utf-8 -*-

import pickle
from emc import EMC
from dqn import DQN
from dst import DST
from usim import USim
from user import User
from utils import *
from config import config as c

class BotRL(object):

	def __init__(self):
		read = lambda fn: pickle.load(open(get_path(c.c['paths'][fn]),'rb'))
		self.movie_db = read('movie_db')
		remove_empty_slots(self.movie_db)
		self.movie_dict = read('movie_dict')
		self.user_goals = read('user_goals')
		self.dst = DST(self.movie_db)
		self.emc = EMC(self.movie_dict)
		self.dqn = DQN(self.dst.s_size)

	def eps_reset(self):
		self.dst.reset()
		u_act = self.usr.reset()
		self.emc.infuse_error(u_act)
		self.dst.update_state_user(u_act)
		self.dqn.reset()

	def next_turn(self, state, warmup=False):
		a_aid,a_act = self.dqn.get_action(state,use_rule=warmup)
		self.dst.update_state_agent(a_act)
		u_act,reward,done,succ = self.usr.step(a_act)
		if not done: self.emc.infuse_error(u_act)
		self.dst.update_state_user(u_act)
		nstate = self.dst.get_state(done)
		self.dqn.add_exp(state,a_aid,reward,nstate,done)
		return nstate, reward, done, succ

	def warmup(self, cnt=0):
		while cnt < c.c['run']['warmup_mem'] and not self.dqn.is_mem_full():
			self.eps_reset()
			state = self.dst.get_state(); done = False
			while not done:
				nstate,_,done,_ = self.next_turn(state,warmup=True)
				state = nstate; cnt += 1
			# self.dst.print_history(); print '-'*10

	def train(self):
		self.usr = USim(self.movie_db,self.user_goals)
		eps = 0; succ_total,reward_total = 0,0; best = 0.0
		while eps < c.c['run']['ep_run_num']:
			self.eps_reset(); eps += 1; done = False
			state = self.dst.get_state()
			while not done:
				nstate,reward,done,succ = self.next_turn(state)
				reward_total += reward
				state = nstate
			if succ: self.dst.print_history(); print '-'*10
			succ_total += succ
			if eps%c.c['run']['train_freq'] == 0:
				succ_rate = 1.*succ_total/c.c['run']['train_freq']
				av_reward = 1.*reward_total/c.c['run']['train_freq']
				print 'eps:{0} succ:{1} reward:{2}'.format(eps,succ_rate,av_reward)
				if succ_rate >= best and succ_rate >= c.c['run']['succ_thres']:
					self.dqn.empty_mem()
				if succ_rate > best:
					best = succ_rate
					self.dqn.save_weights()
				succ_total,reward_total = 0,0
				self.dqn.copy()
				self.dqn.train()

	def test(self):
		self.usr = User()
		eps = 0
		while eps < c.c['run']['ep_run_num']:
			print 'eps:{0}'.format(eps)
			self.eps_reset(); eps += 1; done = False
			state = self.dst.get_state()
			while not done:
				a_aid,a_act = self.dqn.get_action(state)
				self.dst.update_state_agent(a_act)
				u_act,reward,done,succ = self.usr.step(a_act)
				if not done: self.emc.infuse_error(u_act)
				self.dst.update_state_user(u_act)
				state = self.dst.get_state(done)
				# print 'succ:{0} reward:{1}'.format(succ,reward)
		print 'end'

if __name__ == '__main__':
	pass
