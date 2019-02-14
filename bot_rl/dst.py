# -*- coding: utf-8 -*-

import copy, numpy as np
from utils import *
from collections import defaultdict
from config import config as c

class DBQ:
	def __init__(self, database):
		self.db = database
		self.db_index = self.index(self.db)
		self.no_query = c.no_query_keys
		self.match_key = c.usim_def_key

	def index(self, database):
		index = defaultdict(set)
		for i,vdict in database.iteritems():
			for k,v in vdict.iteritems():
				index[(k,v.lower())].add(i)
		return index

	def stat_vals(self, key, res):
		stat = defaultdict(int)
		for i,vdict in res.iteritems():
			if key in vdict: stat[vdict[key]] += 1
		return stat

	def fill_slot(self, informs, slot):
		key = slot.keys()[0]
		res = self.find_reco({k:v for k,v in informs.iteritems() if k != key})
		fill,stat = {}, self.stat_vals(key,res)
		fill[key] = informs[key] = max(stat,key=stat.get) if stat else 'no match'
		return fill

	def find_reco(self, informs):
		options = set(self.db.keys())
		for k,v in informs.iteritems():
			if k in self.no_query or v=='anything': continue
			options = self.db_index[(k,v.lower())]&options
		return {i:self.db[i] for i in options}

	def stat_reco(self, informs):
		stat = defaultdict(int)
		for k,v in informs.iteritems():
			if k in self.no_query: stat[k] = 0; continue
			stat[k] = len(self.db) if v == 'anything' else\
					  len(self.db_index[(k,v.lower())])
		stat['match_all'] = len(self.find_reco(informs))
		return stat

class DST(object):

	def __init__(self, database):
		self.dbq = DBQ(database)
		self.match_key = c.usim_def_key
		self.intents_dict = list_to_dict(c.all_intents)
		self.slots_dict = list_to_dict(c.all_slots)
		self.max_rounds = c.c['run']['max_rounds']
		self.s_size = 2*len(self.intents_dict)+7*len(self.slots_dict)+3+self.max_rounds
		self.none_state = np.zeros(self.s_size)
		self.reset()

	def reset(self):
		self.informs, self.history, self.round_num = {}, [], 0

	def print_history(self):
		for act in self.history: print act

	def update_state_agent(self, a_act):
		if a_act['intent'] == 'inform':
			a_act['inform_slots'] = self.dbq.fill_slot(self.informs,a_act['inform_slots'])
		if a_act['intent'] == 'match_found':
			res = self.dbq.find_reco(self.informs)
			if res:
				k,v = res.items()[0] # random pick
				a_act['inform_slots'] = copy.deepcopy(v)
				a_act['inform_slots'][self.match_key] = str(k)
			else:
				a_act['inform_slots'][self.match_key] = 'no match'
			self.informs[self.match_key] = a_act['inform_slots'][self.match_key]
		a_act.update({'round':self.round_num,'speaker':'Agent'})
		self.history.append(a_act)

	def update_state_user(self, u_act):
		for k,v in u_act['inform_slots'].items():
			self.informs[k] = v
		u_act.update({'round':self.round_num,'speaker':'User'})
		self.history.append(u_act)
		self.round_num += 1

	def get_state(self, done=False):
		if done: return self.none_state
		u_act = self.history[-1]
		m_stat = self.dbq.stat_reco(self.informs)
		a_act = self.history[-2] if len(self.history) > 1 else None
		u_act_rep = np.zeros((len(self.intents_dict),))
		u_inf_slots_rep = np.zeros((len(self.slots_dict),))
		u_req_slots_rep = np.zeros((len(self.slots_dict),))
		u_act_rep[self.intents_dict[u_act['intent']]] = 1.0
		for key in u_act['inform_slots'].keys():
			u_inf_slots_rep[self.slots_dict[key]] = 1.0
		for key in u_act['request_slots'].keys():
			u_req_slots_rep[self.slots_dict[key]] = 1.0
		a_act_rep = np.zeros((len(self.intents_dict),))
		a_inf_slots_rep = np.zeros((len(self.slots_dict),))
		a_req_slots_rep = np.zeros((len(self.slots_dict),))
		if a_act:
			a_act_rep[self.intents_dict[a_act['intent']]] = 1.0
			for key in a_act['inform_slots'].keys():
				a_inf_slots_rep[self.slots_dict[key]] = 1.0
			for key in a_act['request_slots'].keys():
				a_req_slots_rep[self.slots_dict[key]] = 1.0
		curr_slots_rep = np.zeros((len(self.slots_dict),))
		for key in self.informs:
			curr_slots_rep[self.slots_dict[key]] = 1.0
		turn_rep = np.zeros((1,))+self.round_num/5.
		turn_oh_rep = np.zeros((self.max_rounds,))
		turn_oh_rep[self.round_num-1] = 1.0
		kb_cnt_rep = np.zeros((len(self.slots_dict)+1,))+m_stat['match_all']/100.
		for key in m_stat.keys():
			if key in self.slots_dict: kb_cnt_rep[self.slots_dict[key]] = m_stat[key]/100.
		kb_bin_rep = np.zeros((len(self.slots_dict)+1,))+np.sum(m_stat['match_all']>0.)
		for key in m_stat.keys():
			if key in self.slots_dict: kb_bin_rep[self.slots_dict[key]] = np.sum(m_stat[key]>0.)
		state_rep = np.hstack([u_act_rep,u_inf_slots_rep,u_req_slots_rep,\
							   a_act_rep,a_inf_slots_rep,a_req_slots_rep,\
							   curr_slots_rep,turn_rep,turn_oh_rep,kb_cnt_rep,kb_bin_rep]).flatten()
		return state_rep

if __name__ == '__main__':
	pass
