# -*- coding: utf-8 -*-

from utils import *
from config import config as c

class User(object):

	def __init__(self):
		self.max_round = c.c['run']['max_rounds']

	def reset(self):
		return self.get_resp()

	def get_resp(self):
		resp = {'intent':'','inform_slots':{},'request_slots':{}}
		while True:
			c1, c2 = True, True
			string = raw_input('response: ')
			chunks = string.split('/')
			c0 = chunks[0] in c.usim_intents
			resp['intent'] = chunks[0]
			for inf in filter(None,chunks[1].split(',')):
				key,val = inf.split(':')
				if key not in c.all_slots: c1 = False; break
				resp['inform_slots'][key] = val
			for req in filter(None,chunks[2].split(',')):
				if req not in c.all_slots: c2 = False; break
				resp['request_slots'][req] = 'UNK'
			if c0 and c1 and c2: break
		return resp

	def get_succ(self, succ=None):
		while succ not in (-1,0,1):
			succ = int(raw_input('success: '))
		return succ

	def step(self, a_act):
		print 'agent action: {}'.format(a_act)
		resp = {'intent':'','request_slots':{},'inform_slots':{}}
		if a_act['round'] == self.max_round:
			resp['intent'] = 'done'
			succ = c.FAIL
		else:
			resp = self.get_resp()
			succ = self.get_succ()
		return resp, get_reward(succ,self.max_round), succ in (c.FAIL,c.SUCC), succ == 1

if __name__ == '__main__':
	pass
	# example:
	# request/moviename:hail caesar/ticket
	# request//ticket 0
	# inform/theater:amc lowes oak tree 6/ 0
	# inform/numberofpeople:2/ 0
	# inform/city:seattle/ 0
	# thanks// 1
