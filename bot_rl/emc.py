# -*- coding: utf-8 -*-

import random
from config import config as c

class EMC(object):

	def __init__(self, db_dict):
		self.db_dict = db_dict
		for k,v in c.c['emc'].iteritems():
			setattr(self,k,v)
		self.intents = c.usim_intents

	def infuse_error(self, frame):
		slots = frame['inform_slots']
		for key in frame['inform_slots'].keys():
			if random.random() < self.slot_prob:
				if self.slot_mode not in (0,1,2):
					self.slot_mode = random.randint(0,2)
				if self.slot_mode == 0:
					self.slot_value_noise(key,slots)
				if self.slot_mode == 1:
					self.slot_noise(key,slots)
				if self.slot_mode == 2:
					self.slot_del(key,slots)
		if random.random() < self.intent_prob:
			frame['intent'] = random.choice(self.intents)

	def slot_value_noise(self, key, slots):
		slots[key] = random.choice(self.db_dict[key])

	def slot_noise(self, key, slots):
		slots.pop(key)
		key = random.choice(self.db_dict.keys())
		slots[key] = random.choice(self.db_dict[key])

	def slot_del(self, key, slots):
		slots.pop(key)

if __name__ == '__main__':
	pass
