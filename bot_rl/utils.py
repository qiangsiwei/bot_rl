# -*- coding: utf-8 -*-

import os
from config import config as c

get_path = lambda fn:os.path.join(os.path.abspath(os.path.dirname(__file__)),fn)

def remove_empty_slots(d):
	for i in d.keys():
		d[i] = {k:v for k,v in d[i].iteritems() if v}

def list_to_dict(l):
	assert len(l) == len(set(l))
	return {k:i for i,k in enumerate(l)}

def get_reward(succ, max_round):
	reward = -1
	if succ == c.FAIL:
		reward += -max_round
	if succ == c.SUCC:
		reward += 4*max_round
	return reward

if __name__ == '__main__':
	pass
