# -*- coding: utf-8 -*-

class Config(object):
	def __init__(self):
		self.SUCC = +1
		self.FAIL = -1
		self.NO_OUTCOME = 0
		self.c = {
			'paths': {
				'movie_db'    :'../data/movie_db.pkl',
				'movie_dict'  :'../data/movie_dict.pkl',
				'user_goals'  :'../data/user_goals.pkl'
			},
			'run': {
				'model'       :'per_dqn',
				'warmup_mem'  :1000,
				'ep_run_num'  :40000,
				'train_freq'  :100,
				'succ_thres'  :0.3,
				'max_rounds'  :20
			},
			'agent': {
				'save_path'   :'../bot_rl/weights/weights.h5',
				'load_path'   :'../bot_rl/weights/weights.h5',
				'vanilla'     :False,
				'lr'          :0.005,
				'b_size'      :16,
				'epsilon'     :0.0,
				'gamma'       :0.9,
				'h_dim'       :80,
				'max_mem'     :500000
			},
			'emc': {
				'slot_mode'   :0,
				'slot_prob'   :0.05,
				'intent_prob' :0.00}}
		self.usim_intents = ['inform','request','thanks','reject','done']
		self.usim_def_key = 'ticket'
		self.usim_req_key = 'moviename'
		self.agent_inf_slots = ['moviename','theater','starttime','date','genre','state','city','zip','critic_rating',
								'mpaa_rating','distanceconstraints','video_format','theater_chain','price','actor',
								'description','other','numberofkids',self.usim_def_key]
		self.agent_req_slots = ['moviename','theater','starttime','date','genre','state','city','zip','critic_rating',
								'mpaa_rating','distanceconstraints','video_format','theater_chain','price','actor',
								'description','other','numberofkids','numberofpeople']
		self.agent_actions = [
			{'intent':'done','inform_slots':{},'request_slots':{}},
			{'intent':'match_found','inform_slots':{},'request_slots':{}}]
		self.agent_actions.extend([
			{'intent':'inform','inform_slots':{slot:'PLACEHOLDER'},'request_slots':{}}\
			for slot in self.agent_inf_slots if slot != self.usim_def_key])
		self.agent_actions.extend([
			{'intent':'request','inform_slots':{},'request_slots':{slot:'UNK'}}\
			for slot in self.agent_req_slots])
		self.rule_requests = ['moviename','starttime','city','date','theater','numberofpeople']
		self.no_query_keys = ['numberofpeople',self.usim_def_key]
		self.all_intents = ['inform','request','match_found','thanks','reject','done']
		self.all_slots = ['actor','actress','city','critic_rating','date','description','distanceconstraints',
						  'genre','greeting','implicit_value','movie_series','moviename','mpaa_rating',
						  'numberofpeople','numberofkids','other','price','seating','starttime','state',
						  'theater','theater_chain','video_format','zip','result',self.usim_def_key,'mc_list']

config = Config()
