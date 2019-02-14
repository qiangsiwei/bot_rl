# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from bot_rl.bot_rl import BotRL

def train(bot):
	bot.train()

def test(bot):
	bot.test()

if __name__ == '__main__':
	bot = BotRL()
	train(bot)
	# test(bot)
