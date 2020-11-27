#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.turtlebot3_dqn.environment_stage_1_torch_ddpg import Env
from agent_ddpg import *



if __name__ == '__main__':

	state_size = 26
	action_size = 2
	test = False
	cont = False
	current_time = '2020_11_27-15:37:48'
	env = Env(action_size)
	if test:
		agent = ReinforceAgent(env, state_size, action_size, test)
		agent.test_model(10000)
	elif cont:
		agent = ReinforceAgent(env, state_size, action_size, test, cont, current_time)
		agent.train_model()
	else:
		agent = ReinforceAgent(env, state_size, action_size, test, cont)
		agent.train_model()

   
