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

import rospy
import os
import json
import numpy as np
import random
import time
import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_1_torch import Env
from src.turtlebot3_dqn.dqn_model import DQN
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# set the random seed:
torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

# initialize tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
tensorboard = SummaryWriter(log_dir=log_dir)


EPISODES = 3000

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 10000
        self.update_freq = 1
        self.target_update = 2000
        self.target_update_freq = 5000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.max_epsilon = 1.0   
        self.min_epsilon = 0.1
        self.epsilon_decay_step = (self.max_epsilon - self.min_epsilon)/100000
        self.batch_size = 64
        self.train_start = 5000
        self.memory = deque(maxlen=1000000)

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize Q network and target Q network
        self.set_weight_init = True
        self.model =  DQN(self.state_size, self.action_size, 
                            set_init=self.set_weight_init).to(self.device)
        self.target_model = DQN(self.state_size, self.action_size, 
                            set_init=self.set_weight_init).to(self.device)

        # update the target model
        self.updateTargetModel()

        # define loss function and optimizer
        self.loss_fcn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                lr=self.learning_rate)

        # if self.load_model:
        #     self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

        #     with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
        #         param = json.load(outfile)
        #         self.epsilon = param.get('epsilon')

        # saving model and data
        self.save_model_freq = 100
        

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def getAction(self, state):
        # process state
        state_t = torch.Tensor(np.array(state)).to(self.device).unsqueeze(0)

        # obtain q_value from q_net
        self.q_value = self.model(state_t)

        # use epsilon-greedy if test is false
        if np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)

        else:
            # find greedy action
            action = int(torch.argmax(self.q_value))
        
        return action


    def appendMemory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def replayBuffer(self):
         # sample from the replay buffer
        states, actions, rewards, next_states, dones = \
                    zip(*random.sample(self.memory, self.batch_size))

        return states, actions, rewards, next_states, dones


    def trainModel(self):
        # update epsilon value: causing it to decay
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay_step

        ## sample random minibatch from buffer
        states, actions, rewards, next_states, done = self.replayBuffer()

        ## process parameters
        states_t = torch.Tensor(np.array(states)).to(self.device)
        actions_t = torch.Tensor(actions).to(self.device)
        actions_t = actions_t.type(torch.int64).unsqueeze(-1)
        next_states_t = torch.Tensor(np.array(next_states)).to(self.device)

        ## get max Q value for state->next_state using q_target_net
        max_q_values = torch.max(self.target_model(next_states_t), dim=1)[0]

        ## check if the episode terminates in next step
        td_target = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if done[i]:
                td_target[i] = rewards[i]
            else:
                td_target[i] = rewards[i] + self.discount_factor*max_q_values[i]

        ## convert td_target to tensor
        td_target_t = torch.Tensor(td_target).to(self.device)

        ## get current_q_values
        curr_q_values = self.model(states_t).gather(1, actions_t).squeeze()

        ## calculate the loss 
        self.loss = self.loss_fcn(curr_q_values, td_target_t)

        ## perform backprop and update weights
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    # initialize node
    rospy.init_node('turtlebot3_dqn_stage_1')
    # initialize result publisher
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    # initialize get_action publisher
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    # set varaibles
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    # set state and action space sizes
    state_size = 362
    action_size = 5

    # initialize environment
    env = Env(action_size)

    # initialize agent
    agent = ReinforceAgent(state_size, action_size)

    # set variables
    scores, episodes, total_reward = [], [], []
    global_step = 0
    # set start time
    start_time = time.time()

    # main loop: for each episode
    for e in range(agent.load_episode + 1, EPISODES):
        done = False
        state = env.reset()
        score = 0
        
        # inner loop: for each episode step
        for t in range(agent.episode_step):
            # get action
            action = agent.getAction(state)

            # take action and return state, reward, status
            next_state, reward, done = env.step(action)

            # append memory to memory buffer
            agent.appendMemory(state, action, reward, next_state, done)

            ## check if replay buffer is ready:
            if len(agent.memory) >= agent.train_start:
                # learn only after every 8 steps:
                if global_step % agent.update_freq == 0:
                    agent.trainModel()

            ## update the target network parameters
            if global_step % agent.target_update_freq == 0:
                # update the target network
                agent.updateTargetModel()

            # increment score and append reward
            score += reward
            total_reward.append(reward)

            # update state
            state = next_state

            # publish get_action
            get_action.data = [action, score, reward]
            pub_get_action.publish(get_action)

            # save to tensorboard
            num = 30
            tensorboard.add_scalar('step reward', reward, global_step)
            tensorboard.add_scalar('average step reward (over 30 steps)', 
                                    sum(total_reward[-num:])/num, global_step)

            # save model after every N episodes
            if e % agent.save_model_freq == 0 and e != 0:
                torch.save(agent.model.state_dict(), agent.dirPath + str(e) + '.pth')

            # timeout after 1200 steps (robot is just moving in circles or so)
            if t >= 1200: # changed this from 500 to 1200 steps
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = [score, int(torch.argmax(agent.q_value))]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d | score: %.2f | memory: %d | epsilon: %.6f | time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                param_keys = ['epsilon']
                param_values = [agent.epsilon]
                param_dictionary = dict(zip(param_keys, param_values))

                # add to tensorboard
                k = 10
                tensorboard.add_scalar('episode reward', score, e)
                tensorboard.add_scalar('average episode reward (over 10 episodes)', 
                                    sum(scores[-k:])/k, e)
                break

            global_step += 1
            # if global_step % agent.target_update == 0:
            #     rospy.loginfo("UPDATE TARGET NETWORK")
