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
print(sys.path)
from collections import deque
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_1_torch_ddpg import Env
from src.turtlebot3_dqn.ddpg_model import Actor,Critic
from src.turtlebot3_dqn.randomise_action import OrnsteinUhlenbeckProcess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# set the random seed:
torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

# initialize tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs_ddpg_1/' + current_time
tensorboard = SummaryWriter(log_dir=log_dir)



class ReinforceAgent():
    def __init__(self, env, state_size, action_size, test = False):

        self.env = env
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model_ddpg/'+current_time+'_stage_1_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.n_iterations = 10000
        self.episode_step = 10000

        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 10000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.test = test

        self.max_steps = 2500000
        self.annealing_steps = 100000
        self.start_epsilon = 1
        self.end_epsilon_1 = 0.1
        self.end_epsilon_2 = 0.01
        self.epsilon = self.start_epsilon

        self.slope1 = -(self.start_epsilon - self.end_epsilon_1)/self.annealing_steps
        self.constant1 = self.start_epsilon
        self.slope2 = -(self.end_epsilon_1 - self.end_epsilon_2)/(self.max_steps - self.annealing_steps)
        self.constant2 = self.end_epsilon_2 - self.slope2*self.max_steps

        #self.epsilon_decay_step = (self.max_epsilon - self.min_epsilon)/self.annealing_steps
        self.batch_size = 64
        self.tau = 0.001
        self.train_start = 50000
        # self.memory = deque(maxlen=1000000)
        self.buffer = []
        self.buffer_memory = 1000000
        self.batch = []
        self.warmup = 1
        self.steps = 1


        self.ou_theta = 0.15
        self.ou_mu = 0.2
        self.ou_sigma = 0.5
        self.random_noise = OrnsteinUhlenbeckProcess(size=self.action_size, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:',self.device)
        # initialize Q network and target Q network
        
        self.actor = Actor(self.state_size,self.action_size).to(self.device)
        self.critic = Critic(self.state_size,self.action_size).to(self.device)
        
        self.target_actor = Actor(self.state_size,self.action_size).to(self.device)
        self.target_critic = Critic(self.state_size,self.action_size).to(self.device)

        # update the target model
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    

        # define loss function and optimizer
        self.criteria = nn.MSELoss()
        self.actor_optimiser = optim.Adam(self.actor.parameters(),self.learning_rate)
        self.critic_optimiser = optim.Adam(self.critic.parameters(),self.learning_rate)
        self.save_model_freq = 50

        if self.test:
            actor_weights = torch.load(self.dirPath + '1700' + 'actor.pth', map_location = self.device)
            critic_weights = torch.load(self.dirPath + '1700' + 'critic.pth', map_location = self.device)
            self.actor.load_state_dict(actor_weights)
            self.critic.load_state_dict(critic_weights)
            self.actor.eval()
            self.critic.eval()


    def soft_update(self,model):
        if model == "Critic":    
            for current,target in zip(self.critic.parameters(),self.target_critic.parameters()):
                target.data.copy_(target.data *(1 - self.tau) + current.data * self.tau)
        elif model == "Actor":
            for current,target in zip(self.actor.parameters(),self.target_actor.parameters()):
                target.data.copy_(target.data *(1 - self.tau) + current.data * self.tau)



    def getAction(self, state, test = False):
        # process state
        self.actor.eval()
        if not test:
            #self.epsilon-=self.epsilon_decay_step
            
            if self.steps <=  self.annealing_steps:
              self.epsilon = self.steps*self.slope1 + self.constant1
        
            elif self.steps > self.annealing_steps:
              self.epsilon = self.steps*self.slope2 + self.constant2
        
            p = random.random()
            if p < self.epsilon:
            	return self.random_action()

        action = self.actor(torch.from_numpy(np.array([state])).to(self.device)).squeeze(0).detach().cpu().numpy()
        # print(action)
        # action += (1-test)*max(self.epsilon,0)*self.random_noise.sample()
        action = np.clip(action,-1.,1.)
        # print(action)
        return action
    
    
    def random_action(self):
        action = np.random.uniform(-1,1,self.action_size)
        # print(action)
        return action
    
    def appendMemory(self, episode):
        if len(self.buffer) >= self.buffer_memory:
            self.buffer.pop(0)
        self.buffer.append(episode)
        # self.memory.append((state, action, reward, next_state, done))


    def replayBuffer(self):
         # sample from the replay buffer
        batch  = random.sample(self.buffer, self.batch_size)
        self.batch = list(zip(*batch))


    def learn(self):
        # update epsilon value: causing it to decay

        ## sample random minibatch from buffer
        self.replayBuffer()
        self.actor.train()
    
        ## process parameters
        states = torch.from_numpy(np.asarray(self.batch[0])).to(self.device)
        # print(self.batch[1])
        actions = torch.from_numpy(np.asarray(self.batch[1])).to(self.device)
        # print(actions.size())
        rewards = torch.from_numpy(np.asarray(self.batch[2])).to(self.device)
        # print(rewards.size())
        next_states = torch.from_numpy(np.asarray(self.batch[3])).to(self.device)
        dones = torch.from_numpy(np.asarray(self.batch[4])).to(self.device)

        predicted_q_values = self.critic(states,actions).squeeze(1)
        next_state_actions  = self.target_actor(next_states)
        next_state_q_values = self.target_critic(next_states, next_state_actions).squeeze(1).detach()
        next_state_q_values[dones] = 0

        Y = next_state_q_values*self.discount_factor + rewards
        # print(Y.size())

        critic_loss = self.criteria(predicted_q_values.double(),Y.double())
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimiser.step()


        actor_loss = -self.critic(states,self.actor(states)).squeeze(1).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimiser.step()

        self.soft_update("Critic")
        self.soft_update("Actor")

    def train(self):

        rospy.init_node('turtlebot3_dqn_stage_1')
        # initialize result publisher
        pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        # initialize get_action publisher
        pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
        # set varaibles
        result = Float32MultiArray()
        get_action = Float32MultiArray()

        # set variables
        scores, episodes, total_reward = [], [], []
        global_step = 0
        # set start time
        start_time = time.time()

        # main loop: for each episode
        for e in range(self.load_episode + 1, self.n_iterations):
            done = False
            state = self.env.reset()
            score = 0

            for t in range(self.episode_step):

                if e < self.warmup:
                    action = self.random_action()
                else:
                    action = self.getAction(state)

                # print(action.dtype)
                self.steps += 1
                # take action and return state, reward, status
                next_state, reward, done = self.env.step(action)

                # append memory to memory buffer
                self.appendMemory((state, action, reward, next_state, done))

                ## check if replay buffer is ready:
                if e > self.warmup and len(self.buffer) > self.batch_size:
                    self.learn()

            
                # increment score and append reward
                score += reward
                total_reward.append(reward)

                # update state
                state = next_state

                # publish get_action
                get_action.data = [action[0],action[1], score, reward]
                pub_get_action.publish(get_action)

                # save to tensorboard
                num = 30
                tensorboard.add_scalar('step reward', reward, global_step)
                tensorboard.add_scalar('average step reward (over 30 steps)', 
                                        sum(total_reward[-num:])/num, global_step)

                # save model after every N episodes
                if e % self.save_model_freq == 0 and e != 0:
                    torch.save(self.actor.state_dict(), self.dirPath + str(e) + 'actor.pth')
                    torch.save(self.critic.state_dict(), self.dirPath + str(e) + 'critic.pth')

                # timeout after 1200 steps (robot is just moving in circles or so)
                if t >= 500: # changed this from 500 to 1200 steps
                    rospy.loginfo("Time out!!")
                    done = True

                if done:
                    result.data = [score, action]
                    pub_result.publish(result)
                    scores.append(score)
                    episodes.append(e)
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)

                    rospy.loginfo('Ep: %d | score: %.2f | memory: %d | epsilon: %.6f | time: %d:%02d:%02d',
                                  e, score, len(self.buffer), self.epsilon, h, m, s)
                    param_keys = ['epsilon']
                    param_values = [self.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))

                    # add to tensorboard
                    k = 10
                    tensorboard.add_scalar('episode reward', score, e)
                    tensorboard.add_scalar('average episode reward (over 10 episodes)', 
                                        sum(scores[-k:])/k, e)
                    break

                global_step += 1

    def test_model(self, episodes):

        rospy.init_node('turtlebot3_dqn_stage_1')
        # initialize result publisher
        pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        # initialize get_action publisher
        pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
        # set varaibles
        result = Float32MultiArray()
        get_action = Float32MultiArray()


        done = False
        state = self.env.reset()
        score = 0

        for t in range(episodes):
            
            action = self.getAction(state,self.test)

            next_state, reward, done = self.env.step(action)
        
            score += reward

            state = next_state

            get_action.data = [action[0],action[1], score, reward]
            pub_get_action.publish(get_action)

            num = 30


            if done:
                self.env.reset()


