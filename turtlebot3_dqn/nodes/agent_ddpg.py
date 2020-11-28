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

# initialize self.tensorboard
current_time_global = datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")

class ReinforceAgent():
    def __init__(self, env, state_size, action_size, test = False, cont = False, current_time = current_time_global):

        ############## Init Parameters ##############
        self.env = env
        self.current_time = current_time
        self.state_size = state_size
        self.action_size = action_size
        self.test = test
        self.cont = cont

        ############## Publisher ##############
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.result = Float32MultiArray()


        ############## Path Parameters ##############
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model_ddpg/')
        self.model_path = self.dirPath + self.current_time + '/' + 'stage_1_'
        self.log_dir = 'logs_ddpg/' + self.current_time
        self.tensorboard = SummaryWriter(log_dir=self.log_dir)


        ############## Train Parameters ##############
        self.n_iterations = 2000
        self.episode_step = 5000
        self.e = 0
        self.episode_step = 10000
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 64
        self.tau = 0.001
        self.warmup = 10
        self.steps = 1
        self.save_model_freq = 50
        self.buffer_memory = 1000000
        self.buffer = []
        self.batch = []

        ############## Epsilon Parameters ##############
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


        ############## Random_Noise Parameters ##############
        self.ou_theta = 0.15
        self.ou_mu = 0.2
        self.ou_sigma = 0.5
        self.random_noise = OrnsteinUhlenbeckProcess(size=self.action_size, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)


        # print(self.dirPath + self.current_time)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device:',self.device)


        ############## Initialise Networks ##############
        self.actor = Actor(self.state_size,self.action_size).to(self.device)
        self.critic = Critic(self.state_size,self.action_size).to(self.device)
        
        self.target_actor = Actor(self.state_size,self.action_size).to(self.device)
        self.target_critic = Critic(self.state_size,self.action_size).to(self.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
        self.criteria = nn.MSELoss()
        self.actor_optimiser = optim.Adam(self.actor.parameters(),self.learning_rate)
        self.critic_optimiser = optim.Adam(self.critic.parameters(),self.learning_rate)


        ############## Testing ##############

        if self.test:
            print("############## Testing model ##############")
            actor_weights = torch.load(self.model_path + '1700' + '_actor.pth', map_location = self.device)
            critic_weights = torch.load(self.model_path + '1700' + '_critic.pth', map_location = self.device)
            self.actor.load_state_dict(actor_weights)
            self.critic.load_state_dict(critic_weights)
            self.actor.eval()
            self.critic.eval()

        ############## Conitnuing ##############
        if self.cont:
            print("############## Resuming Tarining ##############")
            last_train_memory = torch.load(self.model_path+ 'last_train_memory.tar')
            self.epsilon = last_train_memory['epsilon']
            self.e = last_train_memory['e']
            self.current_time = last_train_memory['current_time']
            self.steps = last_train_memory['steps']
            self.buffer = last_train_memory['buffer']

            last_train_weights = torch.load(self.model_path+ 'last_train_weights.tar', map_location = self.device)
            actor_weights = last_train_weights['actor']
            critic_weights = last_train_weights['critic']
            target_actor_weights = last_train_weights['target_actor']
            target_critic_weights = last_train_weights['target_critic']
            actor_optimiser_weights = last_train_weights['actor_optimiser']
            critic_optimiser_weights = last_train_weights['critic_optimiser']
            self.actor.load_state_dict(actor_weights)
            self.critic.load_state_dict(critic_weights)
            self.target_actor.load_state_dict(target_actor_weights)
            self.target_critic.load_state_dict(target_critic_weights)
            self.actor_optimiser.load_state_dict(actor_optimiser_weights)
            self.critic_optimiser.load_state_dict(critic_optimiser_weights)
            self.actor.train()
            self.critic.train()




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
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimiser.step()


        actor_loss = -self.critic(states,self.actor(states)).squeeze(1).mean()
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimiser.step()

        self.soft_update("Critic")
        self.soft_update("Actor")

    def train_model(self):

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
        prev_e = 0

        # main loop: for each episode
        for e in range(self.e + 1, self.n_iterations):
            done = False
            state = self.env.reset()
            score = 0

            for t in range(self.episode_step):

                if e < self.warmup:
                    action = self.random_action()
                else:
                    action = self.getAction(state)

                self.steps += 1
                self.e = e
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

                # save to self.tensorboard
                num = 30
                self.tensorboard.add_scalar('step reward', reward, global_step)
                self.tensorboard.add_scalar('average step reward (over 30 steps)', 
                                        sum(total_reward[-num:])/num, global_step)

                # save model after every N episodes
                if e % self.save_model_freq == 0 and e != prev_e:
                    print('Saving Model!')
                    self.save_model()
                    prev_e = e 

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
                    # add to self.tensorboard
                    k = 10
                    self.tensorboard.add_scalar('episode reward', score, e)
                    self.tensorboard.add_scalar('average episode reward (over 10 episodes)', 
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

    
    def save_model(self):
        # print(self.dirPath + self.current_time)
        if not os.path.isdir(self.dirPath + self.current_time):
            print('Creating Directory')
            os.makedirs(self.dirPath + self.current_time)

        torch.save(self.actor.state_dict(), self.model_path + str(self.e) + '_actor.pth')
        torch.save(self.critic.state_dict(), self.model_path + str(self.e) + '_critic.pth')
        torch.save({
            'epsilon':self.epsilon,
            'steps':self.steps,
            'buffer':self.buffer,
            'e':self.e,
            'current_time':self.current_time
            }
            ,self.model_path+ 'last_train_memory.tar')
        torch.save({
            'actor':self.actor.state_dict(),
            'critic':self.critic.state_dict(),
            'target_actor':self.target_actor.state_dict(),
            'target_critic':self.target_critic.state_dict(),
            'actor_optimiser':self.actor_optimiser.state_dict(),
            'critic_optimiser':self.critic_optimiser.state_dict()
            }
            ,self.model_path+ 'last_train_weights.tar')




