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

# initialize tensorboard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
tensorboard = SummaryWriter(log_dir=log_dir)



class ReinforceAgent():
    def __init__(self, state_size, action_size):
        
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model_ddpg/stage_1_')
        self.result = Float32MultiArray()

        self.load_model = False
        self.load_episode = 0
        self.n_iterations = 1000
        self.episode_step = 10000

        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 10000
        self.discount_factor = 0.99
        self.learning_rate = 0.0000625
        
        self.epsilon = 1.0
        self.max_epsilon = 1.0   
        self.min_epsilon = 0.1
        self.annealing_steps = 10000

        self.epsilon_decay_step = (self.max_epsilon - self.min_epsilon)/self.annealing_steps
        self.batch_size = 64
        self.tau = 0.001
        self.train_start = 50000
        # self.memory = deque(maxlen=1000000)
        self.buffer = []
        self.buffer_memory = 1000000
        self.batch = []
        self.warmup = 1


        self.ou_theta = 0.15
        self.ou_mu = 0.2
        self.ou_sigma = 0.5
        self.random_noise = OrnsteinUhlenbeckProcess(size=self.action_size, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.save_model_freq = 100

        # if self.load_model:
        #     self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

        #     with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
        #         param = json.load(outfile)
        #         self.epsilon = param.get('epsilon')

        # saving model and data
        

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
        if not test and self.epsilon > self.min_epsilon:
            self.epsilon-=self.epsilon_decay_step

        p = random.random()
        if p < self.epsilon:
        	return self.random_action()

        action = self.actor(torch.from_numpy(np.array([state]))).squeeze(0).detach().numpy()
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


    def trainModel(self):
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
    state_size = 26
    action_size = 2

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
    for e in range(agent.load_episode + 1, agent.n_iterations):
        done = False
        state = env.reset()
        score = 0
        agent.random_noise.reset_states()
    
        # inner loop: for each episode step
        for t in range(agent.episode_step):
            # get action

            if e < agent.warmup:
                action = agent.random_action()
            else:
                action = agent.getAction(state)

            # print(action.dtype)

            # take action and return state, reward, status
            next_state, reward, done = env.step(action)

            # append memory to memory buffer
            agent.appendMemory((state, action, reward, next_state, done))

            ## check if replay buffer is ready:
            if e > agent.warmup and len(agent.buffer) > agent.batch_size:
                agent.trainModel()

        
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
            if e % agent.save_model_freq == 0 and e != 0:
                torch.save(agent.actor.state_dict(), agent.dirPath + str(e) + 'actor.pth')
                torch.save(agent.critic.state_dict(), agent.dirPath + str(e) + 'critic.pth')

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
                              e, score, len(agent.buffer), agent.epsilon, h, m, s)
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
