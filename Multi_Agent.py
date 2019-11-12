# in this file you will find an implementation of the MDDPG 
import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch

from ReplayBuf import ReplayBuf
from  Agent import DDPG_Agent
import torch
import torch.nn as nn
import torch.nn.functional as F


BUFFER_SIZE = int(1e5)             # replay buffer size
BATCH_SIZE = 512  

UPDATE_EVERY_NB_EPISODE  = 5

GAMMA = 0.99

MDPG_UPDATE = 5
CLIP_CRITIC_GRADIENT = True 
TAU = 2e-1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Maddpg():
   
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        super(Maddpg, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.random_seed = random.seed(5.0)
        
        # Instantiate Multiple  Agent
        self.agents = [ DDPG_Agent(state_size,action_size,num_agents) for i in range(num_agents) ]
        
        # Instantiate Memory replay Buffer (shared between agents)
        self.memory = ReplayBuf(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
                  
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise):
        """Return action to perform for each agents (per policy)"""        
        return [ agent.act(state, noise) for agent, state in zip(self.agents, states) ]
    
    def get_agent_data(self, size, num_agents, id_agent, states_actions):
        """
        This returns a batch of Environment states or actions containing the data 
        of only the agent specified as a tensor.
        Input :
           size (int): size of the action space of state spaec to decode
           num_agents (int) : Number of agent in the environment (and for which info hasbeen concatenetaded)
           id_agent (int): index of the agent whose informationis going to be retrieved
           sa (torch.tensor) : Batch of Environment states or actions, each concatenating the info of several 
                               agents (This is sampled from the buffer memmory in the context of MADDPG)
        """
    
        list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(device)    
        return states_actions.index_select(1, list_indices)
                
    
    def step(self, states, actions, rewards, next_states, dones, num_current_episode):
        """ # Save experience in replay memory, and use random sample from buffer to learn"""
 
        states_ = np.array(states).reshape(1,-1).squeeze()
        future_states = np.array(next_states).reshape(1,-1).squeeze()
        actions_ = np.array(actions).reshape(1,-1).squeeze()
        
        self.memory.add(states_,  actions_, rewards, future_states, dones)

        # If enough samples in the replay memory and if it is time to update
        if (len(self.memory) > BATCH_SIZE) and (num_current_episode % UPDATE_EVERY_NB_EPISODE ==0) :
            
           # this is because we are playing tennis so we expect only 2 agents 
            assert(len(self.agents)==2)
            
            # Allow to learn several time in a row in the same episode
            for i in range(MDPG_UPDATE):
                # Sample a batch of experience from the replay buffer 
                experiences = self.memory.sample()   
                # Update Agent #0
                self.maddpg_learn(experiences, own_idx=0, other_idx=1)
                # Sample another batch of experience from the replay buffer 
                experiences = self.memory.sample()   
                # Update Agent #1
                self.maddpg_learn(experiences, own_idx=1, other_idx=0)
                
    
    def maddpg_learn(self, experiences, own_idx, other_idx, gamma=GAMMA):
        """
        Update the policy of the MADDPG "own" agent. The actors have only access to agent own 
        information, whereas the critics have access to all agents information.
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value

        Input :
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            own_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
               
        # Filter out the agent OWN states, actions and next_states batch
        own_states =  self.get_agent_data(self.state_size, self.num_agents, own_idx, states)
        own_actions = self.get_agent_data(self.action_size, self.num_agents, own_idx, actions)
        own_next_states = self.get_agent_data(self.state_size, self.num_agents, own_idx, next_states) 
                
        # Filter out the OTHER agent states, actions and next_states batch
        other_states =  self.get_agent_data(self.state_size, self.num_agents, other_idx, states)
        other_actions = self.get_agent_data(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = self.get_agent_data(self.state_size, self.num_agents, other_idx, next_states)
        
        # Concatenate both agent information (own agent first, other agent in second position)
        all_states=torch.cat((own_states, other_states), dim=1).to(device)
        all_actions=torch.cat((own_actions, other_actions), dim=1).to(device)
        all_next_states=torch.cat((own_next_states, other_next_states), dim=1).to(device)
   
        agent = self.agents[own_idx]
        
            
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models        
        all_next_actions = torch.cat((agent.actor_target(own_states), agent.actor_target(other_states)),
                                     dim =1).to(device) 
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim = 1).to(device)      
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()        
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)                   
    
    
                        
    def checkpoints(self):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'checkpoint_actor_local_' + str(idx) + '.pth'
            critic_local_filename = 'checkpoint_critic_local_' + str(idx) + '.pth'           
            actor_target_filename = 'checkpoint_actor_target_' + str(idx) + '.pth'
            critic_target_filename = 'checkpoint_critic_target_' + str(idx) + '.pth'            
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename)             
            torch.save(agent.actor_target.state_dict(), actor_target_filename) 
            torch.save(agent.critic_target.state_dict(), critic_target_filename)
            
            
    