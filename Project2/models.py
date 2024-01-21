import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributions as dists


# Simple Neural Network  without coms
class Simple_Actor(nn.Module):
    def __init__(self, state_size,num_objects, goal_size, hidden_dim):
        super(Simple_Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size*num_objects+goal_size, hidden_dim,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 2,dtype=float)
        )

    def forward(self, state, goal):
        x = torch.cat((state.view(-1), goal))
        x = self.network(x)
        return x
    
        
# Main network from Emergence of Grounded Compositional Language in Multi-Agent Populations
class Policy_Network(nn.Module):
    def __init__(self,communication_size,num_communication_streams ,state_size,action_space_size, goal_size,memory_size,hidden_size=256):
        super(Policy_Network, self).__init__()
        self.memory_size = memory_size
        self.action_space_size = action_space_size
       
        # initilaze the memory buffer for the communication streams
        self.com_memory = [torch.zeros(memory_size, dtype=torch.float32) for _ in range(num_communication_streams)]
            
        # Define the shared fully-connected processing module for communication streams
        self.shared_communication_module = nn.Sequential(
            nn.Linear(communication_size+memory_size, hidden_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size+memory_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
        # Define the shared fully-connected processing module for pyhisical observation
        self.shared_pyhisical_observation_module = nn.Sequential(
            nn.Linear(state_size, hidden_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, hidden_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
        # Define the pooling layer
        self.pooling_layer = nn.Softmax(dim=0)  
        
        # initilaze the memory buffer for the final layer
        self.final_memory = torch.zeros(memory_size,dtype=float)
        
        # define the final layer
        self.final_layer = nn.Sequential(
            nn.Linear(2*hidden_size+goal_size + memory_size, hidden_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, action_space_size+communication_size+memory_size,dtype=float),
        )
        
    # define forward pass 
    def forward(self, physical_observations,communication_streams, private_goal):

        # Process physical observations
        physical_features = []
        
        # forward passing each state representaion one after another -> weights are shared
        for i,observation in enumerate(physical_observations):
            processed_observation = self.shared_pyhisical_observation_module(observation)
            physical_features.append(processed_observation)

       
        communication_featues = []
        # forward passing each communication stream one after another -> weights are shared
        # each communication stream has its own personal memory vector 
        for i,(com,memory) in enumerate(zip(communication_streams,self.com_memory)):
            
            # concat memeory + com-stream and pass through the shared module
            processed_communications = self.shared_communication_module(torch.cat([com,memory]))
            processed_communication_vector = processed_communications[:-self.memory_size]
            
            communication_featues.append(processed_communication_vector)
            
            # extraxt delta m
            delta_memory = processed_communications[-self.memory_size:]
             
            # Sample a zero-mean Gaussian noise tensor
            epsilon = dists.Normal(torch.zeros_like(memory), torch.ones_like(memory)).sample()
            
            # update memory according to paper
            self.com_memory[i] = torch.tanh(memory+delta_memory+epsilon)
            
        # Pool com and observation features
        pooled_communication_features = torch.sum(self.pooling_layer(torch.stack(communication_featues)),dim=0)
        pooled_physical_features = torch.sum(self.pooling_layer(torch.stack(physical_features)),dim=0)
     
        # Combine features with private goal vector and final memory vector
        combined_features = torch.cat([pooled_communication_features, pooled_physical_features, private_goal,self.final_memory], dim=0)
        
        # forward pass through final layer
        final_output = self.final_layer(combined_features)

        # extraxt delta m
        delta_memory = final_output[-self.memory_size:]
        
        # Sample a zero-mean Gaussian noise tensor
        epsilon = dists.Normal(torch.zeros_like(delta_memory), torch.ones_like(delta_memory)).sample()
        
        # update memory
        self.final_memory = torch.tanh(self.final_memory+delta_memory+epsilon)
        
        # prepare actions
        actions = final_output[:self.action_space_size]
        action_epsilon = dists.Normal(torch.zeros_like(actions), torch.ones_like(actions)).sample()
        action = actions+action_epsilon
        
        
        # prepare communications using gumbel softmax 
        coms = final_output[self.action_space_size-1:self.memory_size+1]
        communication_symbol = F.gumbel_softmax(coms, tau=1, hard=False)
        
        # return action [angle,velocity]  and communcication vector (size: communication_size)
        return action, communication_symbol
        
# simple Crtic Network
class Critic(nn.Module):
    def __init__(self, state_size,num_objects, action_size, hidden_dim):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size*num_objects+action_size, hidden_dim,dtype=float),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1,dtype=float)
        )

    # out: value (for state,action) pair
    def forward(self, state, action):
        # concat the state and action 
        x = torch.cat((state.view(-1), action))
        x = self.network(x)
        return x
    
    