import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import torch.distributions as dists


class Policy_Network(nn.Module):
    def __init__(self,communication_size,num_communication_streams ,state_size,action_space_size, goal_size,memory_size,hidden_size=256):
        super(Policy_Network, self).__init__()
        self.memory_size = memory_size
        self.action_space_size = action_space_size
        # Define the fully-connected processing modules for communication streams and physical entities
        
        # Define the shared LSTM cell for communication streams  
        #self.shared_communication_lstm = nn.LSTM(input_size=communication_size,hidden_size=256,num_layers=2,dropout=0.1)
        
        self.com_memory = []
        for c in range(num_communication_streams):
            self.com_memory.append(torch.zeros(memory_size,dtype=float))
        
        
            
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

        self.final_memory = torch.zeros(memory_size,dtype=float)
        self.final_layer = nn.Sequential(
            nn.Linear(2*hidden_size+goal_size + memory_size, hidden_size,dtype=float),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_size, action_space_size+communication_size+memory_size,dtype=float),
        )
        
        
  
    def forward(self, physical_observations,communication_streams, private_goal):
        
        # Process communication streams
        #communication_features = []
        #for i,stream in enumerate(communication_streams):
        #    processed_stream = self.shared_communication_lstm(stream)
        #    communication_features.append(processed_stream)

        # Process physical observations
        physical_features = []
        for i,observation in enumerate(physical_observations):
            
            
            processed_observation = self.shared_pyhisical_observation_module(observation)
            physical_features.append(processed_observation)

       
        communication_featues = []
        for i,(com,memory) in enumerate(zip(communication_streams,self.com_memory)):
            

            processed_communications = self.shared_communication_module(torch.cat([com,memory]))
          
            
            processed_communication_vector = processed_communications[:-self.memory_size]
            
            
            communication_featues.append(processed_communication_vector)
            
            delta_memory = processed_communications[-self.memory_size:]
            
         
                
            # Sample a zero-mean Gaussian noise tensor
            epsilon = dists.Normal(torch.zeros_like(memory), torch.ones_like(memory)).sample()
            #update memory
            self.com_memory[i] = torch.tanh(memory+delta_memory+epsilon)
            
            
          
       
        
        # Pool features
        pooled_communication_features = torch.sum(self.pooling_layer(torch.stack(communication_featues)),dim=0)
        pooled_physical_features = torch.sum(self.pooling_layer(torch.stack(physical_features)),dim=0)
        
        #print(pooled_communication_features.shape)
        #print(pooled_physical_features.shape)
        #print(private_goal.shape)
        #print(self.final_memory.shape)


        # Combine features
        combined_features = torch.cat([pooled_communication_features, pooled_physical_features, private_goal,self.final_memory], dim=0)
        
       
        
        final_output = self.final_layer(combined_features)
    
    
        delta_memory = final_output[-self.memory_size:]
        
        #update memory
        epsilon = dists.Normal(torch.zeros_like(delta_memory), torch.ones_like(delta_memory)).sample()
        self.final_memory = torch.tanh(self.final_memory+delta_memory+epsilon)
        
        # prepare actions
        actions = final_output[:self.action_space_size]
        action_epsilon = dists.Normal(torch.zeros_like(actions), torch.ones_like(actions)).sample()
        action = actions+action_epsilon
        
        
        # prepare communications
        coms = final_output[self.action_space_size-1:self.memory_size+1]
        communication_symbol = F.gumbel_softmax(coms, tau=1, hard=False)
        
        return action, communication_symbol
        


    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



class Critic(nn.Module):
    def __init__(self, state_size,num_objects, action_size, hidden_dim):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size*num_objects+action_size, hidden_dim,dtype=float),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1,dtype=float)
        )

    def forward(self, state, action):
        x = torch.cat((state.view(-1), action))
        x = self.network(x)
        return x
    
    

# Create an instance of the PolicyNetwork class
#action_space_size = 2
#state_size = 9
#communication_size = 64
#goal_size = 32
#num_communication_streams = 2
#memory_size = 32
#policy_network = Policy_Network(communication_size,num_communication_streams ,state_size,action_space_size, goal_size,memory_size)

# Initialize the network's weights
#torch.nn.init.xavier_uniform_(policy_network.parameters())

# Create input data
#communication_streams = torch.randn(2, 64)
#physical_observations = torch.randn(3, 9)
#private_goal = torch.randn(32)

# Pass the input data through the network
#action, communication_symbol = policy_network(communication_streams, physical_observations, private_goal)

#print(f"Action: {action}")
#print(f"Communication symbol: {communication_symbol}")
