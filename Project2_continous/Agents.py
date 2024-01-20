import random
import numpy as np
from collections import deque
from models import Policy_Network ,Critic,Simple_Actor
import math
import torch
import torch.optim as optim
from utils import normalize_angle ,check_range, adjust_position

# define hyperparameters
LR = 0.01
LANDMARK_RADIUS = 30
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

    
class Agent():
    def __init__(self, pos, size,max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size=2,memory_size=32):
        self.pos = pos
        self.max_velocity = max_velocity
        self.color = color
        self.velocity = 0 # init velocity
        self.theta = math.pi # init angle
        self.n_games = 0 # init n_games
        self.size = size
        self.epsilon = 0 # randomness
        self.scores = []
        
        self.goal = torch.zeros(goal_size,dtype=float) # init private goal vector
        self.com  = torch.zeros(communication_size,dtype=float) # init private communication vector

        # define own model for action, com
        self.model = Policy_Network(communication_size,num_communication_streams ,state_size,action_space_size, goal_size,memory_size,hidden_size=128) 
        
        # define critic for state,action value
        self.critic = Critic(state_size=state_size,num_objects=9,action_size=action_space_size,hidden_dim=128)
        
        # can be igonred
        #self.simpel_model = Simple_Actor(state_size=state_size,num_objects=9,goal_size=goal_size,hidden_dim=128)
        
        # Define the optimizers
        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)
        
    
    def get_own_state(self,world):
        
        #proposed in paper 
        #damping_factor = 0.5
        time_step = 0.1
        x,y = self.pos
        new_x = x + self.velocity * math.cos(self.theta) * time_step
        new_y = y + self.velocity * math.sin(self.theta) * time_step
        
        #calculate physical interaction forces
        
        #look if next position is in contact with:
        #[landmark,prey,predator]  0/1
        l = 0
        for lm in world.landmarks[world.current]:
            if not l:
                l = check_range(new_x,new_y,self.size,lm[0],lm[1],LANDMARK_RADIUS)
                
        prey = 0
        for p in world.prey:
            if not prey and p is not self:
                prey = check_range(new_x,new_y,self.size,p.pos[0],p.pos[1],p.size)
        
        predator = 0
        for p in world.predator:
            if not predator and p is not self:
                predator = check_range(new_x,new_y,self.size,p.pos[0],p.pos[1],p.size)
                
        #alternativly we could calculate the collision forces based on physical interactions or some other way 
        
        
        #Agents observe the relative positions and velocities of the agents, and the positions of the landmarks.
        state = [new_x,new_y,self.velocity,l,prey,predator]+ self.color #theta = orientation is private  
        return state
        
    def get_x_vector(self,world):
        
        # Agents observe the relative positions and velocities of the agents, and the positions of the landmarks.
        state =[self.get_own_state(world)]
        
        x=self.pos[0]
        y=self.pos[1]
        
        # append all predators
        for p in world.predator:
            
            s = p.get_own_state(world)
            s[0] -= x
            s[1] -= y
            
            state.append(s)
        
        # append all prey
        for p in world.prey:
            s = p.get_own_state(world)
            s[0] -= x
            s[1] -= y
            
            state.append(s)
            
        # append the landmarks
        for l in world.landmarks[world.current]:
            
            state.append( [l[0]-x,l[1]-y,0,0,0,0,0,0,255])
        
        # return state [x_1,...,x_n]
        return torch.tensor(state,dtype=float)
    
    def get_action_communication(self, physical_observations,communication_streams,goal):
        
        # define exploration factor
        self.epsilon = 80 - self.n_games
        
        communication_streams = torch.stack(communication_streams,dim=0)

        # get random angle + velocity
        if random.randint(0, 200) < self.epsilon+5: # remain at least 5% exploration rate
            theta = random.uniform(0, 2 * math.pi)
            velocity = random.uniform(0, self.max_velocity)
            prediction = [theta,velocity]
            communication = self.com
            
        # get angle + velocity based on the actor model
        else:
            # pred = theta,velocity
            prediction , communication = self.model(physical_observations,communication_streams,goal)
            
            prediction = prediction.detach().numpy()
            
            prediction[0] = normalize_angle(prediction[0]) # make sure angle is in range
            
            prediction[1] = max(prediction[1],self.max_velocity) # make sure velocity is not over max 
 
        return prediction , communication
    
    
    # can be ignored
    def get_simple_action(self, physical_observations,goal):
        self.epsilon = 80 - self.n_games
        
        # get random angle + velocity
        if random.randint(0, 200) < self.epsilon+5:  # remain at least 5% exploration rate
            theta = random.uniform(0, 2 * math.pi)
            velocity = random.uniform(0, self.max_velocity)
            prediction = [theta,velocity]
           
            
        else:
            # pred = theta,velocity
            
            prediction = self.model(physical_observations,goal)
            
            prediction = prediction.detach().numpy()
            
            prediction[0] = normalize_angle(prediction[0])#  make sure angle is in range
            
            prediction[1] = max(prediction[1],self.max_velocity) # make sure velocity is not over max 

            
        return prediction
    
    #updating position
    def update_pos(self,theta,velocity,world):
        # update veolcity and theta
        self.velocity = velocity
        self.theta = theta
        
        # calculate new position
        new_x = self.pos[0] + self.velocity * math.cos(self.theta) 
        new_y = self.pos[1] + self.velocity * math.sin(self.theta) 
        
        # make sure pos is in bounds
        new_x = max(self.size,min(new_x,SCREEN_WIDTH-self.size))
        new_y = max(self.size,min(new_y,SCREEN_HEIGHT-self.size))
        
        # make sure pos dont intersect with a landmark
        for landmark in world.landmarks[world.current]:
            if check_range(self.pos[0],self.pos[1],self.size,landmark[0],landmark[1],LANDMARK_RADIUS):
                new_x,new_y = adjust_position(self.pos,self.size,landmark,LANDMARK_RADIUS)
        
        # update pos        
        self.pos = [new_x,new_y]
        
    
        
class Predator(Agent):
    def __init__(self,pos, size,max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size=2,memory_size=32):
        super().__init__(pos, size,max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size,memory_size)
   
    # goal vector defines distance to prey
    def update_goal(self,world):
        for i, p in enumerate(world.prey):

            self.goal[i] = np.linalg.norm(np.array(self.pos) - np.array(p.pos))
            
    def get_reward(self,world):
        
        # check for intersection
        hit = any(check_range(self.pos[0], self.pos[1], self.size, p.pos[0], p.pos[1], p.size) for p in world.prey)
        
        # print if catched
        if hit:
            print('catched')
        
        # calc reward 
        reward =  -torch.sum(self.goal)/1000 + 10 * hit
        
        return reward , hit
    
class Prey(Agent):
    def __init__(self,pos,size,max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size=2,memory_size=32):
        super().__init__(pos,size,max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size,memory_size)
        
    # goal vector defines distance to other predators
    def update_goal(self,world):
        for i, p in enumerate(world.predator):
            self.goal[i] = np.linalg.norm(np.array(self.pos) - np.array(p.pos))
    
    def get_reward(self,world):
        
        # check for intersection
        hit = any(check_range(self.pos[0], self.pos[1], self.size, p.pos[0], p.pos[1], p.size) for p in world.predator)
       
        # calc reward 
        reward = torch.sum(self.goal)/1000 - 10 * hit
        
        return reward ,hit