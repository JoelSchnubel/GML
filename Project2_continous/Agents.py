import random
import numpy as np
from collections import deque
from models import Policy_Network, QTrainer
import math
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.01
LANDMARK_RADIUS = 30

def normalize_angle(theta):
    # Ensure theta is within [0, 2*pi)
    return theta % (2 * math.pi)

def check_range(x,y,r, other_x,other_y,other_r):
    distance_to_landmark = math.sqrt((x - other_x)**2 + (y - other_y)**2)
    combined_radius = r + other_r
    return distance_to_landmark <= combined_radius
    

class Agent():
    def __init__(self, pos, max_velocity,color,communication_size,num_communication_streams,state_size,goal_size,action_space_size=2,memory_size=32):
        self.pos = pos
        self.max_velocity = max_velocity
        self.color = color
        self.velocity = 0
        self.theta = math.pi
        self.score = 0
        self.n_games = 0
        self.size = 10
        self.epsilon = 0 # randomness
        self.gamma = 0.95 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        
        self.goal = torch.zeros(goal_size)
        self.com  = torch.zeros(communication_size)

        self.model = Policy_Network(communication_size,num_communication_streams ,state_size,action_space_size, goal_size,memory_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state)) # popleft if MAX_MEMORY is reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states)
        #for state, action, reward, nexrt_state in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state)

    def train_short_memory(self, state, action, reward, next_state):
        self.trainer.train_step(state, action, reward, next_state)
    
    
    def get_own_state(self,world):
        
        #proposed in paper 
        damping_factor = 0.5
        time_step = 0.1
        x,y = self.pos
        new_x = x + self.velocity * math.cos(self.theta) * time_step
        new_y = y + self.velocity * math.sin(self.theta) * time_step
        
        #calculate physical interaction forces
        
        #look if next position is in contact with:
        #[landmark,prey,predator]  0/1
        l = 0
        for lm in world.landmarks:
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
                
        #alternativly we can calculate the collision forces based 
        
        
        
        #Agents observe the relative positions and velocities of the agents, and the positions of the landmarks.
        state = [new_x,new_y,self.velocity,l,prey,predator]+ self.color #theta = orientation is private  
        return state
        
    def get_x_vector(self,world):
        #Agents observe the relative positions and velocities of the agents, and the positions of the landmarks.
        state = self.get_own_state()
        
        x=self.pos[0]
        y=self.pos[1]
        
        #append all predators
        for p in world.predators:
            s = p.get_own_state()
            s[0] -= x
            s[1] -= y
            
            state.append(s)
        
        #append all prey
        for p in world.prey:
            s = p.get_own_state()
            s[0] -= x
            s[1] -= y
            
            state.append(s)
        
        for l in world.landmarks[world.current]:
            
            state.append( [l[0]-x,l[1]-y,0,0,0,0,0,0,255])
        
        return state
    
    
#agent = Agent([1,1],max_velocity=10,color=[255,0,0])
       
#agent.get_own_state()
        
class Predator(Agent):
    def __init__(self):
        super().__init__(self)
   
    #goal vector defines distance to prey
    def update_goal(self,world):
        for i, p in enumerate(world.prey):
            self.goal[i] = np.linalg.norm(self.pos - p.pos)
            
    
    def get_action(self, state , world):
        self.epsilon = 80 - self.n_games
        
        final_move = [0,0]
        if random.randint(0, 200) < self.epsilon:
            self.theta = random.uniform(0, 2 * math.pi)
            self.velocity = random.uniform(0, self.max_velocity)
            
   
        else:
            pass
            #state0 = torch.tensor(state, dtype=torch.float)
            #prediction = self.model(state0)
            #move = torch.argmax(prediction).item()
            #final_move[move] = 1

        return final_move
        pass
   
    
class Prey(Agent):
    def __init__(self):
        super().__init__(self)
        
    #goal vector defines distance to other predators
    def update_goal(self,world):
        for i, p in enumerate(world.predators):
            self.goal[i] = np.linalg.norm(self.pos - p.pos)
        