import torch
import random
import numpy as np
from collections import deque
from enum import Enum
from models import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.015

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    



class Solo_Q_Agent():
    def __init__(self, pos ,perception_radius):
        self.pos = pos #[x,y]
        self.score = 0
        self.done = False
        self.perception_radius = perception_radius
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet((2*self.perception_radius+1)**2, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
   



class Solo_Q_Agent_Rabbit(Solo_Q_Agent):
    
    def __init__(self, pos, perception_radius):
        super().__init__(pos, perception_radius)
    
    def get_state(self, world):
        agent_x, agent_y = self.pos

        # Calculate grid size based on perception radius
        grid_size = 2 * self.perception_radius + 1

        # Initialize the grid
        self.grid = np.zeros((grid_size, grid_size))

        # Update the grid with the presence of the agent
        self.grid[self.perception_radius, self.perception_radius] = 5
        
        
        rabbits_pos = [rabbit.pos for rabbit in world.rabbits]
        foxes_pos = [fox.pos for fox in world.foxes]

        # Update the grid with the presence of rocks and carrots within perception_radius
        for obj_type, obj_list in [('rocks', world.rocks), ('carrots', world.carrots), ('foxes', foxes_pos), ('rabbits', rabbits_pos)]:
            for obj_x, obj_y in obj_list:
                rel_x, rel_y = obj_x - agent_x, obj_y - agent_y
                if abs(rel_x) <= self.perception_radius and abs(rel_y) <= self.perception_radius:
                    grid_x, grid_y = rel_x + self.perception_radius, rel_y + self.perception_radius
                    # Ensure the object is within the grid bounds
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Mark carrot with 1
                        if obj_type == 'carrots':
                            self.grid[grid_y, grid_x] = 1
                        # Mark rock with -1
                        elif obj_type == 'rocks' :
                            self.grid[grid_y, grid_x] = -1
                        # Mark foxes with -10
                        elif obj_type == 'foxes':
                            self.grid[grid_y, grid_x] = -10
                        # Example: Mark other rabbits with a different value, e.g., -5
                        elif obj_type == 'rabbits' and (obj_x, obj_y) != (agent_x, agent_y):
                            self.grid[grid_y, grid_x] = -5
                            
        
        # Mark boundaries with -1
        for i in range(grid_size):
            for j in range(grid_size):
                if (agent_x - self.perception_radius + i) < 0 or (agent_x - self.perception_radius + i) >= world.rows \
                        or (agent_y - self.perception_radius + j) < 0 or (agent_y - self.perception_radius + j) >= world.columns:
                    self.grid[j, i] = -1

        # Flatten the grid to create a fixed-size vector
        state = np.concatenate(self.grid).flatten()


        return np.array(state)
    

class Solo_Q_Agent_Fox(Solo_Q_Agent):
    
    def __init__(self, pos, perception_radius):
        super().__init__(pos, perception_radius)
    
    def get_state(self, world):
        agent_x, agent_y = self.pos

        # Calculate grid size based on perception radius
        grid_size = 2 * self.perception_radius + 1

        # Initialize the grid
        self.grid = np.zeros((grid_size, grid_size))

        # Update the grid with the presence of the agent
        self.grid[self.perception_radius, self.perception_radius] = 5

        rabbits_pos = [rabbit.pos for rabbit in world.rabbits]
        foxes_pos = [fox.pos for fox in world.foxes]
        
        # Update the grid with the presence of rocks and carrots within perception_radius
        for obj_type, obj_list in [('rocks', world.rocks), ('carrots', world.carrots), ('foxes', foxes_pos), ('rabbits', rabbits_pos)]:
            for obj_x, obj_y in obj_list:
                rel_x, rel_y = obj_x - agent_x, obj_y - agent_y
                if abs(rel_x) <= self.perception_radius and abs(rel_y) <= self.perception_radius:
                    grid_x, grid_y = rel_x + self.perception_radius, rel_y + self.perception_radius
                    # Ensure the object is within the grid bounds
                    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                        # Mark carrot with 1
                        if obj_type == 'carrots':
                            self.grid[grid_y, grid_x] = 0
                        # Mark rock with -1
                        elif obj_type == 'rocks' :
                            self.grid[grid_y, grid_x] = -1
                        # Mark foxes with -10
                        elif obj_type == 'rabbits':
                            self.grid[grid_y, grid_x] = 10
                        # Example: Mark other rabbits with a different value, e.g., -5
                        elif obj_type == 'foxes' and (obj_x, obj_y) != (agent_x, agent_y):
                            self.grid[grid_y, grid_x] = -5
                            
        
        # Mark boundaries with -1
        for i in range(grid_size):
            for j in range(grid_size):
                if (agent_x - self.perception_radius + i) < 0 or (agent_x - self.perception_radius + i) >= world.rows \
                        or (agent_y - self.perception_radius + j) < 0 or (agent_y - self.perception_radius + j) >= world.columns:
                    self.grid[j, i] = -1

        # Flatten the grid to create a fixed-size vector
        state = np.concatenate(self.grid).flatten()

        return np.array(state)
    