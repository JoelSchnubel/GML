import pygame
import random
from Agents import Prey,Predator
import numpy as np
import torch
# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED =[255,0,0]
GREEN = [0,255,0]
BLUE = [0,0,255]

# Define window size
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LANDMARK_RADIUS = 30



class Game:
    def __init__(self,width,height,prey=[],predator=[]):
        self.width = width
        self.height = height
        self.landmarks = []
        self.prey = prey
        self.predator = predator
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.current = 0
        self.predator_coms = []
        self.prey_coms = []
        
    def create_landmarks(self,seed,num_worlds,num_landmarks):
 
        random.seed(seed)
        for i in range(num_worlds):
            landmark = []
            for j in range(num_landmarks):
                
                x = random.randint(LANDMARK_RADIUS, SCREEN_WIDTH-LANDMARK_RADIUS)
                y = random.randint(LANDMARK_RADIUS, SCREEN_HEIGHT-LANDMARK_RADIUS)
                landmark.append([x,y])

            self.landmarks.append(landmark)
            
    
    def init_coms(self):
        self.prey_coms = [0] * len(self.prey)
        self.predator_coms = [0] * len(self.predator)
        
    def update(self):

        #get new coms
        for i,p in enumerate(self.prey):
            self.prey_coms[i] = p.com
        
        
        for i,p in enumerate(self.prey):
            state = p.get_x_vector(self)
           
            action , new_coms = p.get_action_communication(self.prey_coms,state)

            p.coms = new_coms
            
            p.update_goal(self)
            reward = p.get_reward(self)
                
            new_state = p.get_x_vector(self)
            p.remember(state,action,reward,new_state,self.prey_coms)
            
            p.train_short_memory(state, action, reward, new_state,self.prey_coms)
        
            
            p.update_pos(action[0],action[1],self)
            #update coms
            self.prey_coms[i] = new_coms
            
            
        #get new coms
        for i,p in enumerate(self.predator):
            self.predator_coms[i] = p.com
        
        
        for i,p in enumerate(self.predator):
            
            state = p.get_x_vector(self)
            
            print('pred1',self.predator_coms[0].shape)
            print('pred2',self.predator_coms[1].shape)
            
            action , new_coms = p.get_action_communication(self.predator_coms,state)

            p.coms = new_coms
            
            p.update_goal(self)
            reward = p.get_reward(self)
                
            new_state = p.get_x_vector(self)
            p.remember(state,action,reward,new_state,self.predator_coms)
            
            p.train_short_memory(state, action, reward, new_state,self.predator_coms)
            
            p.update_pos(action[0],action[1],self)
            #update coms
            self.predator_coms[i] = new_coms

        

        
    
    def pick_new_landmark(self):
        self.current = random.randint(0, len(self.landmarks)-1)
        
    
    def render(self):
        
        self.screen.fill((255, 255, 255))
 
        for landmark in self.landmarks[self.current]:
            pygame.draw.circle(self.screen, BLUE, (int(landmark[0]), int(landmark[1])), LANDMARK_RADIUS)
        
        for p in self.prey:
            pygame.draw.circle(self.screen, p.color, (p.pos[0], p.pos[1]), p.size)
            
        for p in self.predator:
            pygame.draw.circle(self.screen, p.color, (p.pos[0], p.pos[1]), p.size)
            
        pygame.display.flip()
            
       



prey1 = Prey(pos = [SCREEN_WIDTH/2,SCREEN_HEIGHT/2],max_velocity=10,color=GREEN,communication_size=32,num_communication_streams=1,state_size=9,goal_size=2)
predator1 =Predator(pos = [20,20],max_velocity=5,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)
predator2 =Predator(pos = [SCREEN_WIDTH-20,SCREEN_HEIGHT-20],max_velocity=5,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)


game = Game(SCREEN_WIDTH,SCREEN_HEIGHT)
game.create_landmarks(seed=42,num_worlds=3,num_landmarks=5)
game.predator=[predator1,predator2]
game.prey = [prey1]
game.init_coms()


# Game loop
running = True
clock = pygame.time.Clock()
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        
        game.render()
        game.update()
        #game.pick_new_landmark()
        clock.tick(2)
      
    
# Quit Pygame
pygame.quit()