import pygame
import random
from Agents import Prey,Predator
import numpy as np
# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# Define window size
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
LANDMARK_RADIUS = 30


def Gumbel_Softmax(logits , tau=1.0):
    
    z = np.random.gumbel(size=logits.shape)
    
    # Apply Gumbel distortion
    logits += z

    # Apply softmax
    exp_logits = np.exp(logits / tau)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return probs
    



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
                
                x = random.randint(0, SCREEN_WIDTH)
                y = random.randint(0, SCREEN_HEIGHT)
                landmark.append([x,y])

            self.landmarks.append(landmark)
            

    def init_communications(self):
        # need to be same size as x to share weights 
        com_length = len(self.landmarks)*5 + len(self.prey)*9 + len(self.predator)*9
        
        for p in self.prey:
            c = np.random.rand(com_length)
            self.prey_coms.append(Gumbel_Softmax(c))
            
        for p in self.predator:
            c = np.random.rand(com_length)
            self.predator_coms.append(Gumbel_Softmax(c))
    
        
    def update(self):
        pass
        
    def pick_new_landmark(self):
        self.current = random.randint(0, len(self.landmarks)-1)
        
    
    def render(self):
        
        self.screen.fill((255, 255, 255))
 
        for landmark in self.landmarks[self.current]:
            pygame.draw.circle(self.screen, BLUE, (int(landmark[0]), int(landmark[1])), LANDMARK_RADIUS)
        
        for p in self.prey:
            pygame.draw.circle(self.screen, p.color, (p.x, p.y), p.size)
            
        for p in self.predator:
            pygame.draw.circle(self.screen, p.color, (p.x, p.y), p.size)
            
        pygame.display.flip()
            
       

game = Game(SCREEN_WIDTH,SCREEN_HEIGHT)
game.create_landmarks(seed=42,num_worlds=3,num_landmarks=5)

# Game loop
running = True
clock = pygame.time.Clock()
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        
        game.render()
        game.pick_new_landmark()
        clock.tick(2)
      
    
# Quit Pygame
pygame.quit()