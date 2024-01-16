import pygame
import random
from Agents import Prey,Predator
import math
import torch
from collections import deque
import torch.nn.functional as F
import pickle
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
MAX_MEMORY = 100_000

def update_file(file_path,value):
    with open (file_path,'a') as f:
        f.write(str(value) + '\n')

#torch.autograd.set_detect_anomaly(True)
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
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        
        self.best_prey_score = -math.inf
        self.best_predator_score = -math.inf
       
        
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
        
        
    def remember(self,prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states):
        #states : (state,coms,goal)
        #next_staes : (state,coms,goal)
        self.memory.append((prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states))
        
    def train_critics(self,gamma=0.95,batch_size=32):
        print('training critics ...')
        
        #update all preys
        for i,p in enumerate(self.prey):
            
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                mini_sample = self.memory

            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            
            
            next_preadtor_critics = []
            
            for i,pred in enumerate(self.predator):
                
                # get the next predator moves and critic values based on their current parameters
                next_moves = [pred.model(state[i][0],state[i][1],state[i][2])[0] for state in predator_next_states]
                next_preadtor_critics.append([pred.critic(state[i][0], action) for state, action in zip(predator_next_states, next_moves)])
                
            # take  elementwise mean of the predators critics
            combined_list = list(zip(*next_preadtor_critics))
            next_preadtor_critics = [torch.mean(torch.tensor(position_values)).view(-1) for position_values in combined_list]

            # get own moves and critic values based on current parameters
            own_next_moves = [p.model(state[i-1][0],state[i-1][1],state[i-1][2])[0] for state in prey_next_states]
            own_next_criric = [p.critic(state[i-1][0], action) for state, action in zip(prey_next_states, own_next_moves)]
            
            
            # calc target vector y as min of critcs
            #target = torch.tensor(prey_rewards[i-1]) + gamma * torch.min(torch.cat(own_next_criric),torch.cat(next_preadtor_critics))
            
            prey_target = torch.tensor(prey_rewards[i-1]) + gamma * torch.min(torch.cat(own_next_criric), torch.cat(next_preadtor_critics)).detach()
            # get critics based on current state
            prey_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(prey_states, prey_actions)]
            prey_critic_value = torch.cat(prey_critic_value)
            
            # update crtic network
            prey_critic_loss = F.mse_loss(prey_critic_value, prey_target)
            p.critic_optimizer.zero_grad()
            prey_critic_loss.backward(retain_graph=True)
            p.critic_optimizer.step()

            
            # Update the actor network
            #prey_action_loss = -prey_critic_value.mean()
            #prey_action_loss_copy = prey_action_loss.clone()
            #p.actor_optimizer.zero_grad()
            #prey_action_loss_copy.backward()
            #p.actor_optimizer.step()
            
            
           
       
        for i,p in enumerate(self.prey):
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                mini_sample = self.memory

            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            
             
            next_prey_critics = []
            
            for i,prey in enumerate(self.prey):
                
                # get the next predator moves and critic values based on their current parameters
                next_moves = [prey.model(state[i][0],state[i][1],state[i][2])[0] for state in prey_next_states]
                next_prey_critics.append([prey.critic(state[i][0], action) for state, action in zip(prey_next_states, next_moves)])
                
            predator_target = torch.tensor(prey_rewards[i-1]) + gamma * torch.min(torch.cat(own_next_criric), torch.cat(next_preadtor_critics)).detach()
            # get critics based on current state
            predator_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(predator_states, predator_actions)]
            predator_critic_value = torch.cat(predator_critic_value)
       
            # update crtic network
            predator_critic_loss = F.mse_loss(predator_critic_value, predator_target)
            p.critic_optimizer.zero_grad()
            predator_critic_loss.backward(retain_graph=True)
            p.critic_optimizer.step()
            
        
    def train_actors(self,gamma=0.95,batch_size=32):
        print('training actors ...')
        #update all preys
        for i,p in enumerate(self.prey):
            
            #update score
            p.scores.append(p.goal)
            p.n_games += 1
            
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                mini_sample = self.memory

            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            

            # get critics based on current state
            prey_critic_value = [p.critic(state[i][0], torch.tensor(action[i])) for state, action in zip(prey_states, prey_actions)]
            prey_critic_value = torch.cat(prey_critic_value)
            
            # Update the actor network
            prey_action_loss = -prey_critic_value.mean()
            p.actor_optimizer.zero_grad()
            prey_action_loss.backward()
            p.actor_optimizer.step()
        
        
        #update predator actor
        for i,p in enumerate(self.predator):
            
            #update score
            p.scores.append(p.goal)
            p.n_games += 1
            
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                mini_sample = self.memory

            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            

            # get critics based on current state
            predator_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(predator_states, predator_actions)]
            predator_critic_value = torch.cat(predator_critic_value)
            
            # Update the actor network
            predator_action_loss = -predator_critic_value.mean()
            p.actor_optimizer.zero_grad()
            predator_action_loss.backward()
            p.actor_optimizer.step()
            
    
    def eval(self):
        for i,p in  enumerate(self.prey):
            score = float(p.goal.sum())
            
           
            if score > self.best_prey_score:
                self.best_prey_score = score
            
                with open("Project2_continous/model/prey"+str(i)+".pkl", "wb") as f:
                    pickle.dump(p, f)
            
                
            #save the scores
            update_file(file_path="Project2_continous/Scores/prey"+str(i)+".csv",value=score)
            
            
        for i,p in enumerate(self.predator):
            score = float(p.goal.sum())
            
           
            
            if score > self.best_predator_score:
                self.best_predator_score = score
            
                with open("Project2_continous/model/predator"+str(i)+".pkl", "wb") as f:
                    pickle.dump(p, f)
            
                
            #save the scores
            update_file(file_path="Project2_continous/Scores/predator"+str(i)+".csv",value=score)
        
            
            
        
    def update(self):

        old_prey_states = []
        prey_actions = []
        new_prey_coms = []
        new_prey_states = []
        prey_rewards = []
        
        
        old_predator_states = []
        predator_actions = []
        new_predator_coms = []
        new_predator_states = []
        predator_rewards = []
        
            
        # get new coms
        for i,p in enumerate(self.prey):
            self.prey_coms[i] = p.com
            
        for i,p in enumerate(self.predator):
            self.predator_coms[i] = p.com
        
        
        #select actions and coms w.r.t current policy for pry
        for i,p in enumerate(self.prey):
            
            state = p.get_x_vector(self)
            
            old_prey_states.append((state,self.prey_coms[i],p.goal))
    
            action , new_coms = p.get_action_communication(state,self.prey_coms,p.goal)
            p.coms = new_coms
            
            prey_actions.append(action)
            new_prey_coms.append(new_coms)
            
            
            
        
        #select actions and coms w.r.t current policy for predators
        for i,p in enumerate(self.predator):
            
            state = p.get_x_vector(self)
            
            
            old_predator_states.append((state,self.predator_coms[i],p.goal))
    
            action , new_coms = p.get_action_communication(state,self.predator_coms,p.goal)
            p.coms = new_coms
            
            predator_actions.append(action)
            new_predator_coms.append(new_coms)
           
        
        
        
        
        # execute actions 
        for p,action in zip(self.prey,prey_actions):
            #execute the actions
            p.update_pos(action[0],action[1],self)
            
            
        # execute actions 
        for p,action in zip(self.predator,predator_actions):
            #execute the actions
            p.update_pos(action[0],action[1],self)
            
            
        #observe reward and new state
        for i,p in enumerate(self.prey):
        
            #update reward and new state 
            p.update_goal(self)
            reward = p.get_reward(self)
            new_state = p.get_x_vector(self)
            #update coms
           
          
            self.prey_coms[i] = new_prey_coms[i]
            new_prey_states.append((new_state,self.prey_coms,p.goal))
            prey_rewards.append(reward)
            #rememeber
            
            
        #observe reward and new state
        for i,p in enumerate(self.predator):
        
            #update reward and new state 
            p.update_goal(self)
            reward = p.get_reward(self)
            new_state = p.get_x_vector(self)
            #update coms
            
           
            self.predator_coms[i] = new_predator_coms[i]
            new_predator_states.append((new_state,self.predator_coms,p.goal))
            predator_rewards.append(reward)
            #rememeber
         
        self.remember(old_prey_states,prey_actions,prey_rewards,new_prey_states,old_predator_states,predator_actions,predator_rewards,new_predator_states)
        
        
            
     
    def pick_new_landmark(self):
        
        #reset positions
        self.predator[0].pos = [20,20]
        self.predator[1].pos = [SCREEN_WIDTH-20,SCREEN_HEIGHT-20]
        
        self.prey[0].pos = [SCREEN_WIDTH/2,SCREEN_HEIGHT/2]
        
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
            
       



prey1 = Prey(pos = [SCREEN_WIDTH/2,SCREEN_HEIGHT/2],max_velocity=25,color=GREEN,communication_size=32,num_communication_streams=1,state_size=9,goal_size=2)
predator1 =Predator(pos = [20,20],max_velocity=15,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)
predator2 =Predator(pos = [SCREEN_WIDTH-20,SCREEN_HEIGHT-20],max_velocity=15,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)


game = Game(SCREEN_WIDTH,SCREEN_HEIGHT)
game.create_landmarks(seed=42,num_worlds=3,num_landmarks=5)
game.predator=[predator1,predator2]
game.prey = [prey1]
game.init_coms()

max_episode_length = 100
# Game loop
running = True
clock = pygame.time.Clock()
while running:
    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        for i in range(max_episode_length):
            #game.render()
            game.update()
            
            
        game.train_critics(batch_size=32)
        game.train_actors()
        game.eval()
        game.pick_new_landmark()
        
      
    
# Quit Pygame
pygame.quit()