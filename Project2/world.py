import pygame
import random
from Agents import Prey,Predator
import math
import torch
from collections import deque
import torch.nn.functional as F
import pickle
import numpy as np
import csv
from utils import check_range,update_file
import time

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = [255,0,0]
GREEN = [0,255,0]
BLUE = [0,0,255]

# Define window size
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
LANDMARK_RADIUS = 30
MAX_MEMORY = 10_000
BATCH_SIZE = 256

class Game:
    def __init__(self,width,height,prey=[],predator=[]): 
        self.width = width # screen width
        self.height = height # screen height
        self.landmarks = []
        self.prey = prey
        self.predator = predator
        self.screen = pygame.display.set_mode((width, height))
        self.current = 0 # current landmark index 
        self.predator_coms = []
        self.prey_coms = []
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if full
        
        # init scores 
        self.best_prey_score = -math.inf
        self.best_predator_score = -math.inf
        
    # create n_world maps with num_landmarks and storing them. 
    # can be access using self.current 
    # make sure to first create the agents and the create the landmarks 
    def create_landmarks(self,seed,num_worlds,num_landmarks):
        print('creating landmarks ...')
        
        # get positions of all agents
        agent_positions = []
        for p in self.predator:
            agent_positions.append(p.pos)
            
        for p in self.prey:
            agent_positions.append(p.pos)

        size = self.prey[0].size
        
        # set seed
        random.seed(seed)
        
        # create the landmarks
        for _ in range(num_worlds):
            landmark = []
            for _ in range(num_landmarks):
                
                x = random.randint(LANDMARK_RADIUS, SCREEN_WIDTH-LANDMARK_RADIUS)
                y = random.randint(LANDMARK_RADIUS, SCREEN_HEIGHT-LANDMARK_RADIUS)

                #make sure landmarks and agents dont overlap
                while any(check_range(x, y, LANDMARK_RADIUS + 20, pos[0], pos[1], size) for pos in agent_positions):
                    x = random.randint(LANDMARK_RADIUS, SCREEN_WIDTH-LANDMARK_RADIUS)
                    y = random.randint(LANDMARK_RADIUS, SCREEN_HEIGHT-LANDMARK_RADIUS)
                
                    
                landmark.append([x,y])

            self.landmarks.append(landmark)
            
    # initilizing the coms 
    # make sure to first create the agents
    def init_coms(self):
        print('initilize communications ...')
        self.prey_coms = [0] * len(self.prey)
        self.predator_coms = [0] * len(self.predator)
        
    # append state to memory
    def remember(self,prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states):
        # states : (state,coms,goal)
        # next_staes : (state,coms,goal)
        # actions : (theta , velocity)
        self.memory.append((prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states))
        
    # train all the agents 
    def train_critics(self,gamma=0.95,batch_size=BATCH_SIZE,simple_prey=False,simple_predator=False):
        # only train if memory > batchsize
        if len(self.memory) > batch_size:
            print('training critics ...')
        
        # update all prey critics
        for i,p in enumerate(self.prey):
            
            # only train if memory is ar least of size = batch_size
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                return
            
            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            
            # get the next predator moves and critic values based on their current parameters
            next_predator_critics = []
            for i,pred in enumerate(self.predator):
                if simple_predator:
                    next_moves = [pred.simpel_model(state[i][0],state[i][2]) for state in predator_next_states]
                else:
                    next_moves = [pred.model(state[i][0],state[i][1],state[i][2])[0] for state in predator_next_states]
                next_predator_critics.append([pred.critic(state[i][0], action) for state, action in zip(predator_next_states, next_moves)])
                
            # take elementwise mean of the predators critics
            combined_list = list(zip(*next_predator_critics))
            next_predator_critics = [torch.mean(torch.tensor(position_values)).view(-1) for position_values in combined_list]

            # get own moves and critic values based on current parameters
            if simple_prey:
                own_next_moves = [p.simpel_model(state[i-1][0],state[i-1][2]) for state in prey_next_states]
            else:
                own_next_moves = [p.model(state[i-1][0],state[i-1][1],state[i-1][2])[0] for state in prey_next_states]
            prey_own_next_critic = [p.critic(state[i-1][0], action) for state, action in zip(prey_next_states, own_next_moves)]
            
            # calc target vector y as min of critcs        
            prey_target = torch.tensor(prey_rewards[i-1]) + gamma * torch.min(torch.cat(prey_own_next_critic), torch.cat(next_predator_critics)).detach()
            
            # get critics based on current parameters
            prey_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(prey_states, prey_actions)]
            prey_critic_value = torch.cat(prey_critic_value)
            
            # update crtic network
            prey_critic_loss = F.mse_loss(prey_critic_value, prey_target)
            p.critic_optimizer.zero_grad()
            prey_critic_loss.backward(retain_graph=True)
            p.critic_optimizer.step()


        # update all prey predator
        for i,p in enumerate(self.predator):
            
            # only train if memory is ar least of size = batch_size
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                return
    
            # unload the memory
            prey_states,prey_actions,prey_rewards,prey_next_states,predator_states,predator_actions,predator_rewards,predator_next_states= zip(*mini_sample)
            
            # transpose for easier accessing down the line
            predator_rewards = np.transpose(np.array(predator_rewards))
            
            # get the next prey moves and critic values based on their current parameters
            next_prey_critics = []
            for i,prey in enumerate(self.prey):
                
                if simple_prey:
                    next_moves = [prey.simpel_model(state[i][0],state[i][2]) for state in prey_next_states]
                else:
                    next_moves = [prey.model(state[i][0],state[i][1],state[i][2])[0] for state in prey_next_states]
                next_prey_critics.append([prey.critic(state[i][0], action) for state, action in zip(prey_next_states, next_moves)])
            
            combined_list = list(zip(*next_prey_critics))
            next_prey_critics = [torch.mean(torch.tensor(position_values)).view(-1) for position_values in combined_list]
            
            if simple_predator:
                own_next_moves = [p.simpel_model(state[i][0],state[i][2]) for state in predator_next_states]
            else:
                own_next_moves = [p.model(state[i][0],state[i][1],state[i][2])[0] for state in predator_next_states]
                
            predator_own_next_criric = [p.critic(state[i][0], action) for state, action in zip(predator_next_states, own_next_moves)]
                
            # calc target vector y as min of critcs  
            predator_target = torch.tensor(predator_rewards[i]) + gamma * torch.min(torch.cat(predator_own_next_criric), torch.cat(next_prey_critics)).detach()
            
            # get own critics based on current parameters
            predator_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(predator_states, predator_actions)]
            predator_critic_value = torch.cat(predator_critic_value)
       
            # update crtic network
            predator_critic_loss = F.mse_loss(predator_critic_value, predator_target)
            p.critic_optimizer.zero_grad()
            predator_critic_loss.backward(retain_graph=True)
            p.critic_optimizer.step()
            
    
    # train all the agents actors
    def train_actors(self,batch_size=BATCH_SIZE):
        # only train if memory > batchsize
        if len(self.memory) > batch_size:
            print('training actors ...')
            
        # update all preys actors
        for i,p in enumerate(self.prey):
            
            #update score and n_games
            p.scores.append(p.goal)
            p.n_games += 1
            
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                return
                
            # unload the memory
            prey_states,prey_actions,_,_,predator_states,predator_actions,_,_= zip(*mini_sample)
            
            # get critics based on current state
            prey_critic_value = [p.critic(state[i][0], torch.tensor(action[i])) for state, action in zip(prey_states, prey_actions)]
            prey_critic_value = torch.cat(prey_critic_value)
            
            # Update the actor network
            prey_action_loss = -prey_critic_value.mean()
            p.actor_optimizer.zero_grad()
            prey_action_loss.backward()
            p.actor_optimizer.step()
        
        
        # update predator actor
        for i,p in enumerate(self.predator):
            
            #update score and n_games
            p.scores.append(p.goal)
            p.n_games += 1
            
            if len(self.memory) > batch_size:
                mini_sample = random.sample(self.memory, batch_size) # list of tuples
            else:
                return
                
            # unload the memory
            prey_states,prey_actions,_,_,predator_states,predator_actions,_,_= zip(*mini_sample)
        
            # get critics based on current state
            predator_critic_value = [p.critic(state[i-1][0], torch.tensor(action[i-1])) for state, action in zip(predator_states, predator_actions)]
            predator_critic_value = torch.cat(predator_critic_value)
            
            # Update the actor network
            predator_action_loss = -predator_critic_value.mean()
            p.actor_optimizer.zero_grad()
            predator_action_loss.backward()
            p.actor_optimizer.step()
            
            
    # evaluating the models and appending scores to the respectiv .csv file
    def eval(self,prey_name='',predator_name=''):
        
        # eval prey agents
        for i,p in  enumerate(self.prey):
            
            # increase score per game such that later models get more likely stored
            score ,_ = p.get_reward(self)
            score = float(score) + p.n_games

            if score > self.best_prey_score: # update global best score
                self.best_prey_score = score

                # store the model
                with open("Project2/model/prey"+str(i)+str(prey_name)+".pkl", "wb") as f:
                    pickle.dump(p, f)
            
            #save the scores
            update_file(file_path="Project2/Scores/prey"+str(i)+str(prey_name)+".csv",value=score-p.n_games)
            
        # eval predator agents
        for i,p in enumerate(self.predator):
            
             # increase score per game such that later models get more likely stored
            score ,_ = p.get_reward(self)
            score = float(score) + p.n_games
            
            if score > self.best_predator_score: # update global best score
                self.best_predator_score = score
                
                # store the model
                with open("Project2/model/predator"+str(i)+str(predator_name)+".pkl", "wb") as f:
                    pickle.dump(p, f)
            
            #save the scores
            update_file(file_path="Project2/Scores/predator"+str(i)+str(predator_name)+".csv",value=score-p.n_games)
        
            
    # Loading models
    def load_models(self,prey_name='',predator_name=''):
        print('Loading models ...')
        
        for i,p in enumerate(self.predator):
            with open("Project2/model/predator"+str(i)+str(predator_name)+".pkl", "rb") as f:
                predator_model = pickle.load(f)
            self.predator[i] = predator_model
            
            with open ("Project2/Scores/predator"+str(i)+str(predator_name)+".csv",'r') as f:
                reader = csv.reader(f)
                values = [float(row[0]) for row in reader]
                if max(values) > self.best_predator_score:
                    self.best_predator_score = max(values)
            p.num_games = len(values)
                    
        for i,p in enumerate(self.prey):
            with open("Project2/model/prey"+str(i)+str(prey_name)+".pkl", "rb") as f:
                prey_model = pickle.load(f)
            self.prey[i] = prey_model
            
            with open ("Project2/Scores/prey"+str(i)+str(prey_name)+".csv",'r') as f:
                reader = csv.reader(f)
                values = [float(row[0]) for row in reader]
                if max(values) > self.best_prey_score:
                    self.best_prey_score = max(values)
            p.num_games = len(values)
            
    # update the agents     
    def update(self,simple_Prey=False,simple_Predator=False):

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
        
        done = 0
    
        # get new coms
        for i,p in enumerate(self.prey):
            self.prey_coms[i] = p.com
            
        for i,p in enumerate(self.predator):
            self.predator_coms[i] = p.com
        
        
        #select actions and coms w.r.t current policy for prey
        for i,p in enumerate(self.prey):
            
            state = p.get_x_vector(self)
            
            old_prey_states.append((state,self.prey_coms[i],p.goal))

            if simple_Prey:
                action =  p.get_simple_action(state,p.goal)
                new_coms = 0
            else:
                action , new_coms = p.get_action_communication(state,self.prey_coms,p.goal)
            p.coms = new_coms
            
            prey_actions.append(action)
            new_prey_coms.append(new_coms)
            
            
            
        
        #select actions and coms w.r.t current policy for predators
        for i,p in enumerate(self.predator):
            
            state = p.get_x_vector(self)
            
            old_predator_states.append((state,self.predator_coms[i],p.goal))
            if simple_Predator:
                action =  p.get_simple_action(state,p.goal)
                new_coms = 0
            else:
                action , new_coms = p.get_action_communication(state,self.predator_coms,p.goal)
            p.coms = new_coms
            
            predator_actions.append(action)
            new_predator_coms.append(new_coms)
           
        # execute actions 
        for p,action in zip(self.prey,prey_actions):
            p.update_pos(action[0],action[1],self)
            
            
        # execute actions 
        for p,action in zip(self.predator,predator_actions):
            p.update_pos(action[0],action[1],self)
            
            
        #observe reward and new state
        for i,p in enumerate(self.prey):
        
            #update reward and new state 
            p.update_goal(self)
            reward, hit = p.get_reward(self)
            new_state = p.get_x_vector(self)
            
            #update coms
            self.prey_coms[i] = new_prey_coms[i]
            new_prey_states.append((new_state,self.prey_coms,p.goal))
            prey_rewards.append(reward)
        
            done = hit or done
            
            
        # observe reward and new state
        for i,p in enumerate(self.predator):
        
            # update reward and new state 
            p.update_goal(self)
            reward , hit = p.get_reward(self)
            new_state = p.get_x_vector(self)
            
            #update coms
            self.predator_coms[i] = new_predator_coms[i]
            new_predator_states.append((new_state,self.predator_coms,p.goal))
            predator_rewards.append(reward)
            
            done = hit or done
        
        # rememeber
        self.remember(old_prey_states,prey_actions,prey_rewards,new_prey_states,old_predator_states,predator_actions,predator_rewards,new_predator_states)
        
        
        return done

    # pick a new landmark and reset the position of the agents
    def pick_new_landmark(self):
        
        #reset positions
        self.predator[0].pos = [20,20]
        self.predator[1].pos = [SCREEN_WIDTH-20,SCREEN_HEIGHT-20]
        
        self.prey[0].pos = [SCREEN_WIDTH/2,SCREEN_HEIGHT/2]
        
        #change the landmark map
        self.current = random.randint(0, len(self.landmarks)-1)
        
    
    # rendering pygame
    def render(self):
        
        self.screen.fill((255, 255, 255))
 
        for landmark in self.landmarks[self.current]:
            pygame.draw.circle(self.screen, BLUE, (int(landmark[0]), int(landmark[1])), LANDMARK_RADIUS)
        
        for p in self.prey:
            pygame.draw.circle(self.screen, p.color, (p.pos[0], p.pos[1]), p.size)
            
        for p in self.predator:
            pygame.draw.circle(self.screen, p.color, (p.pos[0], p.pos[1]), p.size)
            
        pygame.display.flip()
            
       

if __name__ == '__main__':
    
    # set to True or False wheater you want to load models or not
    # name of the models to load can be set further down the code
    load_models = True
    # set to True or False wheater you want to train or test 
    test_mode = True
    
    max_episode_length = 120
    epochs = 200
    
    # define prey
    prey1 = Prey(pos = [SCREEN_WIDTH/2,SCREEN_HEIGHT/2],size=15 ,max_velocity=25,color=GREEN,communication_size=32,num_communication_streams=1,state_size=9,goal_size=2)

    # define predators
    predator1 =Predator(pos = [20,20],size=15, max_velocity=15,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)
    predator2 =Predator(pos = [SCREEN_WIDTH-20,SCREEN_HEIGHT-20],size=15 ,max_velocity=15,color=RED,communication_size=32,num_communication_streams=2,state_size=9,goal_size=1)

    # init state
    game = Game(SCREEN_WIDTH,SCREEN_HEIGHT)

    # inputing agents to the game world
    game.predator=[predator1,predator2]
    game.prey = [prey1]
    game.create_landmarks(seed=42,num_worlds=3,num_landmarks=5)
    game.init_coms()
    #game.render()
    
    if load_models:
        # set the name of the models you want to load
        # notice they get additionally assigned a number
        game.load_models(predator_name='MADDPG_coms_vs_simple',prey_name='MADDPG_coms_vs_simple')

    # main Loop 
    for _ in range(epochs):
        done = 0
        
        for _ in range(max_episode_length):
            if not done:
                #game.render()
                done = game.update(simple_Predator=False,simple_Prey=True)
        
        if not test_mode:        
            game.train_critics(simple_predator=False,simple_prey=True)
            game.train_actors()
            # set the names of the models to train
            game.eval(predator_name='MADDPG_coms_vs_simple',prey_name='MADDPG_coms_vs_simple')
        else:     
            # these names are just set for the .csv file to append the values.
            # the test models can be ignored 
            game.eval(predator_name='MADDPG_coms_vs_simple_test',prey_name='MADDPG_coms_vs_simple_test')
        
        game.pick_new_landmark()
        
        
        
    # Quit Pygame
    pygame.quit()

  