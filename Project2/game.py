import pygame , random 
import numpy as np
import sys
from Agents import Solo_Q_Agent_Rabbit,Solo_Q_Agent_Fox
from utils import plot
import torch 
import pickle
import math

WIDTH = 30
HEIGHT = 30
FPS = 60
GRID_SIZE = 30


ROCK_IMAGE = pygame.image.load("Project2/Assets/rock.png")
ROCK_IMAGE = pygame.transform.scale(ROCK_IMAGE,(GRID_SIZE,GRID_SIZE))

CARROT_IMAGE = pygame.image.load("Project2/Assets/carrot.png")
CARROT_IMAGE = pygame.transform.scale(CARROT_IMAGE,(GRID_SIZE,GRID_SIZE))

RABBIT_IMAGE = pygame.image.load("Project2/Assets/rabbit2.png")
RABBIT_IMAGE = pygame.transform.scale(RABBIT_IMAGE,(GRID_SIZE,GRID_SIZE))

FOX_IMAGE = pygame.image.load("Project2/Assets/fox.png")
FOX_IMAGE = pygame.transform.scale(FOX_IMAGE,(GRID_SIZE,GRID_SIZE))


def update_file(file_path,values):
    with open (file_path,'a') as f:
        f.write(','.join(map(str, values)) + '\n')

def perlin_noise(width, height):
    # Generate a 2D array of random values
    noise = np.random.rand(width, height)

    # Apply a Perlin noise filter to the array
    perlin_noise = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            neighbors = []
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= i + dx < width and 0 <= j + dy < height:
                        neighbors.append(noise[i + dx, j + dy])
            perlin_noise[i, j] = sum(neighbors) / len(neighbors)

    return perlin_noise



class World:
    def __init__(self,rows,columns,rocks=[],carrots=[],rabbits=[],foxes=[],num_iter=100):
        self.rabbit_plot_scores = []
        self.rabbit_plot_mean_scores = []
        self.rabbit_total_score = 0
        self.rabbit_record_score = -math.inf
        
        self.fox_plot_scores = []
        self.fox_plot_mean_scores = []
        self.fox_total_score = 0
        self.fox_record_score = -math.inf
        
       
        self.num_iter = num_iter
        self.dead_rabbits = []
        self.rabbits = rabbits
        self.foxes = foxes
        self.rows = rows
        self.columns = columns
        self.rocks = rocks
        self.carrots = carrots
        self.display = pygame.display.set_mode((GRID_SIZE * columns, GRID_SIZE * rows))
        pygame.display.set_caption("World")
        
    def generate_world(self,noise_function,rock_intensity,carrot_intensity):
        
        
        rock_noise = noise_function(WIDTH,HEIGHT)
        carrot_noise = noise_function(WIDTH,HEIGHT)
         
        for i in range(WIDTH):
            for j in range(HEIGHT):
                if rock_noise[i, j] > rock_intensity :
                    self.rocks.append([i, j])
                if carrot_noise[i, j] > carrot_intensity :
                    self.carrots.append([i, j])
                    
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)
                
        #make sure at least one carrot is spawned  
        while [x,y]  in self.rocks:
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
                
            self.carrots.append([x,y])
                    
      
        
        
    def move_agent(self,agent,move):
        x, y = agent.pos
        new_pos = None  # Initialize new_pos to None
        old_pos = agent.pos
        
        rabbits_pos = [rabbit.pos for rabbit in world.rabbits]
        foxes_pos = [fox.pos for fox in world.foxes]
 
        #UP
        if move[0] == 1 and y > 0:    
            new_pos = [x, y - 1]
          
        #DOWN
        if move[1] == 1 and y < HEIGHT // GRID_SIZE - 1:    
            new_pos =  [x, y + 1]
                
        #LEFT
        if move[2] == 1 and x > 0:   
            new_pos =  [x-1, y]
                
        #RIGHT
        if move[3] == 1 and x < WIDTH // GRID_SIZE - 1:  
            new_pos =  [x+1, y]
                
        # Check if new position is valid
        if new_pos is not None and new_pos not in self.rocks:
            agent.pos = new_pos
                
        else:
            agent.score += -1
            return -1
            
            
              
        if type(agent) == Solo_Q_Agent_Rabbit:
            if new_pos in self.carrots:
                    #create new carrot
                    agent.score += 10
                    self.carrots.remove(agent.pos)
                    
                    x = random.randint(0, WIDTH - 1)
                    y = random.randint(0, HEIGHT - 1)
            
                    while [x,y]  in self.rocks or [x,y]  in rabbits_pos or [x,y]  in foxes_pos:
                        x = random.randint(0, WIDTH - 1)
                        y = random.randint(0, HEIGHT - 1)
                    self.carrots.append([x,y])
                    return 10
                
            if new_pos in rabbits_pos:
                agent.pos = old_pos
                agent.score += -1
                return -1
                
            if new_pos in foxes_pos:
                    rabbits_pos = [rabbit.pos for rabbit in world.rabbits]
                    self.dead_rabbits.append(agent)
                    agent.done = True
                    self.rabbits.pop(rabbits_pos.index(new_pos))
                    agent.score += -100
                    return -100
        else:
            
            if new_pos in foxes_pos:
                agent.pos = old_pos
                agent.score += -1
                return -1
                
            if new_pos in rabbits_pos:
                    target_rabbit = self.rabbits[rabbits_pos.index(agent.pos)]
                    target_rabbit.done = True
                    target_rabbit.score = -100
                    self.dead_rabbits.append(target_rabbit)
                    self.rabbits.pop(rabbits_pos.index(agent.pos))
                    agent.score += 100

                    return 100
            
        return 0
     
    def update(self): # update + train
        
        #move and train rabbits
        for rabbit in self.rabbits:
            # get old state
            state_old = rabbit.get_lstm_state(self)

            
            # get move
            final_move = rabbit.get_action(state_old)
           
            #move the rabbit and get reward
            reward = self.move_agent(rabbit,final_move)
            
            # perform move and get new state
            #reward, score = game.play_step(final_move)
       
            #get new state
            state_new = rabbit.get_lstm_state(self)

            # train short memory
            rabbit.train_short_memory(state_old, final_move, reward, state_new,rabbit.done)

            # remember
            rabbit.remember(state_old, final_move, reward, state_new,rabbit.done)
            
            
           #move and train rabbits
        for fox in self.foxes:
            # get old state
            state_old = fox.get_lstm_state(self)

            # get move
            final_move = fox.get_action(state_old)

            #move the fox
            reward = self.move_agent(fox,final_move)

            
            state_new = fox.get_lstm_state(self)

            # train short memory
            fox.train_short_memory(state_old, final_move, reward, state_new,fox.done)

            # remember
            fox.remember(state_old, final_move, reward, state_new,fox.done)
            
      
                
        
    def reset(self,noise_function,rock_intensity,carrot_intensity, current_iter):
        self.num_iter = 100
        self.rocks = []
        self.carrots =[]
        
        self.generate_world(noise_function,rock_intensity,carrot_intensity)

        self.rabbits.extend(self.dead_rabbits)
        self.dead_rabbits = []

        print('training long term memory ...')
        scores = []
        for rabbit in self.rabbits:
        
            scores.append(rabbit.score)
            
            rabbit.n_games +=1
            rabbit.train_long_memory()
            
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
            
            while [x,y]  in self.rocks:
                    x = random.randint(0, WIDTH - 1)
                    y = random.randint(0, HEIGHT - 1)
            

            rabbit.pos = [x,y]
            
            
            
        
        score = self.rabbits[np.argmax(scores)].score
            
        if score > self.rabbit_record_score:
            self.rabbit_score = score
            #self.rabbits[np.argmax(scores)].model.save("Solo_Q_Rabbit.pth")
            # Save the model to a file
            with open("Project2/model/Solo_Q_Rabbit.pkl", "wb") as f:
                pickle.dump(self.rabbits[np.argmax(scores)], f)
            
            
        #save the scores
        update_file(file_path='Project2/Scores/Solo_Q_Rabbit.csv',values=scores)
        
        #reset score
        for rabbit in self.rabbits:
            rabbit.score = current_iter
        
        
        #self.rabbit_plot_scores.append(score)
        #self.rabbit_total_score += score
        #mean_score = self.rabbit_total_score / rabbit.n_games
        #self.rabbit_plot_mean_scores.append(mean_score)
        #plot(self.rabbit_plot_scores, self.rabbit_plot_mean_scores)
                
            
        if len(self.foxes) > 1:
            scores = []
            for fox in self.foxes:
                
                scores.append(fox.score)
                
                fox.n_games +=1
                fox.train_long_memory()
                
                x = random.randint(0, WIDTH - 1)
                y = random.randint(0, HEIGHT - 1)

                fox.pos = [x,y]
            
                
            score = self.foxes[np.argmax(scores)].score
                
            if score > self.fox_record_score:
                self.fox_score = score
                #self.foxes[np.argmax(score)].model.save("Solo_Q_Fox.pth")
                with open("Project2/model/Solo_Q_Fox.pkl", "wb") as f:
                    pickle.dump(self.foxes[np.argmax(scores)], f)
                
                
            #save the scores
            update_file(file_path='Project2/Scores/Solo_Q_Fox.csv',values=scores)
            
            #reset scores
            for fox in self.foxes:
                fox.score= current_iter
     
            
    def create_animals(self,num_rabbits,num_foxes, perception_radius,Load):
        rabbit_pos = []
        for i in range(num_rabbits):
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
                    
           
            while (x,y)  in self.rocks:
                x = random.randint(0, WIDTH - 1)
                y = random.randint(0, HEIGHT - 1)
               
             
            if Load:
                print('Loading Rabbit model ...')
                # Load the model from the file
                with open("Project2/model/Solo_Q_Rabbit.pkl", "rb") as f:
                    rabbit_model = pickle.load(f)
                
                self.rabbits.append(rabbit_model)
                rabbit_pos.append(rabbit_model.pos)
            else:
                self.rabbits.append(Solo_Q_Agent_Rabbit(pos = [x,y],perception_radius=perception_radius))
                rabbit_pos.append([x,y])
            
            
        for i in range(num_foxes):
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
                    
            
            while (x,y)  in self.rocks or [x,y] in rabbit_pos:
                x = random.randint(0, WIDTH - 1)
                y = random.randint(0, HEIGHT - 1)
                
            if Load:
                print('Loading Fox model ...')
                # Load the model from the file
                with open("Project2/model/Solo_Q_Fox.pkl", "rb") as f:
                    fox_model = pickle.load(f)
                self.foxes.append(fox_model)
            else:
                self.foxes.append(Solo_Q_Agent_Fox(pos = [x,y],perception_radius=perception_radius))
                    
            
    
        
    def show(self):
        
        
        for row in range(self.rows):
            for col in range(self.columns):
                pygame.draw.rect(self.display,  (0x98, 0xFB, 0x98), (GRID_SIZE * col, GRID_SIZE * row, GRID_SIZE, GRID_SIZE))
                if [col, row] in self.rocks:  
                    world.display.blit(ROCK_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
                elif [col, row] in self.carrots:
                    world.display.blit(CARROT_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
               
        for rabbit in self.rabbits:
              [col,row] = rabbit.pos
              world.display.blit(RABBIT_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
              
        for fox in self.foxes:
              [col,row] = fox.pos
              world.display.blit(FOX_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
            

        pygame.display.update()
        
        

        
# Create the world
world = World(WIDTH, HEIGHT)
world.generate_world(noise_function=perlin_noise,rock_intensity=0.65,carrot_intensity=0.7)
world.create_animals(num_rabbits=2,num_foxes=2,perception_radius=3,Load=True)
world.show()


def main():
    current_iter =99
    pygame.init()
    clock = pygame.time.Clock()
    

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Show the world
        world.update()
        world.show()
        
        world.num_iter -=1
        if world.num_iter == 0:
            world.reset(noise_function=perlin_noise,rock_intensity=0.7,carrot_intensity=0.75,current_iter=current_iter)
            current_iter += 1
            print('starting new iteration ...')

        # Wait for the next frame
        #clock.tick(FPS)


if __name__ == '__main__':
    main()
