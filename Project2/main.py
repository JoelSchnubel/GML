import pygame , random 
import numpy as np
import sys


WIDTH = 30
HEIGHT = 30
FPS = 60
GRID_SIZE = 30


ROCK_IMAGE = pygame.image.load("Assets/rock.png")
ROCK_IMAGE = pygame.transform.scale(ROCK_IMAGE,(GRID_SIZE,GRID_SIZE))

CARROT_IMAGE = pygame.image.load("Assets/carrot.png")
CARROT_IMAGE = pygame.transform.scale(CARROT_IMAGE,(GRID_SIZE,GRID_SIZE))

RABBIT_IMAGE = pygame.image.load("Assets/rabbit2.png")
RABBIT_IMAGE = pygame.transform.scale(RABBIT_IMAGE,(GRID_SIZE,GRID_SIZE))

FOX_IMAGE = pygame.image.load("Assets/fox.png")
FOX_IMAGE = pygame.transform.scale(FOX_IMAGE,(GRID_SIZE,GRID_SIZE))




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
    def __init__(self,rows,columns,rocks=[],carrots=[]):
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
        while (x,y)  in self.rocks:
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
                
            self.carrots.append([x,y])
            
            
    def show(self):
        
    
        for row in range(self.rows):
            for col in range(self.columns):
                pygame.draw.rect(self.display,  (0x98, 0xFB, 0x98), (GRID_SIZE * col, GRID_SIZE * row, GRID_SIZE, GRID_SIZE))
                if [col, row] in self.rocks:  
                    self.display.blit(ROCK_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
                #elif (col, row) in self.carrots:
                #    self.display.blit(CARROT_IMAGE, (GRID_SIZE * col, GRID_SIZE * row))
  

            

        pygame.display.update()


def main():
    pygame.init()
    clock = pygame.time.Clock()
    world = World(WIDTH, HEIGHT)
    world.generate_world(noise_function=perlin_noise,rock_intensity=0.6,carrot_intensity=0.75)


    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        
        world.show()
        

        # Wait for the next frame
        clock.tick(60)

if __name__ == "__main__":
    main()