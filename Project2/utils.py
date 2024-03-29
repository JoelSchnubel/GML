import csv ,os
import matplotlib.pyplot as plt
import math

# run this file to see the plots


# takes as input an array of filepaths and a title for the plot 
# example is given at the bottom
# the plots are additionally stored in the Plots folder 
def plot(file_paths,title):
    plt.figure(figsize=(12, 8))    
    catches = 0
    
    for file_path in file_paths:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            values = [float(row[0]) for row in reader]
    

        for x in values:
            if x > 5 or x < -5:
                catches+=1
        
        # get the name of the file
        name = os.path.basename(file_path)[:-4]
        
        plt.plot(values, label=name)
        
    # Display counts as text on the plot
    plt.text(len(values) -10, 13, f'catches: {catches/(len(file_paths)-1)}', color='black')
    plt.text(len(values) -10, 12, f'catch_rate: {catches/len(values)}', color='black')

    plt.xlabel('n_epochs')
    plt.ylabel('reward')
    plt.title(title)
    plt.legend()
   
    # save the plot:
    path = 'Project2\\Plots\\'+str(title)+'.png'
    plt.savefig(path)
    
    plt.show()
        
        
# Ensure theta is within [0, 2*pi)
def normalize_angle(theta):
    return theta % (2 * math.pi)

# return true if [x,y] with radius r touched [other_x,other_y] with radius other_r
def check_range(x,y,r, other_x,other_y,other_r):
    distance_to_landmark = math.sqrt((x - other_x)**2 + (y - other_y)**2)
    combined_radius = r + other_r
    return int(distance_to_landmark <= combined_radius)

# collision between agent and landmark
def adjust_position(agent_pos, agent_radius, landmark_pos, landmark_radius):
    collision_vector = (agent_pos[0] - landmark_pos[0], agent_pos[1] - landmark_pos[1])
    collision_distance = agent_radius + landmark_radius
    collision_magnitude = math.sqrt(collision_vector[0]**2 + collision_vector[1]**2)

    # Normalize the collision vector
    normalized_collision_vector = (collision_vector[0] / collision_magnitude, collision_vector[1] / collision_magnitude)

    # Move the agent away from the landmark along the collision vector
    new_x = landmark_pos[0] + normalized_collision_vector[0] * collision_distance
    new_y = landmark_pos[1] + normalized_collision_vector[1] * collision_distance
    
    return new_x,new_y

# updateing a csv file by append a value
def update_file(file_path,value):
    with open (file_path,'a') as f:
        f.write(str(value) + '\n')
        

if __name__ == '__main__':    
    
    # coms vs coms
    plot(['Project2/Scores/predator0MADDPG.csv',
        'Project2/Scores/predator1MADDPG.csv',
        'Project2/Scores/prey0MADDPG.csv'
        ],'Training_coms_vs_coms')


    plot(['Project2/Scores/predator0MADDPG_test.csv',
        'Project2/Scores/predator1MADDPG_test.csv',
        'Project2/Scores/prey0MADDPG_test.csv'
        ],'Testing_coms_vs_coms')
    
        
    # simple vs simple 
    plot(['Project2/Scores/predator0MADDPG_simple.csv',
        'Project2/Scores/predator1MADDPG_simple.csv',
        'Project2/Scores/prey0MADDPG_simple.csv'
        ],'Training_simple_vs_simple')


    plot(['Project2/Scores/predator0MADDPG_simple_test.csv',
        'Project2/Scores/predator1MADDPG_simple_test.csv',
        'Project2/Scores/prey0MADDPG_simple_test.csv'
        ],'Testing_simple_vs_simple')
    
    # coms vs simple
    plot(['Project2/Scores/predator0MADDPG_coms_vs_simple.csv',
        'Project2/Scores/predator1MADDPG_coms_vs_simple.csv',
        'Project2/Scores/prey0MADDPG_coms_vs_simple.csv'
        ],'Training_coms_vs_simple')


    plot(['Project2/Scores/predator0MADDPG_coms_vs_simple_test.csv',
        'Project2/Scores/predator1MADDPG_coms_vs_simple_test.csv',
        'Project2/Scores/prey0MADDPG_coms_vs_simple_test.csv'
        ],'Testing_coms_vs_simple')
    
    # simple vs simple GD
    plot(['Project2/Scores/predator0MADDPG_simple_GD.csv',
        'Project2/Scores/predator1MADDPG_simple_GD.csv',
        'Project2/Scores/prey0MADDPG_simple_GD.csv'
        ],'Training_simple_GD')


    plot(['Project2/Scores/predator0MADDPG_simple_GD_test.csv',
        'Project2/Scores/predator1MADDPG_simple_GD_test.csv',
        'Project2/Scores/prey0MADDPG_simple_GD_test.csv'
        ],'Testing_simple_GD')
    
    # simple Adam vs GD
    plot(['Project2/Scores/predator0MADDPG_simple_adam_vs_GD.csv',
        'Project2/Scores/predator1MADDPG_simple_adam_vs_GD.csv',
        'Project2/Scores/prey0MADDPG_simple_adam_vs_GD.csv'
        ],'Training_simple_adam_vs_GD')


    plot(['Project2/Scores/predator0MADDPG_simple_adam_vs_GD_test.csv',
        'Project2/Scores/predator1MADDPG_simple_adam_vs_GD_test.csv',
        'Project2/Scores/prey0MADDPG_simple_adam_vs_GD_test.csv'
        ],'Testing_simple_adam_vs_GD')
    
    # simple GD vs adam
    plot(['Project2/Scores/predator0MADDPG_simple_GD_vs_adam.csv',
        'Project2/Scores/predator1MADDPG_simple_GD_vs_adam.csv',
        'Project2/Scores/prey0MADDPG_simple_GD_vs_adam.csv'
        ],'Training_simple_GD_vs_adam')


    plot(['Project2/Scores/predator0MADDPG_simple_GD_vs_adam_test.csv',
        'Project2/Scores/predator1MADDPG_simple_GD_vs_adam_test.csv',
        'Project2/Scores/prey0MADDPG_simple_GD_vs_adam_test.csv'
        ],'Testing_simple_GD_vs_adam')