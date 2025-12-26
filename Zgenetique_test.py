from Zgenetique2 import Population, Individue
from Zmaze_maker2 import Maze, MazeColor
import os

def clear_exploration():
    path = "logs/explorations"
    files = os.listdir(path)
    for file in files:
        os.remove(f"{path}/{file}")

def tests1():
    clear_exploration()

    fig = maze.get_fig_pixel_map(title="", show_image=show)
    fig.savefig("tests/genetique/init.png")
    

    # parametres
    Population.set_gardee(0.3)
    Population.set_generation(1000)
    Population.set_individues(200)
    Population.set_longueurs_max(20*20)
    Population.set_mutation(0.5)
    Individue.set_offset(0.05)

    population = Population(maze)
    population.simulation(picture_each_x_generation=100)



    fig = maze.get_fig_explored_full_map(title="", show_values=value, show_image=show)
    fig.savefig("tests/genetique/exploredFull.png")
    fig = maze.get_fig_pixel_map(title="", show_image=show)
    fig.savefig("tests/genetique/final.png")


if __name__ == "__main__":
        
    os.makedirs("tests", exist_ok=True)
    os.makedirs("tests/genetique", exist_ok=True)
    show = False
    value = True
    maze = Maze(20, 20)

    tests1()
    
    